# Created by ww at 2021/4/28

import torch


def nearest_point_distance(model_pred, model_targets):
    x_square = model_pred.square().sum(dim=-1)
    y_square = model_targets.square().sum(dim=-1)
    dist = torch.abs(x_square - 2 * model_pred @ model_targets.transpose(-1, -2) + y_square.transpose(-1, -2)).min(dim=-1)[0]
    return dist


def project(xyz, K, RT):
    """
    xyz: [3, N, 3] or [N, 3]
    K: [b, 3, 3]
    RT: [b, 3, 4]
    """
    # xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    # xyz = np.dot(xyz, K.T)
    # xy = xyz[:, :2] / xyz[:, 2:]
    if len(xyz.shape) == 2:
        xyz = xyz.unsqueeze(0)
    xyz = torch.matmul(xyz, RT[..., :3].transpose(-1, -2)) + RT[..., 3:].transpose(-1, -2)
    xyz = torch.matmul(xyz, K.transpose(-1, -2))
    xy = xyz[..., :2] / xyz[..., 2:]
    return xy


def projection_error(pose_pred, pose_target, K, model_xyz):
    """
    2D projection error

    :param pose_pred: shape (b, 3, 4), estimated pose
    :param pose_target: shape (b, 3, 4), target pose
    :param K: shape (b, 3, 3), camera intrinsic parameters
    :param model_xyz: shape (N, 3) or (b, N, 3), 3D points cloud of object
    :return: float

    >>> import torch
    >>> pose_pred = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]).unsqueeze(0)
    >>> pose_target = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]).unsqueeze(0)
    >>> K = torch.tensor([[320, 0, 320], [0, 320, 240], [0, 0, 1]]).unsqueeze(0)
    >>> model_xyz = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).unsqueeze(0)
    >>> projection_error(pose_pred, pose_target, K, model_xyz).item()
    0.0
    """
    pred_xy = project(model_xyz, K, pose_pred)
    target_xy = project(model_xyz, K, pose_target)
    proj_mean_diff = torch.mean(torch.norm(pred_xy - target_xy, dim=-1), dim=-1)
    return proj_mean_diff


def add_error(pose_pred, pose_target, model_xyz, symmetric=False):
    """
    ADD error, and computed faster by using gpu

    :param pose_pred: shape (b, 3, 4), estimated pose
    :param pose_target: shape (b, 3, 4), target pose
    :param model_xyz: shape (N, 3) or (b, N, 3), 3D points cloud of object
    :param symmetric: whether the object is symmetric or not
    :return: float

    >>> import torch
    >>> pose_pred = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1.0]]).unsqueeze(0)
    >>> pose_target = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1.0]]).unsqueeze(0)
    >>> K = torch.tensor([[320, 0, 320], [0, 320, 240], [0, 0, 1.0]]).unsqueeze(0)
    >>> model_xyz = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]).unsqueeze(0)
    >>> add_error(pose_pred, pose_target, model_xyz, True).item()
    0.0
    >>> add_error(pose_pred, pose_target, model_xyz, False).item()
    0.0
    """
    if len(model_xyz.shape) == 2:
        model_xyz = model_xyz.unsqueeze(0)

    model_pred = torch.matmul(model_xyz, pose_pred[..., :3].transpose(-1, -2)) + pose_pred[..., 3].unsqueeze(1)
    model_target = torch.matmul(model_xyz, pose_target[..., :3].transpose(-1, -2)) + pose_target[..., 3].unsqueeze(1)

    if symmetric:
        mean_dist = torch.mean(nearest_point_distance(model_pred, model_target), dim=-1)
    else:
        mean_dist = torch.mean(torch.norm(model_pred - model_target, dim=-1), dim=-1)
    return mean_dist


def angular_error(pose_pred, pose_target):
    """
    Angular error for rotation matrix

    :param pose_pred: shape (b, 3, 4), estimated pose
    :param pose_target: shape (b, 3, 4), target pose
    :return: float

    >>> import torch
    >>> pose_pred = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1.0]]).unsqueeze(0)
    >>> pose_target = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1.0]]).unsqueeze(0)
    >>> angular_error(pose_pred, pose_target).item()
    0.0
    """
    rotation_diff = torch.matmul(pose_pred[..., :3], pose_target[..., :3].transpose(-1, -2))
    degree_err = torch.zeros((rotation_diff.shape[0], ), device=rotation_diff.device, dtype=rotation_diff.dtype)
    for i in range(len(rotation_diff)):
        trace = torch.trace(rotation_diff[i])
        trace = trace if trace <= 3 else 3
        degree = torch.rad2deg(torch.arccos((trace - 1.) / 2.0))
        degree_err[i] = degree

    return degree_err


def translation_error(pose_pred, pose_target):
    """
    Angular error for rotation matrix

    :param pose_pred: shape (b, 3, 4), estimated pose
    :param pose_target: shape (b, 3, 4), target pose
    :return: float

    >>> import torch
    >>> pose_pred = torch.tensor([[1.0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]).unsqueeze(0)
    >>> pose_target = torch.tensor([[1.0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]).unsqueeze(0)
    >>> translation_error(pose_pred, pose_target).item()
    0.0
    >>> pose_pred = torch.tensor([[1.0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 2]]).unsqueeze(0)
    >>> pose_target = torch.tensor([[1.0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]).unsqueeze(0)
    >>> translation_error(pose_pred, pose_target).item()
    1.0
    """
    return torch.norm(pose_pred[..., 3] - pose_target[..., 3], dim=-1)


def projection_2d(pose_pred, pose_target, K, model_xyz, threshold=5):
    """
    2D projection error

    :param pose_pred: shape (b, 3, 4), estimated pose
    :param pose_target: shape (b, 3, 4), target pose
    :param K: shape (b, 3, 3), camera intrinsic parameters
    :param model_xyz: shape (N, 3) or (b, N, 3), 3D points cloud of object
    :param threshold:
    :return: bool

    >>> import torch
    >>> pose_pred = torch.tensor([[1.0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]).unsqueeze(0)
    >>> pose_target = torch.tensor([[1.0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]).unsqueeze(0)
    >>> K = torch.tensor([[320, 0, 320], [0, 320, 240], [0, 0, 1.0]]).unsqueeze(0)
    >>> model_xyz = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]).unsqueeze(0)
    >>> projection_2d(pose_pred, pose_target, K, model_xyz).item()
    True
    """
    return projection_error(pose_pred, pose_target, K, model_xyz) < threshold


def add(pose_pred, pose_target, model_xyz, symmetric=False, threshold=None, model_diameter=None, percentage=0.1):
    """
    ADD error

    :param pose_pred: shape (3, 4), estimated pose
    :param pose_target: shape (3, 4), target pose
    :param model_xyz: shape (N, 3), 3D points cloud of object
    :param symmetric: whether the object is symmetric or not
    :param threshold: distance threshold, 'threshold' and 'model_diameter percentage' is not compatible
    :param model_diameter: the diameter of object
    :param percentage: percentage of model diameter
    :return: bool

    >>> import torch
    >>> import math
    >>> pose_pred = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1.0]]).unsqueeze(0)
    >>> pose_target = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1.0]]).unsqueeze(0)
    >>> K = torch.tensor([[320, 0, 320], [0, 320, 240], [0, 0, 1.0]]).unsqueeze(0)
    >>> model_xyz = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]).unsqueeze(0)
    >>> model_diameter = math.sqrt(3)
    >>> add(pose_pred, pose_target, model_xyz, model_diameter, True).item()
    True
    """
    threshold = model_diameter * percentage if threshold is None else threshold
    return add_error(pose_pred, pose_target, model_xyz, symmetric) < threshold


def cm_degree(pose_pred, pose_target, cm_threshold=5, degree_threshold=5):
    """
    degree and cm error

    :param pose_pred: shape (3, 4), estimated pose
    :param pose_target: shape (3, 4), target pose
    :param cm_threshold: unit is centimeter
    :param degree_threshold:
    :return: bool

    >>> import torch
    >>> pose_pred = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1.0]]).unsqueeze(0)
    >>> pose_target = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1.0]]).unsqueeze(0)
    >>> cm_degree(pose_pred, pose_target).item()
    True
    >>> pose_pred = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 2.0]]).unsqueeze(0)
    >>> pose_target = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1.0]]).unsqueeze(0)
    >>> cm_degree(pose_pred, pose_target).item()
    False
    """
    return torch.logical_and(translation_error(pose_pred, pose_target) < 0.01 * cm_threshold, angular_error(pose_pred, pose_target) < degree_threshold)


def auc(x, y, x_max_limit=1.0, y_max_limit=1.0):
    """
    Area Under Curve

    :param x: shape (n, )
    :param y: shape (n, )
    :param x_max_limit: max limit value of x axis
    :param y_max_limit: max limit value of y axis
    :return:

    >>> x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])
    >>> y = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])
    >>> auc(x, y, 1.0, 1.0).item()
    0.5399999618530273
    >>> x = torch.tensor([0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])
    >>> y = torch.tensor([0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])
    >>> auc(x, y, 1.0, 1.0).item()
    0.5399999618530273
    >>> x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1])
    >>> y = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1])
    >>> auc(x, y, 1.0, 1.0).item()
    0.5499999523162842
    >>> x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])
    >>> y = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])
    >>> auc(x, y, 1.0, 2.0).item()
    0.26999998092651367
    """
    # see https://github.com/yuxng/YCB_Video_toolbox/blob/master/plot_accuracy_keyframe.m
    # remove out range
    mask = x <= x_max_limit
    x = x[mask]
    y = y[mask]

    if len(y) == 0:
        return 0

    mx = torch.cat([torch.tensor([0.0]), x, torch.tensor([x_max_limit])])
    my = torch.cat([torch.tensor([0.0]), y, torch.tensor([y[-1]])])

    for i in range(1, len(my)):
        my[i] = max(my[i], my[i - 1])

    i = torch.where((mx[1:] - mx[0:-1]) != 0.0)[0] + 1
    ap = torch.sum((mx[i] - mx[i - 1]) * my[i]) / x_max_limit / y_max_limit
    return ap


def add_auc(add_errors, max_dist_thre, unit_scale=1.0):
    """
    ADD AUC

    :param add_errors: add error array of samples
    :param max_dist_thre: max error threshold, so threshold is [0, max]
    :param unit_scale: scale difference between add error of sample and threshold, likes mm and cm
    :return:

    >>> x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])
    >>> add_auc(x, 1.0, 1.0).item()
    0.5399999618530273
    """
    add_errors = torch.sort(add_errors, dim=-1)[0] * unit_scale
    accuracy = torch.cumsum(torch.ones_like(add_errors), dim=-1) / len(add_errors)
    return auc(add_errors, accuracy, max_dist_thre, 1.0)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
