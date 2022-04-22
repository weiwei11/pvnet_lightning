# Created by ww at 2021/4/28

import numpy as np

try:
    from ...csrc.nn import nn_utils

    def nearest_point_distance(model_pred, model_targets):
        idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
        dist = np.linalg.norm(model_pred[idxs] - model_targets, 2, 1)
        return dist
except ImportError:
    from scipy import spatial

    def nearest_point_distance(model_pred, model_targets):
        dist_index = spatial.cKDTree(model_pred)
        dist, _ = dist_index.query(model_targets, k=1)
        return dist


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def projection_error(pose_pred, pose_target, K, model_xyz):
    """
    2D projection error

    :param pose_pred: shape (3, 4), estimated pose
    :param pose_target: shape (3, 4), target pose
    :param K: shape (3, 3), camera intrinsic parameters
    :param model_xyz: shape (N, 3), 3D points cloud of object
    :return: float

    >>> import numpy as np
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> K = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]])
    >>> model_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> projection_error(pose_pred, pose_target, K, model_xyz)
    0.0
    """
    pred_xy = project(model_xyz, K, pose_pred)
    target_xy = project(model_xyz, K, pose_target)
    proj_mean_diff = np.mean(np.linalg.norm(pred_xy - target_xy, axis=-1))
    return proj_mean_diff


def add_error(pose_pred, pose_target, model_xyz, symmetric=False):
    """
    ADD error, and computed faster by using gpu

    :param pose_pred: shape (3, 4), estimated pose
    :param pose_target: shape (3, 4), target pose
    :param model_xyz: shape (N, 3), 3D points cloud of object
    :param symmetric: whether the object is symmetric or not
    :return: float

    >>> import numpy as np
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> model_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> add_error(pose_pred, pose_target, model_xyz, True)
    0.0
    """
    model_pred = np.dot(model_xyz, pose_pred[:, :3].T) + pose_pred[:, 3]
    model_target = np.dot(model_xyz, pose_target[:, :3].T) + pose_target[:, 3]

    if symmetric:
        mean_dist = np.mean(nearest_point_distance(model_pred, model_target))
    else:
        mean_dist = np.mean(np.linalg.norm(model_pred - model_target, axis=-1))
    return mean_dist


def angular_error(pose_pred, pose_target):
    """
    Angular error for rotation matrix

    :param pose_pred: shape (3, 4), estimated pose
    :param pose_target: shape (3, 4), target pose
    :return: float

    >>> import numpy as np
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> angular_error(pose_pred, pose_target)
    0.0
    """
    rotation_diff = np.dot(pose_pred[:, :3], pose_target[:, :3].T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    return np.rad2deg(np.arccos((trace - 1.) / 2.0))


def translation_error(pose_pred, pose_target):
    """
    Angular error for rotation matrix

    :param pose_pred: shape (3, 4), estimated pose
    :param pose_target: shape (3, 4), target pose
    :return: float

    >>> import numpy as np
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> translation_error(pose_pred, pose_target)
    0.0
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 2]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> translation_error(pose_pred, pose_target)
    1.0
    """
    return np.linalg.norm(pose_pred[:, 3] - pose_target[:, 3])


def projection_2d(pose_pred, pose_target, K, model_xyz, threshold=5):
    """
    2D projection error

    :param pose_pred: shape (3, 4), estimated pose
    :param pose_target: shape (3, 4), target pose
    :param K: shape (3, 3), camera intrinsic parameters
    :param model_xyz: shape (N, 3), 3D points cloud of object
    :param threshold:
    :return: bool

    >>> import numpy as np
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> K = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]])
    >>> model_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> projection_2d(pose_pred, pose_target, K, model_xyz)
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

    >>> import numpy as np
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> model_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> model_diameter = np.sqrt(3)
    >>> add(pose_pred, pose_target, model_xyz, model_diameter, True)
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

    >>> import numpy as np
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> cm_degree(pose_pred, pose_target)
    True
    >>> import numpy as np
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 2]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> cm_degree(pose_pred, pose_target)
    False
    """
    return translation_error(pose_pred, pose_target) < 0.01 * cm_threshold and angular_error(pose_pred, pose_target) < degree_threshold


def auc(x, y, x_max_limit=1.0, y_max_limit=1.0):
    """
    Area Under Curve

    :param x: shape (n, )
    :param y: shape (n, )
    :param x_max_limit: max limit value of x axis
    :param y_max_limit: max limit value of y axis
    :return:

    >>> x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])
    >>> y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])
    >>> auc(x, y, 1.0, 1.0)
    0.5399999999999999
    >>> x = np.array([0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])
    >>> y = np.array([0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])
    >>> auc(x, y, 1.0, 1.0)
    0.5399999999999999
    >>> x = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1])
    >>> y = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.1])
    >>> auc(x, y, 1.0, 1.0)
    0.5499999999999999
    >>> x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])
    >>> y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])
    >>> auc(x, y, 1.0, 2.0)
    0.26999999999999996
    """
    # see https://github.com/yuxng/YCB_Video_toolbox/blob/master/plot_accuracy_keyframe.m
    # remove out range
    mask = x <= x_max_limit
    x = x[mask]
    y = y[mask]

    if len(y) == 0:
        return 0

    mx = np.concatenate([[0.0], x, [x_max_limit]])
    my = np.concatenate([[0.0], y, [y[-1]]])

    for i in range(1, len(my)):
        my[i] = max(my[i], my[i - 1])

    i = np.where(mx[1:] != mx[0:-1])[0] + 1
    ap = np.sum((mx[i] - mx[i - 1]) * my[i]) / x_max_limit / y_max_limit
    return ap


def add_auc(add_errors, max_dist_thre, unit_scale=1.0):
    """
    ADD AUC

    :param add_errors: add error array of samples
    :param max_dist_thre: max error threshold, so threshold is [0, max]
    :param unit_scale: scale difference between add error of sample and threshold, likes mm and cm
    :return:

    >>> x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1])
    >>> add_auc(x, 1.0, 1.0)
    0.5399999999999999
    """
    add_errors = np.sort(add_errors) * unit_scale
    accuracy = np.cumsum(np.ones_like(add_errors)) / len(add_errors)
    return auc(add_errors, accuracy, max_dist_thre, 1.0)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
