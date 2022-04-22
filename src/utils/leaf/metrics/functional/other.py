# Author: weiwei
import numpy as np


def nn_ap(match_query_kpt, match_ref_id, gt_query_kpt, gt_ref_id, dist_threshold):
    """
    Nearest neighbor average precise

    :param match_query_kpt: shape (M, 2)
    :param match_ref_id: shape (M, )
    :param gt_query_kpt: shape (N, 2)
    :param gt_ref_id: shape (N, )
    :param dist_threshold: float
    :return:

    >>> import numpy as np
    >>> match_query_kpt = np.array([[0, 0], [1, 1]])
    >>> match_ref_id = np.array([0, 2])
    >>> gt_query_kpt = np.array([[0, 0], [1, 0], [0, 1]])
    >>> gt_ref_id = np.array([0, 1, 2])
    >>> gt_query_kpt1 = np.array([[9, 9]])
    >>> gt_ref_id1 = np.array([2])
    >>> nn_ap(match_query_kpt, match_ref_id, gt_query_kpt, gt_ref_id, 0.1)
    0.3333333333333333
    >>> nn_ap(match_query_kpt, match_ref_id, gt_query_kpt, gt_ref_id, 1)
    0.6666666666666666
    >>> nn_ap(match_query_kpt, match_ref_id, gt_query_kpt, gt_ref_id, 2)
    0.6666666666666666
    >>> nn_ap(match_query_kpt, match_ref_id, gt_query_kpt1, gt_ref_id1, 2)
    0.0
    >>> nn_ap(match_query_kpt, match_ref_id, gt_query_kpt1, gt_ref_id1, 20)
    0.5
    """
    # dist = np.linalg.norm(kpt2d - kpt2d_gt[match_idx, :], axis=1)
    # return np.sum(dist <= dist_threshold) / len(kpt2d_gt)

    pred_idx, gt_idx = np.where(match_ref_id[:, None] == gt_ref_id[None, :])
    correct_num = np.sum(np.linalg.norm(match_query_kpt[pred_idx] - gt_query_kpt[gt_idx], axis=1) <= dist_threshold)
    ratio = correct_num / len(np.union1d(match_ref_id, gt_ref_id))
    return ratio


def box2d_iou(box_pred, box_gt):
    """
    Iou for 2D bounding box

    :param box_pred: [x, y, x, y]
    :param box_gt: [x, y, x, y]
    :return: float

    >>> a = [0, 0, 1, 1]
    >>> b = [0, 0, 1, 1]
    >>> box2d_iou(a, b)
    1.0
    >>> a = [0, 0, 1, 1]
    >>> b = [0, 0, 1, 2]
    >>> box2d_iou(a, b)
    0.5
    >>> a = [0, 0, 1, 1]
    >>> b = [1, 1, 2, 2]
    >>> box2d_iou(a, b)
    0.0
    >>> a = [0, 0, 1, 2]
    >>> b = [0, 1, 1, 3]
    >>> box2d_iou(a, b)
    0.3333333333333333
    """
    low_x, low_y = max(box_pred[0], box_gt[0]), max(box_pred[1], box_gt[1])
    high_x, high_y = min(box_pred[2], box_gt[2]), min(box_pred[3], box_gt[3])
    w, h = high_x - low_x, high_y - low_y
    inter = w * h if w > 0 and h > 0 else 0
    union = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1]) + (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1]) - inter
    if union == 0:
        iou = 0
    else:
        iou = inter / union
    return iou