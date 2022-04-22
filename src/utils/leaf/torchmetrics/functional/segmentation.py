# Author: weiwei


def mask_iou(mask_pred, mask_gt):
    """
    Intersection over Union for mask of image

    :param mask_pred: shape (H, W), estimated mask
    :param mask_gt: shape (H, W), target mask
    :return: float

    >>> import numpy as np
    >>> mask_pred = np.array([[1, 0], [0, 1]], dtype=np.bool_)
    >>> mask_gt = np.array([[1, 1], [1, 1]], dtype=np.bool_)
    >>> mask_iou(mask_pred, mask_gt)
    0.5
    """
    return (mask_pred & mask_gt).sum() / (mask_pred | mask_gt).sum()


if __name__ == '__main__':
    import doctest

    doctest.testmod()
