# Author: weiwei
import math
from math import floor

import numpy as np

from ..matrix_transforms import affine_transform as atf


def resize(im_size, new_size):
    """

    :param im_size: (height, width)
    :param new_size: (height, width)
    :return: m is transformation matrix, new_size is new image size
    """
    old_h, old_w = im_size
    new_h, new_w = new_size

    m = atf.scale2matrix([new_w / old_w, new_h / old_h])
    return m, new_size


def add_pad_to_aspect_ratio(im_size, aspect_ratio):
    """

    :param im_size: (height, width)
    :param aspect_ratio: (h, w)
    :return: m is transformation matrix, new_size is new image size
    """
    cur_h, cur_w = im_size

    # compute pad
    h_scale = aspect_ratio[0]
    w_scale = aspect_ratio[1]
    if cur_h / h_scale > cur_w / w_scale:
        new_height = cur_h
        new_width = floor(cur_h / h_scale * w_scale)
        h_pad = 0
        w_pad = (new_width - cur_w) // 2
    else:
        new_height = floor(cur_w / w_scale * h_scale)
        new_width = cur_w
        h_pad = (new_height - cur_h) // 2
        w_pad = 0

    offset = np.array([w_pad, h_pad])
    m = atf.translation2matrix(offset)
    new_size = [new_height, new_width]
    return m, new_size


def crop(im_size, crop_box):
    """

    :param im_size: (height, width)
    :param crop_box: (x1, y1, x2, y2)
    :return: m is transformation matrix, new_size is new image size
    """
    cur_h, cur_w = im_size
    x_beg, y_beg, x_end, y_end = crop_box.tolist() if isinstance(crop_box, np.ndarray) else crop_box

    assert 0 <= x_beg <= cur_w - 1
    assert 0 <= x_end <= cur_w - 1
    assert 0 <= y_beg <= cur_h - 1
    assert 0 <= y_beg <= cur_h - 1
    assert x_beg < x_end and y_beg < y_end

    m = atf.translation2matrix([-x_beg, -y_beg])
    new_size = [y_end - y_beg + 1, x_end - x_beg + 1]
    return m, new_size


def crop_v2(im_size, crop_box):
    """

    :param im_size: (height, width)
    :param crop_box: (x1, y1, x2, y2)
    :return: m is transformation matrix, new_size is new image size
    """
    x_beg, y_beg, x_end, y_end = crop_box.tolist() if isinstance(crop_box, np.ndarray) else crop_box

    assert x_beg < x_end and y_beg < y_end

    m = atf.translation2matrix([-x_beg, -y_beg])
    new_size = [y_end - y_beg + 1, x_end - x_beg + 1]
    return m, new_size


def flip(im_size):
    """

    :param im_size: (height, width)
    :return: m is transformation matrix, new_size is new image size
    """
    cur_h, cur_w = im_size
    m = atf.reflect2matrix([0, 1], [cur_w // 2, 0])
    return m, im_size


def resize_random(im_size, ratio_min, ratio_max):
    """

    :param im_size: (height, width)
    :param ratio_min:
    :param ratio_max:
    :return: m is transformation matrix, new_size is new image size
    """
    ratio = np.random.uniform(ratio_min, ratio_max)
    new_h, new_w = int(ratio * im_size[0]), int(ratio * im_size[1])
    m = atf.scale2matrix([ratio, ratio])
    return m, [new_h, new_w]


def zoom_random(im_size, zoom_ratio_min, zoom_ratio_max, center=None):
    """

    :param im_size: (height, width)
    :param zoom_ratio_min:
    :param zoom_ratio_max:
    :param center: (x, y)
    :return: m is transformation matrix, new_size is new image size
    """
    center = [im_size[1] / 2, im_size[0] / 2] if center is None else center
    center = np.array(center)
    zoom_ratio = np.random.uniform(zoom_ratio_min, zoom_ratio_max)
    m1 = atf.translation2matrix(-center)
    m2 = atf.translation2matrix(center)
    m = m2 @ atf.scale2matrix([zoom_ratio, zoom_ratio]) @ m1
    return m, im_size


def rotate_random(im_size, rot_ang_min, rot_ang_max, center=None):
    """

    :param im_size: (height, width)
    :param rot_ang_min: degree
    :param rot_ang_max: degree
    :param center: (x, y)
    :return: m is transformation matrix, new_size is new image size
    """
    center = [im_size[1] / 2, im_size[0] / 2] if center is None else center
    degree = np.random.uniform(rot_ang_min, rot_ang_max)
    m = atf.rotation2matrix(degree, center)
    return m, im_size


def crop_random(im_size, overlap_ratio=0.8):
    """

    :param im_size: (height, width)
    :param overlap_ratio:
    :return: m is transformation matrix, new_size is new image size
    """
    ratio = math.sqrt(overlap_ratio)
    cur_h, cur_w = im_size
    min_h, min_w = int(cur_h * ratio), int(cur_w * ratio)
    h_min, h_max = 0, cur_h - min_h
    w_min, w_max = 0, cur_w - min_w

    h_beg = np.random.randint(h_min, h_max)
    w_beg = np.random.randint(w_min, w_max)
    h_end = np.random.randint(h_beg + min_h - 1, cur_h)
    w_end = np.random.randint(w_beg + min_w - 1, cur_w)

    return crop(im_size, np.array([w_beg, h_beg, w_end, h_end]))


def crop_random_v2(im_size, new_size):
    """

    :param im_size: (height, width)
    :param new_size: (height, width)
    :return: m is transformation matrix, new_size is new image size
    """
    cur_h, cur_w = im_size
    new_h, new_w = new_size
    h_min, h_max = 0, cur_h - new_h
    w_min, w_max = 0, cur_w - new_w

    h_beg = np.random.randint(h_min, h_max + 1)
    w_beg = np.random.randint(w_min, w_max + 1)
    h_end = h_beg + new_h - 1
    w_end = w_beg + new_w - 1

    return crop(im_size, np.array([w_beg, h_beg, w_end, h_end]))


def rotate_instance_random(im_size, rot_ang_min, rot_ang_max, obj_xs, obj_ys):
    """

    :param im_size: (height, width)
    :param rot_ang_min: degree
    :param rot_ang_max: degree
    :param obj_xs:
    :param obj_ys:
    :return: m is transformation matrix, new_size is new image size
    """
    obj_center = [np.mean(obj_xs), np.mean(obj_ys)]
    return rotate_random(im_size, rot_ang_min, rot_ang_max, obj_center)


def crop_instance_random(im_size, new_size, overlap_ratio, obj_xs, obj_ys):
    """

    :param im_size: (height, width)
    :param new_size: (height, width)
    :param overlap_ratio:
    :param obj_xs:
    :param obj_ys:
    :return: m is transformation matrix, new_size is new image size
    """
    obj_box = np.array([np.min(obj_xs), np.min(obj_ys), np.max(obj_xs), np.max(obj_ys)])  # [w1, h1, w2, h2]
    cur_h, cur_w = im_size
    new_h, new_w = new_size
    obj_h, obj_w = obj_box[3] - obj_box[1], obj_box[2] - obj_box[0]
    cur_h_max, cur_h_min = max(obj_box[3], cur_h), min(obj_box[1], 0)
    cur_w_max, cur_w_min = max(obj_box[2], cur_w), min(obj_box[0], 0)

    assert new_h <= cur_h and new_w <= cur_w

    ratio = math.sqrt(overlap_ratio)
    over_h, over_w = int(obj_h * ratio), int(obj_w * ratio)

    if new_h < over_h:  # new image is smaller than overlap_ratio size, so sample inside object
        h_min, h_max = obj_box[1], obj_box[3] - new_h
    elif new_h < obj_h:  # new image is smaller than object, so sample around object
        # h_min = max(obj_box[1] + over_h - new_h, 0)
        # h_max = min(obj_box[3] - over_h, cur_h - new_h)
        h_min = max(obj_box[1] + over_h - new_h, cur_h_min)
        h_max = min(obj_box[3] - over_h, cur_h_max - new_h)
    else:  # new image is bigger than object, so sample, so sample around object
        # h_min = max(obj_box[1] + over_h - new_h, 0)
        # h_max = min(obj_box[3] - over_h, cur_h - new_h)
        h_min = max(obj_box[1] + over_h - new_h, cur_h_min)
        h_max = min(obj_box[3] - over_h, cur_h_max - new_h)

    if new_w < over_w:  # new image is smaller than overlap_ratio size, so sample inside object
        w_min, w_max = obj_box[0], obj_box[2] - new_w
    elif new_w < obj_w:  # new image is smaller than object, so sample around object
        # w_min = max(obj_box[0] + over_w - new_w, 0)
        # w_max = min(obj_box[2] - over_w, cur_w - new_w)
        w_min = max(obj_box[0] + over_w - new_w, cur_w_min)
        w_max = min(obj_box[2] - over_w, cur_w_max - new_w)
    else:  # new image is bigger than object, so sample, so sample around object
        # w_min = max(obj_box[0] + over_w - new_w, 0)
        # w_max = min(obj_box[2] - over_w, cur_w - new_w)
        w_min = max(obj_box[0] + over_w - new_w, cur_w_min)
        w_max = min(obj_box[2] - over_w, cur_w_max - new_w)

    # random sample
    h_beg = np.random.randint(h_min, h_max+1)
    w_beg = np.random.randint(w_min, w_max+1)
    h_end = h_beg + new_h - 1
    w_end = w_beg + new_w - 1

    return crop_v2(im_size, np.array([w_beg, h_beg, w_end, h_end]))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2
    from aug_transforms import transform_bboxes

    fig = plt.figure(figsize=(19.20, 10.80))
    n_rows, n_cols = 4, 4
    index = 0

    im_size = [240, 320]
    obj_box = np.array([50, 30, 100-1, 150-1], dtype=np.uint8)
    ori_boxes = obj_box.reshape(1, -1)
    ori_img = np.zeros(im_size)
    ori_img[obj_box[1]:obj_box[3]+1, obj_box[0]:obj_box[2]+1] = 255
    title = 'ori_img'
    index += 1
    plt.subplot(n_rows, n_cols, index)
    plt.imshow(ori_img)
    plt.title(title)
    plt.plot(ori_boxes[:, [0, 2]].flatten(), ori_boxes[:, [1, 3]].flatten(), '-r')

    m, new_size = resize(im_size, (480, 640))
    new_img = cv2.warpAffine(ori_img, m[:2, :], tuple(new_size[::-1]))
    title = 'resize'
    index += 1
    plt.subplot(n_rows, n_cols, index)
    plt.imshow(new_img)
    plt.title(title)
    boxes = transform_bboxes(ori_boxes, m)
    plt.plot(boxes[:, [0, 2]].flatten(), boxes[:, [1, 3]].flatten(), '-r')

    m, new_size = add_pad_to_aspect_ratio(im_size, (1, 1))
    new_img = cv2.warpAffine(ori_img, m[:2, :], tuple(new_size[::-1]))
    title = 'add_pad_to_aspect_ratio'
    index += 1
    plt.subplot(n_rows, n_cols, index)
    plt.imshow(new_img)
    plt.title(title)
    boxes = transform_bboxes(ori_boxes, m)
    plt.plot(boxes[:, [0, 2]].flatten(), boxes[:, [1, 3]].flatten(), '-r')

    m, new_size = crop(im_size, obj_box)
    new_img = cv2.warpAffine(ori_img, m[:2, :], tuple(new_size[::-1]))
    title = 'crop'
    index += 1
    plt.subplot(n_rows, n_cols, index)
    plt.imshow(new_img)
    plt.title(title)
    boxes = transform_bboxes(ori_boxes, m)
    plt.plot(boxes[:, [0, 2]].flatten(), boxes[:, [1, 3]].flatten(), '-r')

    m, new_size = flip(im_size)
    new_img = cv2.warpAffine(ori_img, m[:2, :], tuple(new_size[::-1]))
    title = 'flip'
    index += 1
    plt.subplot(n_rows, n_cols, index)
    plt.imshow(new_img)
    plt.title(title)
    boxes = transform_bboxes(ori_boxes, m)
    plt.plot(boxes[:, [0, 2]].flatten(), boxes[:, [1, 3]].flatten(), '-r')

    m, new_size = resize_random(im_size, 0.5, 1.5)
    new_img = cv2.warpAffine(ori_img, m[:2, :], tuple(new_size[::-1]))
    title = 'resize_random'
    index += 1
    plt.subplot(n_rows, n_cols, index)
    plt.imshow(new_img)
    plt.title(title)
    boxes = transform_bboxes(ori_boxes, m)
    plt.plot(boxes[:, [0, 2]].flatten(), boxes[:, [1, 3]].flatten(), '-r')

    m, new_size = zoom_random(im_size, 0.5, 1.5)
    new_img = cv2.warpAffine(ori_img, m[:2, :], tuple(new_size[::-1]))
    title = 'zoom_random'
    index += 1
    plt.subplot(n_rows, n_cols, index)
    plt.imshow(new_img)
    plt.title(title)
    boxes = transform_bboxes(ori_boxes, m)
    plt.plot(boxes[:, [0, 2]].flatten(), boxes[:, [1, 3]].flatten(), '-r')

    m, new_size = rotate_random(im_size, -30, 30)
    new_img = cv2.warpAffine(ori_img, m[:2, :], tuple(new_size[::-1]))
    title = 'rotate_random'
    index += 1
    plt.subplot(n_rows, n_cols, index)
    plt.imshow(new_img)
    plt.title(title)
    boxes = transform_bboxes(ori_boxes, m)
    plt.plot(boxes[:, [0, 2]].flatten(), boxes[:, [1, 3]].flatten(), '-r')

    m, new_size = crop_random(im_size, 0.6)
    new_img = cv2.warpAffine(ori_img, m[:2, :], tuple(new_size[::-1]))
    title = 'crop_random'
    index += 1
    plt.subplot(n_rows, n_cols, index)
    plt.imshow(new_img)
    plt.title(title)
    boxes = transform_bboxes(ori_boxes, m)
    plt.plot(boxes[:, [0, 2]].flatten(), boxes[:, [1, 3]].flatten(), '-r')

    m, new_size = rotate_instance_random(im_size, -30, 30, obj_box[[0, 2]], obj_box[[1, 3]])
    new_img = cv2.warpAffine(ori_img, m[:2, :], tuple(new_size[::-1]))
    title = 'rotate_instance_random'
    index += 1
    plt.subplot(n_rows, n_cols, index)
    plt.imshow(new_img)
    plt.title(title)
    boxes = transform_bboxes(ori_boxes, m)
    plt.plot(boxes[:, [0, 2]].flatten(), boxes[:, [1, 3]].flatten(), '-r')

    m, new_size = crop_instance_random(im_size, [200, 300], 0.6, obj_box[[0, 2]], obj_box[[1, 3]])
    new_img = cv2.warpAffine(ori_img, m[:2, :], tuple(new_size[::-1]))
    title = 'crop_instance_random'
    index += 1
    plt.subplot(n_rows, n_cols, index)
    plt.imshow(new_img)
    plt.title(title)
    boxes = transform_bboxes(ori_boxes, m)
    plt.plot(boxes[:, [0, 2]].flatten(), boxes[:, [1, 3]].flatten(), '-r')

    m, new_size = crop_instance_random(im_size, [100, 40], 0.6, obj_box[[0, 2]], obj_box[[1, 3]])
    new_img = cv2.warpAffine(ori_img, m[:2, :], tuple(new_size[::-1]))
    title = 'crop_instance_random'
    index += 1
    plt.subplot(n_rows, n_cols, index)
    plt.imshow(new_img)
    plt.title(title)
    boxes = transform_bboxes(ori_boxes, m)
    plt.plot(boxes[:, [0, 2]].flatten(), boxes[:, [1, 3]].flatten(), '-r')

    m, new_size = crop_instance_random(im_size, [70, 20], 0.6, obj_box[[0, 2]], obj_box[[1, 3]])
    new_img = cv2.warpAffine(ori_img, m[:2, :], tuple(new_size[::-1]))
    title = 'crop_instance_random'
    index += 1
    plt.subplot(n_rows, n_cols, index)
    plt.imshow(new_img)
    plt.title(title)
    boxes = transform_bboxes(ori_boxes, m)
    plt.plot(boxes[:, [0, 2]].flatten(), boxes[:, [1, 3]].flatten(), '-r')

    plt.show()
