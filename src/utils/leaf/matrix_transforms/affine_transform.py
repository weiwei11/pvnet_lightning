# Author: weiwei


import numpy as np


# see https://zhuanlan.zhihu.com/p/80852438


def translation2matrix(translation):
    """

    :param translation: np.array([x y])
    :return: matrix shape is (3, 3)
    """
    m = np.eye(3)
    m[0, 2] = translation[0]
    m[1, 2] = translation[1]
    return m


def scale2matrix(scale):
    """

    :param scale: np.array([x y])
    :return: matrix shape is (3, 3)
    """
    m = np.eye(3)
    m[0, 0] = scale[0]
    m[1, 1] = scale[1]
    return m


def rotation2matrix(degree, center):
    """

    :param degree: degree of angle (0 ~ 360) or (-180 ~ 180)
    :param center: rotation center, np.array([x, y])
    :return: matrix shape is (3, 3)
    """
    center = np.array(center)
    m = np.eye(3)
    d = degree / 180.0 * np.pi
    m[0, 0] = np.cos(d)
    m[0, 1] = -np.sin(d)
    m[1, 0] = np.sin(d)
    m[1, 1] = np.cos(d)
    return translation2matrix(center) @ m @ translation2matrix(-center)


def reflect2matrix(direction, offset):
    """

    :param direction: direction of reflection axis
    :param offset: offset of reflection axis to original point
    :return: matrix shape is (3, 3)
    """
    m = np.eye(3)
    m[1, 1] = -1
    degree = np.arccos(direction[0] / np.linalg.norm(direction)) / np.pi * 180
    center = np.array([0, 0])

    m_t1 = translation2matrix(-np.array(offset))
    m_t2 = translation2matrix(np.array(offset))
    m_r1 = rotation2matrix(-degree, center)
    m_r2 = rotation2matrix(degree, center)
    return m_t2 @ m_r2 @ m @ m_r1 @ m_t1


def shear2matrix(dx, dy, x_limit=1, y_limit=1):
    """

    :param dx: offset of x
    :param dy: offset of y
    :param x_limit: max value of x
    :param y_limit: max value of y
    :return: matrix shape is (3, 3)
    """
    m = np.eye(3)
    dx = dx / y_limit
    dy = dy / x_limit
    m[0, 1] = dx
    m[1, 0] = dy
    return m


def affine_transform(points, t):
    """

    :param points: shape is (n, 2)
    :param t: shape is (2, 3) or (3, 3)
    :return:
    """
    new_points = np.dot(np.array(points), t[:2, :2].T) + t[:2, 2]
    return new_points


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # square = np.array([[0, 0],
    #                    [5, 0],
    #                    [5, 10],
    #                    [0, 10]])
    # idx = [0, 1, 2, 3, 0]
    # s = square[idx, :]
    #
    # # plt.subplot(2, 3, 1)
    # plt.plot(s[:, 0], s[:, 1], 'r')
    #
    # translation = np.array([-5, -2])
    # s1 = affine_transform(s, translation2matrix(translation))
    # # plt.subplot(2, 3, 2)
    # plt.plot(s1[:, 0], s1[:, 1], 'y')
    #
    # scale = np.array([2, 0.5])
    # s2 = affine_transform(s, scale2matrix(scale))
    # # plt.subplot(2, 3, 3)
    # plt.plot(s2[:, 0], s2[:, 1], 'g')
    #
    # degree = -45
    # center = np.array([2.5, 5])
    # s3 = affine_transform(s, rotation2matrix(degree, center))
    # # plt.subplot(2, 3, 4)
    # plt.plot(s3[:, 0], s3[:, 1], 'b')
    #
    # # direction = np.array([1, 0])
    # # offset = np.array([10, 10])
    # # direction = np.array([0, 1])
    # # offset = np.array([5, 5])
    # direction = np.array([5, 5])
    # offset = np.array([0, 1])
    # s4 = affine_transform(s, reflect2matrix(direction, offset))
    # # plt.subplot(2, 3, 5)
    # plt.plot(s4[:, 0], s4[:, 1], 'm')
    #
    # dx, dy = 5, 5
    # x_limit, y_limit = 5, 10
    # s5 = affine_transform(s, shear(dx, dy, x_limit, y_limit))
    # # plt.subplot(2, 3, 6)
    # plt.plot(s5[:, 0], s5[:, 1], 'k')
    #
    # plt.axis('equal')
    # plt.show()

    # image
    import cv2
    size = (100, 100)
    img = np.ones((50, 50, 3)).astype(np.uint8) * 100
    plt.subplot(2, 3, 1)
    plt.imshow(img)

    translation = np.array([10, 20])
    s1 = translation2matrix(translation)
    img1 = cv2.warpAffine(img, s1[:2, :], size, flags=cv2.INTER_LINEAR)
    plt.subplot(2, 3, 2)
    plt.imshow(img1)

    scale = np.array([2, 0.4])
    s2 = scale2matrix(scale)
    img2 = cv2.warpAffine(img, s2[:2, :], size, flags=cv2.INTER_LINEAR)
    plt.subplot(2, 3, 3)
    plt.imshow(img2)

    degree = -30
    center = np.array([25, 25])
    s3 = rotation2matrix(degree, center)
    img3 = cv2.warpAffine(img, s3[:2, :], size, flags=cv2.INTER_LINEAR)
    plt.subplot(2, 3, 4)
    plt.imshow(img3)

    # direction = np.array([1, 0])
    # offset = np.array([10, 10])
    # direction = np.array([0, 1])
    # offset = np.array([5, 5])
    direction = np.array([20, 20])
    offset = np.array([10, 20])
    s4 = reflect2matrix(direction, offset)
    img4 = cv2.warpAffine(img, s4[:2, :], size, flags=cv2.INTER_LINEAR)
    plt.subplot(2, 3, 5)
    plt.imshow(img4)

    dx, dy = 5, 10
    x_limit, y_limit = 50, 50
    s5 = shear2matrix(dx, dy, x_limit, y_limit)
    img5 = cv2.warpAffine(img, s5[:2, :], size, flags=cv2.INTER_LINEAR)
    plt.subplot(2, 3, 6)
    plt.imshow(img5)

    # plt.axis('equal')
    plt.show()

    # from torch.nn import functional as F
    # import torch
    # img = torch.randn((1, 1, 100, 100))
    # # img = torch.ones((1, 1, 100, 100)) * 100
    # # coordinate range (-1, 1)
    # a = torch.from_numpy(translation2matrix(np.array([1, 1]))).inverse()[:2, :].to(torch.float32)
    # a = torch.from_numpy(rotation2matrix(45, np.array([0, 0]))).inverse()[:2, :].to(torch.float32)
    # a = torch.from_numpy(reflect2matrix(np.array([-1, -0.5]), np.array([0, 0]))).inverse()[:2, :].to(torch.float32)
    # a = torch.from_numpy(scale2matrix(np.array([0.8, 0.5]))).inverse()[:2, :].to(torch.float32)
    # a = torch.from_numpy(shear(0.5, 0.5, 1, 1)).inverse()[:2, :].to(torch.float32)
    # grid = F.affine_grid(a.unsqueeze(0), [1, 1, 200, 200], align_corners=False)
    # img1 = F.grid_sample(img, grid, align_corners=False)
    # plt.imshow(img1[0].numpy().transpose(1, 2, 0))
    # plt.show()
