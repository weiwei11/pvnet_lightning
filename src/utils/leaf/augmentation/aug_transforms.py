# Author: weiwei
import cv2
import numpy as np

from ..matrix_transforms.affine_transform import affine_transform


def transform_image(image, m, new_size, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0):
    new_img = cv2.warpAffine(image, m[:2, :], (new_size[1], new_size[0]), flags=interpolation,
                             borderMode=borderMode, borderValue=borderValue)
    return new_img


def transform_points2d(pts2d, m):
    new_pts2d = affine_transform(pts2d, m)
    return new_pts2d


def transform_bboxes(bboxes, m):
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    new_lt = affine_transform(np.column_stack([x1, y1]), m)
    new_lb = affine_transform(np.column_stack([x1, y2]), m)
    new_rt = affine_transform(np.column_stack([x2, y1]), m)
    new_rb = affine_transform(np.column_stack([x2, y2]), m)

    x1, x2, x3, x4 = new_lt[:, 0], new_lb[:, 0], new_rt[:, 0], new_rb[:, 0]
    y1, y2, y3, y4 = new_lt[:, 1], new_lb[:, 1], new_rt[:, 1], new_rb[:, 1]
    x, y = np.column_stack([x1, x2, x3, x4]), np.column_stack([y1, y2, y3, y4])
    new_bboxes = np.column_stack([np.min(x, axis=-1), np.min(y, axis=-1),
                                  np.max(x, axis=-1), np.max(y, axis=-1)])

    return new_bboxes


try:
    import kornia
    import torch

    def transform_image_pth(images, m, new_size, mode='bilinear', padding_mode='zeros', fill_value=torch.zeros(3), align_corners=True):
        new_img = kornia.geometry.warp_affine(images, m[:, :2, :], (new_size[0], new_size[1]), mode=mode,
                                              padding_mode=padding_mode, fill_value=fill_value, align_corners=align_corners)
        return new_img


    def transform_points2d_pth(pts2d, m):
        new_pts2d = torch.matmul(pts2d, m[:, :2, :2].transpose(-1, -2)) + m[:, :2, 2]
        return new_pts2d


    def transform_bboxes_pth(bboxes, m):
        x1, y1, x2, y2 = bboxes[:, :, 0], bboxes[:, :, 1], bboxes[:, :, 2], bboxes[:, :, 3]
        new_lt = transform_points2d_pth(torch.stack([x1, y1], dim=-1), m)
        new_lb = transform_points2d_pth(torch.stack([x1, y2], dim=-1), m)
        new_rt = transform_points2d_pth(torch.stack([x2, y1], dim=-1), m)
        new_rb = transform_points2d_pth(torch.stack([x2, y2], dim=-1), m)

        x1, x2, x3, x4 = new_lt[:, :, 0], new_lb[:, :, 0], new_rt[:, :, 0], new_rb[:, :, 0]
        y1, y2, y3, y4 = new_lt[:, :, 1], new_lb[:, :, 1], new_rt[:, :, 1], new_rb[:, :, 1]
        x, y = torch.stack([x1, x2, x3, x4], dim=-1), torch.stack([y1, y2, y3, y4], dim=-1)
        new_bboxes = torch.stack([torch.min(x, dim=-1).values, torch.min(y, dim=-1).values, torch.max(x, dim=-1).values, torch.max(y, dim=-1).values], dim=-1)

        return new_bboxes
except ImportError:
    pass


if __name__ == '__main__':
    from leaf.matrix_transforms.affine_transform import translation2matrix, rotation2matrix, reflect2matrix, scale2matrix, shear2matrix
    from torch.nn import functional as F
    import torch
    import matplotlib.pyplot as plt

    img = torch.randn((1, 1, 100, 100))
    point = torch.tensor([[0, 0], [0, 100], [100, 0], [100, 100]])[None].to(torch.float32)
    bbox = torch.tensor([[0, 0, 100, 100]])[None].to(torch.float32)

    m1 = torch.from_numpy(translation2matrix(np.array([100, 100])))[:2, :].to(torch.float32)
    m2 = torch.from_numpy(rotation2matrix(45, np.array([0, 0])))[:2, :].to(torch.float32)
    m3 = torch.from_numpy(reflect2matrix(np.array([1, 0.5]), np.array([0, 0])))[:2, :].to(torch.float32)
    m4 = torch.from_numpy(scale2matrix(np.array([0.8, 0.5])))[:2, :].to(torch.float32)
    m5 = torch.from_numpy(shear2matrix(0.5, 0.5, 1, 1))[:2, :].to(torch.float32)

    img1 = transform_image_pth(img, m1[None], (200, 200))
    img2 = transform_image_pth(img, m2[None], (200, 200))
    img3 = transform_image_pth(img, m3[None], (200, 200))
    img4 = transform_image_pth(img, m4[None], (200, 200))
    img5 = transform_image_pth(img, m5[None], (200, 200))

    point1 = transform_points2d_pth(point, m1[None])
    point2 = transform_points2d_pth(point, m2[None])
    point3 = transform_points2d_pth(point, m3[None])
    point4 = transform_points2d_pth(point, m4[None])
    point5 = transform_points2d_pth(point, m5[None])

    bbox1 = transform_bboxes_pth(bbox, m1[None])
    bbox2 = transform_bboxes_pth(bbox, m2[None])
    bbox3 = transform_bboxes_pth(bbox, m3[None])
    bbox4 = transform_bboxes_pth(bbox, m4[None])
    bbox5 = transform_bboxes_pth(bbox, m5[None])

    plt.subplot(2, 3, 1)
    plt.imshow(img[0].numpy().transpose(1, 2, 0))
    plt.plot(*point[0].numpy().T, '.')
    plt.plot(*bbox[0].numpy().reshape(2, 2).T, '.r')
    plt.subplot(2, 3, 2)
    plt.imshow(img1[0].numpy().transpose(1, 2, 0))
    plt.plot(*point1[0].numpy().T, '.')
    plt.plot(*bbox1[0].numpy().reshape(2, 2).T, '.r')
    plt.subplot(2, 3, 3)
    plt.imshow(img2[0].numpy().transpose(1, 2, 0))
    plt.plot(*point2[0].numpy().T, '.')
    plt.plot(*bbox2[0].numpy().reshape(2, 2).T, '.r')
    plt.subplot(2, 3, 4)
    plt.imshow(img3[0].numpy().transpose(1, 2, 0))
    plt.plot(*point3[0].numpy().T, '.')
    plt.plot(*bbox3[0].numpy().reshape(2, 2).T, '.r')
    plt.subplot(2, 3, 5)
    plt.imshow(img4[0].numpy().transpose(1, 2, 0))
    plt.plot(*point4[0].numpy().T, '.')
    plt.plot(*bbox4[0].numpy().reshape(2, 2).T, '.r')
    plt.subplot(2, 3, 6)
    plt.imshow(img5[0].numpy().transpose(1, 2, 0))
    plt.plot(*point5[0].numpy().T, '.')
    plt.plot(*bbox5[0].numpy().reshape(2, 2).T, '.r')
    plt.show()

    import time
    image = np.random.randn(480, 640, 3)
    m = translation2matrix(np.array([100.0, 100.0]))[:2, :3]
    image_tensor = torch.from_numpy(image).cuda().permute(2, 0, 1)[None]
    m_tensor = torch.from_numpy(m).cuda()[None]
    it_num = 1000

    t1 = time.time()
    for i in range(it_num):
        img = transform_image(image, m, (480, 640))
    print(img.shape)
    t2 = time.time()
    print('numpy time: ', t2 - t1)

    for i in range(it_num):
        img = transform_image_pth(image_tensor, m_tensor, (480, 640))
    print(img.shape)
    torch.cuda.synchronize()
    t2 = time.time()
    print('cuda time: ', t2 - t1)
