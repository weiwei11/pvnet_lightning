# Author: weiwei
import numpy as np
import affine_transform as af
from lib.utils.base_utils import project


def rotation2affine(degree, K):
    """

    :param degree: degree of angle (0 ~ 360) or (-180 ~ 180)
    :param K: camera intrinsic matrix
    :return: matrix shape is (3, 3)
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    m1 = af.translation2matrix(-np.array([cx, cy]))
    m2 = af.translation2matrix(np.array([cx, cy]))

    m = np.eye(3)
    d = degree / 180.0 * np.pi
    m[0, 0] = np.cos(d)
    m[0, 1] = -np.sin(d) * fx / fy
    m[1, 0] = np.sin(d) * fy / fx
    m[1, 1] = np.cos(d)
    return m2 @ m @ m1


def translation2affine(translation, K):
    """

    :param translation: np.array([x y])
    :param K: camera intrinsic matrix
    :return: matrix shape is (3, 3)
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    m1 = af.translation2matrix(-np.array([cx, cy]))
    m2 = af.translation2matrix(np.array([cx, cy]))

    m = np.eye(3)
    m[0, 2] = translation[0] * fx
    m[1, 2] = translation[1] * fy
    return m2 @ m @ m1


def rotation2pose(degree):
    """

    :param degree: degree of angle (0 ~ 360) or (-180 ~ 180)
    :return: matrix shape is (3, 3)
    """
    m = np.eye(4)
    d = degree / 180.0 * np.pi
    m[0, 0] = np.cos(d)
    m[0, 1] = -np.sin(d)
    m[1, 0] = np.sin(d)
    m[1, 1] = np.cos(d)
    return m


def translation2pose(translation):
    """

    :param translation: np.array([x y])
    :return: matrix shape is (3, 3)
    """
    m = np.eye(4)
    m[0, 3] = translation[0]
    m[1, 3] = translation[1]
    return m


if __name__ == '__main__':
    # from lib.utils.vsd import inout
    # from lib.datasets.dataset_catalog import DatasetCatalog
    from pycocotools.coco import COCO
    from PIL import Image
    import matplotlib.pyplot as plt
    import cv2

    degree = -30
    offset = np.array([0.1, 0.1])
    im_size = (640, 480)
    cls_type = 'cat'

    ann_file = 'data/linemod/{}/test.json'.format(cls_type)
    coco = COCO(ann_file)
    for img_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anno = coco.loadAnns(ann_ids)[0]

        path = coco.loadImgs(int(img_id))[0]['file_name']
        image = np.asarray(Image.open(path))
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)

        K = np.array(anno['K'])
        pose = np.array(anno['pose'])
        kpt_2d = project(kpt_3d, K, pose)

        # rotation
        m4 = rotation2pose(degree)
        m3 = rotation2affine(degree, K)
        warp_image1 = cv2.warpAffine(image, m3[:2, :], im_size, flags=cv2.INTER_LINEAR)
        warp_kpt_2d1 = kpt_2d @ m3[:2, :2].T + m3[:2, 2:].T
        xyz = np.dot(kpt_3d, pose[:, :3].T) + pose[:, 3:].T
        xyz = np.dot(xyz, m4[:3, :3].T)
        xyz = np.dot(xyz, K.T)
        xy1 = xyz[:, :2] / xyz[:, 2:]

        # translation
        m4 = translation2pose(offset)
        m3 = translation2affine(offset, K)
        warp_image2 = cv2.warpAffine(image, m3[:2, :], im_size, flags=cv2.INTER_LINEAR)
        warp_kpt_2d2 = kpt_2d @ m3[:2, :2].T + m3[:2, 2:].T
        xyz = np.dot(kpt_3d, pose[:, :3].T) + pose[:, 3:].T
        xyz = np.dot(xyz, m4[:3, :3].T) + m4[:3, 3:].T
        xyz = np.dot(xyz, K.T)
        xy2 = xyz[:, :2] / xyz[:, 2:]

        print(np.linalg.norm(xy1 - warp_kpt_2d1, axis=-1))
        print(np.linalg.norm(xy2 - warp_kpt_2d2, axis=-1))

        plt.figure(figsize=(19.20, 10.80))
        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.subplot(2, 2, 2)
        plt.imshow(warp_image1)
        plt.plot(warp_kpt_2d1[:, 0], warp_kpt_2d1[:, 1], '.b')
        plt.plot(xy1[:, 0], xy1[:, 1], '+r')
        plt.subplot(2, 2, 3)
        plt.imshow(warp_image2)
        plt.plot(warp_kpt_2d2[:, 0], warp_kpt_2d2[:, 1], '.b')
        plt.plot(xy2[:, 0], xy2[:, 1], '+r')
        plt.show()
