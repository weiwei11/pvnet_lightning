# Author: weiwei
import numpy as np

from lib.utils.base_utils import project


def rotation2matrix(degree, fx, fy):
    """

    :param degree: degree of angle (0 ~ 360) or (-180 ~ 180)
    :param fx: camera intrinsic matrix
    :param fy: camera intrinsic matrix
    :return: matrix shape is (3, 3)
    """
    m = np.eye(3)
    d = degree / 180.0 * np.pi
    m[0, 0] = np.cos(d)
    m[0, 1] = -np.sin(d) * fy / fx
    m[1, 0] = np.sin(d) * fx / fy
    m[1, 1] = np.cos(d)
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


def translation2matrix(translation, fx, fy):
    """

    :param translation: np.array([x y])
    :param fx: camera intrinsic matrix
    :param fy: camera intrinsic matrix
    :return: matrix shape is (3, 3)
    """
    m = np.eye(3)
    m[0, 2] = translation[0] / fx
    m[1, 2] = translation[1] / fy
    return m


if __name__ == '__main__':
    # from lib.utils.vsd import inout
    # from lib.datasets.dataset_catalog import DatasetCatalog
    from pycocotools.coco import COCO
    from PIL import Image
    import matplotlib.pyplot as plt
    from lib.utils.matrix_transforms import affine_transform
    import cv2

    # error_eps = 20
    # obj_path = 'data/linemod/cat/cat.ply'
    # model = inout.load_ply(obj_path)
    # model['pts'] = model['pts'] * 1000.

    degree = -30
    s = np.array([1.5, 1.5])
    offset = np.array([100, 100])
    im_size = (640, 480)
    cls_type = 'cat'

    ann_file = 'data/linemod/{}/test.json'.format(cls_type)
    # dataset_info = DatasetCatalog.get('LinemodTest')
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

        fx, fy = K[0, 0], K[1, 1]
        center = np.array([K[0, 2], K[1, 2]])

        # rotation
        m2 = affine_transform.rotation2matrix(degree, center)
        m3 = rotation2matrix(degree, fx, fy)
        warp_image1 = cv2.warpAffine(image, m2[:2, :], im_size, flags=cv2.INTER_LINEAR)
        warp_kpt_2d1 = kpt_2d @ m2[:2, :2].T + m2[:2, 2:].T
        xyz = np.dot(kpt_3d, pose[:, :3].T) + pose[:, 3:].T
        xyz = np.dot(xyz, m3.T)
        xyz = np.dot(xyz, K.T)
        xy1 = xyz[:, :2] / xyz[:, 2:]

        # scale
        t1 = affine_transform.translation2matrix(-center)
        t2 = affine_transform.translation2matrix(center)
        m2 = t2 @ affine_transform.scale2matrix(s) @ t1
        m3 = scale2matrix(s)
        warp_image2 = cv2.warpAffine(image, m2[:2, :], im_size, flags=cv2.INTER_LINEAR)
        warp_kpt_2d2 = kpt_2d @ m2[:2, :2].T + m2[:2, 2:].T
        xyz = np.dot(kpt_3d, pose[:, :3].T) + pose[:, 3:].T
        xyz = np.dot(xyz, m3.T)
        xyz = np.dot(xyz, K.T)
        xy2 = xyz[:, :2] / xyz[:, 2:]

        # translation
        m2 = t2 @ affine_transform.translation2matrix(offset) @ t1
        m3 = translation2matrix(offset, fx, fy)
        warp_image3 = cv2.warpAffine(image, m2[:2, :], im_size, flags=cv2.INTER_LINEAR)
        warp_kpt_2d3 = kpt_2d @ m2[:2, :2].T + m2[:2, 2:].T
        xyz = np.dot(kpt_3d, pose[:, :3].T) + pose[:, 3:].T
        xyz = np.dot(xyz, m3.T)
        xyz = np.dot(xyz, K.T)
        xy3 = xyz[:, :2] / xyz[:, 2:]

        print(np.linalg.norm(xy1 - warp_kpt_2d1, axis=-1))
        print(np.linalg.norm(xy2 - warp_kpt_2d2, axis=-1))
        print(np.linalg.norm(xy3 - warp_kpt_2d3, axis=-1))

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
        plt.subplot(2, 2, 4)
        plt.imshow(warp_image3)
        plt.plot(warp_kpt_2d3[:, 0], warp_kpt_2d3[:, 1], '.b')
        plt.plot(xy3[:, 0], xy3[:, 1], '+r')
        plt.show()

        # m3 = t2 @ rotation2matrix(degree, fy, fx) @ t1
        # img = cv2.warpAffine(image, m3[:2, :], im_size, flags=cv2.INTER_LINEAR)
        # plt.imshow(img)
        # plt.show()
