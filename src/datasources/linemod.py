import hydra.utils
import torch.utils.data as data
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import os

from src.utils.pvnet import pvnet_data_utils
from lib.vendor.bop_toolkit.bop_toolkit_lib import inout


mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3).astype(np.float32)
std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3).astype(np.float32)

diameters = {
    'cat': 15.2633,
    'ape': 9.74298,
    'benchvise': 28.6908,
    'bowl': 17.1185,
    'cam': 17.1593,
    'can': 19.3416,
    'cup': 12.5961,
    'driller': 25.9425,
    'duck': 10.7131,
    'eggbox': 17.6364,
    'glue': 16.4857,
    'holepuncher': 14.8204,
    'iron': 30.3153,
    'lamp': 28.5155,
    'phone': 20.8394,
    'BingMaYong': 8.050211798455988443e-01 * 100,
    'CQJXW': 8.832011775354469130e-01 * 100,
    'jiepu': 2.4795272e-01 * 100,
    'lifan': 7.1044780e+00 * 100,
    'YADIPro': 1.925804403359801187 * 100,
    'Coffee': 0.33867007771339336 * 100,
    'Tiguan': 0.2881067036629352 * 100,
    'Wangzai': 0.12981391938864068 * 100
}

linemod_cls_names = ['ape', 'cam', 'cat', 'duck', 'glue', 'iron', 'phone', 'benchvise', 'can', 'driller', 'eggbox', 'holepuncher', 'lamp',
                     'BingMaYong', 'CQJXW', 'jiepu', 'lifan', 'YADIPro', 'Coffee', 'Tiguan', 'Wangzai']

linemod_K = np.array([[572.4114, 0., 325.2611],
                     [0., 573.57043, 242.04899],
                     [0., 0., 1.]])


blender_K = np.array([[700., 0., 320.],
                     [0., 700., 240.],
                     [0., 0., 1.]])


class Linemod(data.Dataset):
    def __init__(self, obj_cls, data_root, ann_file, obj_file, im_size, cache_root=None):
        super().__init__()

        self.data_root = data_root

        # load coco data
        self.coco = COCO(ann_file)
        self.img_ids = np.array(sorted(self.coco.getImgIds()))
        self.anno_ids = self.coco.getAnnIds(imgIds=self.img_ids)

        path = self.coco.loadImgs(self.img_ids)
        annotations = self.coco.loadAnns(self.anno_ids)

        self.img_paths = [x['file_name'] for x in path]
        self.mask_paths = [x['mask_path'] for x in annotations]
        self.kpt_3ds = list(map(lambda x: np.concatenate([x['fps_3d'], [x['center_3d']]]), annotations))
        self.kpt_2ds = list(map(lambda x: np.concatenate([x['fps_2d'], [x['center_2d']]]), annotations))
        self.poses = [np.array(x['pose']) for x in annotations]
        self.K = [np.array(x['K']) for x in annotations]
        self.corner_3ds = [np.array(x['corner_3d']) for x in annotations]
        self.corner_2ds = [np.array(x['corner_2d']) for x in annotations]

        self.types = [x['type'] for x in annotations]
        self.classes = [x['cls'] for x in annotations]

        # model
        self.obj_cls = obj_cls
        self.model = inout.load_ply(obj_file)
        self.model_diameter = diameters[obj_cls] / 100
        self.model_scale = 1
        if obj_cls in ['eggbox', 'glue']:
            self.is_symmetric = True
        else:
            self.is_symmetric = False

        self.img_mean = mean
        self.img_std = std

        # if cache_root is not None:
        #     makedirs(cache_root)

    def load_model(self, index=None):
        return self.model

    def load_image(self, index):
        return Image.open(os.path.join(hydra.utils.get_original_cwd(), self.img_paths[index]))

    def load_mask(self, index):
        cls_idx = linemod_cls_names.index(self.classes[index]) + 1
        mask = pvnet_data_utils.read_linemod_mask(os.path.join(self.data_root, self.mask_paths[index]), self.types[index], cls_idx)
        return mask

    def __getitem__(self, index):
        return {
            'img_id': self.img_ids[index],
            'kpt_2d': self.kpt_2ds[index], 'kpt_3d': self.kpt_3ds[index],
            'corner_2d': self.corner_2ds[index], 'corner_3d': self.corner_3ds[index],
            'K': self.K[index], 'pose': self.poses[index],
        }

    def __len__(self):
        return len(self.img_ids)
