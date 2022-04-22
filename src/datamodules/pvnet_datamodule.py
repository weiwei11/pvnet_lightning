from typing import Optional

import torch.utils.data as data
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import numpy as np
from torch.utils.data import SequentialSampler, RandomSampler

from src.datamodules.components.samplers import ImageSizeBatchSampler
from src.datamodules.components.transforms import Compose, RandomBlur, ColorJitter, ToTensor, Normalize
from src.utils.pvnet import pvnet_data_utils
from src.datamodules.components.augmentation import crop_or_padding_to_fixed_size, rotate_instance, crop_resize_instance_v1
import pytorch_lightning as pl


class PVNetDataset(data.Dataset):

    def __init__(self, datasource: data.Dataset, split, aug_config=None):
        super().__init__()

        self.datasource = datasource
        self.split = split

        self._transforms = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._color_augment = Compose(
            [
                RandomBlur(0.5),
                ColorJitter(0.1, 0.1, 0.05, 0.05),
            ]
        )
        self.aug = aug_config

    def __getitem__(self, index_tuple):
        index, height, width = index_tuple

        img_id = self.datasource.img_ids[index]
        img = self.datasource.load_image(index)
        mask = self.datasource.load_mask(index)
        kpt_2d = self.datasource.kpt_2ds[index]
        kpt_3d = self.datasource.kpt_3ds[index]
        corner_3d = self.datasource.corner_3ds[index]
        corner_2d = self.datasource.corner_2ds[index]
        pose = self.datasource.poses[index]
        K = self.datasource.K[index]

        img = np.asarray(img).astype(np.uint8)

        if self.split == 'train':
            inp, kpt_2d, mask = self.augment(img, mask, kpt_2d, height, width)
            inp, kpt_2d, mask = self._color_augment(inp, kpt_2d, mask)
        else:
            inp = img

        # vertex = pvnet_data_utils.compute_vertex(mask, kpt_2d, cfg.train.is_norm).transpose(2, 0, 1)
        # _, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(inp)
        # ax1.plot(kpt_2d[:, 0], kpt_2d[:, 1], '.')
        # ax2.imshow(mask * vertex[0, :, :])
        # # ax2.imshow(vertex[0])
        # # plt.savefig(os.path.join('/home/ww/Desktop/img_samples/test', f'{img_id}.png'))
        # # plt.clf()
        # plt.show()

        if self._transforms is not None:
            inp, kpt_2d, mask = self._transforms(inp, kpt_2d, mask)

        vertex = pvnet_data_utils.compute_vertex(mask, kpt_2d, True).transpose(2, 0, 1)

        ret = {'inp': inp, 'mask': mask.astype(np.uint8), 'vertex': vertex, 'img_id': img_id,
               'kpt_2d': kpt_2d.astype(np.float32), 'kpt_3d': kpt_3d,
               'corner_2d': corner_2d, 'corner_3d': corner_3d,
               'pose': pose, 'K': K,
               'meta': {}}

        return ret

    def __len__(self):
        return len(self.datasource)

    def augment(self, img, mask, kpt_2d, height, width):
        # add one column to kpt_2d for convenience to calculate
        hcoords = np.concatenate((kpt_2d, np.ones((len(kpt_2d), 1))), axis=-1)
        img = np.asarray(img).astype(np.uint8)
        foreground = np.sum(mask)
        # randomly mask out to add occlusion
        if foreground > 0:
            img, mask, hcoords = rotate_instance(img, mask, hcoords, self.aug.rotate_min, self.aug.rotate_max)
            img, mask, hcoords = crop_resize_instance_v1(img, mask, hcoords, height, width,
                                                         self.aug.overlap_ratio,
                                                         self.aug.resize_ratio_min,
                                                         self.aug.resize_ratio_max)
        else:
            img, mask = crop_or_padding_to_fixed_size(img, mask, height, width)
        kpt_2d = hcoords[:, :2]

        return img, kpt_2d, mask


class PVNetDataModule(pl.LightningDataModule):
    def __init__(self, train_datasource: data.Dataset, test_datasource: data.Dataset,
                 train_batch_size: int, test_batch_size: int, aug_config: DictConfig, num_workers=8):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.train_datasource = train_datasource
        self.test_datasource = test_datasource

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = PVNetDataset(self.train_datasource, 'train', aug_config=self.hparams.aug_config)
            self.val_dataset = PVNetDataset(self.test_datasource, 'test')

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = PVNetDataset(self.test_datasource, 'test')

        if stage == "predict" or stage is None:
            pass

    def train_dataloader(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_sampler = ImageSizeBatchSampler(train_sampler, self.hparams.train_batch_size, False, 256, 480, 640)
        return data.DataLoader(self.train_dataset, batch_sampler=train_sampler, pin_memory=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        test_sampler = SequentialSampler(self.val_dataset)
        test_sampler = ImageSizeBatchSampler(test_sampler, self.hparams.test_batch_size, False, 256, 480, 640)
        return data.DataLoader(self.val_dataset, batch_sampler=test_sampler, pin_memory=True, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        test_sampler = SequentialSampler(self.test_dataset)
        test_sampler = ImageSizeBatchSampler(test_sampler, self.hparams.test_batch_size, False, 256, 480, 640)
        return data.DataLoader(self.test_dataset, batch_sampler=test_sampler, pin_memory=True, num_workers=self.hparams.num_workers)

    # def predict_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count(), pin_memory=True)
