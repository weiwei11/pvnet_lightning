from math import floor

import numpy as np
import random
import torchvision
import cv2
from PIL import Image
import collections


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, kpts=None, mask=None):
        for t in self.transforms:
            img, kpts, mask = t(img, kpts, mask)
        return img, kpts, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):

    def __call__(self, img, kpts, mask):
        return np.asarray(img).astype(np.float32) / 255., kpts, mask


class Normalize(object):

    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, img, kpts, mask):
        img -= self.mean
        img /= self.std
        if self.to_bgr:
            img = img.transpose(2, 0, 1).astype(np.float32)
        return img, kpts, mask


class ColorJitter(object):

    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue, )

    def __call__(self, image, kpts, mask):
        image = np.asarray(self.color_jitter(Image.fromarray(np.ascontiguousarray(image, np.uint8))))
        return image, kpts, mask


class RandomBlur(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, kpts, mask):
        if random.random() < self.prob:
            sigma = np.random.choice([3, 5, 7, 9])
            image = cv2.GaussianBlur(image, (sigma, sigma), 0)
        return image, kpts, mask


class Resize(object):
    """
    Resize the input opencv image to given size by using opencv resize.

    Args:
        size: new size of image, it is [h w].
        interpolation: the type of interpolation.
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, kpts, mask):
        """
        Args:
            image:
            kpts:
            mask
        Returns:
            new_image:
            new_kpts:
            new_mask:
        """
        assert (image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1])

        new_image = cv2.resize(image, (self.size[1], self.size[0]), interpolation=self.interpolation)

        # points change position
        new_kpts = np.zeros_like(kpts)
        new_kpts[:, 0] = kpts[:, 0] / (image.shape[1] / self.size[1])
        new_kpts[:, 1] = kpts[:, 1] / (image.shape[0] / self.size[0])

        new_mask = cv2.resize(mask, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)

        # cv2.imshow('1', new_image)
        # cv2.imshow('2', new_mask * 255)
        # cv2.waitKey()
        # print(image.shape[1] / self.size[1], image.shape[0] / self.size[0])
        # print(new_kpts)
        # print(recover_kpts_2d, 'weiwei')
        # global recover_kpts_2d
        # recover_kpts_2d.scale = image.shape[1] / self.size[1]
        scale = image.shape[1] / self.size[1]

        return new_image, new_kpts, new_mask, scale


class AddPad(object):
    """
    Add pad to make height and width of image has specific scale.

    Args:
        scale: specific scale that height and width,
                such as (3, 4), it's means height/width is 3/4 after pad adding
        fill_mode: the mode to fill pad.
            'mean'
            'max'
            'min'
            'constant': this choice should product 'constant_value' parameter
    """

    def __init__(self, scale, fill_mode='mean', **kwargs):
        if fill_mode == 'mean' or fill_mode == 'max' or fill_mode == 'min':
            self.scale = scale
            self.fill_mode = fill_mode
        elif fill_mode == 'constant':
            self.scale = scale
            self.fill_mode = fill_mode
            self.constant_value = kwargs['constant_value']
        else:
            raise ValueError('The AddPad mode is unknown!\n')

    def __call__(self, image, kpts, mask):
        """
        Args:
            image:
            kpts:
            mask:
        Returns:
            new_image:
            new_kpts:
            new_mask:
            pad:
        """
        image = np.array(image)
        assert (image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1])

        height, width, channel = image.shape

        # compute pad
        h_scale = self.scale[0]
        w_scale = self.scale[1]
        if height / h_scale > width / w_scale:
            new_height = height
            new_width = floor(height / h_scale * w_scale)
            h_pad = 0
            w_pad = (new_width - width) // 2
        else:
            new_height = floor(width / w_scale * h_scale)
            new_width = width
            h_pad = (new_height - height) // 2
            w_pad = 0
        new_image = np.ones([new_height, new_width, channel], image.dtype)
        new_mask = np.zeros([new_height, new_width], mask.dtype)

        # set different value
        if self.fill_mode == 'mean':
            channel_mean = np.mean(np.reshape(image, (height * width, channel)), 0)
            new_image = new_image * channel_mean
        elif self.fill_mode == 'max':
            channel_max = np.max(np.reshape(image, (height * width, channel)), 0)
            new_image = new_image * channel_max
        elif self.fill_mode == 'min':
            channel_min = np.min(np.reshape(image, (height * width, channel)), 0)
            new_image = new_image * channel_min
        elif self.fill_mode == 'constant':
            new_image = new_image * self.constant_value
        else:
            raise ValueError('The mode is unknow!', self.fill_mode)

        # copy original image
        new_image[h_pad:h_pad + height, w_pad:w_pad + width, :] = image

        # points change position
        new_kpts = kpts + np.array([w_pad, h_pad])
        # copy original mask
        new_mask[h_pad:h_pad + height, w_pad:w_pad + width] = mask

        # print(h_pad, w_pad)
        # print(new_kpts)
        # print(recover_kpts_2d)
        # global recover_kpts_2d
        # recover_kpts_2d.offset = np.array([w_pad, h_pad])
        offset = np.array([w_pad, h_pad])

        return np.uint8(new_image), new_kpts, new_mask, offset


def make_transforms(cfg, is_train):
    if is_train is True:
        transform = Compose(
            [
                RandomBlur(0.5),
                ColorJitter(0.1, 0.1, 0.05, 0.05),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return transform


def make_preprocess():
    # preprocess = Compose(
    #     [
    #         AddPad([3, 4], 'mean'),
    #         Resize([480, 640])
    #     ]
    # )

    def preprocess(img, kpts_2d, mask):
        add_pad = AddPad([3, 4], 'mean')
        resize = Resize([480, 640])
        img, kpts_2d, mask, offset = add_pad(img, kpts_2d, mask)
        img, kpts_2d, mask, scale = resize(img, kpts_2d, mask)
        return img, kpts_2d, mask, offset, scale

    return preprocess
