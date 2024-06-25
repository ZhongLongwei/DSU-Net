import cv2
import numpy as np
import numbers
import random
import torch
from torchvision import transforms as T, transforms
from torchvision.transforms import functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as TF

def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=0)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class randomcrop(object):
    """Crop the given PIL Image and mask at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lab):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lab (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image and mask.
        """
        if self.padding > 0:
            img = TF.pad(img, self.padding)
            lab = TF.pad(lab, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = TF.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
            lab = TF.pad(lab, (int((1 + self.size[1] - lab.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = TF.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))
            lab = TF.pad(lab, (0, int((1 + self.size[0] - lab.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)

        return TF.crop(img, i, j, h, w), TF.crop(lab, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class Shades_of_gray(object):
    def __call__(self, img, target):
        img = np.array(img)
        # 获取图像尺寸
        # 计算每个通道的平均值
        avg_r = np.mean(img[:, :, 0])
        avg_g = np.mean(img[:, :, 1])
        avg_b = np.mean(img[:, :, 2])

        # 计算增益因子，使得每个通道的平均值相等
        gray_world_r = avg_g / avg_r
        gray_world_g = 1.0
        gray_world_b = avg_g / avg_b

        # 对每个像素进行增益调整
        img[:, :, 0] = np.minimum(255, np.multiply(img[:, :, 0], gray_world_r))
        img[:, :, 2] = np.minimum(255, np.multiply(img[:, :, 2], gray_world_b))
        return img, target


class RandomRotation_and_ColorJitter(object):
    def __call__(self, image, target):
        transform = A.Compose([
            A.VerticalFlip(p=0.5),  # 沿y轴垂直翻转
            A.HorizontalFlip(p=0.5),  # 沿x轴垂直翻转
            A.Rotate(limit=15, p=0.5),  # 随机旋转角度范围为-15到15度
            # A.RandomCrop(height=150, width=150, p=0.5),
            # A.PadIfNeeded(min_height=224, min_width=224, value=0)

            # A.RandomBrightnessContrast(brightness_limit=(-0.03, 0.03), contrast_limit=(-0.03, 0.03), p=0.5),
            # # 随机改变亮度和对比度
            # A.HueSaturationValue(hue_shift_limit=(-3, 3), sat_shift_limit=(-3, 3), val_shift_limit=(-3, 3), p=0.5),
            # # 随机改变色调、饱和度和值
        ])
        transform3 = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.03, 0.03), contrast_limit=(-0.03, 0.03), p=0.5),
            # 随机改变亮度和对比度
            A.HueSaturationValue(hue_shift_limit=(-3, 3), sat_shift_limit=(-3, 3), val_shift_limit=(-3, 3), p=0.5),
            # 随机改变色调、饱和度和值
        ])

        # transforms_adjust_brightness_contrast = transforms.Compose([
        #     transforms.ColorJitter(brightness=(0.97, 1.03), contrast=(0.97, 1.03), saturation=(0.97, 1.03),
        #                            hue=(-0.03, 0.03))
        # ])
        transform1 = transform(image=np.array(image), mask=np.array(target))
        image = transform1['image']
        target = transform1['mask']
        transform2 = transform3(image=np.array(image))
        image = transform2['image']
        # image = transforms_adjust_brightness_contrast(image)

        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target1 = np.array(target) / 255
        target = torch.as_tensor(target1, dtype=torch.float32)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
