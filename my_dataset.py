import os

import cv2
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from torch.utils.data import Dataset


class PH2Dataset(Dataset):
    def __init__(self, root: str, train: str, transforms=None):
        super(PH2Dataset, self).__init__()
        self.flag = train
        # self.flag = "ph2"
        data_root = os.path.join(root, "2017", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        # img_names = [i for i in os.listdir(os.path.join(data_root, "image")) if i.endswith(".bmp")]
        img_names = [i for i in os.listdir(os.path.join(data_root, "image")) if i.endswith(".png")]
        img_names.sort()
        self.image = [os.path.join(data_root, "image", i) for i in img_names]
        # label_names = [i for i in os.listdir(os.path.join(data_root, "label")) if i.endswith(".bmp")]
        label_names = [i for i in os.listdir(os.path.join(data_root, "label")) if i.endswith(".png")]
        label_names.sort()
        self.label = [os.path.join(data_root, "label", i) for i in label_names]
        # check files
        for i in self.label:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.image[idx]).convert('RGB')
        label = Image.open(self.label[idx]).convert('L')

        if self.transforms is not None:
            img, label = self.transforms(img, label)
        label = label.unsqueeze(0)

        return img, label

    def __len__(self):
        return len(self.image)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)  # 填充
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
