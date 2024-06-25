import os
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F


def resize2():
    root = 'D:\FAT-NET\ISIC2018\ISIC2018_Task1-2_Training_Input\ISIC2018_Task1-2_Training_Input'
    b1 = [i for i in os.listdir(os.path.join(root)) if i.endswith(".jpg")]
    b1.sort()
    root1 = 'D:\FAT-NET\ISIC2018\ISIC2018_Task1_Training_GroundTruth\ISIC2018_Task1_Training_GroundTruth'
    b2 = [i for i in os.listdir(os.path.join(root1)) if i.endswith(".png")]
    b2.sort()
    image_root = [os.path.join(root, i) for i in b1]
    label_root = [os.path.join(root1, i) for i in b2]
    rootest = 'D:\FAT-NET\ISIC2018\ISIC2018_Task1-2_Test_Input\ISIC2018_Task1-2_Test_Input'
    b3 = [i for i in os.listdir(os.path.join(rootest)) if i.endswith(".jpg")]
    b3.sort()
    rootest1 = 'D:\FAT-NET\ISIC2018\ISIC2018_Task1_Test_GroundTruth\ISIC2018_Task1_Test_GroundTruth'
    b4 = [i for i in os.listdir(os.path.join(rootest1)) if i.endswith(".png")]
    b4.sort()
    image_root1 = [os.path.join(rootest, i) for i in b3]
    label_root1 = [os.path.join(rootest1, i) for i in b4]
    rootval = 'D:\FAT-NET\ISIC2018\ISIC2018_Task1-2_Validation_Input\ISIC2018_Task1-2_Validation_Input'
    b5 = [i for i in os.listdir(os.path.join(rootval)) if i.endswith(".jpg")]
    b5.sort()
    rootval1 = 'D:\FAT-NET\ISIC2018\ISIC2018_Task1_Validation_GroundTruth\ISIC2018_Task1_Validation_GroundTruth'
    b6 = [i for i in os.listdir(os.path.join(rootval1)) if i.endswith(".png")]
    b6.sort()
    image_root2 = [os.path.join(rootval, i) for i in b5]
    label_root2 = [os.path.join(rootval1, i) for i in b6]
    for i, (image_path, labels) in enumerate(zip(image_root, label_root)):
        img = Image.open(image_path).convert('RGB')
        label1 = Image.open(labels).convert('L')
        img = img.resize((224, 224), Image.NEAREST)
        label1 = label1.resize((224, 224), Image.NEAREST)
        img.save(os.path.join("./2018/train/image", f'{b1[i].split(".")[0]}.png'))
        label1.save(os.path.join("./2018/train/label", f'{b2[i].split(".")[0]}.png'))

    for i, (image_path, labels) in enumerate(zip(image_root2, label_root2)):
        img = Image.open(image_path).convert('RGB')
        label1 = Image.open(labels).convert('L')
        img = img.resize((224, 224), Image.NEAREST)
        label1 = label1.resize((224, 224), Image.NEAREST)
        img.save(os.path.join("./2018/val/image", f'{b5[i].split(".")[0]}.png'))
        label1.save(os.path.join("./2018/val/label", f'{b6[i].split(".")[0]}.png'))

    for i, (image_path, labels) in enumerate(zip(image_root1, label_root1)):
        img = Image.open(image_path).convert('RGB')
        label1 = Image.open(labels).convert('L')
        img = img.resize((224, 224), Image.NEAREST)
        label1 = label1.resize((224, 224), Image.NEAREST)
        img.save(os.path.join("./2018/test/image", f'{b3[i].split(".")[0]}.png'))
        label1.save(os.path.join("./2018/test/label", f'{b4[i].split(".")[0]}.png'))


if __name__ == '__main__':
    resize2()
