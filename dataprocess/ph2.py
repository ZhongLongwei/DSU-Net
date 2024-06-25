import os
from random import shuffle
import random
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

random.seed(42)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def datachuli():
    set_seed(2024)
    root = 'D:\FAT-NET\PH2\PH2Dataset\PH2Dataset\PH2 Dataset images'
    b1 = [i for i in os.listdir(os.path.join(root))]
    random.shuffle(b1)
    image_root = [os.path.join(root, i, f'{i}_Dermoscopic_Image', f'{i}.bmp') for i in b1]
    image = [Image.open(i) for i in image_root]
    label_root = [os.path.join(root, i, f'{i}_lesion', f'{i}_lesion.bmp') for i in b1]
    for i in range(140):
        img = Image.open(image_root[i]).convert('RGB')
        label1 = Image.open(label_root[i]).convert('L')
        img = img.resize((224, 224), Image.NEAREST)
        label1 = label1.resize((224, 224), Image.NEAREST)
        img.save(os.path.join("./ph2/train/image", f'{b1[i]}.bmp'))
        label1.save(os.path.join("./ph2/train/label", f'{b1[i]}_lesion.bmp'))

    for i in range(140, 160):
        img = Image.open(image_root[i]).convert('RGB')
        label1 = Image.open(label_root[i]).convert('L')
        img = img.resize((224, 224), Image.NEAREST)
        label1 = label1.resize((224, 224), Image.NEAREST)
        img.save(os.path.join("./ph2/val/image", f'{b1[i]}.bmp'))
        label1.save(os.path.join("./ph2/val/label", f'{b1[i]}_lesion.bmp'))

    for i in range(160, 200):
        img = Image.open(image_root[i]).convert('RGB')
        label1 = Image.open(label_root[i]).convert('L')
        img = img.resize((224, 224), Image.NEAREST)
        label1 = label1.resize((224, 224), Image.NEAREST)
        img.save(os.path.join("./ph2/test/image", f'{b1[i]}.bmp'))
        label1.save(os.path.join("./ph2/test/label", f'{b1[i]}_lesion.bmp'))

    print(len(image))



if __name__ == '__main__':
    datachuli()
