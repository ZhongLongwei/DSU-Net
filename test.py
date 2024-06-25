import os
import time
import datetime
import matplotlib.pyplot as plt
import torch
from torch.backends import cudnn
from torch.utils import data
from DSU_Net import DSUNet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import PH2Dataset
import transforms as T
import random
import numpy as np
from train_utils.train_and_eval import modeltest


class SegmentationPresettest:
    def __init__(self, mean, std):
        self.transforms = T.Compose([
            # T.Shades_of_gray(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def seed_torch(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # torch.use_deterministic_algorithms(True)


def get_transform(mean, std):
    return SegmentationPresettest(mean=mean, std=std)


def create_model(num_classes):
    model = DSUNet(n_channels=3, n_classes=num_classes)
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    # 设置随机种子
    seed = 42
    print("随机种子:", seed)
    seed_torch(seed)

    test_dataset = PH2Dataset(args.data_path,
                              train='test',
                              transforms=get_transform(mean=mean, std=std))
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=1,
                                  num_workers=1,
                                  pin_memory=True,
                                  collate_fn=test_dataset.collate_fn)

    model = create_model(num_classes=num_classes)
    # load weights
    weights_path = "./save_weights/2017_best_modelDSUNet.pth"
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    results_file = "2017-results{}.txt".format(seed, model.__class__.__name__)
    start_time = time.time()
    test_confmat, test_dice, testloss = modeltest(model, test_loader, device=device, num_classes=num_classes)
    testloss = testloss.cpu() / len(test_loader)
    test_info = str(test_confmat)
    print('\033[91mthis is test\033[0m')
    print(test_info)
    print(f"dice coefficient: {test_dice:.3f}")
    # write into txt
    with open(results_file, "a") as f:
        train_info = f"test dice coefficient: {test_dice:.3f}\n"
        f.write(train_info + test_info + "\n\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=2, type=int)

    parser.add_argument("--epochs", default=50, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')  # 程序中断后，用指向上一次的权重重新开始训练
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')  # 保存最佳训练权重
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/', help='checkpoint path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
