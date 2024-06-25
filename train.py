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

class SegmentationPresetTrain:
    def __init__(self, mean, std):
        trans = []
        trans.extend([
            # T.Shades_of_gray(),
            T.RandomRotation_and_ColorJitter(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
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


def get_transform(train, mean, std):
    if train:
        return SegmentationPresetTrain(mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


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
    train_dataset = PH2Dataset(args.data_path,
                               train='train',
                               transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = PH2Dataset(args.data_path,
                             train='val',
                             transforms=get_transform(train=False, mean=mean, std=std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   shuffle=True,
                                   pin_memory=True,
                                   collate_fn=train_dataset.collate_fn)

    val_loader = data.DataLoader(val_dataset,
                                 batch_size=1,
                                 num_workers=1,
                                 pin_memory=True,
                                 collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes)
    model.to(device)
    # 用来保存训练以及验证过程中信息
    results_file = "2017-{}results{}{}.txt".format(seed, model.__class__.__name__,
                                                   datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    learning_rate = 0.0001
    betas = (0.9, 0.99)
    weight_decay = 0.00005
    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate, betas=betas, eps=1e-08,
                                  weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # # 创建学习率更新策略，这里是每个step或者epoch更新一次
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best = 1.0
    epoch1 = 0
    test1 = 0.0
    start_time = time.time()
    train_loss = []
    val_loss = []
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr, trainloss = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                                   lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        train_loss.append(trainloss / len(train_loader))
        confmat, val_dice, valloss = evaluate(model, val_loader, device=device, num_classes=num_classes,
                                              lr_scheduler=lr_scheduler)
        valloss = valloss.cpu() / len(val_loader)
        val_loss.append(valloss)
        val_info = str(confmat)
        print('\033[91mthis is val\033[0m')
        print(val_info)
        print(f"dice coefficient: {val_dice:.3f}")
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"val dice coefficient: {val_dice:.3f}\n" \
                         f"val loss: {valloss:.4f}\n"
            f.write(train_info + "\n\n")

        best1 = valloss
        if args.save_best is True:
            if best > best1:
                best = best1
                test1 = val_dice
                epoch1 = epoch
            else:
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, "save_weights/2017_best_model{}.pth".format(model.__class__.__name__))
        else:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))
    with open(results_file, "a") as f:
        train_down = f"epoch {epoch1} is the best\n" \
                     f"best  {best}\n" \
                     f"val dice {test1}\n"
        f.write(train_down)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("epoch {} is the best".format(epoch1))
    print("best  {}".format(best))
    print("val dice {}".format(test1))
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
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")
    main(args)
