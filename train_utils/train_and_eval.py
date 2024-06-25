import numpy as np
import torch
from torch import nn

import train_utils.distributed_utils as utils
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        # print(input_tensor.size())
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            temp_prob = torch.unsqueeze(temp_prob, 1)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        num = target.size(0)
        score = score.view(num, -1)
        target = target.view(num, -1)
        smooth = 1e-6
        intersection = (score * target)
        dice = (2. * intersection.sum(1) + smooth) / (score.sum(1) + target.sum(1) + smooth)
        loss = 1 - dice.sum() / num
        # smooth = 1e-7
        # intersect = torch.sum(score * target)
        # y_sum = torch.sum(target * target)
        # z_sum = torch.sum(score * score)
        # loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        # loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        # print(inputs.size(),target.size())
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100,
              epoch=None):
    losses = {}
    for name, x in inputs.items():
        bce = F.binary_cross_entropy(x, target)
        num = target.size(0)
        x = x.view(num, -1)
        target1 = target.view(num, -1)
        smooth = 1e-7
        intersection = (x * target1)
        dice = (2. * intersection.sum(1) + smooth) / (x.sum(1) + target1.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        loss = 0.6 * bce + 0.4 * dice
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']
    return 2.0 * losses['out'] + 1.6 * losses['outs']


def evaluate(model, data_loader, device, num_classes, lr_scheduler):
    model.eval()
    loss2 = 0.0
    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([0.2, 0.8], device=device)
    else:
        loss_weight = None
    confmat1 = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    with torch.no_grad():
        # 这里的img_index对于的是哪一张图像的索引
        # iterable: data_loader 表示不断加载数据出来，一次取出一个batch_size的img
        #           由于这里自定义了batch，设置了collate_fn函数，所以还会输出其他内容: targets, paths, shapes, img_index
        # print_freq: 100 表示100轮打印一次参数
        # header: 开头打印的内容，这里为"Test: "
        for image, target in metric_logger.log_every(data_loader, 1000, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target, loss_weight, num_classes=num_classes)
            loss2 += loss
            output = output['out']
            if num_classes > 1:
                output = torch.argmax(torch.softmax(output, dim=1), dim=1).float()
            else:
                output = (output > 0.5).float()
            confmat1.update(target, output)
            dice.update(output, target)
        # lr_scheduler.step(loss1)
        # confmat1.reduce_from_all_processes()
        # dice.reduce_from_all_processes()

    return confmat1, dice.value.item(), loss2


def modeltest(model, data_loader, device, num_classes):
    model.eval()
    loss3 = 0.0
    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([0.2, 0.8], device=device)
    else:
        loss_weight = None
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        # 这里的img_index对于的是哪一张图像的索引
        # iterable: data_loader 表示不断加载数据出来，一次取出一个batch_size的img
        #           由于这里自定义了batch，设置了collate_fn函数，所以还会输出其他内容: targets, paths, shapes, img_index
        # print_freq: 100 表示100轮打印一次参数
        # header: 开头打印的内容，这里为"Test: "
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            # loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)
            loss = criterion(output, target, loss_weight, num_classes=num_classes)
            loss3 += loss
            output = output['out']
            if num_classes > 1:
                output = torch.argmax(torch.softmax(output, dim=1), dim=1).float()
            else:
                output = (output > 0.5).float()
            # output = (output > 0.5).float()
            # confmat.update(target.flatten(), output.argmax(1).flatten())
            confmat.update(target, output)
            dice.update(output, target)
        # confmat.reduce_from_all_processes()
        # dice.reduce_from_all_processes()

    return confmat, dice.value.item(), loss3


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([0.2, 0.8], device=device)
    else:
        loss_weight = None
    loss1 = 0.0

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, loss_weight, num_classes=num_classes, epoch=epoch)
            loss1 += loss
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        # lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    lr_scheduler.step()

    return metric_logger.meters["loss"].global_avg, lr, loss1.detach().cpu().numpy()


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """

        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy

            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    def lr(epoch):
        return (1 - ((epoch) / (epochs + 1))) ** 0.9
        # return (1 - ((epoch + 1) / (epochs + 2))) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
