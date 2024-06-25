from collections import defaultdict, deque
import datetime
import time
import torch
import torch.distributed as dist

import errno
import os


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    # 更新数值
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    # #synchronize_between_processes：同步进程数值
    # dist.barrier()：阻塞进程，等待所有进程完成计算
    # dist.all_reduce():把所有节点上计算好的数值进行累加，然后传递给所有的节点。
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    # 中位数
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None
        self.acc = 0.  # Accuracy
        self.SE = 0.  # Sensitivity (Recall)
        self.SP = 0.  # Specificity
        self.iou = 0.  # Jaccard Similarity
        self.DC = 0.  # Dice Coefficient
        self.count = 0

    def update(self, target, pred):
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        TP = torch.dot(pred, target)
        TN = torch.dot(1 - pred, 1 - target)
        FP = torch.dot(pred, 1 - target)
        FN = torch.dot(1 - pred, target)
        self.acc += float(TN + TP) / float(TP + TN + FP + FN) if float(TP + TN + FP + FN) != 0 else 0
        self.SE += float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        self.SP += float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        self.DC += float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        self.iou += float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
        self.count += 1
        # output = (output >= 0.5).int()
        # n = self.num_classes + 1
        # if self.mat is None:
        #     # 创建混淆矩阵
        #     self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        # with torch.no_grad():
        #     # 寻找GT中为目标的像素索引
        #     k = (target >= 0) & (target < n)
        #     # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
        #     inds = n * target[k].to(torch.int64) + output[k]
        #     self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        # h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)

        # acc_global = torch.diag(h).sum() / h.sum()
        # # 计算每个类别的准确率
        # sp, se = torch.diag(h) / h.sum(0)
        # acc = torch.diag(h) / h.sum(1)
        # # 计算每个类别预测与真实目标的iou
        # iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        # dice = 2 * torch.diag(h) / (h.sum(1) + h.sum(0))
        acc = self.acc / self.count
        SE = self.SE / self.count
        SP = self.SP / self.count
        iou = self.iou / self.count
        DC = self.DC / self.count
        return acc, SE, SP, iou, DC

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        # acc_global, acc, iu, se, sp, dice = self.compute()
        acc, SE, SP, iou, DC = self.compute()
        return (
            'acc: {:.4f}\n'
            'SE {}\n'
            'SP: {}\n'
            'IoU: {}\n'
            'dice: {}').format(
            acc * 100,
            SE * 100,
            SP * 100,
            iou * 100,
            DC * 100)
        # return (
        #     'global correct: {:.4f}\n'
        #     'average row correct: {}\n'
        #     'se: {}\n'
        #     'sp: {}\n'
        #     'dice {}\n'
        #     'IoU: {}\n'
        #     'mean IoU: {:.4f}').format(
        #     acc_global.item() * 100,
        #     ['{:.4f}'.format(i) for i in (acc * 100).tolist()],
        #     se.item() * 100,
        #     sp.item() * 100,
        #     ['{:.4f}'.format(i) for i in (dice * 100).tolist()],
        #     ['{:.4f}'.format(i) for i in (iu * 100).tolist()],
        #     iu.mean().item() * 100)


class DiceCoefficient(object):
    def __init__(self, num_classes: int = 2, ignore_index: int = -100):
        self.cumulative_dice = None
        # self.cumulative_iou = None
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.count = None

    def update(self, pred, target):
        if self.cumulative_dice is None:
            self.cumulative_dice = torch.zeros(1, dtype=pred.dtype, device=pred.device)
            # self.cumulative_iou = torch.zeros(1, dtype=pred.dtype, device=pred.device)
        if self.count is None:
            self.count = torch.zeros(1, dtype=pred.dtype, device=pred.device)
        # compute the Dice score, ignoring background
        # pred = F.one_hot(pred.argmax(dim=1), self.num_classes).permute(0, 3, 1, 2).float()
        # dice_target = build_target(target, self.num_classes, self.ignore_index)
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        inter = torch.dot(pred, target)
        sets_sum = torch.sum(pred) + torch.sum(target)
        # iou = (sets_sum - inter)
        self.cumulative_dice += (2 * inter) / sets_sum
        # self.cumulative_iou += inter / iou
        self.count += 1

    @property
    def value(self):
        if self.count == 0:
            return 0
        else:
            return self.cumulative_dice / self.count

    def reset(self):
        if self.cumulative_dice is not None:
            self.cumulative_dice.zero_()

        if self.count is not None:
            self.count.zeros_()

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.cumulative_dice)
        # torch.distributed.all_reduce(self.cumulative_iou)
        torch.distributed.all_reduce(self.count)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        # 存储参数变量，且不断更新
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    # 自动保留字典遍历，记录其更新信息
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    # 打印本身返回的内容
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    # 收集所有进程的统计信息
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)
