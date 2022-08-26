import math
import numpy as np
from typing import Dict

import torch
import torch.optim as optim


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val:float, n:int=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args:Dict, optimizer:Any, epoch:int, lr:float):
    """Learning rate adjustment methods

    Args:
        args (Dict): Parsed arguments
        optimizer (Any): optimizer
        epoch (int): Current epoch
        lr (float): the value of the learning rate
    """
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate**3)
        lr = eta_min + (lr - eta_min) * \
            (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate**steps)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_learning_rate(args:Dict, epoch:int, batch_id:int, total_batches:int, optimizer:Any):
    """Learning rate warmup method

    Args:
        args (Dict): Parse arguments
        epoch (int): Current epoch
        batch_id (int): The number of the current batch.
        total_batches (int): The number of total batch. 
        optimizer (Any): optimizer
    """
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def set_optimizer(opt, model:Any):
    """Initialize the optimizer.

    Args:
        opt (_type_): Parsed arguments. 
    """

    optimizer = optim.Adam(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    return optimizer


def save_model(model:Any, optimizer:Any, opt:Dict, epoch:int, save_file:str):
    """Save the model

    Args:
        save_file (str): The address to save the model. 
    """

    print("==> Saving...")
    state = {
        "opt": opt,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, save_file)
    del state
