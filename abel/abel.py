import torch
from torch import nn, optim

from abel.utils import get_weight_norm

import warnings


class ABEL(optim.lr_scheduler._LRScheduler):
    """
    Automatic, Bouncing into Equilibration Learning rate scheduler.
    
    Args:
        optimizer (torch.optim.Optimizer): torch based optimizer
        decay (float): LR decay(default=0.1)
        last_epoch (int): Last executed epoch(default=-1)
        current_norm (torch.Tensor): current weight norm of model(default=None)
        norm_t_1 (torch.Tensor): t-1 weight norm of model(default=None)
        norm_t_2 (torch.Tensor): t-2 weight norm of model(default=None)
        verbose (bool): Verbosity(default=False)
    """

    def __init__(self, optimizer, decay: float=0.1, last_epoch: int=-1, current_norm: torch.Tensor=None, norm_t_1: torch.Tensor=None, norm_t_2: torch.Tensor=None, verbose: bool=False):
        self.decay = decay

        self.current_norm = current_norm
        self.norm_t_1 = norm_t_1
        self.norm_t_2 = norm_t_2
        self.reached_min = False
        self.decay_level = 1.

        if current_norm is None:
            self.current_norm = get_weight_norm(optimizer.param_groups)
            self.norm_t_1 = norm_t_1
            self.norm_t_2 = norm_t_2

        super(ABEL, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if (self.last_epoch == 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.decay ** (self.decay_level)
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        if self.last_epoch >= 2:
            assert self.current_norm is not None
            assert self.norm_t_1 is not None
            assert self.norm_t_2 is not None
            
            if (self.current_norm - self.norm_t_1) * (self.norm_t_1 - self.norm_t_2) < 0:
                if self.reached_min:
                    self.reached_min = False
                    self.decay_level += 1
                    return [base_lr * self.decay ** (self.decay_level)
                            for base_lr in self.base_lrs]
                else:
                    self.reached_min = True
                    return [base_lr * self.decay ** (self.decay_level)
                            for base_lr in self.base_lrs]
        
        return [base_lr
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.last_epoch >= 2:
            self.norm_t_2 = self.norm_t_1
            self.norm_t_1 = self.current_norm
            self.current_norm = get_weight_norm(self.optimizer.param_groups)
        elif self.last_epoch == 1:
            self.norm_t_1 = self.current_norm
            self.norm_t_2 = self.current_norm
            self.current_norm = get_weight_norm(self.optimizer.param_groups)

        super(ABEL, self).step(epoch)
