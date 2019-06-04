from collections import defaultdict

import torch
from torch.optim import Optimizer


class ExtraOptimizer:
    def __init__(self, optimizer: Optimizer, subsampling=1):
        self.optimizer = optimizer
        self.param_groups = optimizer.param_groups

        self.subsampling = subsampling

        self.extra_state = defaultdict(dict)
        for group in self.param_groups:
            for p in group['params']:
                self.extra_state['backup'][p] = p.data.clone()

    def share_memory(self):
        self.optimizer.share_memory()
        for group in self.param_groups:
            for p in group['params']:
                self.extra_state['backup'][p].share_memory_()

    def _save_backup(self):
        for group in self.param_groups:
            for p in group['params']:
                self.extra_state['backup'][p].copy_(p.data)

    def deextrapolate(self):
        for group in self.param_groups:
            for p in group['params']:
                p.data.copy_(self.extra_state['backup'][p])

    def state_dict(self):
        for group in self.param_groups:
            for p in group['params']:
                self.optimizer.state[p] = dict(**self.optimizer.state[p],
                                               **self.extra_state[p])
        state_dict = self.optimizer.state_dict()
        for group in self.param_groups:
            for p in group['params']:
                for key in self.extra_state[p]:
                    self.optimizer.state[p].pop(key)
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group['params']:
                for key in self.extra_state[p]:
                    self.extra_state[key][p] = self.optimizer.state[p].pop(key)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _transform_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad /= self.subsampling

    def step(self, closure=None, extrapolate=False):
        update = not extrapolate
        if update:
            self.deextrapolate()
        self._transform_grad()
        loss = self.optimizer.step(closure)
        if update:
            self._save_backup()
        return loss


class ExtraOptimizerVR(ExtraOptimizer):
    def __init__(self, optimizer: Optimizer, subsampling=1):
        super().__init__(optimizer, subsampling)
        for group in self.param_groups:
            for p in group['params']:
                self.extra_state['grad_table'][p] = None

    def _transform_grad(self):
        super()._transform_grad()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad.clone()
                    if self.extra_state['grad_table'][p] is None:
                        self.extra_state['grad_table'][p] = grad
                    else:
                        p.grad -= (1 - self.subsampling) * self.extra_state['grad_table'][p]
                        self.extra_state['grad_table'][p].copy_(grad)
                else:
                    # Use the grad stat
                    if self.extra_state['grad_table'][p] is not None:
                        p.grad = self.extra_state['grad_table'][p]