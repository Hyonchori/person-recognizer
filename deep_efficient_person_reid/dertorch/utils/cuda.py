""" CUDA / AMP utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch

try:
    from apex import amp
    has_apex = True
except ImportError:
    amp = None
    has_apex = False


class ApexScaler:
    state_dict_key = "amp"

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False):
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(create_graph=create_graph)
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                amp.master_params(optimizer), clip_grad)

    def step(self, optimizer):
        optimizer.step()

    def state_dict(self):
        if 'state_dict' in amp.__dict__:
            return amp.state_dict()

    def load_state_dict(self, state_dict):
        if 'load_state_dict' in amp.__dict__:
            amp.load_state_dict(state_dict)


class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)

    def step(self, optimizer, clip_grad=None, parameters=None):
        if clip_grad is not None:
            assert parameters is not None
            # unscale the gradients of optimizer's assigned params in-place
            self._scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
