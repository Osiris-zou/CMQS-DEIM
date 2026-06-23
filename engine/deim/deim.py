"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.

CMQS modification: propagate the current training epoch to the decoder so that
beta(t) and T_exit follow the configured curriculum schedule.
"""
# Modified by the CMQS-DEIM authors, 2026.
# Changes include explicit propagation of training targets and the current
# epoch from the detector wrapper to the decoder.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from ..core import register

__all__ = ['DEIM']


@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder']

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, targets=None, epoch=None):
        x = self.backbone(x)
        x = self.encoder(x)
        return self.decoder(x, targets, epoch=epoch)

    def deploy(self):
        self.eval()
        for module in self.modules():
            if hasattr(module, 'convert_to_deploy'):
                module.convert_to_deploy()
        return self
