from typing import List

import torch
import torch.nn as nn
from argus import Model
from argus.utils import deep_to, deep_detach
from argus.engine import State

from src.detect.yolox.loss import YOLOXLoss
from src.detect.yolox.yolox import YOLOX
from src.detect.yolox.yolo_head import YOLOXHead
from src.detect.yolox.yolo_pafpn import YOLOPAFPN
from src.detect.yolox.postprocess import NMSPostprocessor


class YOLOXArgus(nn.Module):
    """YOLOX model module.

    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, num_classes: int = 80,
                 in_channels: List[int] = [256, 512, 1024],
                 depth: float = 1.00,
                 width: float = 1.00,
                 act: str = "silu",
                 use_l1: bool = False,
                 freeze_encoder: bool = False):
        super().__init__()

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        backbone = YOLOPAFPN(depth, width,
                             in_channels=in_channels, act=act)
        if freeze_encoder:
            for param in backbone.parameters():
                param.requires_grad = False
        head = YOLOXHead(num_classes, width,
                         in_channels=in_channels, act=act, use_l1=use_l1)
        self.model = YOLOX(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class YOLOXMetaModel(Model):
    nn_module = YOLOXArgus
    loss = YOLOXLoss
    optimizer = torch.optim.Adam
    prediction_transform = NMSPostprocessor

    def __init__(self, params):
        super().__init__(params)
        self.amp = (False if 'amp' not in self.params
                    else bool(self.params['amp']))
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None

    def train_step(self, batch, state: State) -> dict:
        self.train()
        self.optimizer.zero_grad()
        inp_tensor, target = deep_to(batch, device=self.device,
                                     non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.amp):
            prediction = self.nn_module(inp_tensor)
        loss = self.loss(prediction, target)
        loss, loss_iou, loss_obj, loss_l1, _, iou = loss

        if self.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        prediction = self.prediction_transform(prediction)
        return {
            'prediction': prediction,
            'target': target,
            'loss': loss.item(),
            'metrics': {
                'loss_iou': loss_iou.item(),
                'loss_obj': loss_obj.item(),
                'loss_l1': loss_l1,
                'iou': iou.item()
            }
        }

    def val_step(self, batch, state: State) -> dict:
        self.eval()
        with torch.no_grad():
            inp_tensor, target = deep_to(batch, device=self.device,
                                         non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.amp):
                prediction = self.nn_module(inp_tensor)
            loss = self.loss(prediction, target)
            loss, loss_iou, loss_obj, loss_l1, _, iou = loss

            prediction = deep_detach(prediction)
            target = deep_detach(target)
            prediction = self.prediction_transform(prediction)
            return {
                'prediction': prediction,
                'target': target,
                'loss': loss.item(),
                'metrics': {
                    'loss_iou': loss_iou.item(),
                    'loss_obj': loss_obj.item(),
                    'loss_l1': loss_l1,
                    'iou': iou.item()
                }
            }
