import math

import torch
import torch.nn as nn

from src.detect.yolox.utils import meshgrid
from src.detect.yolox.network_blocks import DWConv, BaseConv


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        use_l1: bool = False
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        # self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        # self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            # self.cls_convs.append(
            #     nn.Sequential(
            #         *[
            #             Conv(
            #                 in_channels=int(256 * width),
            #                 out_channels=int(256 * width),
            #                 ksize=3,
            #                 stride=1,
            #                 act=act,
            #             ),
            #             Conv(
            #                 in_channels=int(256 * width),
            #                 out_channels=int(256 * width),
            #                 ksize=3,
            #                 stride=1,
            #                 act=act,
            #             ),
            #         ]
            #     )
            # )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            # self.cls_preds.append(
            #     nn.Conv2d(
            #         in_channels=int(256 * width),
            #         out_channels=self.num_classes,
            #         kernel_size=1,
            #         stride=1,
            #         padding=0,
            #     )
            # )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = use_l1
        self.strides = strides
        self.grids = []

    def initialize_biases(self, prior_prob):
        # for conv in self.cls_preds:
        #     b = conv.bias.view(1, -1)
        #     b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        #     conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (reg_conv, stride_this_level, x) in enumerate(
            zip(self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            # cls_x = x
            reg_x = x

            # cls_feat = cls_conv(cls_x)
            # cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # if self.training:
            output = torch.cat([reg_output, obj_output], 1)
            output, grid = self.get_output_and_grid(
                output, k, stride_this_level, xin[0].type()
            )
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.zeros(1, grid.shape[1])
                .fill_(stride_this_level)
                .type_as(xin[0])
            )
            if self.use_l1:
                batch_size = reg_output.shape[0]
                hsize, wsize = reg_output.shape[-2:]
                reg_output = reg_output.view(
                    batch_size, 1, 4, hsize, wsize
                )
                reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                    batch_size, -1, 4
                )
                origin_preds.append(reg_output.clone())

            # else:
            #     output = torch.cat([reg_output, obj_output.sigmoid()], 1)

            outputs.append(output)

        # if self.training:
        return torch.cat(outputs, 1), x_shifts, y_shifts, expanded_strides,\
            origin_preds

        # else:
        #     self.hw = [x.shape[-2:] for x in outputs]
        #     [batch, n_anchors_all, 85]
        #     outputs = torch.cat(
        #         [x.flatten(start_dim=2) for x in outputs], dim=2
        #     ).permute(0, 2, 1)
        #     if self.decode_in_inference:
        #         return self.decode_outputs(outputs, dtype=xin[0].type())
        #     else:
        #         return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]

        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, hsize * wsize, -1)

        grid = None
        if len(self.grids) > k:
            grid = self.grids[k]

        if grid is None or grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(
                1, 1, hsize, wsize, 2).type(dtype).to(output.device)
            if len(self.grids) > k:
                self.grids[k] = grid
            else:
                self.grids.append(grid)

        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype).to(outputs.device)
        strides = torch.cat(strides, dim=1).type(dtype).to(outputs.device)

        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:]
        ], dim=-1)
        return outputs
