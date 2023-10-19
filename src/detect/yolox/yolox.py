import torch
import torch.nn as nn

from src.detect.yolox.yolo_head import YOLOXHead
from src.detect.yolox.yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone: nn.Module,
                 head: nn.Module):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(0)

        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        outputs = self.head(fpn_outs)

        # if self.training:
        #     assert targets is not None
        #     loss, iou_loss, conf_loss,  l1_loss, num_fg, iou = self.head(
        #         fpn_outs, targets, x
        #     )
        #     outputs = {
        #         "total_loss": loss,
        #         "iou_loss": iou_loss,
        #         "l1_loss": l1_loss,
        #         "conf_loss": conf_loss,
        #         "num_fg": num_fg,
        #         "iou": iou
        #     }
        # else:
        #     outputs = self.head(fpn_outs)

        return outputs


if __name__ == "__main__":
    model = YOLOX()
    inp = torch.randn(1, 3, 640, 640)
    print(model(inp))
