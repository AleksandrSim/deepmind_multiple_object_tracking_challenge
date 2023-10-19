import numpy as np
import torch
import torchvision.ops as ops


class NMSPostprocessor:
    def __init__(self,
                 conf_thre: float = 0.3,
                 nms_thre: float = 0.45):
        self.conf_thre = conf_thre
        self.nms_thre = nms_thre
        self.class_agnostic = True

    def __call__(self, pred):
        prediction = pred[0]
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
        prediction[:, :, 4] = torch.nn.functional.sigmoid(prediction[:, :, 4])

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(
                image_pred[:, 4: 5], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] >= self.conf_thre).squeeze()
            # Original was: (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat(
                (image_pred[:, :4], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue
            # NMS
            if self.class_agnostic:
                nms_out_index = ops.nms(
                    detections[:, :4],
                    detections[:, 4],
                    self.nms_thre,
                )
            else:
                # It doesn't work as we don't have classes
                nms_out_index = ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4],
                    detections[:, 6],
                    self.nms_thre,
                )

            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output


class SWPostprocessor:
    """Sliding window detections processor.

    Developed with some inspiration from ideas of SAHI.

    Args:
        pred_size (tuple[int, int]): size of each tile (W, H).
        margin (int, optional): Margins (in px) to exclude all predictions touching
            crop boundary. Defaults to 20.
        nms_threshold (float, optional): NMS threshold. Defaults to 0.2.
    """

    def __init__(self, pred_size: tuple, margin: int = 20,
                 nms_threshold: float = 0.2):

        self.pred_size = pred_size
        self.margin = margin
        self.nms_threshold = nms_threshold

    def __call__(self, preds: list,
                 tiles: list):
        assert len(preds) == len(tiles)
        pred_tiles = []
        for pred, tile in zip(preds, tiles):
            if pred is not None and pred.shape[0] > 0:
                # Clean up bboxes, which are touching the image boundary
                x0, y0, x1, y1 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
                conf = pred[:, 4]
                outside_boundary_mask = \
                    (x0 < self.margin)\
                    | (x0 > (self.pred_size[0] - self.margin))\
                    | (x1 < self.margin)\
                    | (x1 > (self.pred_size[0] - self.margin)) \
                    | (y0 < self.margin) \
                    | (y0 > (self.pred_size[1] - self.margin)) \
                    | (y1 < self.margin) \
                    | (y1 > (self.pred_size[1] - self.margin)) \
                    | (conf < 0.1)
                pred = pred[~outside_boundary_mask]
                pred[:, 0:4:2] += tile[0]
                pred[:, 1:4:2] += tile[1]
                pred_tiles.append(pred)
        if len(pred_tiles) > 0:
            pred = torch.concat(pred_tiles, 0)
            nms_out_index = ops.nms(
                pred[:, :4],
                pred[:, 4],
                self.nms_threshold,
            )
            pred = pred[nms_out_index]
            return pred
