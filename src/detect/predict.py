import os
import json
import argparse
from typing import TypeVar

import cv2
import numpy as np
import torch
import torchvision.ops as ops
from argus import Model, load_model

from src.detect.utils import Tiler
from src.detect.transforms import val_transform
from src.detect.yolox.metamodel import YOLOXMetaModel
from src.detect.yolox.postprocess import SWPostprocessor

MODEL_INPUT = (640, 640)
BBOX_COLOR = (255, 0, 0)


def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection on Video')
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='Path to the argus model file')
    parser.add_argument('-i', '--video_path', type=str, required=True,
                        help="Path to the input video")
    parser.add_argument('-vis', '--visualization_video_path', type=str,
                        default=None, help='Path to save visualization video')
    parser.add_argument('-o', '--json_path', type=str, required=True,
                        help='Path to save predictions as JSON')
    parser.add_argument('-d', '--device', type=str, default='cuda:0',
                        help='Device to use for the model inference')
    args = parser.parse_args()
    return args


TensorLike = TypeVar("TensorLike", np.ndarray, torch.Tensor)


class Predictor:
    """Model predictor for YOLOX detector.

    Args:
        model_path (str): Path to the Argus model.
        device (str, optional): Device to use for inference. Defaults to
            'cuda:0'.
        input_size (tuple[int, int], optional): Input shape of the model
            (W, H). Defaults to MODEL_INPUT.
        frame_size (tuple[int, int], optional): Original frame shape (W, H).
            Defaults to MODEL_INPUT.
        use_tiles (bool, optional): Whether to use tiles for prediction.
            Defaults to False.
        tiler_overlap: (float, optional): Minimal desired overlap as a
            fraction of the tile side length. Defaults to 0.5.
        margin (int, optional): Margins (in px) to exclude all predictions
            touching crop boundary. Used if use_tiles=True. Defaults to 20.
        sw_nms_threshold (float, optional): NMS threshold for the tiles
            predictions. Used if use_tiles=True. Defaults to 0.2.
        nms_threshold (float, optional): NMS threshold for integration of tiled
            and standard detection. Used if use_tiles=True. Defaults to 0.75.
    """

    def __init__(self, model_path: str, device: str = 'cuda:0',
                 input_size: tuple = MODEL_INPUT,
                 frame_size: tuple = MODEL_INPUT,
                 use_tiles: bool = True,
                 tiler_overlap: float = 0.5,
                 magrin: int = 20,
                 sw_nms_threshold: float = 0.2,
                 nms_threshold: float = 0.75
                 ):
        self.device = device
        self.input_size = input_size
        self.frame_size = frame_size
        self.use_tiles = use_tiles
        self.nms_threshold = nms_threshold
        self.model = self._load_model(model_path)
        self.transform = val_transform(MODEL_INPUT, 127)
        self.scale = min(self.input_size[0] / self.frame_size[0],
                         self.input_size[1] / self.frame_size[1])
        new_w = int(self.frame_size[0] * self.scale)
        new_h = int(self.frame_size[1] * self.scale)
        self.x0y0 = (int((self.input_size[0] - new_w) / 2),
                     int((self.input_size[1] - new_h) / 2))
        self.tiler = Tiler(frame_size, input_size, tiler_overlap)
        self.sw_processor = SWPostprocessor(
            pred_size=input_size, margin=magrin,
            nms_threshold=sw_nms_threshold)

    def _load_model(self, model_path: str) -> Model:
        assert os.path.exists(model_path), f'Model {model_path} does not exist'
        print(self.device)
        model = load_model(model_path, device=self.device)
        model.eval()
        return model

    def _prepare_tensor(self, frame: np.ndarray) -> torch.Tensor:
        tensor = self.transform(frame, None)[0]
        if self.use_tiles:
            tiles = []
            for tile in self.tiler.tiles:
                x0, y0 = tile
                tiles.append(self.transform(
                    frame[y0:y0+self.input_size[1],
                          x0:x0+self.input_size[0], :])[0])
            tensor = torch.stack([tensor, *tiles])
        else:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _process_tiled_predictions(
            self, general_preds: TensorLike,
            tiled_preds) -> TensorLike:
        preds = general_preds
        pred_tiles = self.sw_processor(tiled_preds, self.tiler.tiles)
        if pred_tiles is not None and general_preds is not None:
            preds = torch.cat([general_preds, pred_tiles], 0)
            preds_idx = ops.nms(preds[:, :4], preds[:, 4], self.nms_threshold)
            preds = preds[preds_idx]
        return preds

    def predict_img(self, frame: np.ndarray) -> np.ndarray:
        """Make prediction on a single image.

        Args:
            frame (np.ndarray): _description_

        Returns:
            np.ndarray: Prediction of shape (N_bbox, 6). Each bbox is
                represented by x1, y1, x2, y2, object confidence.

        """
        res = self.model.predict(self._prepare_tensor(frame))

        preds = self._rescale_pred(res[0])
        if self.use_tiles:
            preds = self._process_tiled_predictions(preds, res[1:])
        preds = self._filter_pred(preds[:, :5]).cpu().numpy()
        return preds

    def predict_batch(self, frames):
        tensors = [self._prepare_tensor(frame) for frame in frames]
        tensor = torch.cat(tensors, 0)
        raw_pred = self.model.predict(tensor)
        n_per_frame = 1  # Number of predictions, correspond to each frame
        if self.use_tiles:
            n_per_frame += len(self.tiler.tiles)
        res_list = [raw_pred[i:i + n_per_frame]
                    for i in range(0, len(raw_pred), n_per_frame)]
        preds_list = []
        for res in res_list:
            pred = self._rescale_pred(res[0])
            if self.use_tiles:
                pred = self._process_tiled_predictions(pred, res[1:])
            preds_list.append(self._filter_pred(pred[:, :5]).cpu().numpy())
        return preds_list

    def _rescale_pred(self, pred: TensorLike) -> TensorLike:
        pred[:, 0:4:2] = (pred[:, 0:4:2] - self.x0y0[0]) / self.scale
        pred[:, 1:4:2] = (pred[:, 1:4:2] - self.x0y0[1]) / self.scale
        return pred

    def _filter_pred(self, pred: TensorLike) -> TensorLike:
        # Remove zero-sized preds
        x_mask = (pred[:, 2] - pred[:, 0] >= 1.0) \
            * (pred[:, 2] <= self.frame_size[0]) * (pred[:, 0] >= 0.0)
        y_mask = (pred[:, 3] - pred[:, 1] >= 1.0) \
            * (pred[:, 3] <= self.frame_size[1]) * (pred[:, 1] >= 0.0)
        return pred[x_mask*y_mask]


def main():
    args = parse_args()

    if args.device.startswith('cuda'):
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'video_codec;h264_cuvid'

    cap = cv2.VideoCapture(args.video_path, cv2.CAP_FFMPEG)

    # Get video frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    predictor = Predictor(model_path=args.model_path, device=args.device,
                          frame_size=(frame_width, frame_height))

    # Prepare video writer for visualization if requested
    if args.visualization_video_path:
        out = cv2.VideoWriter(args.visualization_video_path,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, (frame_width, frame_height))

    # Initialize predictions list
    predictions: dict[int, list[dict]] = {}
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        detected_objects = predictor.predict_img(frame)

        # Visualize detections on the frame
        for obj in detected_objects:
            x1, y1, x2, y2 = map(int, obj[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, 2)

        predictions[i] = [{'bbox': list(map(float, pred[:4])),
                           'conf': float(pred[4])}
                          for pred in detected_objects]

        if args.visualization_video_path:
            out.write(frame)

        i += 1

    # Release video capture and writer
    cap.release()
    if args.visualization_video_path:
        out.release()

    with open(args.json_path, "w") as json_file:
        json.dump(predictions, json_file)


if __name__ == "__main__":
    main()
