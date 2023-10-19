import os
import sys
import argparse
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize

# We assume fast-reid was cloned there and patched for python 3.11
sys.path.append('')  # we assume fast_reid project folder exists under run_scripts


from src.strong_sort.run_scripts.fast_reid.fastreid.config import get_cfg
from src.strong_sort.run_scripts.fast_reid.fastreid.engine import DefaultPredictor


class ReidPredictor:
    def __init__(self, cfg_path: str,
                 model_path: str,
                 input_size: tuple,
                 device: str = 'cuda:0'):
        assert os.path.exists(cfg_path)
        assert os.path.exists(model_path)
        self.input_size = input_size
        self.device = device
        self.cfg = get_cfg()
        self.cfg.merge_from_file(cfg_path)
        self.cfg.MODEL.DEVICE = self.device
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.freeze()
        self.model = DefaultPredictor(self.cfg)
        self.norm = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        torch.backends.cudnn.benchmark = self.cfg.CUDNN_BENCHMARK

    def _preprocess(self, img: np.ndarray, preds: list)\
            -> torch.Tensor:
        tensor = torch.from_numpy(img).to(self.device, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).moveaxis(-1, 1)  # 1x3xHxW
        # tensor = self.norm(tensor)  # Seems to be unnessesary
        tensors = []
        for pred in preds:
            x1, y1, x2, y2 = map(int, pred['bbox'])
            patch = tensor[:, :, y1:y2, x1:x2]
            patch = F.interpolate(patch, self.input_size, mode='bilinear',
                                  align_corners=True)
            tensors.append(patch)
        return torch.cat(tensors)

    def _postprocess(self, features: torch.Tensor) -> np.ndarray:
        features = F.normalize(features)
        features = features
        return features

    def predict(self, img: np.ndarray, preds: list):
        tensor = self._preprocess(img, preds)
        res = self.model(tensor)
        res = self._postprocess(res)
        return res


if __name__ == '__main__':
    cfg_path = '/workdir/data/ext_models/bagtricks_S50.yml'
    model_path = '/workdir/data/ext_models/DukeMTMC_BoT-S50.pth'
    test_img_path1 = '/workdir/data/video/test_img/img0.png'
    bboxes1 = [{'bbox': [712, 387, 903, 683]},
               {'bbox': [1106, 174, 1637, 728]},
               {'bbox': [1014, 390, 1101, 548]}]

    test_img_path2 = '/workdir/data/video/test_img/img1.png'
    bboxes2 = [{'bbox': [708, 386, 900, 680]},
               {'bbox': [1218, 305, 1464, 636]},
               {'bbox': [1014, 390, 1101, 548]}]

    predictor = ReidPredictor(cfg_path, model_path,
                              input_size=(256, 128), device='cpu')

    img1 = cv2.imread(test_img_path1)
    img2 = cv2.imread(test_img_path2)

    res1 = predictor.predict(img1.copy(), bboxes1)
    res2 = predictor.predict(img2.copy(), bboxes2)

    # print(np.linalg.norm(res, axis=1))
    print(F.cosine_similarity(res1, res2, dim=1))
