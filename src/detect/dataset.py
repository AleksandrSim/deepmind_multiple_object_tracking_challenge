import os
import json
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class DetectDataset(Dataset):
    def __init__(self, img_dir: str, annot_file: str,
                 samples_range: Tuple[float, float] = (0.0, 1.0),
                 transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.annot_file = annot_file
        self.transform = transform
        with open(annot_file, 'r') as f:
            data = json.load(f)

        i = 0
        raw_data = {}
        for video in data:
            if len(data[video]) > 1:
                for frame_id in data[video]:
                    img_path = os.path.join(img_dir, video, frame_id + '.png')
                    if not os.path.exists(img_path):
                        continue
                    raw_data[i] = {
                        'img_path': img_path,
                        'annot': data[video][frame_id]
                    }
                    i += 1
        N_objects = len(raw_data)
        start_key = max(0, int(round(N_objects * samples_range[0])))
        end_key = min(int(round(N_objects * samples_range[1])), N_objects - 1)
        self.data = {k-start_key: v for k, v in raw_data.items()
                     if start_key <= k <= end_key}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx]['img_path'])

        # 0 - placeholder for classes as in the original YOLOX
        bboxes = [(0, *item['bounding_boxes'])
                  for item in self.data[idx]['annot'] if not item['is_masked']]
        annot = np.array(bboxes, dtype=np.float32)
        annot[:, 1::2] *= img.shape[1]
        annot[:, 2::2] *= img.shape[0]
        if self.transform is not None:
            img, annot = self.transform(img, annot)
        return img, annot


if __name__ == '__main__':
    img_dir = '/workdir/data/datasets/train/'
    annot_file = '/workdir/data/annot/converted_train.json'
    dataset = DetectDataset(img_dir, annot_file, samples_range=(0.0, 0.2))
    img, annot = dataset[0]
    print('Dataset size:', len(dataset))
    print('Image shape:', img.shape)
    print('Annotation:\n', annot)
