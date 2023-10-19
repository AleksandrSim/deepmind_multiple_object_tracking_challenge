import os
import json
import warnings
from collections import defaultdict

import cv2
from tqdm import tqdm

LABELS_PATH = '/workdir/data/annot/all_train.json'
VIDEOS_PATH = '/workdir/data/train_videos/'
DATASETS_PATH = '/workdir/data/datasets/train/'


def main():
    with open(LABELS_PATH, 'r') as json_file:
        data = json.load(json_file)

    for video in tqdm(data):
        tracking_annot = data[video]['object_tracking']
        frames_with_annot = defaultdict(int)  # frame_id: number_of_annotations
        for obj in tracking_annot:
            for frame in obj['frame_ids']:
                frames_with_annot[frame] += 1
        video_file_path = os.path.join(VIDEOS_PATH, video+'.mp4')
        cap = cv2.VideoCapture(video_file_path)
        dataset_path = os.path.join(DATASETS_PATH, video)
        os.makedirs(dataset_path, exist_ok=True)
        for frame_id in sorted(k for k, v in frames_with_annot.items() if v > 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                output_path = os.path.join(dataset_path, f'{frame_id}.png')
                cv2.imwrite(output_path, frame)
            else:
                warnings.warn(
                    (f"Unable to read frame {frame_id} from {video}"))
        cap.release()
        del cap


if __name__ == "__main__":
    main()
