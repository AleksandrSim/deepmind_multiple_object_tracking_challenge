import os
import json
import sys
import numpy as np
from collections import deque
# Your function for calculating Euclidean distance

import optuna

sys.path.append('')
from src.strong_sort.load_config import load_config
import cv2


from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class MovementDetector:
    seed: int = 1234
    n_samples: int = 1000
    decay_rate: float = 0.2
    spatial_highpass_filter_width: int = 10
    _ixs: np.ndarray = None
    _lowpass_img: np.ndarray = None

    def get_moving_score(self, image) -> float:
        im_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im_grey_highpass = im_grey - cv2.boxFilter(im_grey, ddepth=-1, ksize=(self.spatial_highpass_filter_width, self.spatial_highpass_filter_width))
        im_flat = im_grey_highpass.reshape(-1)
        if self._ixs is None:
            self._ixs = np.random.RandomState(self.seed).choice(len(im_flat), size=min(len(im_flat), self.n_samples), replace=False)
            if self._lowpass_img is None:
                self._lowpass_img = np.zeros_like(im_flat)
        historical_values = self._lowpass_img[self._ixs]
        current_values = im_flat[self._ixs]
        pixel_correlation = np.corrcoef(historical_values, current_values)
        self._lowpass_img = self.decay_rate * im_flat + (1 - self.decay_rate) * self._lowpass_img
        return 1 - pixel_correlation[0, 1]
    

class CameraMovement:
    def __init__(self, predictions_path, json_file=None):
        self.predictions_path = predictions_path
        self.json_file = json_file
        self.data = None
        if self.json_file:
            with open(self.json_file, 'r') as f:
                self.data = json.load(f)
        
        self.predictions_folder = os.listdir(self.predictions_path)
        self.predictions_folder = [i for i in self.predictions_folder if os.path.join(self.predictions_path, i).endswith('.mp4')]
        self.moving_cameras = {False: 0, True: 0}
        self.estimated_moving_cameras = {False: 0, True: 0}
        self.final =[]

    def check_json(self, moving_score_threshold=0.83, num_frames_to_check=1000):
        correct_moving = 0
        incorrect_moving = 0
        correct_non_moving = 0
        incorrect_non_moving = 0

        for idx, video in enumerate(self.predictions_folder):
            video = video.replace('.mp4','')
            if idx ==20:
                break
            print(f'woring on video {video}')
            if self.data:
                actual_is_moving = self.data[video]['metadata']['is_camera_moving'] == 1
                self.moving_cameras[actual_is_moving] += 1

            video_path = os.path.join(self.predictions_path, video +'.mp4') 
            print(video_path) # Replace with actual video path

            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                movement_detector = MovementDetector()
                frame_count = 0
                moving_frame_count = 0

                while frame_count < num_frames_to_check:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    moving_score = movement_detector.get_moving_score(frame)
                    
                    if moving_score > moving_score_threshold:
                        moving_frame_count += 1

                    frame_count += 1

                cap.release()

                estimated_is_moving = moving_frame_count / num_frames_to_check > 0.03 
                if estimated_is_moving:
                    self.final.append(video)

                self.estimated_moving_cameras[estimated_is_moving] += 1

                if self.data:
                    if estimated_is_moving == actual_is_moving:
                        if estimated_is_moving:
                            correct_moving += 1
                        else:
                            correct_non_moving += 1
                    else:
                        if estimated_is_moving:
                            incorrect_moving += 1
                        else:
                            incorrect_non_moving += 1

                    error = incorrect_moving + incorrect_non_moving

                    print(f"Correctly classified as moving: {correct_moving}")
                    print(f"Incorrectly classified as moving: {incorrect_moving}")
                    print(f"Correctly classified as non-moving: {correct_non_moving}")
                    print(f"Incorrectly classified as non-moving: {incorrect_non_moving}")
                


if __name__ == '__main__':
    cfg = load_config()
    cam = CameraMovement(predictions_path='/Users/aleksandrsimonyan/Desktop/deepmind_updated/data/videos/videos/')
    cam.check_json()
    full_path = os.path.join('/Users', 'aleksandrsimonyan', 'Desktop', 'deepmind_updated', 'data', 'moving.txt')  # Removed leading /
    with open(full_path, 'w') as f:
        for video in cam.final:  # use cam.final instead of self.final
            f.write(video)
            f.write('\n')