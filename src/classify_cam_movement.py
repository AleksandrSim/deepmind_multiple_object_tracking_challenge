import os
import json
import sys
import numpy as np
from collections import deque
# Your function for calculating Euclidean distance

import optuna

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Calculate the centroid for a given frame
def calculate_frame_centroid(frame_detections):
    total_x, total_y, count = 0, 0, 0
    for detection in frame_detections:
        x1, y1, x2, y2 = detection["bbox"]
        centroid_x, centroid_y = (x1 + x2) / 2, (y1 + y2) / 2
        total_x += centroid_x
        total_y += centroid_y
        count += 1
    if count == 0:
        return None, None
    return total_x / count, total_y / count

# Check if the camera is moving based on object detections
def is_camera_moving_based_on_detections(detections, max_acceptable_distance=500, num_frames_to_check=4, distance_moved= 20, ratio = 0.8):
    significant_shifts = 0
    total_frames = len(detections.keys()) - 1

    sorted_frames = sorted(detections.keys(), key=int)
    
    last_centroids = deque(maxlen=num_frames_to_check)
    
    for i, frame in enumerate(sorted_frames):
        centroid_x, centroid_y = calculate_frame_centroid(detections[frame])
        
        if centroid_x is None or centroid_y is None:
            continue

        if len(last_centroids) == num_frames_to_check:
            avg_last_centroid_x = np.mean([x for x, _ in last_centroids])
            avg_last_centroid_y = np.mean([y for _, y in last_centroids])

            distance_moved = euclidean_distance(avg_last_centroid_x, avg_last_centroid_y, centroid_x, centroid_y)
            
            # Ignore centroids that are too far apart to be the same object
            if distance_moved > max_acceptable_distance:
                continue

            if distance_moved > 25:  # Threshold, adjust as needed
                significant_shifts += 1

        last_centroids.append((centroid_x, centroid_y))
    
    if total_frames == 0:
        return False
    
    if significant_shifts / total_frames > ratio:  # At least 80% of frames showed significant shifts
        return True
    
    return False

sys.path.append('')
from src.strong_sort.load_config import load_config


def objective(trial):
    # Hyperparameters to be optimized
    max_acceptable_distance = trial.suggest_int('max_acceptable_distance', 300, 600)
    num_frames_to_check = trial.suggest_int('num_frames_to_check', 2, 6)
    distance_moved_threshold = trial.suggest_int('distance_moved_threshold', 20, 60)
    percentage_threshold = trial.suggest_float('percentage_threshold', 0.6, 1.0)

    cam = CameraMovement('/Users/aleksandrsimonyan/Desktop/deepmind_updated/data/raw_preds/valid_detect/',
                         os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['training_json']))

    error = cam.check_json(max_acceptable_distance, num_frames_to_check, distance_moved_threshold, percentage_threshold)
    
    return error
class CameraMovement:
    def __init__(self, predictions_path, json_file=None):
        self.predictions_path = predictions_path
        self.json_file = json_file
        with open(self.json_file, 'r') as f:
            self.data = json.load(f)
        self.predictions_folder = os.listdir(self.predictions_path)
        self.predictions_folder = [i for i in self.predictions_folder if os.path.isdir(os.path.join(self.predictions_path, i))]
        self.moving_cameras = {False: 0, True: 0}
        self.estimated_moving_cameras = {False: 0, True: 0}  # Initialize this

    def check_json(self, max_acceptable_distance, num_frames_to_check, distance_moved_threshold, percentage_threshold):
        correct_moving = 0
        incorrect_moving = 0
        correct_non_moving = 0
        incorrect_non_moving = 0

        for video in self.data.keys():
            actual_is_moving = self.data[video]['metadata']['is_camera_moving'] == 1
            self.moving_cameras[actual_is_moving] += 1

            tracker_file_path = os.path.join(self.predictions_path, video, 'detections.json')
            if os.path.exists(tracker_file_path):
                with open(tracker_file_path, 'r') as f:
                    detections = json.load(f)
                estimated_is_moving = is_camera_moving_based_on_detections(detections, max_acceptable_distance, num_frames_to_check, distance_moved_threshold, percentage_threshold)
                self.estimated_moving_cameras[estimated_is_moving] += 1

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
        return error
if __name__ == '__main__':
    cfg = load_config()
    
    # Optimize
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)  # Number of trials

    # Best hyperparameters
    print(f"The best hyperparameters are {study.best_params} with a error of {study.best_value}.")