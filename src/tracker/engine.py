import logging

import yaml
import numpy as np
import sys
sys.path.append('')
import os
from  src.strong_sort.deep_sort_custom import nn_matching
from src.strong_sort.deep_sort_custom.track import Track
from src.tracker.visualizer import Visualizer
from src.strong_sort.deep_sort_custom.tracker import Tracker
from src.strong_sort.deep_sort_custom.detection import Detection
from src.tracker.cosine_distance import CosineDistance


class TrackerEngine:
    """Run multi-target tracker on a particular sample.

    Args:
        cfg: (dict): Configuration dictionary ['arguments'] part of it.
        detection_file (str): Path to the detections file.
        output_file (str): Path to the tracking output file. This file will
            contain the tracking results on completion.
        min_confidence (float): Detection confidence threshold. Disregard all detections that have
            a confidence lower than this value.
        min_detection_height (int): Detection height threshold. Disregard all detections that have
            a height lower than this value.
        visualizer (Visualizer | None, optional): Class to use for
            visualization, if provided. Defaults to None.

    """

    def __init__(self,
                 cfg: dict,
                 detection_file: str,
                 output_file: str,
                 min_confidence: float,
                 min_detection_height: int,
                 visualizer: Visualizer,
                 track_id_once: bool = True):
        # If track_id_once is True - the track id from GT is given only once to
        # the Tracker, all the following detections are provided with negative
        # track id.
        self.cfg = cfg
        self.detection_file = detection_file
        self.output_file = output_file
        self.min_confidence = min_confidence
        self.min_detection_height = min_detection_height
        self.visualizer = None
        if visualizer is not None:
            self.visualizer = visualizer
            self.visualizer.start()
        self.known_tracks: list[int] = []
        self.track_id_once = track_id_once

        metric = nn_matching.NearestNeighborDistanceMetric(
            CosineDistance(use_torch=False,  device=cfg['device']),
            cfg['max_cosine_distance'], cfg['nn_budget'])

        self.dl_preds = np.load(self.detection_file, allow_pickle=True).item()
        # TODO Check it, maybe we can have more tracks
        max_tracks = max(len(p) for p in self.dl_preds.values())
        self.tracker = Tracker(metric, cfg['max_iou_distance'],
                               cfg['max_age'], cfg['n_init'], max_tracks)
        if cfg['logger_path'] is not None:
            log_filename = f"{cfg['logger_path']}.txt"
            logging.basicConfig(filename=log_filename, level=logging.DEBUG)

    def _get_detections(self, frame_idx: int):
        detection_list = []
        for i, pred in enumerate(self.dl_preds[frame_idx]):
            x1, y1, x2, y2 = pred['bbox']
            confidence = pred['conf']
            track_id = pred['track_id'] if pred['track_id'] != -1 else -i-1
            if self.track_id_once and track_id >= 0:
                if track_id in self.known_tracks:
                    track_id = -i-1
                else:
                    self.known_tracks.append(track_id)
            embedding = None
            if 'embedding' in pred:
                embedding = pred['embedding']
            else:
                embedding = np.ones(2048) / 2048.0
            if confidence > self.min_confidence\
                    and y2-y1 > self.min_detection_height:
                detection_list.append(
                    Detection([x1, y1, x2-x1, y2-y1],
                              confidence, embedding, track_id))
        return detection_list

    def _save_results(self, results: list):
        with open(self.output_file, 'w') as f:
            for row in results:
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                    row[0], row[1], row[2], row[3], row[4], row[5]), file=f)

    def _visualize(self, frame_idx: int, detections,
                   tracks):
        if self.visualizer is not None:
            self.visualizer.draw(frame_idx, detections, tracks)

    def run(self):
        results = []
        for frame_idx in self.dl_preds:
            detections = self._get_detections(frame_idx)
            self.tracker.predict()
            self.tracker.update(detections, frame_idx)
            for track in self.tracker.tracks:
                if track.is_confirmed():
                    bbox = track.to_tlwh()
                    results.append([frame_idx, track.track_id,
                                    bbox[0], bbox[1], bbox[2], bbox[3]])
                    # print(results[-1])
            self._visualize(frame_idx, detections, self.tracker.tracks)

        self._save_results(results)
        if self.visualizer is not None:
            self.visualizer.draw(None, None, None)
            self.visualizer.release()


if __name__ == '__main__':
    with open('config.yml', 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    video_name = 'video_1000'

    video_path = os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['video_path'], video_name +'.mp4')
    output_video_path =   os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['sequence_folder'],f'{video_name}/{video_name}_pred_strongsort.mp4')
    visualizer = Visualizer(video_path, output_video_path)

    sequence_path = os.path.join(cfg['COMMON_PREFIX'],cfg['prepare_files']['sequence_folder'], video_name)

    strongsort_engine = TrackerEngine(
        detection_file= os.path.join(sequence_path, 'predictions.npy' ),
        output_file= os.path.join(sequence_path, 'tracker.txt'),
        min_confidence=0.0,
        min_detection_height=0,
        visualizer=visualizer,
        cfg=cfg['arguments'],
        track_id_once=True,
    )
    strongsort_engine.run()
