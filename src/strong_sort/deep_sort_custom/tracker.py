from __future__ import absolute_import

import logging

import yaml
import numpy as np

from . import iou_matching, kalman_filter, linear_assignment
from .track import Track, TrackState

with open("config.yml", 'r') as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.9, max_age=1000, n_init=1,
                 max_track_id=0, generate_ids=False):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.tracks: list[Track] = []
        self._next_id = max_track_id + 1000
        self.gt_id_map: dict[int, int] = {}
        self.generate_ids = generate_ids

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict()

    def camera_update(self, video, frame):
        for track in self.tracks:
            track.camera_update(video, frame)

    def update(self, detections, frame_id=None):
        """Perform measurement update and track management."""

        # Run matching cascade.
        # print("Detections before tracker update: ", [d.track_id for d in detections if d.track_id >0])
        matches, unmatched_tracks, unmatched_detections = self._match(
            detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            # print('Match', self.tracks[track_idx].track_id,
            #       detections[detection_idx].track_id)
            this_track = self.tracks[track_idx]
            this_detection = detections[detection_idx]
            this_track.update(this_detection)
            if this_detection.track_id >= 0:  # This is a GT detection
                if this_track.track_id != this_detection.track_id:  # Mismatch
                    if this_detection.track_id not in self.gt_id_map:
                        this_track.track_id = this_detection.track_id
                        this_track.gt = True
                        self.gt_id_map[this_track.track_id] = track_idx
                    else:
                        print('Mismatch, but existed!')

        for track_idx in unmatched_tracks:
            unmatched_track = self.tracks[track_idx]
            if unmatched_track.gt:
                # Tracks with GT without detection, considered stationary
                # TODO: Check if the track is within image, based on Kalman
                # prediction
                unmatched_track.update_constant()
                # print('Unmatched track', unmatched_track.track_id,
                #       unmatched_track.mean)
            else:
                self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        # self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            # TODO: Check how to handle EMA
            # if not opt.EMA:
            #     track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                cost_matrix, tracks, dets, track_indices, detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):

        track_id = detection.track_id
        if track_id >= 0:  # It's a ground-truth track
            print(f'currenlty initializing {track_id}')

            if track_id not in self.gt_id_map:
                # print("Initiating track for ID: ", detection.track_id)
                track = Track(detection, track_id, 0, self.max_age, gt=True)
                track.state = TrackState.Confirmed
                self.tracks.append(track)
                self.gt_id_map[track_id] = len(self.tracks) - 1
            else:
                # print("Forse update track from GT: ", detection.track_id)
                self.tracks[self.gt_id_map[track_id]].update_force(detection)
        else:  # It's a predicted track
            if self.generate_ids:
                self.tracks.append(Track(
                    detection, self._next_id, self.n_init, self.max_age,
                    gt=False
                ))
                self._next_id += 1
