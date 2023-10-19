from __future__ import absolute_import

import logging

import yaml
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

from . import kalman_filter

with open("config.yml", 'r') as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
INFTY_COST = 1e+5


def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):

    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    cost_matrix_ = cost_matrix.copy()
    if cfg['arguments']['logger_path']:
        logging.info(f"Initial cost matrix:\n{cost_matrix}")

    indices = linear_assignment(cost_matrix_)
    indices = np.asarray(indices)
    indices = np.transpose(indices)
    if cfg['arguments']['logger_path']:
        logging.info(f"Indices after linear assignment:\n{indices}")

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
            if cfg['arguments']['logger_path']:
                logging.info(f"Detection {detection_idx} is unmatched.")

    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
            if cfg['arguments']['logger_path']:

                logging.info(
                    f"Unmatched due to cost: Track {track_idx}, Detection {detection_idx}")
                logging.info(
                    f"Matched: Track {track_idx}, Detection {detection_idx}")

        else:
            matches.append((track_idx, detection_idx))
            if cfg['arguments']['logger_path']:

                logging.info(
                    f"Matched: Track {track_idx} with Detection {detection_idx}. Reason: Cost {cost_matrix[row, col]} <= Max distance {max_distance}")

    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):

    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    if cfg['arguments']['woC']:
        track_indices_l = [
            k for k in track_indices
            # if tracks[k].time_since_update == 1 + level
        ]
        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)
        matches += matches_l
    else:
        for level in range(cascade_depth):
            if len(unmatched_detections) == 0:  # No detections left
                break

            track_indices_l = [
                k for k in track_indices
                if tracks[k].time_since_update == 1 + level
            ]
            if len(track_indices_l) == 0:  # Nothing to match at this level
                continue

            matches_l, _, unmatched_detections = \
                min_cost_matching(
                    distance_metric, max_distance, tracks, detections,
                    track_indices_l, unmatched_detections)
            matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    logging.info(
        f"Final unmatched tracks in matching_cascade: {unmatched_tracks}")

    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
        cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    assert not only_position
    gating_threshold = kalman_filter.chi2inv95[4]
    measurements = np.asarray([detections[i].to_xyah()
                              for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = track.kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
        if cfg['arguments']['woC']:
            cost_matrix[row] = cfg['arguments']['MC_lambda'] * cost_matrix[row] + \
                (1 - cfg['arguments']['MC_lambda']) * gating_distance

    return cost_matrix
