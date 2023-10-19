# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import os

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort_custom import nn_matching
from deep_sort_custom.detection import Detection
from deep_sort_custom.tracker import Tracker
import argparse
import matplotlib.pyplot as plt
import logging
import yaml

with open("config.yml", 'r') as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    video_name = os.path.basename(sequence_dir)
    video_name = video_name.replace('_sequence_dir', '')

    groundtruth_file = os.path.join(sequence_dir,video_name + '_gt.npy')

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.load(groundtruth_file)


    video_path = os.path.join(cfg['COMMON_PREFIX'], cfg['arguments']['video_directory'], video_name +'.mp4')
    video_capture = cv2.VideoCapture(video_path)
    ret, image = video_capture.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_size = image.shape

    min_frame_idx = int(detections[:, 0].min())
    max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")

    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None


    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
#        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms,
        "sequence_full_path":sequence_dir
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int16)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, track_id, feature = row[2:6], row[6], row[1], row[10:]  # Extract track_id from the second column
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature, track_id))  # Include track_id
    return detection_list

def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, save=False):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine',
        max_cosine_distance,
        nn_budget
    )
    ids = seq_info['detections'][:,1]
    max_track_id = int(max([i for i in ids ]))
# Initialize logging at the beginning of the run function, or your script
    if cfg['arguments']['logger_path'] is not None:
        # Ensure the path ends with '.txt'
        if not cfg['arguments']['logger_path'].endswith('.txt'):
            print(f"Warning: The tracker_path {cfg['arguments']['logger_path']} does not have a '.txt' extension.")
        # Combine path, video name, and extension
        log_filename = f"{cfg['arguments']['logger_path']}_.txt"
        logging.basicConfig(filename=log_filename, level=logging.DEBUG)



    print(cfg['arguments']['max_iou_distance'])
    tracker = Tracker(metric, cfg['arguments']['max_iou_distance'], 
                      cfg['arguments']['max_age'], cfg['arguments']['n_init'],  max_track_id)
    results = []

    def frame_callback(vis, frame_idx):
        # print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
                        boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        if cfg['arguments']['ECC']:
            tracker.camera_update(sequence_dir.split('/')[-1], frame_idx)

        tracker.predict()
        tracker.update(detections, frame_idx)
        video_name = seq_info['sequence_name']

        video_name = video_name.replace('_sequence_dir', '')
        video_path = os.path.join(cfg['COMMON_PREFIX'], cfg['arguments']['video_directory'],video_name + '.mp4')
 

        if cfg['arguments']['logger_path'] is not None:
            logging.info(f"Processing video: {video_name}")


#        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # Update visualization.
        if display:
            video_capture = cv2.VideoCapture(video_path)

            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, image = video_capture.read()
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed(): #or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                    frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if save:
        visualizer = visualization.Visualization(seq_info, update_ms=int(cfg['arguments']['update_mls']),
                                                 dir_to_save = seq_info['sequence_full_path'])
    elif display and not save:
        visualizer = visualization.Visualization(seq_info, update_ms=int(cfg['arguments']['update_mls']))
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)

def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.7)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
#    gather_sequence_info('/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/video_1_sequnce_dir/',
#                         '/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/video_1_sequnce_dir/video_1.npy')
    print('fd')
    

    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display, args.save)
