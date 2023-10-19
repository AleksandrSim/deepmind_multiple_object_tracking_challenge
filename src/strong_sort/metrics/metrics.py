import os
import numpy as np
from typing import List, Dict, Any, Tuple
import json

from typing import List, Dict, Any, Tuple
import numpy as np
import os
#!pip install git+https://github.com/votchallenge/vot-toolkit-python
from vot.region import calculate_overlaps as calculate_region_overlaps
from vot.region import Polygon, Rectangle, Special

class PredictionLoader:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path
    def load_predictions(self) -> Dict[str, List[Dict[str, Any]]]:
        predictions = {}
        for filename in os.listdir(self.directory_path):

            if filename.endswith(".txt"):  # Ensure it's a .txt file
                
                video_id = os.path.splitext(filename)[0]
                video_id = video_id.replace('_output', '')
                with open(os.path.join(self.directory_path, filename), 'r') as file:
                    lines = file.readlines()
                    tracks = []
                    for line in lines:
                        components = line.strip().split(',')
                        frame_id, track_id = int(components[0]), int(components[1])
                        x1, y1, x2, y2 = map(float, components[2:6])
                        tracks.append({'frame_id': frame_id, 'track_id': track_id, 'bounding_box': [x1, y1, x2, y2]})
                    predictions[video_id] = tracks
        return predictions

def bbox2region(bbox: np.array) -> Rectangle:
    if len(bbox) == 1:
        return Special(bbox[0])
    elif len(bbox) == 4:
        return Rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
    else:
        return Polygon([(x_, y_) for x_, y_ in zip(bbox[::2], bbox[1::2])])

def get_start_frame(track_arr: List[List[float]]) -> int:
    if not track_arr or np.count_nonzero(track_arr) == 0:
        raise ValueError('Track is empty or has no non-zero elements')
    return np.nonzero(track_arr)[0][0]

def get_start_info(track: Dict[str, Any]) -> Dict[str, Any]:
    track_start_idx = get_start_frame(track['initial_tracking_box'])
    track_start_id = track['frame_ids'][track_start_idx]
    track_start_bb = track['bounding_boxes'][track_start_idx]
    return {'start_id': track_start_id, 'start_bounding_box': track_start_bb, 'start_idx': track_start_idx}

def filter_pred_boxes(pred_bb: np.ndarray, pred_fid: np.ndarray, gt_fid: np.ndarray):
    pred_idx = np.isin(pred_fid, gt_fid).nonzero()[0]
    filter_pred_bb = pred_bb[pred_idx]
    filter_pred_fid = pred_fid[pred_idx]
    return filter_pred_bb, filter_pred_fid

def trajectory2region(trajectory: List) -> List:
  traj_region = []
  for bbox in trajectory:
    traj_region.append(bbox2region(bbox))
  return traj_region

def calc_accuracy(gt_trajectory: List, pred_trajectory: List) -> float:

  pred_traj_region = trajectory2region(pred_trajectory)
  gt_traj_region = trajectory2region(gt_trajectory)

  overlaps = np.array(calculate_region_overlaps(pred_traj_region,
                                                gt_traj_region))
  mask = np.ones(len(overlaps), dtype=bool)
  return np.mean(overlaps[mask]) if any(mask) else 0.
def run_iou_modified(boxes: Dict[str, Any], db_dict: Dict[str, Any], filter=True, moving= False, stationary=False) -> Dict[str, Any]:
  """Calculate IoU per track and per video.

  Calculate Intersection over Union (IoU) for predicted and ground truth
    bounding boxes for all tracks in the provided outputs.

  Args:
    boxes (Dict): Dict containing predicted and label bounding boxes for each
      video. Boxes must be in format [x1,y1,x2,y2].
    db_dict (Dict): Dict containing annotations.

  Returns:
    Dict: A dictionary with video IDs as keys and
      a list of IoU scores as values.
  """
  all_vot_iou = {}
  for vid_id, pred_tracks in boxes.items():
    gt_tracks = db_dict[vid_id]['object_tracking']

    video_iou = {}
    for pred_track in pred_tracks:

      
#      if pred_track['id'] >=  len(gt_tracks):
#         id = pred_track['id']
#         print(f'prediction_id_ does not exist in that case continue {id}')
#         continue
      gt_track = gt_tracks[pred_track['id']]
#      if gt_track['label']=='pants' or gt_track['label']=='cup' or gt_track['label']=='bottle' \
#      or gt_track['label']=='curtains' or gt_track['label']=='socket':
#          continue

      if moving:
         movement = gt_track['moving']

      # check track IDs
      assert pred_track['id'] == gt_track['id']

      start_info = get_start_info(gt_track)
      start_idx = start_info['start_idx']
      # get bounding boxes from frame ID were tracking is supposed to start +1
      gt_bb = np.array(gt_track['bounding_boxes'])[start_idx+1:]
  

      gt_fid = gt_track['frame_ids'][start_idx+1:]

      # weird case where only one box is labelled
      if not gt_fid:
        continue

      pred_bb = np.array(pred_track['bounding_boxes'])

      if stationary:
        if 'moving' in gt_track:
            if gt_track['moving']==False:
                pred_bb = np.array([gt_track['bounding_boxes'][0] for i in range(len(pred_bb))])
      if int(gt_track['frame_ids'][start_idx])<=30 and int(gt_track['id']) !=0 :
        print('0 index')
        pred_bb = np.array([gt_track['bounding_boxes'][start_idx] for i in range(len(pred_bb))])
  
      pred_fid = np.array(pred_track['frame_ids'])
      # filter predicted trajectory for frame IDs where we have annotations
      if filter:
        pred_bb, pred_fid = filter_pred_boxes(pred_bb, pred_fid, gt_fid)

      # check for missing frame IDs in prediction

      missing_idx = np.where(np.isin(gt_fid, pred_fid, invert=True))[0]
      if missing_idx.size != 0:
        print(gt_track['label'])
        id = pred_track['id']

        raise ValueError(f'Missing IDs from object trajectory: {missing_idx}')
      if len(gt_bb) != len(pred_bb):
        raise ValueError('Missing boxes in predictions')

      #  convert y2,x2,y1,x1 [0,1] to x1,y1,w,h in pixel space
      [height, width] = db_dict[vid_id]['metadata']['resolution']



      pred_w = pred_bb[:, 2] - pred_bb[:, 0]
      pred_h = pred_bb[:, 3] - pred_bb[:, 1]
      
      if stationary and 'moving' in gt_track and gt_track['moving'] == False:
        pred_bb = np.stack([pred_bb[:, 0]*width, pred_bb[:, 1]*height,pred_w*width, pred_h*height], axis=1)
  
      else:       
        pred_bb = np.stack([pred_bb[:, 0]*width, pred_bb[:, 1]*height,
                            pred_w*width, pred_h*height], axis=1)


      gt_w = gt_bb[:, 2] - gt_bb[:, 0]
      gt_h = gt_bb[:, 3] - gt_bb[:, 1]
      gt_bb = np.stack([gt_bb[:, 0]*width, gt_bb[:, 1]*height,
                        gt_w*width, gt_h*height], axis=1)

      # compute IoU per track
      iou = calc_accuracy(gt_bb, pred_bb)
      label = gt_track['label']
      video_iou[pred_track['id']] = iou
      if moving:
        video_iou[pred_track['id']] = (iou, movement)

    all_vot_iou[vid_id] = video_iou
  return all_vot_iou

class Evaluator:
    def __init__(self, ground_truth_path: str):
        self.ground_truth = self._load_ground_truth(ground_truth_path)

    @staticmethod
    def _load_ground_truth(ground_truth_path: str) -> dict:
        with open(ground_truth_path, 'r') as f:
            data = json.load(f)

        return data

    @staticmethod
    def _convert_predictions(predictions: dict) -> dict:
        converted_predictions = {}
        for video_id, tracks in predictions.items():
            video_id = video_id.replace('_exp','')
            grouped_tracks = {}
            for track in tracks:
                track_id = track['track_id']
                if track_id not in grouped_tracks:
                    grouped_tracks[track_id] = {'frame_ids': [], 'bounding_boxes': []}
                grouped_tracks[track_id]['frame_ids'].append(track['frame_id'])
                grouped_tracks[track_id]['bounding_boxes'].append(track['bounding_box'])
            video_tracks = []
            for track_id, track_info in grouped_tracks.items():
                video_tracks.append({'id': track_id, 'frame_ids': track_info['frame_ids'], 'bounding_boxes': track_info['bounding_boxes']})
            converted_predictions[video_id] = video_tracks
        return converted_predictions

    def evaluate(self, predictions: dict, sequence) -> dict:
        print(sequence)
        
        formatted_predictions = self._convert_predictions(predictions)
        results = run_iou(formatted_predictions, self.ground_truth)
        return results

def summarise_results(labels, results):
  """Summarise the results according to camera movement.

  Summarise the results of a dataset by calculating average IoU scores
  across all videos, videos with a static camera and videos with a moving
  camera.

  Args:
    labels (Dict): A dictionary containing metadata and
      information about the dataset.
    results (Dict): A dictionary containing IoU scores
      for each video in the dataset.
  """
  all_ious = []
  # aggregate performance based on camera motion for analysis
  static_ious = []
  moving_ious = []

  static_objects =[]
  moving_objects=[]

  for vid, iou_dict in results.items():
    ious_original = list(iou_dict.values())

    ious =[]
    
    for iou in ious_original:
        if type(iou) ==tuple:
            ious.append(iou[0])

            if iou[1]==True:
               moving_objects.append(iou[0])
            else:
               static_objects.append(iou[0])
        else:
          ious.append(iou)
        
    if not ious:
      continue

    all_ious.append(np.mean(ious))

    if labels[vid]['metadata']['is_camera_moving']:
      moving_ious.append(np.mean(ious))
    else:
      static_ious.append(np.mean(ious))

    

  if all_ious:
    print(f"""Average IoU across all videos in dataset:
          {np.array(all_ious).mean():.3f}""")
  if static_ious:
    print(f"""Average IoU across static camera videos in dataset:
          {np.array(static_ious).mean():.3f}""")
  if moving_ious:
    print(f"""Average IoU across moving camera videos in dataset:
          {np.array(moving_ious).mean():.3f}""")
    

  if moving_objects:
    print(f"""Average IoU across moving objects:
          {np.array(moving_objects).mean():.3f}""")
    
  if static_objects:
    print(f"""Average IoU across non-moving objects:
          {np.array(static_objects).mean():.3f}""")
    
  return np.array(moving_objects).mean()

     
if __name__=='__main__':

  main_sequence_path = 'Users/aleksandrsimonyan/Desktop/deepmind_updated/data/preds/'
  ground_truth_path = '/Users/aleksandrsimonyan/Desktop/deepmind_updated/data/preds/training_classes.json'


  predictions_json = '/Users/aleksandrsimonyan/Downloads/final_predictions_extrapolated.json'


#  with open(predictions_json, 'r') as f :
#      pred_data = json.load(f)

  with open( ground_truth_path, 'r') as f :
      gt_data = json.load(f)
#  results = run_iou(pred_data, gt_data,moving=True, stationary=True)
#  print(results)
#  summarise_results(gt_data, results)

  predictions_json = '/Users/aleksandrsimonyan/Desktop/deepmind_updated/data/preds/final_predictions_extrapolated_normalized_stationary.json'
  
  with open(predictions_json, 'r') as f :
      pred_data = json.load(f)

  def run_iou(boxes: Dict[str, Any], db_dict: Dict[str, Any], filter=True) -> Dict[str, Any]:
    """Calculate IoU per track and per video.

    Calculate Intersection over Union (IoU) for predicted and ground truth
      bounding boxes for all tracks in the provided outputs.

    Args:
      boxes (Dict): Dict containing predicted and label bounding boxes for each
        video. Boxes must be in format [x1,y1,x2,y2].
      db_dict (Dict): Dict containing annotations.

    Returns:
      Dict: A dictionary with video IDs as keys and
        a list of IoU scores as values.
    """
    all_vot_iou = {}
    for vid_id, pred_tracks in boxes.items():
      print(vid_id)
      gt_tracks = db_dict[vid_id]['object_tracking']

      video_iou = {}
      for pred_track in pred_tracks:
        gt_track = gt_tracks[pred_track['id']]
        # check track IDs
        assert pred_track['id'] == gt_track['id']

        start_info = get_start_info(gt_track)
        start_idx = start_info['start_idx']
        # get bounding boxes from frame ID were tracking is supposed to start +1
        gt_bb = np.array(gt_track['bounding_boxes'])[start_idx+1:]
        gt_fid = gt_track['frame_ids'][start_idx+1:]
        # weird case where only one box is labelled
        if not gt_fid:
          continue

        pred_bb = np.array(pred_track['bounding_boxes'])
        pred_fid = np.array(pred_track['frame_ids'])
        # filter predicted trajectory for frame IDs where we have annotations
        if filter:
          pred_bb, pred_fid = filter_pred_boxes(pred_bb, pred_fid, gt_fid)

        # check for missing frame IDs in prediction
        missing_idx = np.where(np.isin(gt_fid, pred_fid, invert=True))[0]
        if missing_idx.size != 0:
          raise ValueError(f'Missing IDs from object trajectory: {missing_idx}')
        if len(gt_bb) != len(pred_bb):
          raise ValueError('Missing boxes in predictions')

        #  convert y2,x2,y1,x1 [0,1] to x1,y1,w,h in pixel space
        [height, width] = db_dict[vid_id]['metadata']['resolution']
        pred_w = pred_bb[:, 2] - pred_bb[:, 0]
        pred_h = pred_bb[:, 3] - pred_bb[:, 1]
        pred_bb = np.stack([pred_bb[:, 0]*width, pred_bb[:, 1]*height,
                            pred_w*width, pred_h*height], axis=1)
        gt_w = gt_bb[:, 2] - gt_bb[:, 0]
        gt_h = gt_bb[:, 3] - gt_bb[:, 1]
        gt_bb = np.stack([gt_bb[:, 0]*width, gt_bb[:, 1]*height,
                          gt_w*width, gt_h*height], axis=1)

        # compute IoU per track
        iou = calc_accuracy(gt_bb, pred_bb)
        video_iou[pred_track['id']] = iou

      all_vot_iou[vid_id] = video_iou

    return all_vot_iou
  

  results = run_iou(pred_data, gt_data)
  print(results)
  summarise_results(gt_data, results)