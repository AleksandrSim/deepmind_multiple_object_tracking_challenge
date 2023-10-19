# Standard library imports
import os
import sys
import json
from datetime import datetime

# Third-party imports
import cv2
import numpy as np
import torch
from PIL import Image

# Local imports
sys.path.append('')  # make sure to specify the path
from src.tracker.engine import TrackerEngine
from src.tracker.visualizer import Visualizer
from src.tracker.cosine_distance import CosineDistance
from src.strong_sort.load_config import load_config
from src.strong_sort.deep_sort_custom import nn_matching
from src.strong_sort.deep_sort_custom.track import Track
from src.strong_sort.deep_sort_custom.detection import Detection
from src.strong_sort.deep_sort_custom.tracker import Tracker
from src.strong_sort.scripts.original_to_mot import MOTCConverter
from src.strong_sort.run_scripts.final_format_conversion import ConvertFinalFormat
#from src.strong_sort.metrics.comparison import run_iou_modified
from src.strong_sort.scripts.get_moving_objects import BoundingBoxProcessor
from src.strong_sort.metrics.metrics import summarise_results, run_iou_modified
from tqdm import tqdm

from src.strong_sort.scripts.predict_video import predict, parse_gt, parse_args
import time



cfg = load_config()


class PrepareFiles:
    def __init__(self, training_json, video_path, yolo_model_path, specific_videos,
                 sequence_folder, features_model_weights, out_logs_path, debug_yolo= False, iou_thresh=0.96,
                  config_file='/Users/aleksandrsimonyan/fast-reid/configs/DukeMTMC/bagtricks_S50.yml',
                  overwrite_strong_sort=False, validation=False):
        self.training_json = training_json
        self.video_path = video_path

        self.yolo_model_path = yolo_model_path

        self.specific_videos = specific_videos
        self.sequence_folder =sequence_folder
        self.sequences =  os.listdir(self.sequence_folder)
        self.sequences= [i for i in self.sequences if os.path.isdir(os.path.join(self.sequence_folder, i))]
        if self.specific_videos:
            self.sequences = [i for i in self.sequences if i in self.specific_videos]


        self.features_model_weights = features_model_weights
        self.out_logs_path = out_logs_path
        self.debug_yolo = debug_yolo
        self.iou_thresh = iou_thresh
        self.config_file = config_file

        self.overwrite_strong_sort = overwrite_strong_sort
        self.validation = validation


    def prepare_detections(self):
        args = parse_args()
        gt_data = {}
        if args.gt_path is not None:
            with open(args.gt_path, 'r') as file:
                gt_data = json.load(file)

        video_dir = os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['video_path'])
        video_files = os.listdir(video_dir)
        args.model_path =  os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['yolo_model_path'])
        args.device = cfg['arguments']['device']
        args.reid_predictor_config = cfg['prepare_files']['config_file']
        args.reid_model_path = os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['features_model_weights'])

        if self.specific_videos:
            video_files = [i for i in video_files if i.replace('.mp4','') in self.specific_videos]
        base_preds_dir = os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['sequence_folder'])
        for video_file in tqdm(sorted(video_files)):
            if not video_file.endswith('.mp4'):
                continue
            video_name = video_file.split('.')[0]
            gt_video_data = gt_data.get(video_name, {})
            s_time = time.time()
            predict(
                model_path=args.model_path,
                video_path=os.path.join(video_dir, video_file),
                predictions_dir=os.path.join(base_preds_dir, video_name),
                reid_predictor_config=args.reid_predictor_config,
                reid_model_path=args.reid_model_path,
                device=args.device,
                visualization=args.visualization,
                gt_data=gt_video_data
            )
            proc_time = time.time() - s_time
            print(f'{video_file} Processing time: {round(proc_time, 2)} s')


    def run_strong_sort(self, cfg):
        
        for folder in self.sequences:
            print(f'currenly running tracker for  {folder}')

            video_name = folder
            print(folder)

            video_path = os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['video_path'], folder +'.mp4')
            output_video_path =   os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['sequence_folder'],f'{video_name}/{video_name}_pred_strongsort.mp4')

            if cfg['arguments']['display']:
                visualizer = Visualizer(video_path, output_video_path)
            else:
                visualizer= None

            sequence_path = os.path.join(cfg['COMMON_PREFIX'],cfg['prepare_files']['sequence_folder'], video_name)

            strongsort_engine = TrackerEngine(
                detection_file= os.path.join(sequence_path, 'predictions.npy' ),
                output_file= os.path.join(sequence_path, 'tracker.txt'),
                min_confidence=cfg['arguments']['min_confidence'],
                min_detection_height=cfg['arguments']['min_detection_height'],
                visualizer=visualizer,
                cfg=cfg['arguments'],
                track_id_once=True,
            )
            strongsort_engine.run()
            
    def generate_final(self):
        print('assemling final prediction json')
        converter = ConvertFinalFormat(self.sequence_folder, self.training_json, cfg['arguments']['GSI'], specific_videos=self.specific_videos)
        converter.iterate_videos()
        converter.fill_first_track()
        converter.extrapolate()

    def generate_training_classes(self,cfg, validation=True):
        if  validation:
            processor = BoundingBoxProcessor(
            os.path.join(self.sequence_folder, 'final_predictions_extrapolated_normalized.json'), iou_threshold=float(cfg['prepare_files']['iou_thresh']), validation=validation)
            processor.process_videos()
            processor.save_output(os.path.join(self.sequence_folder, 'final_predictions_extrapolated_normalized_stationary.json'))
        else:
            processor = BoundingBoxProcessor(
            self.training_json, iou_threshold=float(cfg['prepare_files']['iou_thresh']), validation=validation)
            processor.process_videos()
            processor.save_output(os.path.join(self.sequence_folder, 'training_classes.json'))

    def write_results_to_json(self, results):
        for key, value in results.items():
            print(key)
            output_path = os.path.join(self.sequence_folder, key, 'results.json')
            with open(output_path, 'w') as f:
                json.dump(value, f)


    def normalize_predictions(self):
        print(f'generating prediction normalization')
        with open(os.path.join(self.sequence_folder,'final_predictions_extrapolated.json'), 'r') as f :
            pred_data = json.load(f)
        with open(self.training_json, 'r') as f :
            gt_data = json.load(f)

        for video in pred_data.keys():
            h, w = gt_data[video]['metadata']['resolution']

            for idx in range(len(pred_data[video])):
                new_bounding_boxes = []

                for bounding_box in pred_data[video][idx]['bounding_boxes']:
                    x1, y1, x2, y2 = bounding_box
                    normalized_x1 = x1 / w
                    normalized_y1 = y1 / h
                    normalized_x2 = x2 / w
                    normalized_y2 = y2 / h

                    normalized_bounding_box = [normalized_x1, normalized_y1, normalized_x2, normalized_y2]
                    new_bounding_boxes.append(normalized_bounding_box)
                pred_data[video][idx]['bounding_boxes'] = new_bounding_boxes
        with open(os.path.join(self.sequence_folder, 'final_predictions_extrapolated_normalized.json'), 'w') as f:
            json.dump(pred_data, f, indent=2)


    def read_and_filter_data(self):
        with open(os.path.join(self.sequence_folder, 'final_predictions_extrapolated_normalized_stationary.json'), 'r') as f:
            pred_data = json.load(f)
        with open(os.path.join(self.sequence_folder, 'training_classes.json'), 'r') as f:
            gt_data = json.load(f)

        if self.specific_videos:
            new = {}
            for video in self.specific_videos:
                new[video] = pred_data[video]
            pred_data = new

        return pred_data, gt_data

    
    def calculate_moving_objects(self):
        print(f'generating training json with moving/non-moving objects')

        pred_data, gt_data = self.read_and_filter_data()
        results = run_iou_modified(pred_data, gt_data, moving=True, stationary=False)
        
        # Write results to JSON files in sequence folder

        self.write_results_to_json(results)

        return summarise_results(gt_data, results)            
            

if __name__=='__main__':
    cfg = load_config()
    prepare_files = PrepareFiles(
        training_json=os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['training_json']),
        video_path=os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['video_path']),
        sequence_folder=os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['sequence_folder']),
        yolo_model_path=os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['yolo_model_path']),
        specific_videos=cfg['prepare_files']['specific_videos'],
        features_model_weights=os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['features_model_weights']),
        out_logs_path=os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['out_logs_path']),
        debug_yolo=cfg['prepare_files']['debug_yolo'],
        iou_thresh=cfg['prepare_files']['iou_thresh'],
        config_file=os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['config_file']),
        overwrite_strong_sort=False,
    )

#    prepare_files.prepare_detections()
#    prepare_files.run_strong_sort(cfg)
    #Before generating Final, the file system assumes that you have .txt files which is result from strong_sort algorithm
    prepare_files.generate_final()

    prepare_files.normalize_predictions()
    prepare_files.generate_training_classes(cfg, validation=False)
    prepare_files.generate_training_classes(cfg, validation=True)


    prepare_files.read_and_filter_data()
    final_score = prepare_files.calculate_moving_objects()
    print(f'final_score___IOU for Moving objects {final_score}')



