import cv2
import os
import json
import sys

sys.path.append('src/strong_sort')

from tools.raw_yolo_detections import YOL, compute_iou
from others.generate_detections import setup, get_model, get_transform
import numpy as np
from fastreid.config import get_cfg
from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from datetime import datetime
from PIL import Image
import torch
from scripts.original_to_mot import MOTCConverter
from run_scripts.final_format_conversion import ConvertFinalFormat
from strong_sort import StrongSort
from load_config import load_config
from metrics.comparison import run_iou
from scripts.get_moving_objects import BoundingBoxProcessor






class PrepareFiles:
    def __init__(self, training_json, video_path, yolo_model_path, specific_videos,
                 sequence_folder, features_model_weights, out_logs_path, debug_yolo= False, iou_thresh=0.96,
                  config_file='/Users/aleksandrsimonyan/fast-reid/configs/DukeMTMC/bagtricks_S50.yml',
                  overwrite_strong_sort=False):
        self.training_json = training_json
        self.video_path = video_path

        self.yolo_model_path = yolo_model_path

        self.specific_videos = specific_videos
        self.sequence_folder =sequence_folder
        self.features_model_weights = features_model_weights
        self.out_logs_path = out_logs_path
        self.debug_yolo = debug_yolo
        self.iou_thresh = iou_thresh
        self.config_file = config_file
        print(self.config_file)

        self.overwrite_strong_sort = overwrite_strong_sort

    def prepare_detections(self):        
        with open( self.training_json, 'r') as f:
            data = json.load(f)
        videos = list(data.keys())

        if self.specific_videos:
            videos = self.specific_videos
        
        os.makedirs(self.sequence_folder, exist_ok=True)

        for video in videos:
            print(f'currently working on video {video}')
            if not os.path.exists(os.path.join(self.video_path, video + '.mp4')):
                print(f'caution the {video}  does not exist given the path')
                continue

            result_folder = os.path.join(self.sequence_folder, video + '_sequence_dir')
            print(result_folder)
            os.makedirs(result_folder, exist_ok=True)
            if video + "_crops.npy" in os.listdir(result_folder):
                continue

            processor = YOL(self.yolo_model_path, self.training_json, debug=self.debug_yolo, video=video, iou_threshold=self.iou_thresh, pretrained=True, video_path=self.video_path)
            processor.process_frames_folder(result_folder)

    def generate_feature(self, config_file='/Users/aleksandrsimonyan/fast-reid/configs/DukeMTMC/bagtricks_S50.yml',
                        output_dir_logs ='/Users/aleksandrsimonyan/Desktop/yolo_weights/predictions/fast_reid/logs'):
        
        sequences= os.listdir(self.sequence_folder)
        sequences_list= [i for i in sequences if i.endswith('dir')]
        for video in sequences_list:

            video_name = video.replace('_sequence_dir', "")
#            if video_name not in cfg['specific_videos']:
#                continue
            file_system = os.listdir(os.path.join(self.sequence_folder, video))
            if (video_name + '.npy' in file_system) or (video_name +'_crops.npy' not in file_system) :
                print(file_system)
                print('exists')
                continue

            print(datetime.now())
            '''配置信息'''
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
            args = default_argument_parser().parse_args()
            args.eval_only = True
            args.config_file = config_file
        #    print("Command Line Args:", args)
            cfg = setup(args)
            cfg.defrost()
            cfg.MODEL.DEVICE ='cpu'

            cfg.MODEL.BACKBONE.PRETRAIN = False
            cfg.MODEL.WEIGHTS = self.features_model_weights
            cfg.OUTPUT_DIR = output_dir_logs

            # cfg.MODEL.BACKBONE.WITH_IBN = False
            thres_score = 0.5
            sequence_dir = os.path.join(self.sequence_folder, video)

#            root_videos = self.video_path
            dir_out_det = sequence_dir
            model = get_model(cfg)

            transform = get_transform((256, 128))

            print('processing the video {}...'.format(video))
            detections = np.load(os.path.join(sequence_dir, video_name  + '_crops.npy'))
            detections = detections[detections[:, 6] >= thres_score]
            mim_frame, max_frame = int(min(detections[:, 0])), int(max(detections[:, 0]))
            list_res = list()

            cap = cv2.VideoCapture(os.path.join(self.video_path, video_name +'.mp4'))
            for frame in range(mim_frame, max_frame + 1):
                print(f'frame_{frame}')
                # print('  processing the frame {}...'.format(frame))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                rate, img = cap.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                detections_frame = detections[detections[:, 0] == frame]
                batch = [img.crop((b[2], b[3], b[2] + b[4], b[3] + b[5])) for b in detections_frame]
                batch = [transform(patch) * 255. for patch in batch]
                if batch:
                    batch = torch.stack(batch, dim=0)
                    outputs = model(batch).detach().cpu().numpy()
                    list_res.append(np.c_[(detections_frame, outputs)])
            res = np.concatenate(list_res, axis=0)
            video = video.replace('_sequence_dir', '')
            np.save(os.path.join(dir_out_det, video + '.npy'), res, allow_pickle=False)
        print(datetime.now())

    def generate_gt(self):
        sequences= os.listdir(self.sequence_folder)
        sequences_list= [i for i in sequences if i.endswith('dir')]

        for video in sequences_list:
            sequence_dir = os.path.join(self.sequence_folder, video)
            video_name = video.replace('_sequence_dir', "")
#            if video_name not in cfg['specific_videos']:
#                continue

            if video_name + '_gt.npy' in os.listdir(sequence_dir):
                continue

            converter = MOTCConverter(self.training_json,self.video_path, sequence_dir, video_name)
            converter.convert_to_motc()

    def run_strong_sort(self, cfg):
        sequences = os.listdir(self.sequence_folder)
        sequences_list = [i for i in sequences if i.endswith('dir')]

        for video in sequences_list:
            cfg['dataset_options']['sequences']['train'] = [video]
            # Continue with existing logic
            sequence_dir = os.path.join(self.sequence_folder, video)
            video_name = video.replace('_sequence_dir', "")
            if cfg['prepare_files']['specific_videos']  :
                if video_name not in cfg['prepare_files']['specific_videos']:
                    print(f'{video_name} video_name is filtered')
                    continue

            sequence_dir_files = os.listdir(sequence_dir)

            if video_name + '.npy' not in sequence_dir_files:
                print(f'Feature files are missing for {video_name}')
                continue  # Use 'continue' to keep processing other videos

            if video_name + '_exp_output.txt' in sequence_dir_files and not self.overwrite_strong_sort:
                print(f'Files are already there {video_name}')
                continue  # Use 'continue' to keep processing other videos
            
            strong_sort_instance = StrongSort(cfg)
            strong_sort_instance.main()
            
    def generate_final(self):
        print('assemling final prediction json')
        converter = ConvertFinalFormat(self.sequence_folder, self.training_json)
        converter.iterate_videos()
        converter.fill_first_track()
        converter.extrapolate()

    def generate_training_classes(self,cfg):
        processor = BoundingBoxProcessor(self.training_json, iou_threshold=float(cfg['prepare_files']['iou_thresh']))
        processor.process_videos()
        processor.save_output(os.path.join(self.sequence_folder, 'training_classes.json'))

    def calculate_moving_objects(self):
        print(f'generating training json with moving/ non moving objects')
        with open(os.path.join(self.sequence_folder,'final_predictions_extrapolated.json'), 'r') as f :
            pred_data = json.load(f)
        with open( os.path.join(self.sequence_folder, 'training_classes.json'), 'r') as f :
            gt_data = json.load(f)

        results = run_iou(pred_data, gt_data,moving=True, stationary=True)
        moving_objects=[]
        for vid, iou_dict in results.items():
            ious_original = list(iou_dict.values())
            ious =[]
            for iou in ious_original:
                if type(iou) ==tuple:
                    ious.append(iou[0])
                    if iou[1]==True:
                        moving_objects.append(iou[0])
        return np.array(moving_objects).mean()
    
    def normalize_predictions(self):
        print(f'generating prediction normalization')
        with open(os.path.join(self.sequence_folder,'final_predictions_extrapolated.json'), 'r') as f :
            pred_data = json.load(f)
        with open(os.path.join(self.sequence_folder, 'training_classes.json'), 'r') as f :
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
        overwrite_strong_sort=False
    )


    prepare_files.prepare_detections()
    prepare_files.generate_feature(config_file='/Users/aleksandrsimonyan/fast-reid/configs/DukeMTMC/bagtricks_S50.yml') #path should be inside main fasteid path file structure
    prepare_files.generate_gt()
    prepare_files.run_strong_sort(cfg)
    #Before generating Final, the file system assumes that you have .txt files which is result from strong_sort algorithm
    prepare_files.generate_training_classes(cfg)
    prepare_files.generate_final()
    final_score = prepare_files.calculate_moving_objects()
    prepare_files.normalize_predictions()
    print(f'final_score___IOU for Moving objects {final_score}')




