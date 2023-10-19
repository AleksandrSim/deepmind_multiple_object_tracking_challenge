import os
import json
import time
import queue
import argparse
import threading

import cv2
import numpy as np
import sys
sys.path.append('')

from src.tools.iou import merge_bboxes
from src.reid.predict import ReidPredictor
from src.detect.predict import Predictor

from src.strong_sort.load_config import load_config

cfg = load_config()
MODEL_INPUT = (640, 640)
BBOX_COLOR = (255, 0, 0)
MAX_QUEUE_SIZE = 8
REPLACE_IOU_THRESHOLD = 0.8  # Threshold to replace bbox with the data from GT
SLEEP_TIME = 0.0001
REID_INPUT_SIZE = (256, 128)

DETECTIONS_JSON = 'detections.json'
EMBEDDINGS_FILE = 'predictions.npy'



def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection on Video')
    parser.add_argument('-m', '--model_path', type=str, required=False,
                        help='Path to the argus model file')
    parser.add_argument('-i', '--video_path', type=str, required=False,
                        help="Path to the input video")
    parser.add_argument('-vis', '--visualization', action='store_true',
                        default=False, help='Save visualization video')
    parser.add_argument('-o', '--predictions_dir', type=str, required=False,
                        help='Path to predictions folder')
    parser.add_argument('-rc', '--reid_predictor_config', type=str,
                        default='/workdir/data/ext_models/bagtricks_S50.yml',
                        help='Path to fast-reid model yml file')
    parser.add_argument('-rm', '--reid_model_path', type=str,
                        default='/workdir/data/ext_models/DukeMTMC_BoT-S50.pth',
                        help='Path to fast-reid model')
    parser.add_argument('-d', '--device', type=str, default='cuda:0',
                        help='Device to use for the model inference')
    parser.add_argument('-gt', '--gt_path', type=str, default=None,
                        help="Path to the gt file")
    return parser.parse_args()


def predict(model_path: str,
            video_path: str,
            predictions_dir: str,
            reid_predictor_config: str,
            reid_model_path: str,
            device: str = 'cuda:0',
            visualization: bool = False,
            gt_data= None):

    frame_queue = queue.Queue(MAX_QUEUE_SIZE)
    prediction_queue = queue.Queue(MAX_QUEUE_SIZE)
    visualize_queue = queue.Queue(MAX_QUEUE_SIZE)

    # Video reader
    if device.startswith('cuda'):
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'video_codec;h264_nvenc'
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Models
    predictor = Predictor(model_path=model_path, device=device,
                          frame_size=(frame_width, frame_height))
    reid_model = ReidPredictor(reid_predictor_config, reid_model_path,
                               input_size=REID_INPUT_SIZE, device=device)

    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir, exist_ok=True)

    # Detection predictions
    predictions: dict[int, list[dict]] = {}
    detection_json_path = os.path.join(predictions_dir, DETECTIONS_JSON)
    if os.path.exists(detection_json_path):
        with open(detection_json_path, 'r') as file:
            predictions_raw = json.load(file)
        predictions = {int(key): value for key, value in
                       predictions_raw.items()}
    embeddings_file = os.path.join(predictions_dir, EMBEDDINGS_FILE)

    # GT data
    if gt_data is None:
        dt_data = {}
    else:
        for bboxes in gt_data.values():
            bboxes[:, 0:4:2] = bboxes[:, 0:4:2] * frame_width
            bboxes[:, 1:4:2] = bboxes[:, 1:4:2] * frame_height

    # Visualization
    if visualization:
        video_name = os.path.basename(video_path)
        visualization_video_path = os.path.join(
            predictions_dir, '_pred.'.join(video_name.split('.')))
        video_writer = cv2.VideoWriter(visualization_video_path,
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       fps, (frame_width, frame_height))

    def match_annot(pred: np.array, frame_id: int):
        if frame_id in gt_data:
            pred = merge_bboxes(pred, gt_data[frame_id].copy(),
                                REPLACE_IOU_THRESHOLD)
        return pred

    def read_frames():
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put((i, frame))
            i += 1
            time.sleep(SLEEP_TIME)
        frame_queue.put(None)  # Mark the end of frames
        cap.release()

    def detection_stream():
        stopped = False
        while not stopped:
            frames: list[tuple[int, np.ndarray]] = []
            while frame_queue.qsize() > 0:
                sample = frame_queue.get()
                if sample is not None:
                    frames.append(sample)
                else:
                    prediction_queue.put(None)
                    stopped = True
                    print('detection_stream stop')
                    break
            if len(frames) > 0:
                frames_to_pred = [f[0] for f in frames
                                  if f[0] not in predictions]
                if len(frames_to_pred) > 0:
                    preds = predictor.predict_batch([f[1] for f in frames
                                                     if f[0] in frames_to_pred])
                    for frame_idx, pred in zip(frames_to_pred, preds):
                        pred = np.pad(pred, ((0, 0), (0, 1)),
                                      constant_values=-1)  # Add track id
                        pred = match_annot(pred, frame_idx)
                        predictions[frame_idx] = [
                            {'bbox': list(map(float, p[:4])),
                             'conf': float(p[4]),
                             'track_id': int(p[5])}
                            for p in pred]
                for frame in frames:
                    prediction_queue.put(
                        (frame[0], frame[1], predictions[frame[0]]))
            time.sleep(SLEEP_TIME)

    def reid_stream():
        stopped = False
        while not stopped:
            preds = []
            while prediction_queue.qsize() > 0:
                sample = prediction_queue.get()
                if sample is not None:
                    preds.append(sample)
                    visualize_queue.put(sample)
                else:
                    visualize_queue.put(None)
                    stopped = True
                    print('reid_stream stop')
                    break
            if len(preds) > 0:
                for pred in preds:
                    frame_idx, frame, pred_dict = pred
                    if len(pred_dict) > 0:
                        embs = reid_model.predict(frame.copy(), pred_dict)
                        embs = embs.cpu().data.numpy()
                        for i, emb in enumerate(embs):
                            predictions[frame_idx][i]['embedding'] = emb
            time.sleep(SLEEP_TIME)

    def write_frames():
        while True:
            if visualize_queue.qsize() > 0:
                data = visualize_queue.get()
                if data is None:
                    print('write_frames stop')
                    break
                if visualization:
                    _, frame, pred = data
                    for obj in pred:
                        x1, y1, x2, y2 = map(int, obj['bbox'])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, 2)
                    video_writer.write(frame)
                else:
                    del data
            time.sleep(SLEEP_TIME)

    read_thread = threading.Thread(target=read_frames)
    predict_thread = threading.Thread(target=detection_stream)
    reid_thread = threading.Thread(target=reid_stream)
    write_thread = threading.Thread(target=write_frames)

    read_thread.start()
    predict_thread.start()
    reid_thread.start()
    write_thread.start()

    read_thread.join()
    predict_thread.join()
    reid_thread.join()
    write_thread.join()
    if visualization:
        video_writer.release()
    cap.release()

    # Save predictions
    bk_postfix = f'{time.time()}.bk'
    if os.path.exists(embeddings_file):
        os.rename(embeddings_file, embeddings_file+bk_postfix)
    print('Saving embeddings')
    np.save(embeddings_file, predictions)
    if os.path.exists(detection_json_path):
        os.rename(detection_json_path, detection_json_path+bk_postfix)
    with open(detection_json_path, "w") as file:
        for frame_data in predictions.values():
            for pred in frame_data:
                if 'embedding' in pred:
                    del pred['embedding']
        json.dump(predictions, file, indent=2)


def parse_gt(gt_data: dict, video_name: str):
    res: dict = {}
    if video_name in gt_data:
        gt_raw = gt_data[video_name]
        for frame_id, det in gt_raw.items():
            bboxes = []
            for sample in det:
                bboxes.append([*sample['bounding_boxes'], 1.0, sample['id']])
            if len(bboxes) > 0:
                res[int(frame_id)] = np.array(bboxes)
                print(int(frame_id), res[int(frame_id)].shape)
    return res


if __name__ == '__main__':
    args = parse_args()
    gt_data = {}
    print(os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['training_json']))

    with open(os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['training_json']), 'r') as file:
            gt_data = json.load(file)
    # Note: GT data will be added with confidence = 1.0

    video_name = 'video_1125'

    gt_video_data = parse_gt(gt_data, video_name)
    

    s_time = time.time()
    args.model_path = os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['yolo_model_path'])
    args.video_path_path = os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['video_path'], 'video_1125.mp4')
    args.prediction_dir = os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['sequence_folder'], 'video_1125')
    args.reid_model_path = os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['features_model_weights'])
    args.device = cfg['arguments']['device']
    
    

    predict(model_path=args.model_path,
            video_path=args.video_path,
            predictions_dir=args.predictions_dir,
            reid_predictor_config=args.reid_predictor_config,
            reid_model_path=args.reid_model_path,
            device=args.device,
            visualization=args.visualization,
            gt_data=gt_video_data)
    print(f'Processing time: {round(s_time-time.time(), 2)} s')