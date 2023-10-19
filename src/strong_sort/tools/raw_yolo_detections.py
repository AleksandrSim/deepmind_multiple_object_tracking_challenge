import torch
from ultralytics import YOLO
import cv2
import os
import numpy as np
import json
import sys
sys.path.append("/Users/aleksandrsimonyan/Documents/GitHub/deepmind_object_tracking_challenge")
from src.detect.predict import Predictor


MODEL_PATH = '/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/YOLOXl-009_003-019-0.839273.pth'
MODEL_INPUT = (640, 640)
BBOX_COLOR = (255, 0, 0)


def compute_iou(box1, box2):
    """Computes the Intersection-Over-Union of two bounding boxes."""
    x1, y1, x2, y2 = box1
    xa, ya, xb, yb = box2

    x2,y2 = x1+x2, y1+y2
    xb,yb= xa+xb, ya+yb

    # Determine the coordinates of the intersection rectangle
    xA = max(x1, xa)
    yA = max(y1, ya)
    xB = min(x2, xb)
    yB = min(y2, yb)

    # Compute the area of the intersection rectangle
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both bounding boxes
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (xb - xa + 1) * (yb - ya + 1)

    # Compute the Intersection Over Union
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

class YOL:
    def __init__(self, model_path, ground_truth_path=None, conf_thresh=0.003, debug=False, video ='video_1', iou_threshold=0.99, first_frame=True, pretrained=True, video_path='/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/videos/'):
        self.iou_threshold = iou_threshold
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.debug = debug
        self.first_frame =first_frame
        self.pretrained= pretrained

        if pretrained:
            self.model= Predictor(model_path=model_path,
                      input_size=(640,640), device='cpu')
        

        # Load ground truth data if the path is provided
        self.ground_truth = {}
        if ground_truth_path:
            with open(ground_truth_path, 'r') as f:
                data = json.load(f)
                for obj in data[video]['object_tracking']:
                    index_of_initial_box = obj['initial_tracking_box'].index(1)
                    frame_id = obj['frame_ids'][index_of_initial_box]
                    bbox = obj['bounding_boxes'][index_of_initial_box]
                    obj_id = obj['id']
                    if frame_id not in self.ground_truth:
                        self.ground_truth[frame_id] = []
                    self.ground_truth[frame_id].append((bbox, obj_id))  # Store bbox and ID together
        self.video_path = video_path
        self.video = video

    def process_frames_folder(self, result_folder, desired_size=(640, 640)):
        print(self.ground_truth.keys())
        if not os.path.exists(result_folder):
            os.makedirs(result_folder, exist_ok=True)

        all_detections = []
#        frames_list = os.listdir(frames_folder)
#        frames_list = [i for i in frames_list if i != '.DS_Store']
#        frames_list = sorted(frames_list)

#        for idx in range(len(frames_list)):
#            filename = str(idx) +'.png'
#            frame_path = os.path.join(frames_folder, filename)
#            assert os.path.exists(frame_path)#

#            print(f'filename_{filename}')

            #if filename == '.DS_Store':
#                continue

            # Read the image and keep a reference to the original

        
        cap = cv2.VideoCapture(os.path.join(self.video_path, self.video + '.mp4'))
        idx = 0

        while True:
            # Read frame from the video
            ret, frame = cap.read()
            if not ret:
                print("Reached the end of the video.")
                break
            # Convert BGR frame to RGB frame
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        

#            original_frame = cv2.imread(frame_path)
            height,width,_ =  original_frame.shape

            original_h, original_w = original_frame.shape[:2]
            print(f'currently processing frame {str(idx)}')

            source = 'Predicted'
            # If this frame has ground truth data, use it. Otherwise, use YOLO for detection
            if idx in self.ground_truth:
                # Load ground truth bounding boxes
                boxes_and_ids = self.ground_truth[idx]
                j_boxes = [item[0] for item in boxes_and_ids]
                boxes_abs = []
                for bbox in j_boxes:
                    bb_left = bbox[0] * width
                    bb_top = bbox[1] * height
                    bb_right = (bbox[2] - bbox[0]) * width
                    bb_bottom = (bbox[3] - bbox[1]) * height
                    boxes_abs.append([bb_left, bb_top, bb_right, bb_bottom])

                gt_boxes = boxes_abs
                gt_ids = [item[1] for item in boxes_and_ids]
                gt_scores = np.ones((len(gt_boxes), 1))  # dummy scores
                gt_color = (0, 0, 255)  # Color for ground truth boxes in debug mode
                gt_source = 'train.json'

                # Detect using YOLO
                resized_frame = cv2.resize(original_frame, desired_size)
                if not self.pretrained:
                    pred_boxes, pred_scores = self.detect_objects(resized_frame)
                else:
                    pred_boxes, pred_scores = self.detect_objects_trained(resized_frame)


                pred_ids = [-1 * ( i + 1) for i in range(max(gt_ids), max(gt_ids) +len(pred_boxes))]  # Local object ids for detected boxes
                pred_color = (0, 255, 0)  # Color for YOLO-detected boxes in debug mode

                # Rescale detections back to original image size
                pred_boxes[:, [0, 2]] *= original_w / desired_size[0]
                pred_boxes[:, [1, 3]] *= original_h / desired_size[1]

                # Now we will compare IoU between each prediction and all ground truth boxes
                # If IoU is low, we keep the prediction. Otherwise, it's likely a duplicate of the ground truth.
                final_boxes, final_ids, final_scores, final_colors, final_sources = [], [], [], [], []
                for pred_box, pred_score, pred_id in zip(pred_boxes, pred_scores, pred_ids):
                    ious = [compute_iou(pred_box, gt_box) for gt_box in gt_boxes]
                    if not any([iou > self.iou_threshold for iou in ious]):
                        final_boxes.append(pred_box)
                        final_ids.append(pred_id)
                        final_scores.append(pred_score)

                # Append ground truth boxes to the final lists
                final_boxes.extend(gt_boxes)
                final_ids.extend(gt_ids)
                final_scores.extend(gt_scores)
#                final_colors.extend([gt_color] * len(gt_boxes))
                final_sources.extend([gt_source] * len(gt_boxes))

                scores = final_scores
                boxes = final_boxes
                ids = final_ids
                colors = final_colors
            else:
                resized_frame = cv2.resize(original_frame, desired_size)
                if not self.pretrained:
                    boxes, scores = self.detect_objects(resized_frame)
                else:
                    boxes, scores = self.detect_objects_trained(resized_frame)

                ids = [-1 * (i+1) for i in range(len(boxes))]  # Local object ids for detected boxes
                color = (0, 255, 0)  # Color for YOLO-detected boxes in debug mode

                # Rescale detections back to original image size
                boxes[:, [0, 2]] *= original_w / desired_size[0]
                boxes[:, [1, 3]] *= original_h / desired_size[1]

                detections = np.concatenate([boxes, scores], axis=1)

            if self.debug and idx in self.ground_truth:
                print(scores)
                for box, obj_id,  score in zip(boxes, ids,  scores):
                    print(score)
                    
                    if score <0.999:
                        color = (0,255,255)
                        source = 'predictions'
                    else:
                        color = (0,0,255)
                        source = 'GT'
                    x, y, w, h = map(int, box)
                    cv2.rectangle(original_frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(original_frame, f"{obj_id} ({source})", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(original_frame, str(idx), (10, 10 + cv2.getTextSize(str(idx), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                cv2.imshow('Debugging Frame', original_frame)
                cv2.waitKey(0)

            # Concatenate boxes, scores, and filename, and write results to all_detections list
            detections = np.concatenate([boxes, scores], axis=1)
            frame_id = idx
            for i, detection in enumerate(detections):
                x, y, w, h, score = detection
                obj_id = ids[i]
                all_detections.append([idx, obj_id, x, y, w, h, score])
            idx+=1


        # Convert the list to a NumPy array and save
        all_detections = np.array(all_detections)
        dummy = np.full((all_detections.shape[0], 3), -1)
        all_detections = np.concatenate((all_detections, dummy), axis=1)
        name =os.path.basename(result_folder).split('_')
        result_filename = os.path.join(result_folder, name[0]+"_"+ name[1]+"_"+ 'crops.npy')
        np.save(result_filename, all_detections)
        print(f"Results saved to {result_filename}")

    def detect_objects(self, frame):
        # Prepare input
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        # Run the model
        with torch.no_grad():
            outputs = self.model(img)
            outputs = outputs[0].boxes.boxes.numpy()
            boxes, scores = outputs[..., :4], outputs[..., 4:5]

        # Convert boxes to [x, y, w, h]
        boxes[:, 2:4] -= boxes[:, :2]
        return boxes, scores
    
    def detect_objects_trained(self, frame):
        img =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        outputs = self.model.predict_img(frame, rescale=False)
        boxes, scores = outputs[..., :4], outputs[..., 4:5]
        boxes[:, 2:4] -= boxes[:, :2]        
        return boxes, scores

        
    
    



if __name__ == "__main__":

    '''
    model = Predictor(model_path='/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/YOLOXl-009_003-019-0.839273.pth',
                      input_size=(640,640), device='cpu')
    

    image = cv2.imread('/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/video_1_frames/0.png')
    image = cv2.resize(image, (640,640))
    predictions = model.predict_img(image, rescale=False)
    print(predictions)


#    for game in video
     '''

    # Example usage
    yolo_model_path = "yolov8n.pt"
    
    ground_truth_json_path = "/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/sequences/train/video_1_sequence_dir/all_train.json"
    parent_folder = '/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/sequences/train/'

    with open( ground_truth_json_path, 'r') as f:
        data = json.load(f)
    videos = list(data.keys())
    
    for vid in videos:
        if vid != 'video_1':
            continue
        result_folder = os.path.join(parent_folder, vid + '_sequence_dir')
        frames_folder = os.path.join(result_folder, vid+'_frames')
        print(frames_folder)
        if not os.path.exists(frames_folder):
            print(vid)
            continue
        print('stexem')
        processor = YOL(MODEL_PATH, ground_truth_json_path, debug=True, video=vid, iou_threshold=0.96, pretrained=True)
        processor.process_frames_folder(result_folder)

