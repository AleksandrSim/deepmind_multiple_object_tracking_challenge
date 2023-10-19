import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
class ObjectDetector:
    def __init__(self, model, window_size=(640, 640), overlap_fraction=0.3, threshold=0.8, non_max_thresh=0.2, adjusted_window=True):
        self.model = model
        self.window_size = window_size
        self.overlap_fraction = overlap_fraction
        self.threshold = threshold
        self.non_max_thresh = non_max_thresh

    def adjusted_window(self, image, overlap_fraction, crop_size):
        height, width, _ = image.shape
        
        x_step = int(crop_size * (1 - overlap_fraction))
        y_step = int(crop_size * (1 - overlap_fraction))

        x_steps = (width - crop_size) // x_step
        y_steps = (height - crop_size) // y_step

        x_step = (width - crop_size) // (x_steps + 1)
        y_step = (height - crop_size) // (y_steps + 1)

        windows = []
        for y_pos in range(0, height - crop_size + 1, y_step):
            for x_pos in range(0, width - crop_size + 1, x_step):
                window = self._take_crop(image, y_pos, x_pos, crop_size)
                windows.append((x_pos, y_pos, window))
        return windows

    def _take_crop(self, image, y_pos, x_pos, crop_size):
        return image[y_pos: y_pos + crop_size, x_pos: x_pos + crop_size]

    def sliding_window(self, image):
        overlap_size = (int(self.window_size[0] * self.overlap_fraction),
                        int(self.window_size[1] * self.overlap_fraction))

        y_step = self.window_size[1] - overlap_size[1]
        x_step = self.window_size[0] - overlap_size[0]

        windows = []
        # Add windows for the main grid
        for y_pos in range(0, image.shape[0] - self.window_size[1] + 1, y_step):
            for x_pos in range(0, image.shape[1] - self.window_size[0] + 1, x_step):
                window = image[y_pos: y_pos + self.window_size[1], x_pos: x_pos + self.window_size[0]]
                windows.append((x_pos, y_pos, window))

        # Add extra windows to cover right and bottom borders if they're missed
        if image.shape[1] % self.window_size[0] != 0:
            x_pos = image.shape[1] - self.window_size[0]
            for y_pos in range(0, image.shape[0] - self.window_size[1] + 1, y_step):
                window = image[y_pos: y_pos + self.window_size[1], x_pos: x_pos + self.window_size[0]]
                windows.append((x_pos, y_pos, window))

        if image.shape[0] % self.window_size[1] != 0:
            y_pos = image.shape[0] - self.window_size[1]
            for x_pos in range(0, image.shape[1] - self.window_size[0] + 1, x_step):
                window = image[y_pos: y_pos + self.window_size[1], x_pos: x_pos + self.window_size[0]]
                windows.append((x_pos, y_pos, window))

        return windows
    def detect_objects(self, frame, debug=True):
        # Prepare input
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        # Run the model
        with torch.no_grad():
            outputs = self.model(img)
            outputs= outputs[0].boxes.boxes.numpy()
            boxes, scores = outputs[..., :4], outputs[..., 4:5]

        # Debugging visualization
        if debug:
            for box in boxes:
                x, y, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            plt.imshow(frame)
            plt.show()

        print(boxes)
        return boxes[:, :4], scores 

    def convert_to_original_scale(self, boxes, x_offset, y_offset):
        return [[x + x_offset, y + y_offset, x2 + x_offset, y2 + y_offset] for x, y, x2, y2 in boxes]

    def custom_non_max_suppression(self, boxes, scores, threshold=0.1):
        if len(boxes) == 0:
            return []
        
        print("Scores shape:", scores.shape) # Debugging print
        print("Boxes shape:", boxes.shape) # Debugging print

        boxes = boxes.astype("float")
        pick = []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            intersection = w * h
            union = area[i] + area[idxs[:last]] - intersection
            iou = intersection / union

            print("IoUs:", iou) # Debugging print

            idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > threshold)[0])))

        return boxes[pick].astype("int"), scores[pick]


    def detect(self, image, debug=True):
        all_boxes = []
        all_scores = []

        windows = self.adjusted_window(image, self.overlap_fraction, self.window_size[0])
#        windows = self.sliding_window(image)

        for (x, y, window) in windows:
            print(window.shape)

            detected_boxes, detected_scores = self.detect_objects(window, debug=True)
            detected_boxes_original_scale = self.convert_to_original_scale(detected_boxes, x, y)
            all_boxes.extend(detected_boxes_original_scale)
            all_scores.extend(detected_scores)

        # Debugging before non-max suppression
        if debug:
            debug_image = np.copy(image)
            for box in all_boxes:
                try:
                    x, y, x2, y2 = map(int, box)
                    cv2.rectangle(debug_image, (x, y), (x2, y2), (0, 255, 0), 2)
                except TypeError as e:
                    print(f"An error occurred while processing the box {box}: {e}")
                    print(f"Box shape: {np.shape(box)}, Box contents: {box}")

            plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
            plt.title("Before Non-Max Suppression")
            plt.show()
            print(f"Total boxes before non-max suppression: {len(all_boxes)}")

        all_scores = np.array(all_scores).flatten() # Flatten the scores
        filtered_boxes, filtered_scores = self.custom_non_max_suppression(np.array(all_boxes), np.array(all_scores), self.non_max_thresh)
        return filtered_boxes, filtered_scores

if __name__=='__main__':
    model_y = YOLO("yolov8n.yaml")  # build a new model from scratch
    model_y = YOLO("yolov8n.pt") 
    detector = ObjectDetector(model=model_y)
    image_path = '/Users/aleksandrsimonyan/Desktop/yolo_weights/frames_orig/frame_0006.png'
    image = cv2.imread(image_path)
    plt.imshow(image)
    plt.show()

    filtered_boxes, filtered_scores = detector.detect(image, debug=True)
    image = cv2.imread(image_path)

    # Drawing the bounding boxes on the original image
    for box in filtered_boxes:
        try:
            x, y, x2, y2 = map(int, box)
            cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
        except TypeError as e:
            print(f"An error occurred while processing the box {box}: {e}")
            print(f"Box shape: {np.shape(box)}, Box contents: {box}")

    # Displaying the original image with bounding boxes
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image with Bounding Boxes After non max suppresion")
    plt.show()