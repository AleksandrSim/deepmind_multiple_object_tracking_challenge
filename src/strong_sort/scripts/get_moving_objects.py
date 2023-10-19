import json
from tqdm import tqdm

class BoundingBoxProcessor:

    def __init__(self, input_path, iou_threshold=0.9, validation=True):
        self.input_path = input_path
        self.iou_threshold = iou_threshold
        self.validation = validation
        self.read_data()

    def read_data(self):
        with open(self.input_path, 'r') as f:
            self.data = json.load(f)

    def save_output(self, output_path):
        with open(output_path, 'w') as f:
            index = 14
            json.dump(self.data, f, indent=2)

    def compute_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        xa, ya, xb, yb = box2

        x2, y2 = x1 + x2, y1 + y2
        xb, yb = xa + xb, ya + yb

        xA = max(x1, xa)
        yA = max(y1, ya)
        xB = min(x2, xb)
        yB = min(y2, yb)

        inter_area = max(0, xB - xA) * max(0, yB - yA)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (xb - xa) * (yb - ya)
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    def process_frame(self, video_name, track):
        is_moving = False
        if self.validation:
            current_track = self.data[video_name][track]
        else:
            current_track = self.data[video_name]['object_tracking'][track]
        bounding_boxes = current_track['bounding_boxes']

        
        for idx in range(1, len(bounding_boxes)):
            if self.compute_iou(bounding_boxes[idx], bounding_boxes[idx-1]) < self.iou_threshold:
                    is_moving = True
                    break

        current_track['moving'] = is_moving
#        if self.validation:
#            if not is_moving:
#               current_track['bounding_boxes'] = [current_track['bounding_boxes'][0] for idx in range(len(current_track['bounding_boxes']))]


    def process_videos(self):
        for video_name in self.data.keys():
            print(video_name)
            if self.validation:
                for track in range(len(self.data[video_name])):
                    self.process_frame(video_name, track)
            else:
                for track in range(len(self.data[video_name]['object_tracking'])):
                    self.process_frame(video_name, track)


if __name__ == '__main__':
    input_path = "/Users/aleksandrsimonyan/Desktop/deepmind_updated/data/final_predictions_extrapolated_normalized.json"  # Replace this with the path to your input JSON file
    output_path = "/Users/aleksandrsimonyan/Desktop/deepmind_updated/data/final_predictions_extrapolated_normalized_stationary.json"  # Replace this with the path where you want to save the output JSON file
    processor = BoundingBoxProcessor(input_path, iou_threshold=0.84, validation=True)
    processor.process_videos()
    processor.save_output(output_path)
