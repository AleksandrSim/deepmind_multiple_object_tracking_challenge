import os
import json
from collections import defaultdict


class DataTransformer:
    def __init__(self, input_path, output_path):
        with open(input_path, 'r') as f:
            self.data = json.load(f)
        self.output_path = output_path

    def transform_data(self):
        transformed_data = {}
        for video_name, video_data in self.data.items():
            print(video_name)

            # Extracting object tracking information
            object_tracking = video_data['object_tracking']

            # Initializing a dictionary for transformed object tracking data
            transformed_object_tracking = defaultdict(list)

            # Transforming the object tracking data
            for item in object_tracking:
                for frame_idx, frame_id in enumerate(item['frame_ids']):
                    transformed_item = {
                        'id': item['id'],
                        'label': item['label'],
                        'is_occluder': int(item['is_occluder']),
                        'bounding_boxes': item['bounding_boxes'][frame_idx],
                        'initial_tracking_box': item['initial_tracking_box'][frame_idx],
                        'timestamps': item['timestamps'][frame_idx],
                        'is_masked': item['is_masked'][frame_idx]}

                    transformed_object_tracking[frame_id].append(
                        transformed_item)

            transformed_data[video_name] = transformed_object_tracking

        self.transformed_data = transformed_data

    def save(self):
        with open(self.output_path, 'w') as f:
            json.dump(self.transformed_data, f, indent=2)


if __name__ == '__main__':
    path = '/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/all_train.json'
    output = '/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/return.json'

    transform = DataTransformer(path, output)
    transform.transform_data()
    transform.save()

    