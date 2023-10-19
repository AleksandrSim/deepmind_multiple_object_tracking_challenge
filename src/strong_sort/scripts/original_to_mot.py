import json
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
class MOTCConverter:
    def __init__(self, json_path, video_folder_path, output_folder, video_name):
        self.video_name = video_name
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.video_folder_path = video_folder_path
        self.output_folder = output_folder

    def _get_video_shape(self, video_name):
        import cv2
        video_path = os.path.join(self.video_folder_path, video_name + ".mp4")
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return width, height

    def convert_to_motc(self):
            width, height = self._get_video_shape(self.video_name)
            tracking_data =  self.data[self.video_name]['object_tracking']

            all_frames = []

            for obj in tracking_data:
                obj_id = obj['id']
                for frame_id, bbox in zip(obj['frame_ids'], obj['bounding_boxes']):
                    # Rescale the bounding boxes
                    bb_left = bbox[0] * width
                    bb_top = bbox[1] * height
                    bb_width = (bbox[2] - bbox[0]) * width
                    bb_height = (bbox[3] - bbox[1]) * height

                    all_frames.append([frame_id, obj_id, bb_left, bb_top, bb_width, bb_height, 1, -1, -1, -1])

            all_frames = sorted(all_frames, key=lambda x: x[0])  # sort by frame id

            # Save as numpy array
            np_path = os.path.join(self.output_folder, self.video_name +'_gt')
            np.save(np_path, all_frames)

            print(f"Saved {self.video_name} data to {np_path}")


    def debug_display(self, video_name, frame_id):
        video_path = os.path.join(self.video_folder_path, self.video_name + ".mp4")
        cap = cv2.VideoCapture(video_path)
        
        # Move to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id-1)  # 0-indexed

        ret, frame = cap.read()
        if not ret:
            print("Failed to retrieve the frame.")
            return

        # Draw bounding boxes for the given frame_id
        all_frames = np.load(os.path.join(self.output_folder, video_name + ".npy"), allow_pickle=True)
        # Filter out bounding boxes of the desired frame
        for data in all_frames:
            if int(data[0]) == frame_id:
                bb_left, bb_top, bb_width, bb_height = data[2:6]
                color = (0, 255, 0)  # Green color
                cv2.rectangle(frame, (int(bb_left), int(bb_top)), (int(bb_left + bb_width), int(bb_top + bb_height)), color, 2)
                cv2.putText(frame, str(int(data[1])), (int(bb_left), int(bb_top)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Convert BGR to RGB for matplotlib display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        plt.imshow(frame)
        plt.title(f'Frame ID: {frame_id}')
        plt.show()

        cap.release()
if __name__=='__main__':
    converter = MOTCConverter('/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/all_train.json', 
                              '/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/videos', '/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/ground_truth_mopt')
#    converter.convert_to_motc()

    video_name_to_debug = "video_1"  # replace with the desired video name
    frame_id_to_debug = 0  # replace with the desired frame ID
    converter.debug_display(video_name_to_debug, frame_id_to_debug)