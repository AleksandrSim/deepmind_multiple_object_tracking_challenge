import cv2
import os
import json
class FrameExtractor:
    
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def extract_frames(self):
        vidcap = cv2.VideoCapture(self.video_path)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        success, image = vidcap.read()
        count = 0
        while success:
            output_path = os.path.join(self.output_dir, f"{count}.png")
            cv2.imwrite(output_path, image)
            success, image = vidcap.read()
            count += 1
            print(f"Processed {count}/{total_frames} frames", end='\r')
        vidcap.release()
        print("\nExtraction completed!")

if __name__ == '__main__':
    video_path_parent = '/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/videos/'
    output_dir = '/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/sequences/train/video_1_sequence_dir/video_1_frames'
    par_path = '/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/sequences/train/'

    ground_truth_json_path = "/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/sequences/train/video_1_sequence_dir/all_train.json"

    with open( ground_truth_json_path, 'r') as f:
        data = json.load(f)
    videos = list(data.keys())

    for idx, vid in enumerate(videos):
        if idx ==10:
            break
        result_folder = os.path.join(par_path, vid + '_sequence_dir')
        frames_folder = os.path.join(result_folder, vid+'_frames')

        if not os.path.exists(result_folder):
            os.makedirs(result_folder, exist_ok=True)
        if not os.path.exists(result_folder):
            os.makedirs(frames_folder, exist_ok=True)
        
        video_path = os.path.join(video_path_parent, vid +'.mp4')
        frame_ext = FrameExtractor(video_path,frames_folder )
        frame_ext.extract_frames()