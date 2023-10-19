import json
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import cv2
from itertools import combinations
import os

class VideoDataEDA:
    def __init__(self, data, all_videos=False):
        self.data = data
        self.label_counts = defaultdict(int)
        self.frame_counts = defaultdict(int)
        self.video_counts = defaultdict(Counter)
        self.label_distribution = defaultdict(int)
        if all_videos:
            self.label_distribution_all = defaultdict(int)

    def analyze(self):
        for video_key, video_value in self.data.items():
            for frame in video_value['object_tracking']:
                self.label_counts[frame['label']] += 1
                self.video_counts[video_key][frame['label']] += 1
        for count in self.video_counts.values():
            self.frame_counts[sum(count.values())] += 1

    def plot_counts(self, counts, title, xlabel, ylabel, top_n=None):
        if top_n is not None:
            counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_n])
        print(f'Unique classes count { len(set(counts.keys())) }')
        keys = list(counts.keys())
        values = [counts[key] for key in keys]
        plt.figure(figsize=(10, 8))
        plt.bar(keys, values)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=90)
        plt.show()


    def get_frame_bboxes(self, video_key, rem_30=False):
        object_tracking = self.data[video_key]['object_tracking']
        frame_bboxes = defaultdict(list)
        for track in object_tracking:
            initial_bbox_flag = True
            for frame_id, bbox, is_masked, initial_tracking in zip(track['frame_ids'], track['bounding_boxes'], track['is_masked'], track['initial_tracking_box']):
                # Add bbox, label and masked status to this frame's list of bboxes
                if rem_30 and frame_id % 30 != 0:
                    continue
                # Count the number of instances of each label in the current frame
                label_counts = defaultdict(int)
                for other_bbox in frame_bboxes[frame_id]:
                    label_counts[other_bbox[1].split("_")[0]] += 1
                # Append an index to the label if there are multiple instances of it in the frame
                label = track['label']
                label_counts[label] += 1
                if label_counts[label] > 1:
                    label += f'_{label_counts[label]}'
                frame_bboxes[frame_id].append((bbox, label, is_masked, initial_tracking, initial_bbox_flag))
                if initial_tracking and initial_bbox_flag:
                    initial_bbox_flag = False
        frame_bboxes= {i:k for i,k in sorted(frame_bboxes.items(), key = lambda x: x[0] )}
        return frame_bboxes

    def visualize_frames(self,  video_path, frame_bboxes):
        video_capture = cv2.VideoCapture(video_path)
        if video_capture.isOpened():
            for frame_id, bboxes in frame_bboxes.items():
                # Read the frame_id frame
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = video_capture.read()
                if not ret:
                    break

                # Draw each bbox on the frame
                for bbox, label, is_masked, initial_tracking, initial_bbox_flag in bboxes:
                    # Denormalize the bbox coordinates
                    img_height, img_width, _ = frame.shape
                    print(img_height)
                    print(img_height)
                    x1 = int(bbox[0] * img_width)
                    y1 = int(bbox[1] * img_height)
                    x2 = int(bbox[2] * img_width)
                    y2 = int(bbox[3] * img_height)

                    # Use blue color for initial tracking box, green color for masked objects, and red color for non-masked objects
                    if initial_tracking and initial_bbox_flag:
                        color = (255, 0, 0) # blue color for initial box
                        cv2.putText(frame, f'Initial appearance of {label}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    elif is_masked:
                        color = (0, 255, 0) # green for masked
                    else:
                        color = (0, 0, 255) # red for non-masked

                    # Draw rectangle on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # Put label on frame   
                    cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Display the frame using matplotlib after all bboxes have been drawn
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.show()

            video_capture.release()
            cv2.destroyAllWindows()
        else:
            print(f'Failed to open video: {video_path}')

    def compute_label_distribution(self, frame_bboxes, save_comb=2):
            self.label_distribution = defaultdict(int)
            for frame_id, bboxes in frame_bboxes.items():
                labels = sorted(list(set([bbox[1] for bbox in bboxes]))) # removes duplicates
                # count individual labels
                self.label_distribution.update(Counter(labels))
                # count combinations of labels of every length
                for r in range(2, min(len(labels)+1, 2)): 
                    for combo in combinations(labels, r): # generates all combinations of r labels
                        self.label_distribution[', '.join(combo)] += 1
                # count all labels together
                self.label_distribution[', '.join(labels)] += 1



    def plot_label_distribution(self):
        self.plot_counts(self.label_distribution, 'Distribution of Labels in Frames', 'Labels', 'Count')

    def save_label_distribution(self, out_path, video_key):
        self.label_distribution = {i:k for i,k in sorted(self.label_distribution.items(), key= lambda x :x[1]) }
        with open(os.path.join(out_path, video_key +'.json'), 'w') as w:
            json.dump(self.label_distribution, w, indent=1)


if __name__=='__main__':
    videos = os.listdir('/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/videos')
    out_path = '/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/video_stats'
    files = os.listdir(out_path)
    with open('/Users/aleksandrsimonyan/Desktop/all_train.json') as json_file:
    with open('/Users/aleksandrsimonyan/Desktop/all_train.json') as json_file:
            data = json.load(json_file)
    for video in videos:
        print(video[:-4]+ '.json')
 #       if (video[:-4]+ '.json') in files:
 #           continue


        print(f'now processing_ {video}')
        video_key = video[:-4]
        video_key ='video_1'

        video_data_eda = VideoDataEDA(data)
        video_data_eda.analyze()
        frame_bboxes = video_data_eda.get_frame_bboxes(video_key, rem_30=True)
        video_data_eda.visualize_frames('/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/videos/' + video_key +'.mp4', frame_bboxes)
#        video_data_eda.compute_label_distribution(frame_bboxes)
#        video_data_eda.save_label_distribution(out_path, video_key)