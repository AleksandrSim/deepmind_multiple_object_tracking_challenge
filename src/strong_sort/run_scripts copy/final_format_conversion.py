import json 
import numpy as np
import os

import sys
sys.path.append('src/strong_sort/')
from src.strong_sort.GSI import GSInterpolation, GaussianSmooth, LinearInterpolation

class ConvertFinalFormat:
    def __init__(self,sequence_dir, training_json, GSI=False, specific_videos=None):

        self.training_json = training_json
        self.sequence_dir = sequence_dir
        self.GSI = GSI
        self.specific_videos = specific_videos

    def iterate_videos(self):
        sequences = os.listdir(self.sequence_dir)
        sequences = [i for i in sequences if os.path.isdir(os.path.join(self.sequence_dir, i))]
        if self.specific_videos:
            sequences = [i for i in sequences if i in self.specific_videos]

        final_dic= {}

        for seq in sequences:

            curr_dir = os.path.join(self.sequence_dir, seq)
            if 'tracker.txt' not in os.listdir(curr_dir):
                continue

            video_name = seq
            original = np.genfromtxt(os.path.join(curr_dir, 'tracker.txt'), delimiter=',', dtype=float)
#            if self.GSI:
#                print('inside GSI')
#                GSInterpolation(path_in=os.path.join(curr_dir, 'tracker.txt'),
#                path_out=os.path.join(curr_dir, 'tracker_extraplolated.txt'), interval=20, tau=5)
#                original = np.genfromtxt(os.path.join(curr_dir, 'tracker_extraplolated.txt'), delimiter=',', dtype=float)

            converted = self.convert(original)
            final_dic[video_name]= converted

        with open(os.path.join( self.sequence_dir, 'final_predictions.json'), 'w') as f:
            json.dump(final_dic,f)
        self.final_dic = final_dic

    def convert(self, original):
        final = []

        track_ids = np.unique(original[:, 1])

        for track in track_ids:
            track_data = original[original[:, 1] == track]
            bounding_boxes = []
            frame_ids = []
            for row in track_data:
                frame_id, track_id = int(row[0]), int(row[1])
                x1, y1, w, h = row[2:6]
                bounding_boxes.append([x1, y1, x1+w, y1+h])
                frame_ids.append(frame_id)
            track_info = {'id': int(track), 'bounding_boxes': bounding_boxes, 'frame_ids': frame_ids}

            final.append(track_info)
        return final
            

    def fill_first_track(self):
            with open(self.training_json, 'r') as f:
                training_data = json.load(f)

            for video in self.final_dic.keys():
                [height, width] = training_data[video]['metadata']['resolution']

                pred_data = self.final_dic[video]  # List of dictionaries
                pred_ids = {d['id']: d for d in pred_data}  # Creates a lookup dictionary for IDs in pred_data

                for data in training_data[video]['object_tracking']:
                    id = data['id']
                    if 'initial_tracking_box' in data:
                        first_frame = np.argmax(data["initial_tracking_box"])
                        first_frame_n = data['frame_ids'][first_frame]
                        bbox = data['bounding_boxes'][first_frame]
                        bbox[0], bbox[2] = bbox[0] * width, bbox[2] * width
                        bbox[1], bbox[3] = bbox[1] * height, bbox[3] * height

                    else:
                        first_frame_n = data['frame_ids'][0]
                        bbox = data['bounding_boxes'][0]
                        bbox[0], bbox[2] = bbox[0] * width, bbox[2] * width
                        bbox[1], bbox[3] = bbox[1] * height, bbox[3] * height


                    if id not in pred_ids:
                        # If id is not in pred_data, create and append a new dict for that id
                        pred_data.append({
                            'id': id,
                            'frame_ids': [first_frame_n],
                            'bounding_boxes': [bbox]
                        })
                    else:
                        # If id exists, first check if first_frame_n is already present
                        if first_frame_n in pred_ids[id]['frame_ids']:
                            idx = pred_ids[id]['frame_ids'].index(first_frame_n)
                            del pred_ids[id]['frame_ids'][idx]
                            del pred_ids[id]['bounding_boxes'][idx]

                        # Update 'frame_ids' and 'bounding_boxes' for that id
                        pred_ids[id]['frame_ids'].append(first_frame_n)
                        pred_ids[id]['bounding_boxes'].append(bbox)

                        # Sort 'frame_ids' and rearrange 'bounding_boxes' based on the sorted 'frame_ids'
                        sort_indices = np.argsort(pred_ids[id]['frame_ids'])
                        pred_ids[id]['frame_ids'] = [pred_ids[id]['frame_ids'][i] for i in sort_indices]
                        pred_ids[id]['bounding_boxes'] = [pred_ids[id]['bounding_boxes'][i] for i in sort_indices]

    # No need for a return statement if you're modifying self.final_dic directly

    def extrapolate(self):
        extrapolated_final = {}
        
        for video, converted in self.final_dic.items():

            # Determine the min and max frame for the entire video
            all_frame_ids = [frame_id for track_info in converted for frame_id in track_info['frame_ids']]
            min_frame, max_frame = min(all_frame_ids), max(all_frame_ids)
            print(min_frame)
            print(max_frame)
            
            extrapolated_final[video] = []
            
            for track_info in converted:
                frame_ids = track_info['frame_ids']
                bounding_boxes = track_info['bounding_boxes']

                first_bbox = bounding_boxes[0]

                new_frame_ids = []
                new_bounding_boxes = []

                for i in range(min_frame, max_frame + 5):
                    if i in frame_ids:
                        new_frame_ids.append(i)#
#                        new_bounding_boxes.append(first_bbox)

                        new_bounding_boxes.append(bounding_boxes[frame_ids.index(i)])
                    else:
                        new_frame_ids.append(i)
                        new_bounding_boxes.append(first_bbox)

                extrapolated_track_info = {'id': track_info['id'], 'bounding_boxes': new_bounding_boxes, 'frame_ids': new_frame_ids}
                extrapolated_final[video].append(extrapolated_track_info)

        with open(os.path.join(self.sequence_dir, 'final_predictions_extrapolated.json'), 'w') as f:
            json.dump(extrapolated_final, f)

    

if __name__=='__main__':
    conv = ConvertFinalFormat('/Users/aleksandrsimonyan/Desktop/test/')
    conv.iterate_videos()
    conv.extrapolate(conv.final_dic())


