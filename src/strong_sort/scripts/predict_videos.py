# It is a metascript to make it easier to process allthe videos from a folder
import os
import json
import time

from tqdm import tqdm

from scripts.predict_video import predict, parse_gt, parse_args
import yaml


from strong_sort.scripts import load_config

if __name__ == '__main__':
    args = parse_args()
    gt_data = {}
    if args.gt_path is not None:
        with open(args.gt_path, 'r') as file:
            gt_data = json.load(file)

    video_dir = '/Users/aleksandrsimonyan/Desktop/deepmind_updated/data/preds/train'
    base_preds_dir = '/Users/aleksandrsimonyan/Desktop/deepmind_updated/data/preds/'
    for video_file in tqdm(sorted(os.listdir(video_dir))):
        if video_file.endswith('.mp4'):
            video_name = video_file.split('.')[0]
            gt_video_data = parse_gt(gt_data, video_name)
            s_time = time.time()
            predict(model_path=args.model_path,
                    video_path=os.path.join(video_dir, video_file),
                    predictions_dir=os.path.join(base_preds_dir, video_name),
                    reid_predictor_config=args.reid_predictor_config,
                    reid_model_path=args.reid_model_path,
                    device=args.device,
                    visualization=args.visualization,
                    gt_data=gt_video_data)
            proc_time = time.time() - s_time
            print(f'{video_file} Processing time: {round(proc_time, 2)} s')
