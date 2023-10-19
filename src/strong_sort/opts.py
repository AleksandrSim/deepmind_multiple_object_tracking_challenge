"""
@Author: Du Yunhao
@Filename: opts.py
@Contact: dyh_bupt@163.com
@Time: 2022/2/28 19:41
@Discription: opts
"""
import json
import argparse
from os.path import join



MAIN_DIR_PATH='/Users/aleksandrsimonyan/Desktop/test/video_1_sequence_dir/'


data = {
    'sequences': {
        'train':[
            'video_9431_sequence_dir',
           'video_1_sequence_dir']

    }}

class opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
#        self.parser.add_argument(
#            'dataset',
#            type=str,
#            help='MOT17 or MOT20',
#            required=False
#        )
#        self.parser.add_argument(
#            '--video-directory',
#            type=str,
#            help='path to the video collection',
#            default='/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/videos/'
#        )
 #       self.parser.add_argument(
#            'mode',
#            type=str,
#            help='val or train',
#            required=False
#        )
        self.parser.add_argument(
            '--BoT',
            action='store_false',
            help='Replacing the original feature extractor with BoT'
        )
        self.parser.add_argument(
            '--ECC',
            action='store_true',
            help='CMC model'
        )
        self.parser.add_argument(
            '--NSA',
            action='store_true',
            help='NSA Kalman filter'
        )
        self.parser.add_argument(
            '--EMA',
            action='store_true',
            help='EMA feature updating mechanism'
        )
        self.parser.add_argument(
            '--MC',
            action='store_true',
            help='Matching with both appearance and motion cost'
        )
        self.parser.add_argument(
            '--woC',
            action='store_true',
            help='Replace the matching cascade with vanilla matching'
        )
        self.parser.add_argument(
            '--AFLINK',
            action='store_true',
            help='Appearance-Free Link'
        )
        self.parser.add_argument(
            '--GSI',
            action='store_true',
            help='Gaussian-smoothed Interpolation'
        )
        self.parser.add_argument(
            '--root_dataset',
            default='/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind'
        )
        self.parser.add_argument(
            '--path_AFLink',
            default='/Users/aleksandrsimonyan/Desktop/deepmind_files/deepmind/afflink/AFLink_epoch20.pth'
        )
        self.parser.add_argument(
            '--dir_save',
            default=MAIN_DIR_PATH
        )
        self.parser.add_argument(
            '--EMA_alpha',
            default=0.9
        )
        self.parser.add_argument(
            '--MC_lambda',
            default=0.98)

        self.parser.add_argument(
        '--max_iou_distance',
        default=0.9
        ),
        self.parser.add_argument(
        '--max_age',
        default=1500
        ),
        self.parser.add_argument(
        '--n_init',
        default=10
        )
        self.parser.add_argument(
            '--loger-path',
            default=None
        )
        self.parser.add_argument(
        '--display',
        default=False
        )

    def parse(self, args=''):
        if args == '':
          opt = self.parser.parse_args()
        else:
          opt = self.parser.parse_args(args)
        opt.min_confidence = 0.00001
        opt.nms_max_overlap = 0.9
        opt.min_detection_height = 0
        opt.dir_dets = MAIN_DIR_PATH
        if opt.BoT:
            opt.max_cosine_distance =  10
#            opt.dir_dets = 
        else:
            opt.max_cosine_distance = 10
#            opt.dir_dets = 
        if opt.MC:
            opt.max_cosine_distance += 10
        if opt.EMA:
            opt.nn_budget = 500
        else:
            opt.nn_budget = 500
        if opt.ECC:
            path_ECC = '/data/dyh/results/StrongSORT_Git/{}_ECC_{}.json'.format(opt.dataset, opt.mode)
            opt.ecc = json.load(open(path_ECC))
        opt.sequences = data[opt.dataset][opt.mode]
        opt.dir_dataset = join(
            opt.root_dataset,
            opt.dataset,
            'train' if opt.mode == 'train' else 'test'
        )
        return opt
    
opt = opts().parse()


if __name__ == "__main__":
    arguments = ["sequences", "val", "--BoT", "--ECC"]  # this mimics the command line arguments
    options = opts().parse(args=arguments)
    print(options)
