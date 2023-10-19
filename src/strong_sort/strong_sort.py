
import yaml
import os
import warnings
import yaml
import torch
from os.path import join

# from opts import opt
from deep_sort_modified import run
from AFLink.AppFreeLink import PostLinker, AFLink, LinkData
from GSI import GSInterpolation
from load_config import load_config



class StrongSort:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.dataset = None

        if self.cfg['arguments']['AFLINK']:
            self.model = PostLinker()
            self.model.load_state_dict(
                torch.load(join(self.cfg['COMMON_PREFIX'],self.cfg['arguments']['path_AFLink']), 
                                        map_location=torch.device('cpu')))
            self.dataset = LinkData('', '')

    def process_video(self, seq):
        detection_name = str(seq)
        video_name = detection_name.replace('_sequence_dir', '')
        print(f'Processing video {seq}...')
        path_save = join(self.cfg['COMMON_PREFIX'],self.cfg['arguments']['root_dataset'], seq, f"{video_name}_exp_output.txt")
        seq_dir = join(self.cfg['COMMON_PREFIX'],self.cfg['arguments']['root_dataset'], seq)

        run(
            sequence_dir=join(self.cfg['COMMON_PREFIX'],self.cfg['arguments']['root_dataset'], seq),
            detection_file=join(self.cfg['COMMON_PREFIX'],seq_dir, f"{video_name}.npy"),
            output_file=path_save,
            min_confidence=self.cfg['arguments']['min_confidence'],
            nms_max_overlap=self.cfg['arguments']['nms_max_overlap'],
            min_detection_height=self.cfg['arguments']['min_detection_height'],
            max_cosine_distance=self.cfg['arguments']['max_cosine_distance'],
            nn_budget=self.cfg['arguments']['nn_budget'],
            display=self.cfg['arguments']['display'],
            save = self.cfg['arguments']['save_video']
        )

        if self.cfg['arguments']['AFLINK']:
            linker = AFLink(
                path_in=path_save,
                path_out=path_save,
                model=self.model,
                dataset=self.dataset,
                thrT=(0, 30),
                thrS=75,
                thrP=0.05
            )
            linker.link()

        if self.cfg['arguments']['GSI']:
            GSInterpolation(
                path_in=path_save,
                path_out=path_save,
                interval=50,
                tau=10
            )

    def main(self):
        warnings.filterwarnings("ignore")
        print('aflink_ashxatuma')

        for i, seq in enumerate(self.cfg['dataset_options']['sequences']['train'], start=1):
            self.process_video(seq)


if __name__ == '__main__':
    cfg = load_config()
    print(cfg)
    strong_sort_instance = StrongSort(cfg)
    strong_sort_instance.main()




