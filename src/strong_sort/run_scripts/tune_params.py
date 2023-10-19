import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from generate_files import PrepareFiles
import os
from load_config import load_config
import copy
static_cfg = load_config()

@hydra.main(config_path="conf", config_name="conf")
def main(cfg: DictConfig) -> None:
    # Convert DictConfig to a native Python dictionary
    dynamic_config = OmegaConf.to_container(cfg, resolve=True)


    new_cfg_dict = copy.deepcopy(static_cfg)
    # Update arguments in new_cfg_dict if they exist in dynamic_config
    for param, value in static_cfg['arguments'].items():
        if param in dynamic_config:
            new_cfg_dict['arguments'][param] = dynamic_config[param]
        else:
            new_cfg_dict['arguments'][param] = value  # Optional, as this value is already in new_cfg_dict

    # Now, new_cfg_dict contains the updated configuration, and all other keys are preserved
    print('Updated configuration:', new_cfg_dict)

    prepare_files = PrepareFiles(
        training_json=os.path.join(new_cfg_dict['COMMON_PREFIX'], new_cfg_dict['prepare_files']['training_json']),
        video_path=os.path.join(new_cfg_dict['COMMON_PREFIX'], new_cfg_dict['prepare_files']['video_path']),
        sequence_folder=os.path.join(new_cfg_dict['COMMON_PREFIX'], new_cfg_dict['prepare_files']['sequence_folder']),
        yolo_model_path=os.path.join(new_cfg_dict['COMMON_PREFIX'], new_cfg_dict['prepare_files']['yolo_model_path']),
        specific_videos=new_cfg_dict['prepare_files']['specific_videos'],
        features_model_weights=os.path.join(new_cfg_dict['COMMON_PREFIX'], new_cfg_dict['prepare_files']['features_model_weights']),
        out_logs_path=os.path.join(new_cfg_dict['COMMON_PREFIX'], new_cfg_dict['prepare_files']['out_logs_path']),
        debug_yolo=new_cfg_dict['prepare_files']['debug_yolo'],
        iou_thresh=new_cfg_dict['prepare_files']['iou_thresh'],
        config_file=os.path.join(new_cfg_dict['COMMON_PREFIX'], new_cfg_dict['prepare_files']['config_file']),
        overwrite_strong_sort=True
    )

    prepare_files.run_strong_sort(new_cfg_dict)
    prepare_files.generate_training_classes(new_cfg_dict)
    prepare_files.generate_final()
    final_score = prepare_files.calculate_moving_objects()
    print(f'final_score___IOU for Moving objects {final_score}')
    print(f'Tested with parameters: {OmegaConf.to_yaml(new_cfg_dict)}')
    return final_score  # Return the value that you want to maximize

if __name__ == "__main__":
    main()