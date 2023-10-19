import os
import json
import logging
import optuna
from omegaconf import DictConfig, OmegaConf
from generate_files import PrepareFiles
from load_config import load_config


def run_pipeline(cfg):
    prepare_files = PrepareFiles(
        training_json=os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['training_json']),
        video_path=os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['video_path']),
        sequence_folder=os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['sequence_folder']),
        yolo_model_path=os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['yolo_model_path']),
        specific_videos=cfg['prepare_files']['specific_videos'],
        features_model_weights=os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['features_model_weights']),
        out_logs_path=os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['out_logs_path']),
        debug_yolo=cfg['prepare_files']['debug_yolo'],
        iou_thresh=cfg['prepare_files']['iou_thresh'],
        config_file=os.path.join(cfg['COMMON_PREFIX'], cfg['prepare_files']['config_file']),
        overwrite_strong_sort=True
    )
    prepare_files.run_strong_sort()
    prepare_files.generate_training_classes()
    prepare_files.generate_final()
    return prepare_files.calculate_moving_objects()


def objective(trial):
    logger = logging.getLogger(__name__)
    dynamic_params = {"AFLINK": trial.suggest_categorical('AFLINK', [True, False]),
                      "BoT": trial.suggest_categorical('BoT', [True, False]),
                      "ECC": trial.suggest_categorical('ECC', [True, False]),
#                      "EMA": trial.suggest_categorical('EMA', [True, False]),
#                      "EMA_alpha": trial.suggest_uniform('EMA_alpha', 0.8, 1.0),
#                      "GSI": trial.suggest_categorical('GSI', [True, False]),   TRUE does not work for some reason
#                      "MC": trial.suggest_categorical('MC', [True, False]),
#                      "MC_lambda": trial.suggest_uniform('MC_lambda', 0.95, 1.0),
#                      "NSA": trial.suggest_categorical('NSA', [True, False]),
#                      "max_age": trial.suggest_int('max_age', 1000, 2000),
#                      "max_cosine_distance": trial.suggest_int('max_cosine_distance', 0, 1),
#                      "max_iou_distance": trial.suggest_uniform('max_iou_distance', 0.2, 1.0),
                      "min_confidence": trial.suggest_uniform('min_confidence', 0.8, 1),
#                      "min_detection_height": trial.suggest_int('min_detection_height', 0, 20),
                      "n_init": trial.suggest_int('n_init', 5, 10000),
#                      "nms_max_overlap": trial.suggest_uniform('nms_max_overlap', 0.8, 1.0),
                      "nn_budget": trial.suggest_int('nn_budget', 5, 55),
                      "woC": trial.suggest_categorical('woC', [True, False])}
    
    dynamic_cfg = OmegaConf.create(dynamic_params)
    static_cfg = load_config()
    for dynamic_param in dynamic_cfg:
        if dynamic_param not in static_cfg['arguments']:
            print('somethign si wrong')
        else:
            static_cfg['arguments'][dynamic_param] = dynamic_params[dynamic_param]
            print(f'dynamic_param {dynamic_param} is owerritten')
            static_cfg['arguments'][dynamic_param]
            
    print(static_cfg)

    final_score = run_pipeline(static_cfg)
    logger.info(f"Trial {trial.number} completed with final_score: {final_score} and parameters: {dynamic_params}")
    all_results.append({"parameters": dynamic_params, "final_score": final_score})
    return final_score


def main():
    logger = logging.getLogger('OptunaExperimentation')

    logger.setLevel(logging.INFO)
    static_cfg = load_config()
    log_dir = os.path.join(static_cfg['COMMON_PREFIX'], static_cfg['prepare_files']['sequence_folder'])
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'optuna_log.txt')

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    global all_results
    all_results = []

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    logger.info(f"Best params: {study.best_params}, Best score: {study.best_value}")
    
    result_file_path = os.path.join(log_dir, 'final_optimization_results.json')
    with open(result_file_path, 'a') as f:
        json.dump(all_results, f)
if __name__ == "__main__":
    main()
