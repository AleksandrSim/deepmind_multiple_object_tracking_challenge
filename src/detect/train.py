import os
from typing import Tuple, Callable, Optional

import hydra
from argus import load_model
from torch import load, compile
from omegaconf import OmegaConf, DictConfig
from torch.backends import cudnn
from argus.callbacks import Checkpoint, EarlyStopping, LoggingToFile, \
    MonitorCheckpoint, ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.argus_utils import load_compatible_weights
from src.detect.dataset import DetectDataset
from src.detect.transforms import val_transform, train_transform
from src.detect.yolox.metrics import IOUMetric

CONFIG_PATH = '/workdir/src/detect/train_config.yaml'


def get_loader(data_params: DictConfig, dataset_params: DictConfig,
               transform: Optional[Callable] = None, shuffle: bool = False)\
        -> DataLoader:
    dataset = DetectDataset(img_dir=dataset_params.img_dir,
                            annot_file=dataset_params.annot_file,
                            samples_range=dataset_params.range,
                            transform=transform)
    loader = DataLoader(
        dataset, batch_size=data_params.batch_size,
        num_workers=data_params.num_workers,
        pin_memory=data_params.pin_memory,
        shuffle=shuffle)
    return loader


@hydra.main(version_base=None, config_path=os.path.dirname(CONFIG_PATH),
            config_name=os.path.splitext(os.path.basename(CONFIG_PATH))[0])
def main(cfg: DictConfig) -> None:
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    model = hydra.utils.instantiate(cfg.model)
    train_trns = train_transform(cfg.data_params.input_size,
                                 cfg.data_params.fill_value,
                                 cfg.data_params.max_labels)
    val_trns = val_transform(cfg.data_params.input_size,
                             cfg.data_params.fill_value,
                             cfg.data_params.max_labels)
    train_loader = get_loader(cfg.data_params, cfg.data_params.train,
                              train_trns, True)
    val_loader = get_loader(cfg.data_params, cfg.data_params.val,
                            val_trns, False)
    print('Train images:', len(train_loader.dataset))
    print('Val images:', len(val_loader.dataset))
    experiment_name = cfg.metadata.experiment_name
    run_name = cfg.metadata.run_name
    save_dir = f'/workdir/data/experiments/{experiment_name}_{run_name}'
    metrics = [IOUMetric()]

    callbacks = [
        EarlyStopping(patience=cfg.train_params.early_stopping_epochs,
                      monitor=cfg.train_params.monitor_metric,
                      better=cfg.train_params.monitor_metric_better),
        ReduceLROnPlateau(factor=cfg.train_params.reduce_lr_factor,
                          patience=cfg.train_params.reduce_lr_patience,
                          monitor=cfg.train_params.monitor_metric,
                          better=cfg.train_params.monitor_metric_better),
        MonitorCheckpoint(save_dir, max_saves=3,
                          monitor=cfg.train_params.monitor_metric,
                          better=cfg.train_params.monitor_metric_better),
        Checkpoint(save_dir, max_saves=2, save_after_exception=True),
        LoggingToFile(os.path.join(save_dir, 'log.txt')),
    ]

    OmegaConf.save(cfg, os.path.join(save_dir, os.path.basename(CONFIG_PATH)))
    pretrain_path = cfg.model.params.pretrain
    if pretrain_path is not None:
        if os.path.exists(pretrain_path):
            if cfg.model.params.pretrain_convert:
                model_pretrain = load(pretrain_path)
                model = load_compatible_weights(model_pretrain, model)
            else:
                model = load_model(pretrain_path,
                                   nn_module=cfg.model.params.nn_module,
                                   device=cfg.model.params.device)
            model.set_lr(cfg.model.params.optimizer.lr)
    if cfg.train_params.use_compile:
        model.nn_module = compile(model.nn_module)
    model.fit(train_loader, val_loader=val_loader, metrics_on_train=False,
              num_epochs=cfg.train_params.max_epochs,
              callbacks=callbacks,
              metrics=metrics)


if __name__ == "__main__":
    main()
