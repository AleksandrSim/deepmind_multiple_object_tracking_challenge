import torch
from argus.engine import State
from argus.metrics.metric import Metric


class IOUMetric(Metric):
    """Pseudo metrics, which takes values from step_output and computes average
        IOU and other metrics and loses to use for training monitoring.
    """

    name = 'iou'
    better = 'max'

    def __init__(self):
        super().__init__()
        self.n_elements = 0
        self.obj_sum = 0.0
        self.l1_sum = 0.0
        self.iou_loss_sum = 0.0
        self.iou_sum = 0.0

    def reset(self):
        self.n_elements = 0
        self.obj_sum = 0.0
        self.l1_sum = 0.0
        self.iou_loss_sum = 0.0
        self.iou_sum = 0.0

    def update(self, step_output: dict):
        metrics = step_output['metrics']
        batch_size = step_output['target'].shape[0]
        self.n_elements += batch_size
        self.obj_sum += metrics['loss_obj'] * batch_size
        self.l1_sum += metrics['loss_l1'] * batch_size
        self.iou_loss_sum += metrics['loss_iou'] * batch_size
        self.iou_sum += metrics['iou'] * batch_size

    def compute(self) -> float:
        if self.n_elements > 0:
            return self.iou_sum / self.n_elements
        else:
            return float('inf')

    def epoch_complete(self, state: State):
        name_prefix = f'{state.phase}_' if state.phase else ''
        with torch.no_grad():
            iou = self.compute()
        obj_loss, l1, iou_loss = float('inf'), float('inf'), float('inf')
        if self.n_elements > 0:
            obj_loss = self.obj_sum / self.n_elements
            l1 = self.l1_sum / self.n_elements
            iou_loss = self.iou_loss_sum / self.n_elements
        state.metrics[f'{name_prefix}obj_loss'] = obj_loss
        state.metrics[f'{name_prefix}l1'] = l1
        state.metrics[f'{name_prefix}iou_loss'] = iou_loss
        state.metrics[f'{name_prefix}{self.name}'] = iou
