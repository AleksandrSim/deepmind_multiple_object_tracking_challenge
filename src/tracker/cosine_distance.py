import numpy as np
import torch
import torch.nn.functional as F


class CosineDistance:
    def __init__(self, use_torch: bool = False, device: str = 'cuda:0'):
        self.use_torch = use_torch
        self.device = device

    def _cosine_distance_np(self, x, y):
        distances = 1. - np.dot(x, y.T)
        return distances.min(axis=0)

    def _cosine_distance_torch(self, x, y):
        x = torch.tensor(np.array(x)).to(self.device)
        y = torch.tensor(np.array(y)).to(self.device)
        distances = 1.0 - F.cosine_similarity(x.unsqueeze(1), y, dim=-1)
        return distances.min(axis=0)[0].cpu().numpy()

    def __call__(self,
                 x,
                 y):
        if self.use_torch:
            return self._cosine_distance_torch(x, y)
        return self._cosine_distance_np(x, y)
