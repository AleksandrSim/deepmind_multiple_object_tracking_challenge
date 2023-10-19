
import random
from typing import Any, List, Tuple, Callable

import cv2
import numpy as np
import torch
from torchvision import transforms as T

from src.detect.utils import xyxy2cxcywh

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomCrop:
    def __init__(self, min_size: Tuple[int, int] = (640, 640)):
        self.min_size = min_size

    def __call__(self, image: np.ndarray, target: Any = None):
        h, w, _ = image.shape
        x0, y0 = np.random.randint(
            low=(0, 0),
            high=(w-self.min_size[0], h-self.min_size[1]))
        x1, y1 = np.random.randint(
            low=(x0+self.min_size[0], y0+self.min_size[1]),
            high=(w, h))
        image = image[y0:y1, x0:x1, :]
        target[:, 1::2] -= x0
        target[:, 1::2] = np.clip(target[:, 1::2], 0, x1-x0)
        target[:, 2::2] -= y0
        target[:, 2::2] = np.clip(target[:, 2::2], 0, y1-y0)
        return image, target


class ColorAugment:
    def __init__(self,
                 brightness: Tuple[float, float] = (0.8, 1.2),
                 color: Tuple[float, float] = (0.8, 1.2),
                 contrast: Tuple[float, float] = (0.8, 1.2)):
        self.brightness = brightness
        self.color = color
        self.contrast = contrast

    def __call__(self, image: np.ndarray, target: Any = None)\
            -> Tuple[np.ndarray, Any]:
        image = image.astype(float)
        random_colors = np.random.uniform(
            self.brightness[0], self.brightness[1])\
            * np.random.uniform(self.color[0], self.color[1], 3)
        for i in range(3):
            image[:, :, i] = image[:, :, i] * random_colors[i]
        mean = image.mean(axis=(0, 1))
        contrast = np.random.uniform(self.contrast[0], self.contrast[1])
        image = (image - mean) * contrast + mean
        image = np.clip(image, 0.0, 255.0)
        image = image.astype(np.uint8)
        return image, target


class GaussNoise:
    def __init__(self, sigma_sq: float = 30.0):
        self.sigma_sq = sigma_sq

    def __call__(self, image: np.ndarray, target: Any = None)\
            -> Tuple[np.ndarray, Any]:
        image = image.astype(int)
        h, w, c = image.shape
        gauss = np.random.normal(0, np.random.uniform(0.0, self.sigma_sq),
                                 (h, w, c))
        image = image + gauss
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        return image, target


class Flip:
    def __call__(self, image: np.ndarray, target: Any = None)\
            -> Tuple[np.ndarray, Any]:
        image = cv2.flip(image, 1)
        if target is not None:
            target[:, 1] = image.shape[1] - target[:, 1]
        return image, target


class ScalePadded:
    def __init__(self, input_size: Tuple[int, int] = (640, 640),
                 fill_value: int = 0, max_labels: int = 128) -> None:
        super().__init__()
        self.input_shape = (input_size[0], input_size[1], 3)
        self.fill_value = fill_value
        self.max_labels = max_labels

    def __call__(self, image: np.ndarray, target: Any = None)\
            -> Tuple[np.ndarray, Any]:

        padded_img = np.ones(self.input_shape, dtype=np.uint8)\
            * self.fill_value

        r = min(self.input_shape[0] / image.shape[0],
                self.input_shape[1] / image.shape[1])
        new_w = int(image.shape[1] * r)
        new_h = int(image.shape[0] * r)
        resized_img = cv2.resize(
            image, (new_w, new_h),
            interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        x0 = int((self.input_shape[1] - new_w) / 2)
        y0 = int((self.input_shape[0] - new_h) / 2)
        padded_img[y0:y0+new_h, x0:x0+new_w, :] = resized_img

        if target is not None:
            labels = target[:, 0].copy()
            boxes = target[:, 1:].copy()
            if len(boxes) > 0:
                boxes_cxcy = xyxy2cxcywh(boxes).astype(float)
                boxes_cxcy *= r
                boxes_cxcy[:, 0] += x0
                boxes_cxcy[:, 1] += y0

                mask = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
                boxes_cxcy = boxes_cxcy[mask]
                labels = labels[mask]
                labels = np.expand_dims(labels, 1)
                target = np.hstack((labels, boxes_cxcy))
            else:
                target = np.zeros((self.max_labels, 5))

            padded_target = np.zeros((self.max_labels, 5))
            padded_target[range(len(target))[:self.max_labels]]\
                = target[:self.max_labels]
            target = np.ascontiguousarray(
                padded_target, dtype=np.float32)

        return padded_img, target


class ToTensor:
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=mean, std=std)

    def __call__(self, image: np.ndarray, target: Any = None)\
            -> Tuple[np.ndarray, Any]:
        image = self.to_tensor(image)
        if target is not None:
            target = torch.from_numpy(target)
        return image, target


class UseWithProb:
    """Apply transform with a given probability for data augmentation.

    Args:
        transform (Callable): Transform to apply.
        prob (float, optional): Probability of the transform. Should be in
            range [0..1]. Defaults to 0.5.
    """

    def __init__(self, transform: Callable, prob: float = 0.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image: np.ndarray, target: Any = None)\
            -> Tuple[np.ndarray, Any]:
        if random.random() < self.prob:
            image, target = self.transform(image, target)
        return image, target


class ComposeTransform:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, target: Any = None):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


def train_transform(input_size: Tuple[int, int] = (640, 640),
                    fill_value: int = 0, max_labels: int = 128):
    transforms = ComposeTransform([
        UseWithProb(RandomCrop(min_size=input_size), 0.5),
        ScalePadded(input_size, fill_value, max_labels),
        UseWithProb(ColorAugment(), 0.33),
        UseWithProb(GaussNoise(), 0.1),
        UseWithProb(Flip(), 0.5),
        ToTensor()
    ])
    return transforms


def val_transform(input_size: Tuple[int, int] = (640, 640),
                  fill_value: int = 0, max_labels: int = 128):
    transforms = ComposeTransform([
        ScalePadded(input_size, fill_value, max_labels),
        ToTensor()
    ])
    return transforms
