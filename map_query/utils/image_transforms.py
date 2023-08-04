import pathlib
import numpy as np
from typing import List

from PIL import Image, ImageFilter
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import Compose


def normalise(sample: torch.Tensor) -> torch.Tensor:
    return (sample - sample.min()) / (sample.max() - sample.min())

class ToFloat(object):
    def __init__(self):
        pass

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        return sample.float()

class Normalise(object):
    def __init__(self):
        pass

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        return normalise(sample)

class TransfromTemplate(object):
    def __init__(self):
        pass

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        return sample

def load_tile(tile_path:pathlib.Path, mode='L') -> Image.Image:
    return Image.open(tile_path).convert(mode)

def load_tile_as_array(tile_path:pathlib.Path, mode='L') -> np.ndarray:
    return np.array(load_tile(tile_path, mode))

def image_filter_gaussian_blur():
    return ImageFilter.GaussianBlur(radius=1)

def load_and_preprocess(tile_path:pathlib.Path, mode='L', transforms = [image_filter_gaussian_blur()]) -> np.ndarray:
    tile = load_tile(tile_path, mode)
    for transform in transforms:
        tile = tile.filter(transform)
    return np.array(tile, np.uint8)

def dilate(src:np.ndarray, dilate_size=1):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilate_size + 1, 2 * dilate_size + 1),
                                    (dilate_size, dilate_size))
    return cv2.dilate(src.astype('uint8'), element)

def erode(src:np.ndarray, dilate_size=1):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_size + 1, 2 * dilate_size + 1), (dilate_size, dilate_size))
    return cv2.erode(src.astype('uint8'), element)

def parse_transformations_list(transformation_list:List):
    trafos = []

    for transformation in transformation_list:
        if transformation[0] == 'ToFloat':
            parsed_transformed = ToFloat()
        elif transformation[0] == 'Normalise':
            parsed_transformed = Normalise()

        else:
            raise NotImplementedError
        trafos.append(parsed_transformed)
    return Compose(trafos)