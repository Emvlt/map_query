import pathlib
import numpy as np
from PIL import Image, ImageFilter

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

