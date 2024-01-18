from typing import List

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