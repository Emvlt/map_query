from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pandas as pd
import numpy as np

from map_query.utils.image_transforms import load_tile_as_array, erode, dilate

class TrainingThumbnails(Dataset):
    def __init__(self,
        path_to_training_folder:Path,
        thumbnail_transforms=None,
        ) -> None:

        self.path_to_training_folder = path_to_training_folder
        self.thumbnail_transforms = thumbnail_transforms

    def __len__(self):
        return int(len(list(self.path_to_training_folder.glob('*'))) / 2)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        path_to_input  = self.path_to_training_folder.joinpath(f'input_{index}.npy')
        path_to_target = self.path_to_training_folder.joinpath(f'target_{index}.npy')

        np_input = np.load(path_to_input)
        np_tgt   = np.load(path_to_target)

        np_input = np.concatenate(
                (
                    erode(np_input[:]),
                    np_input,
                    dilate(np_input)
                 ), axis = 0)

        ipt = torch.from_numpy(np_input)
        tgt = torch.from_numpy(np_tgt)

        if self.thumbnail_transforms is not None:
            ipt = self.thumbnail_transforms(ipt)
            tgt = self.thumbnail_transforms(tgt)

        k = np.random.randint(0,4)
        ipt = torch.rot90(ipt, k, dims=(1,2))
        tgt = torch.rot90(tgt, k, dims=(1,2))

        return ipt, tgt

class Thumbnails(Dataset):
    def __init__(self,
                 tile_information:pd.Series,
                 tile_transform = None,
                 thumbnail_transforms = None,
                 verbose = True) -> None:

        self.tile_shape = (tile_information['height'], tile_information['width'])

        tile_path = Path(tile_information['tile_path'])
        self.tile = load_tile_as_array(tile_path)

        if tile_transform == 'maphis_transform':
            tile_to_thumbnail = {
            (7200,10800):{
                'height_padding':121,
                'width_padding':169,
                'height_stride':50,
                'width_stride':50,
                'n_cols':23,
                'n_rows':15
            },
            (7590,11400):{
                'height_padding':157,
                'width_padding':100,
                'height_stride':50,
                'width_stride':50,
                'n_cols':24,
                'n_rows':16
            }}
            self.tile_dict  = tile_to_thumbnail[self.tile_shape]
            self.thumbnail_size = 512

            self.padded_tile_shape = (
                tile_information['height'] + 2*self.tile_dict['height_padding'],
                tile_information['width'] + 2*self.tile_dict['width_padding'],
                )

            tile_padding = nn.ConstantPad2d((
                self.tile_dict['width_padding'], self.tile_dict['width_padding'],
                self.tile_dict['height_padding'], self.tile_dict['height_padding']), 0)


            transformed_tile = np.concatenate(
                (np.expand_dims(erode(self.tile), 0),
                 np.expand_dims(self.tile, 0),
                 np.expand_dims(dilate(self.tile), 0)
                 ), axis = 0)

            self.tensor_tile = tile_padding(torch.from_numpy(transformed_tile))

            self.compute_thumbnails_coordinate()
        else:
            raise NotImplementedError

        self.thumbnail_transforms = thumbnail_transforms

    def compute_thumbnails_coordinate(self):
        self.thumbnail_coordinates = {}
        n_thumbnails = 0
        for row_index in range(self.tile_dict['n_rows']+1):
            h_low  = (self.thumbnail_size - self.tile_dict['height_stride'])*row_index

            h_high = self.thumbnail_size*(row_index+1) - self.tile_dict['height_stride']*row_index

            for col_index in range(self.tile_dict['n_cols']+1):
                w_low  = (self.thumbnail_size - self.tile_dict['width_stride'])*col_index
                w_high = self.thumbnail_size*(col_index+1) - self.tile_dict['width_stride']*col_index
                self.thumbnail_coordinates[n_thumbnails] = {'w_low':w_low, 'w_high':w_high, 'h_low':h_low, 'h_high':h_high}
                n_thumbnails += 1

    def __len__(self):
        return len(self.thumbnail_coordinates)

    def __getitem__(self, index):
        thumbnail = self.tensor_tile[:, self.thumbnail_coordinates[index]['h_low']:self.thumbnail_coordinates[index]['h_high'],
            self.thumbnail_coordinates[index]['w_low']:self.thumbnail_coordinates[index]['w_high']]
        if self.thumbnail_transforms is not None:
            thumbnail = self.thumbnail_transforms(thumbnail)
        return thumbnail, index