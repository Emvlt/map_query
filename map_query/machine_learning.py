from pathlib import Path
import math

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from map_query.utils.image_transforms import load_tile_as_array, erode, dilate


class Thumbnails(Dataset):
    def __init__(self,
                 tile_information:pd.Series,
                 tile_transform = None,
                 thumbnail_transforms = None,
                 verbose = True) -> None:

        tile_shape = (tile_information['height'], tile_information['width'])

        tile_path = Path(tile_information['tile_path'])
        tile = load_tile_as_array(tile_path)

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
            self.tile_dict  = tile_to_thumbnail[tile_shape]
            self.thumbnail_size = 512

            self.padded_tile_shape = (
                tile_information['height'] + 2*self.tile_dict['height_padding'],
                tile_information['width'] + 2*self.tile_dict['width_padding'],
                )

            tile_padding = nn.ConstantPad2d((
                self.tile_dict['width_padding'], self.tile_dict['width_padding'],
                self.tile_dict['height_padding'], self.tile_dict['height_padding']), 0)


            transformed_tile = np.concatenate(
                (np.expand_dims(erode(tile), 0),
                 np.expand_dims(tile, 0),
                 np.expand_dims(dilate(tile), 0)
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

class Down2D(nn.Module):
    """Down sampling unit of factor 2

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            filter_size (int): size of the filter of the conv layers, odd integer
    """
    def __init__(self, in_channels:int, out_channels:int, filter_size:int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels,  in_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Conv2d(in_channels,  out_channels, filter_size, 1, int((filter_size-1) / 2)),
            nn.LeakyReLU(negative_slope = 0.1),
            nn.Conv2d(out_channels, out_channels, filter_size, 1, int((filter_size - 1) / 2)),
            nn.LeakyReLU(negative_slope = 0.1)
        )

    def forward(self, input_tensor:torch.Tensor) -> torch.Tensor:
        """forward function of the Down2D module: input -> output

        Args:
            input_tensor (torch.Tensor): input tensor
        Returns:
            torch.Tensor: output tensor
        """
        return self.down(input_tensor)

class Up2D(nn.Module):
    """Up sampling unit of factor 2

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
    """
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.unpooling2d = nn.ConvTranspose2d(in_channels, in_channels, 4, stride = 2, padding = 1)
        self.conv1 = nn.Conv2d(in_channels,  out_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2 * out_channels, out_channels, 3, stride=1, padding=1)
        self.l_relu = nn.LeakyReLU(negative_slope = 0.1)

    def forward(self, input_tensor:torch.Tensor, skp_connection:torch.Tensor) -> torch.Tensor:
        """forward function of the Up2D module: input -> output

        Args:
            input_tensor (torch.Tensor): input tensor
            skp_connection (torch.Tensor): input from downsampling path

        Returns:
            torch.Tensor: output tensor
        """
        x_0 = self.l_relu(self.unpooling2d(input_tensor))
        x_1 = self.l_relu(self.conv1(x_0))
        return self.l_relu(self.conv2(torch.cat((x_1, skp_connection), 1)))

class Unet2D(nn.Module):
    """Definition of the 2D unet
    """
    def __init__(self, in_channels:int, out_channels:int, ngf:int):
        super().__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(in_channels, ngf, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(ngf, ngf, 5, stride=1, padding=2)
        self.down1 = Down2D(ngf, 2*ngf, 5)
        self.down2 = Down2D(2*ngf, 4*ngf, 3)
        self.down3 = Down2D(4*ngf, 8*ngf, 3)
        self.down4 = Down2D(8*ngf, 16*ngf, 3)
        self.down5 = Down2D(16*ngf, 32*ngf, 3)
        self.down6 = Down2D(32*ngf, 64*ngf, 3)
        self.down7 = Down2D(64*ngf, 64*ngf, 3)
        self.up1   = Up2D(64*ngf, 64*ngf)
        self.up2   = Up2D(64*ngf, 32*ngf)
        self.up3   = Up2D(32*ngf, 16*ngf)
        self.up4   = Up2D(16*ngf, 8*ngf)
        self.up5   = Up2D(8*ngf, 4*ngf)
        self.up6   = Up2D(4*ngf, 2*ngf)
        self.up7   = Up2D(2*ngf, ngf)
        self.conv3 = nn.Conv2d(ngf, in_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.l_relu = nn.LeakyReLU(negative_slope=0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor :torch.Tensor):
        s_0  = self.l_relu(self.conv1(input_tensor))
        s_1 = self.l_relu(self.conv2(s_0))
        s_2 = self.down1(s_1)
        s_3 = self.down2(s_2)
        s_4 = self.down3(s_3)
        s_5 = self.down4(s_4)
        s_6 = self.down5(s_5)
        s_7 = self.down6(s_6)
        u_0 = self.down7(s_7)
        u_1 = self.up1(u_0, s_7)
        u_2 = self.up2(u_1, s_6)
        u_3 = self.up3(u_2, s_5)
        u_4 = self.up4(u_3, s_4)
        u_5 = self.up5(u_4, s_3)
        u_6 = self.up6(u_5, s_2)
        u_7 = self.up7(u_6, s_1)
        y_0 = self.l_relu(self.conv3(u_7))
        y_1 = self.sigmoid(self.conv4(y_0))
        return y_1

class SegmentationModel(nn.Module):
    def __init__(self, parameters_dict:dict):
        super().__init__()
        ## Assert that all parameters are here:
        for param_name in ['ngf', 'n_gabor_filters', 'support_sizes']:
            if not parameters_dict[param_name]:
                raise KeyError (f'{param_name} is missing')
        self.ngf           = parameters_dict['ngf']
        self.n_gabor_filters  = parameters_dict['n_gabor_filters']
        self.support_sizes    = parameters_dict['support_sizes']
        self.n_input_channels = parameters_dict['n_input_channels']
        self.n_output_channels = parameters_dict['n_output_channels']
        self.train_mode = parameters_dict['train_mode']
        self.gabor_filters  = nn.ModuleDict({
            f'{support_size}': nn.Conv2d(
                self.n_input_channels,
                int(self.n_gabor_filters/len(self.support_sizes)),
                support_size,
                stride = 1,
                padding=int((support_size-1)/2), padding_mode='reflect' ) for support_size in self.support_sizes
            })

        for param in self.gabor_filters.parameters():
            param.requires_grad = False
        self.set_gabor_filters_values()

        self.network = Unet2D(self.n_gabor_filters, self.n_output_channels, self.ngf)

    def set_gabor_filters_values(self, theta_range = 90):
        """Set the gabor filters values of the nn.module dictionnary

        Args:
            theta_range (int, optional): Angles at which to instantiate the filters. Defaults to 180.
        """
        thetas = torch.linspace(0, theta_range, int(self.ngf/len(self.support_sizes)))
        for support_size in self.support_sizes:
            filters = GaborFilters(support_size)
            for indextheta, theta in enumerate(thetas):
                self.gabor_filters[f'{support_size}'].weight[indextheta][0] = nn.parameter.Parameter(filters.get_filter(theta), requires_grad=self.train_mode) #type:ignore

    def forward(self, input_tensor:torch.Tensor):
        c_5  = self.gabor_filters['5'](input_tensor)
        c_7  = self.gabor_filters['7'](input_tensor)
        c_9  = self.gabor_filters['9'](input_tensor)
        c_11 = self.gabor_filters['11'](input_tensor)
        filtered_input = torch.cat((c_5,c_7,c_9,c_11),1)
        return self.network(filtered_input)

class GaborFilters():
    """Class defition of the gabor filters"""
    def __init__(self, support_size:int, frequency=1/8, sigma=3) -> None:
        """Initialise Gabor filters for fixed frequency and support size and sigma

        Args:
            support_size (int): Size of the gabor filter, odd integer
            frequency (_type_, optional): Frequency of the Gabor filter. Defaults to 1/8.
            sigma (int, optional): Deviation of the Gabor filter. Defaults to 3.
        """
        self.grid_x, self.grid_y = torch.meshgrid(torch.arange(-math.floor(support_size/2),math.ceil(support_size/2)), torch.arange(-math.floor(support_size/2),math.ceil(support_size/2)), indexing='ij')
        self.frequency = frequency
        self.sigma_squared = sigma**2

    def get_filter(self, theta:torch.Tensor) -> torch.Tensor:
        """Returns a (self.grid_x.shape, self.grid_y.shape) sized matrix containing the Gabor filter values for the and Theta

        Args:
            theta (float): angle, in radians, at which the filter is returned

        Returns:
            np.float32: The Gabor filter values
        """
        g_filter = torch.cos(2*3.1415*self.frequency*(self.grid_x*torch.cos(theta) + self.grid_y*torch.sin(theta)))*torch.exp(-(self.grid_x*self.grid_x+self.grid_y*self.grid_y)/(2*self.sigma_squared))
        return g_filter/torch.linalg.norm(g_filter)
