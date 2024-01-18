import math
from typing import List, Dict

import torch
import torch.nn as nn

def load_model(model_dict:Dict, device:torch.device) ->torch.nn.Module:
     ###  Load the model from the model_dict parameters
    print('\t Loading model...')
    if model_dict['name'] == 'maphis_segmentation':
        model = SegmentationModel(model_dict)
    else:
        raise NotImplementedError

    if 'model_load_path' in model_dict:
        print(f'Loading model at {model_dict["model_load_path"]}')
        model.load_state_dict(torch.load(model_dict['model_load_path'], map_location=device))

    return model

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
        self.activation_layer = nn.Sigmoid()

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
        y_1 = self.activation_layer(self.conv4(y_0))
        return y_1

class SegmentationModel(nn.Module):
    def __init__(self, parameters_dict:dict):
        super().__init__()
        ### Gabor filters parameters
        self.support_sizes:List        = parameters_dict['support_sizes']
        ## Spatial resolution parameters
        self.frequency_range:List     = parameters_dict['frequency_range']
        self.frequency_resolution:int = parameters_dict['frequency_resolution']
        ## Angular resolution parameters
        self.angular_range:List     = parameters_dict['angular_range']
        self.angular_resolution:int =  parameters_dict['angular_resolution']
        ### Unet parameters
        self.ngf:int                = parameters_dict['ngf']
        self.n_input_channels:int   = parameters_dict['n_input_channels']
        self.n_output_channels:int  = parameters_dict['n_output_channels']
        ### For each support size, there is
        ## frequency_resolution*angular_resolution filters
        self.gabor_filters  = nn.ModuleDict({
            f'{support_size}': nn.Conv2d(
                self.n_input_channels,
                self.frequency_resolution*self.angular_resolution,
                support_size,
                stride = 1,
                bias = False,
                padding=int((support_size-1)/2), padding_mode='reflect' ) for support_size in self.support_sizes
            })
        self.set_gabor_filters_values()

        self.unet = Unet2D(
            len(self.support_sizes)*self.frequency_resolution*self.angular_resolution,
            self.n_output_channels,
            self.ngf
            )

    def set_gabor_filters_values(self):
        """Set the gabor filters values of the nn.module dictionnary

        Args:
            theta_range (float, optional): angular range of the filters, in radians
        """
        angular_space = torch.linspace(self.angular_range[0], self.angular_range[1], self.angular_resolution)
        frequential_space = torch.linspace(self.frequency_range[0], self.frequency_range[1], self.frequency_resolution)
        with torch.no_grad():
            for support_size in self.support_sizes:
                filters = GaborFilters(support_size)
                for angular_index, theta in enumerate(angular_space):
                    for spatial_index,frequency in enumerate(frequential_space):
                        weight_index = (angular_index * self.angular_resolution) + spatial_index

                        self.gabor_filters[f'{support_size}'].weight[weight_index][0] = nn.parameter.Parameter(  #type:ignore
                        filters.get_filter(theta, frequency ), requires_grad=True
                        )

    def forward(self, input_tensor:torch.Tensor):
        return self.unet(
            torch.cat(
                [self.gabor_filters[f'{i}'](input_tensor) for i in self.support_sizes]
                ,1)
            )

class GaborFilters():
    """Class defition of the gabor filters"""
    def __init__(self, support_size:int, sigma=3) -> None:
        """Initialise Gabor filters for fixed frequency and support size and sigma

        Args:
            support_size (int): Size of the gabor filter, odd integer
            frequency (_type_, optional): Frequency of the Gabor filter. Defaults to 1/8.
            sigma (int, optional): Deviation of the Gabor filter. Defaults to 3.
        """
        self.grid_x, self.grid_y = torch.meshgrid(torch.arange(-math.floor(support_size/2),math.ceil(support_size/2)), torch.arange(-math.floor(support_size/2),math.ceil(support_size/2)), indexing='ij')
        self.sigma_squared = sigma**2

    def get_filter(self, theta:torch.Tensor, frequency:torch.Tensor) -> torch.Tensor:
        """Returns a (self.grid_x.shape, self.grid_y.shape) sized matrix containing the Gabor filter values for the and Theta

        Args:
            theta (float): angle, in radians, at which the filter is returned

        Returns:
            np.float32: The Gabor filter values
        """
        g_filter = torch.cos(2*3.1415*frequency*(self.grid_x*torch.cos(theta) + self.grid_y*torch.sin(theta)))*torch.exp(-(self.grid_x*self.grid_x+self.grid_y*self.grid_y)/(2*self.sigma_squared))
        return g_filter/torch.linalg.norm(g_filter)
