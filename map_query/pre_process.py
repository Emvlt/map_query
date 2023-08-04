from typing import Dict, Tuple
from pathlib import Path

import pandas as pd
import geopandas as gpd

from map_query.utils.image_transforms import load_tile_as_array, np
from map_query.utils.geodata import pd, unpack_crs, process_city_geodata, create_city_tile_coordinates

def pre_process_city(
    paths:Dict[str,Path],
    city_name:str,
    pre_process_dict
    ) -> Tuple[Path, gpd.GeoDataFrame]:
    print(f'Pre-processing {city_name}')
    ### Unpacking preprocess_dict
    map_image_file_extension = pre_process_dict['map_image_file_extension']

    geo_data_file_extension = pre_process_dict['geo_data_file_extension']
    geo_data_file_extension_driver = pre_process_dict['geo_data_file_extension_driver']

    raw_city_path = paths['raw'].joinpath(city_name)

    processed_city_path = paths['processed'].joinpath(city_name)
    processed_city_path.mkdir(exist_ok=True, parents=True)
    dataframe_path = processed_city_path.joinpath(f'{city_name}{geo_data_file_extension}')

    print(f'Checking if dataframe already exists at {dataframe_path}')
    if dataframe_path.is_file():
        print(f'Folder at {dataframe_path} already processed, passing..')
        city_geo_dataframe:gpd.GeoDataFrame = gpd.read_file(dataframe_path) #type:ignore
    else:
        print('Collecting tiles...')
        list_of_tiles = list(raw_city_path.glob(f'*{map_image_file_extension}'))
        city_dict = {
            'tile_name':[],
            'width':[],
            'height':[],
            'tile_path':[],
            'crs':[],
            'geometry':[],
            }

        for tile_path in list_of_tiles:
            print(f'Processing tile {tile_path.stem}')

            tile_name = tile_path.stem
            city_dict['tile_name'].append(tile_name)

            tile = load_tile_as_array(tile_path)
            n_pixels_x = np.shape(tile)[1]
            n_pixels_y = np.shape(tile)[0]
            city_dict['width'].append(n_pixels_x)
            city_dict['height'].append(n_pixels_y)

            city_dict['tile_path'].append(tile_path)

            additional_tile_file = list(raw_city_path.glob(f'{tile_name}*'))
            for tile_file_path in additional_tile_file:
                city_dict = process_city_geodata(tile_file_path, city_dict, (n_pixels_x, n_pixels_y))

        crs = unpack_crs(city_dict)

        city_dataframe = pd.DataFrame(city_dict)
        city_dataframe.drop(columns='crs', inplace=True)
        city_dataframe = create_city_tile_coordinates(city_dataframe)
        city_geo_dataframe = gpd.GeoDataFrame(city_dataframe, geometry='geometry', crs=crs) #type:ignore
        print(city_dataframe)
        city_geo_dataframe.to_file(dataframe_path, driver=geo_data_file_extension_driver)

    return dataframe_path, city_geo_dataframe