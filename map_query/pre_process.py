from typing import Dict, Tuple, List
from pathlib import Path

from shapely import Polygon
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from map_query.image_transforms import load_tile_as_array, np

def compute_tile_shape(tile_path:Path) -> Tuple:
    print(f'Computing tile shape from tile at {tile_path}')
    tile = load_tile_as_array(tile_path)
    n_pixels_x = np.shape(tile)[1]
    n_pixels_y = np.shape(tile)[0]
    return (n_pixels_x, n_pixels_y)

def process_tfw_file(file_path:Path, tile_width, tile_height) -> dict:
    tfw_raw = open(file_path, 'r').read()
    x_diff = float(tfw_raw.split("\n")[0])
    y_diff = float(tfw_raw.split("\n")[3])
    west_boundary = float(tfw_raw.split("\n")[4])
    north_boundary = float(tfw_raw.split("\n")[5])
    east_boundary = west_boundary + (tile_width - 1) * x_diff
    south_boundary = north_boundary + (tile_height - 1) * y_diff
    return {
        'west_boundary':west_boundary,
        'north_boundary':north_boundary,
        'east_boundary':east_boundary,
        'south_boundary':south_boundary,
        'x_diff':x_diff, 'y_diff':y_diff,
        'lattitude_length': (tile_width - 1) * x_diff,
        'longitude_length':(tile_height - 1) * y_diff}

def process_city_geodata(tile_file_path:Path, map_geodata_file_extension:str, city_dict:Dict[str,List], tile_shape:Tuple) -> Dict[str,List]:
    if tile_file_path.suffix == map_geodata_file_extension:
        if map_geodata_file_extension == '.tfw':
            geo_dict = process_tfw_file(tile_file_path, tile_shape[0], tile_shape[1])
            city_dict['geometry'].append(Polygon([
                (geo_dict['west_boundary'] ,geo_dict['north_boundary']),
                (geo_dict['east_boundary'], geo_dict['north_boundary']),
                (geo_dict['east_boundary'], geo_dict['south_boundary']),
                (geo_dict['west_boundary'], geo_dict['south_boundary'])
            ]))
        else:
            raise NotImplementedError

        for key, item in geo_dict.items():
            if key in city_dict.keys():
                city_dict[key].append(item)
            else:
                city_dict[key] = [item]
    return city_dict


def create_tile_coord(dataframe:pd.DataFrame) -> pd.DataFrame:
    ## Get west and north boundaries
    west_boundary  = dataframe['west_boundary'].min()
    north_boundary = dataframe['north_boundary'].max()

    dataframe['tile_x_coord'] = (dataframe['west_boundary'] - west_boundary)/dataframe['lattitude_length']
    dataframe['tile_y_coord'] = (dataframe['north_boundary'] - north_boundary)/dataframe['longitude_length']

    return dataframe

def pre_process_city(
    paths:Dict[str,Path],
    city_name:str,
    pre_process_dict
    ) -> Tuple[Path, gpd.GeoDataFrame]:
    print(f'Pre-processing {city_name}')
    ### Unpacking preprocess_dict
    map_image_file_extension = pre_process_dict['map_image_file_extension']
    map_geodata_file_extension = pre_process_dict['map_geodata_file_extension']
    crs = pre_process_dict['crs']
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
            'tile_width':[],
            'tile_height':[],
            'geometry':[]
            }

        for tile_path in list_of_tiles:
            print(f'Processing tile {tile_path.stem}')

            tile_name = tile_path.stem
            city_dict['tile_name'].append(tile_name)

            tile_shape = compute_tile_shape(tile_path)
            city_dict['tile_width'].append(tile_shape[0])
            city_dict['tile_height'].append(tile_shape[1])

            additional_tile_file = list(raw_city_path.glob(f'{tile_name}*'))
            for tile_file_path in additional_tile_file:
                city_dict = process_city_geodata(tile_file_path, map_geodata_file_extension, city_dict, tile_shape)

        city_dataframe = pd.DataFrame(city_dict)
        city_dataframe = create_tile_coord(city_dataframe)
        city_geo_dataframe = gpd.GeoDataFrame(city_dataframe, geometry='geometry', crs=crs) #type:ignore

        print(f'Saving geo dataframe at {dataframe_path}')
        city_geo_dataframe.to_file(dataframe_path, driver=geo_data_file_extension_driver)

    return dataframe_path, city_geo_dataframe