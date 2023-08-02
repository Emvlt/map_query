from pathlib import Path
from typing import Dict, List, Tuple
from itertools import groupby

import pandas as pd
from shapely import Polygon

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def process_tfw_file(file_path:Path, tile_width, tile_height) -> Dict:
    tfw_raw = open(file_path, 'r').read()
    x_diff = float(tfw_raw.split("\n")[0])
    y_diff = float(tfw_raw.split("\n")[3])
    west_boundary = float(tfw_raw.split("\n")[4])
    north_boundary = float(tfw_raw.split("\n")[5])
    east_boundary = west_boundary + (tile_width - 1) * x_diff
    south_boundary = north_boundary + (tile_height - 1) * y_diff
    return {
        'west':west_boundary,
        'north':north_boundary,
        'east':east_boundary,
        'south':south_boundary,
        'x_diff':x_diff, 'y_diff':y_diff,
        'lat_length': (tile_width - 1) * x_diff,
        'lon_length':(tile_height - 1) * y_diff}

def process_prj_file(tile_file_path:Path) ->Dict:
    prj_string = open(tile_file_path, 'r').read()
    return {'crs':prj_string}

def process_city_geodata(tile_file_path:Path, city_dict:Dict[str,List], tile_shape:Tuple) -> Dict[str,List]:
    if tile_file_path.suffix == '.tfw':
        geo_dict = process_tfw_file(tile_file_path, tile_shape[0], tile_shape[1])
        city_dict['geometry'].append(Polygon([
            (geo_dict['west'] ,geo_dict['north']),
            (geo_dict['east'], geo_dict['north']),
            (geo_dict['east'], geo_dict['south']),
            (geo_dict['west'], geo_dict['south'])
        ]))
    elif tile_file_path.suffix == '.prj':
        geo_dict = process_prj_file(tile_file_path)
    else:
        geo_dict = {}

    for key, item in geo_dict.items():
        if key in city_dict.keys():
            city_dict[key].append(item)
        else:
            city_dict[key] = [item]

    return city_dict

def create_city_tile_coordinates(dataframe:pd.DataFrame) -> pd.DataFrame:
    ## Get west and north boundaries
    west_boundary  = dataframe['west'].min()
    north_boundary = dataframe['north'].max()

    dataframe['tile_x'] = ((dataframe['west'] - west_boundary)/dataframe['lat_length']).astype(int)
    dataframe['tile_y'] = ((dataframe['north'] - north_boundary)/dataframe['lon_length']).astype(int)

    return dataframe

def unpack_crs(city_dict:Dict):
    assert 'crs' in city_dict, 'There is no crs in the extracted geodata'
    crs_list = city_dict['crs']
    assert crs_list is not None, 'Crs is specified as from_file, but no crs_list provided'
    assert all_equal(crs_list), 'Crs is inconsistent across tiles'
    return crs_list[0]
