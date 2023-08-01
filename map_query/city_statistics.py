from typing import Dict
from pathlib import Path

import pandas as pd
import geopandas as gpd

def compute_feature_density(paths:Dict[str, Path], city_name:str, operation_dict:Dict):
    ### Unpacking paths
    processed_city_path = paths['processed'].joinpath(city_name)

    ### Unpack operation dict
    feature_list = operation_dict['features']
    geo_data_file_extension = operation_dict['geo_data_file_extension']

    city_feature_statistics_path = processed_city_path.joinpath(f'{city_name}_statistics{geo_data_file_extension}')

    if city_feature_statistics_path.is_file():
        city_feature_statistics:gpd.GeoDataFrame = gpd.GeoDataFrame.from_file(city_feature_statistics_path)

    else:
        ### Load city dataframe
        city_geodf_path = processed_city_path.joinpath(f'{city_name}{geo_data_file_extension}')
        city_feature_statistics:gpd.GeoDataFrame = gpd.GeoDataFrame.from_file(city_geodf_path)
        ### Load city feature_dataframe
        city_feature_geodf_path = processed_city_path.joinpath(f'{city_name}_features{geo_data_file_extension}')
        city_feature :gpd.GeoDataFrame = gpd.GeoDataFrame.from_file(city_feature_geodf_path)

        for feature_name in feature_list:
            feature_count = []
            for tile_name in city_feature_statistics['tile_name']:
                occurences = (city_feature['tile_name'] == tile_name) & (city_feature['feature'] == feature_name)
                print(occurences)
                feature_count.append(occurences.sum())
            city_feature_statistics[feature_name] = pd.Series(feature_count)

    if 'geo_data_file_extension_driver' in operation_dict:
        city_feature_statistics.to_file(city_feature_statistics_path, driver=operation_dict['geo_data_file_extension_driver'])
    else:
        city_feature_statistics.to_file(city_feature_statistics_path)
