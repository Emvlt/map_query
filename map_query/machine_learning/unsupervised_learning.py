from typing import Dict
from pathlib import Path

from shapely import Point
import geopandas as gpd

def unsupervised_feature_learning(paths:Dict[str, Path], city_name:str, operation_dict:Dict):
    ### Unpack path
    path_to_feature_folder = paths['processed'].joinpath(city_name)
    ### Unpack essentials
    essentials = operation_dict['essentials']
    feature_names = essentials['feature_names']
    feature_identifiers = essentials['features_identifiers']
    features_file_name = essentials['features_file_name']
    geo_data_file_extension = essentials['geo_data_file_extension']
    geo_data_file_extension_driver = essentials['geo_data_file_extension_driver']
    proximity_buffer = essentials['proximity_buffer']
    processed_feature_file = path_to_feature_folder.joinpath(f'{city_name}_processed_features{geo_data_file_extension}')
    ### Load feature file
    if processed_feature_file.is_file():
        feature_dataframe:gpd.GeoDataFrame = gpd.GeoDataFrame.from_file(processed_feature_file)
    else:
        path_to_feature_file = path_to_feature_folder.joinpath(city_name + features_file_name + geo_data_file_extension)
        assert path_to_feature_file.is_file(), f'There is no file at {path_to_feature_file}'
        feature_dataframe:gpd.GeoDataFrame = gpd.GeoDataFrame.from_file(path_to_feature_file)

    ### Compute city centroid
    if 'centroid_d' in feature_dataframe:
        print(f'Computing {city_name} centroid, this might take time...')
        centroid = feature_dataframe.dissolve().centroid[0]
        ### Compute distance from city centroid
        feature_dataframe['centroid_d'] = feature_dataframe.distance(centroid)
        feature_dataframe.to_file(processed_feature_file, driver=geo_data_file_extension_driver)

    ### Compute nearby features
    for feature_name, feature_id in zip(feature_names, feature_identifiers):
        print(f'Computing number of {feature_name} in a square of {proximity_buffer} pixels')

    return 0