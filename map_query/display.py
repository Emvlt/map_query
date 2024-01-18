from typing import Dict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import geopandas as gpd

def display_city(paths:Dict[str, Path], city_name:str, operation_dict:Dict):
    ### Unpacking paths
    processed_city_path = paths['processed'].joinpath(city_name)
    plot_save_path = paths['plots'].joinpath(city_name)
    plot_save_path.mkdir(exist_ok=True, parents=True)

    ### Unpack operation dict
    geo_data_file_extension = operation_dict['geo_data_file_extension']

    city_dataframe_path = processed_city_path.joinpath(f'{city_name}{geo_data_file_extension}')
    city_dataframe:gpd.GeoDataFrame = gpd.GeoDataFrame.from_file(city_dataframe_path)

    fig, ax = plt.subplots(1, 1, figsize=(8,8), dpi=80)
    city_dataframe.plot(ax=ax, color='lightgray', edgecolor='blue')
    plt.savefig(f'{plot_save_path.joinpath(f"{city_name}.jpg")}')
    plt.clf()

def display_feature_density(paths:Dict[str, Path], city_name:str, operation_dict:Dict):
    ### Unpacking paths
    processed_city_path = paths['processed'].joinpath(city_name)
    plot_save_path = paths['plots'].joinpath(city_name)
    plot_save_path.mkdir(exist_ok=True, parents=True)

    ### Unpack operation dict
    feature_list = operation_dict['feature_names']
    geo_data_file_extension = operation_dict['geo_data_file_extension']

    city_feature_statistics_path = processed_city_path.joinpath(f'{city_name}_statistics{geo_data_file_extension}')
    city_feature_statistics:gpd.GeoDataFrame = gpd.GeoDataFrame.from_file(city_feature_statistics_path)

    print(city_feature_statistics)

    for feature_name in feature_list:
        city_feature_statistics.plot(
            column = feature_name,
            legend_kwds={"label": f"{feature_name} density in {city_name}", "orientation": "horizontal"}
        )
        plt.savefig(f'{plot_save_path.joinpath(f"{feature_name}_density.jpg")}')
        plt.clf()

def display_features(paths:Dict[str, Path], city_name:str, operation_dict:Dict):
    ### Unpacking paths
    processed_city_path = paths['processed'].joinpath(city_name)
    plot_save_path = paths['plots'].joinpath(city_name)
    plot_save_path.mkdir(exist_ok=True, parents=True)

    ### Unpack operation dict
    process_name = operation_dict['extraction_process']
    geo_data_file_extension = operation_dict['geo_data_file_extension']

    city_features_path = processed_city_path.joinpath(f'{city_name}_{process_name}_features{geo_data_file_extension}')
    city_features:gpd.GeoDataFrame = gpd.GeoDataFrame.from_file(city_features_path)

    city_features.plot(
            column = 'feature'
        )
    plt.show()
    plt.savefig(f'{plot_save_path.joinpath(f"features.jpg")}')

