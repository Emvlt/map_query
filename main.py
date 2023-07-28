from pathlib import Path
import argparse
from typing import Dict
import toml

from consistency_checker import check_configuration_file_consistency

from map_query.pre_process import pre_process_city
from map_query.template_matching import template_matching
from map_query.city_statistics import compute_feature_density
from map_query.display import display_city, display_features, display_feature_density

F_NAME_to_FUNCTION={
    'pre_processing':pre_process_city,
    'template_matching':template_matching,
    'compute_feature_density':compute_feature_density,
    'display_feature_density':display_feature_density,
    'display_features':display_features,
    'display_city':display_city
}

if __name__ == '__main__':
    config_file = toml.load('project_configuration.toml')

    check_configuration_file_consistency(config_file)
    ### Project name
    project_name = config_file['project_data']['project_name']

    ### Paths of the project, indicate where are the files located
    # datasets
    #   --- labels
    #        --- templates
    #   --- processed
    #   --- raw
    # plots
    paths = {
        'raw':Path(config_file['paths']['raw']).joinpath(project_name),
        'processed':Path(config_file['paths']['processed']).joinpath(project_name),
        'plots':Path(config_file['paths']['plots']).joinpath(project_name)
    }

    ### Cities of interest
    city_names = config_file['project_data']['city_names']

    ### Operations to perform
    operations:Dict = config_file['operations']

    print('Checking project folders')
    for path_name, path in paths.items():
        if not path.is_dir():
            print(f'Making {path_name} dataset folder')
            path.mkdir(exist_ok=True, parents=True)

    for operation_name, operation_dict in operations.items():
        print(f'Executing {operation_name}...')
        for city_name in city_names:
            F_NAME_to_FUNCTION[operation_name](paths, city_name, operation_dict)

