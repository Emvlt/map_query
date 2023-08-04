from typing import List, Dict

def check_paths(config_dict:Dict):
    assert 'paths' in config_dict.keys(), 'Specify the projects paths'
    assert 'raw' in config_dict['paths'].keys(), 'Specify the raw data path'
    assert 'processed' in config_dict['paths'].keys(), 'Specify the processed data path'
    assert 'plots' in config_dict['paths'].keys(), 'Specify the plots data path'

def check_project_data(config_dict:Dict):
    assert 'project_data' in config_dict.keys(), 'Specify the projects data information'
    assert 'project_name' in config_dict['project_data'].keys(), 'Specify the raw data path'
    assert 'city_names' in config_dict['project_data'].keys(), 'Specify the processed data path'

def check_pre_processing(operation_dict:Dict):
    assert 'map_image_file_extension' in operation_dict.keys(), 'Specify map image file extension'
    assert 'geo_data_file_extension' in operation_dict.keys(), 'Specify geodata file extension'
    assert 'geo_data_file_extension_driver' in operation_dict.keys(), 'Specify geodata file extension driver'

def check_display_city(operation_dict:Dict):
    assert 'figure_name' in operation_dict.keys(), 'Specify the figure name'
    assert 'geo_data_file_extension' in operation_dict.keys(), 'Specify geodata file extension'

def check_compute_feature_density(operation_dict:Dict):
    assert 'feature_names' in operation_dict.keys(), 'Specify the list of feature names to process'
    assert 'geo_data_file_extension' in operation_dict.keys(), 'Specify geodata file extension'
    assert 'geo_data_file_extension_driver' in operation_dict.keys(), 'Specify geodata file extension driver'

def check_display_feature_density(operation_dict:Dict):
    assert 'figure_name' in operation_dict.keys(), 'Specify the figure name'
    assert 'feature_names' in operation_dict.keys(), 'Specify the list of feature names to process'

def check_display_features(operation_dict:Dict):
    assert 'figure_name' in operation_dict.keys(), 'Specify the figure name'
    assert 'feature_names' in operation_dict.keys(), 'Specify the list of feature names to process'

def check_operations(config_dict:Dict):
    assert 'operations' in config_dict.keys(), 'Specify the operations information'
    for operation_name, operation_dict in config_dict['operations'].items():
        if operation_name == 'pre_processing':
            check_pre_processing(operation_dict)
        elif operation_name == 'display_city':
            check_display_city(operation_dict)
        elif operation_name == 'extract_features':
            check_extract_features(operation_dict)
        elif operation_name == 'compute_feature_density':
            check_compute_feature_density(operation_dict)
        elif operation_name == 'display_features_density':
            check_display_features_density(operation_dict)
        elif operation_name == 'display_features':
            check_display_features(operation_dict)
        else:
            raise NotImplementedError



def check_configuration_file_consistency(config_dict:Dict):
    print('Checking the provided project configuration file...')
    ### Check paths
    check_paths(config_dict)
    ### Check the project data
    check_project_data(config_dict)
    ### Check operations


    print(u'The provided project configuration file is consistent \u2713')
