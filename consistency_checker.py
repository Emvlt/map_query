from typing import List, Dict

def check_paths(config_dict:Dict):
    assert 'paths' in config_dict, 'Specify the projects paths'
    assert 'raw' in config_dict['paths'], 'Specify the raw data path'
    assert 'processed' in config_dict['paths'], 'Specify the processed data path'
    assert 'plots' in config_dict['paths'], 'Specify the plots data path'

def check_project_data(config_dict:Dict):
    assert 'project_data' in config_dict, 'Specify the projects data information'
    assert 'project_name' in config_dict['project_data'], 'Specify the raw data path'
    assert 'city_names' in config_dict['project_data'], 'Specify the processed data path'

def check_pre_processing(operation_dict:Dict):
    assert 'map_image_file_extension' in operation_dict, 'Specify map image file extension'
    assert 'geo_data_file_extension' in operation_dict, 'Specify geodata file extension'
    assert 'geo_data_file_extension_driver' in operation_dict, 'Specify geodata file extension driver'

def check_display_city(operation_dict:Dict):
    assert 'figure_name' in operation_dict, 'Specify the figure name'
    assert 'geo_data_file_extension' in operation_dict, 'Specify geodata file extension'

def check_compute_feature_density(operation_dict:Dict):
    assert 'feature_names' in operation_dict, 'Specify the list of feature names to process'
    assert 'extraction_process' in operation_dict, 'Specify the extraction process used to obtain the features'
    assert 'geo_data_file_extension' in operation_dict, 'Specify geodata file extension'
    assert 'geo_data_file_extension_driver' in operation_dict, 'Specify geodata file extension driver'

def check_display_features_density(operation_dict:Dict):
    assert 'figure_name' in operation_dict, 'Specify the figure name'
    assert 'feature_names' in operation_dict, 'Specify the list of feature names to process'
    assert 'geo_data_file_extension' in operation_dict, 'Specify geodata file extension'

def check_display_features(operation_dict:Dict):
    assert 'figure_name' in operation_dict, 'Specify the figure name'
    assert 'feature_names' in operation_dict, 'Specify the list of feature names to process'
    assert 'geo_data_file_extension' in operation_dict, 'Specify geodata file extension'

def check_template_matching(process_dict:Dict):
    assert 'pixel_tolerance' in process_dict, 'Specify pixel tolerance int argument'
    assert 'detection_threshold' in process_dict, 'Specify detection_threshold float argument'
    assert 'templates_path' in process_dict, 'Specify templates path, e.g: datasets/labels/templates'
    assert 'template_names' in process_dict, 'Specify template_names arguments, e.g:["bench_mark", "man_hole", ...]'
    assert 'template_file_extension' in process_dict, 'Specify template file extension, e.g: ".jpg"'
    assert 'city_overwrite' in process_dict, 'Specify if the process overwrites the potentially existing city results (boolean argument)'
    assert 'tile_overwrite' in process_dict, 'Specify if the process overwrites the potentially existing tile results (boolean argument)'

def check_hand_labelled_feature_extraction(process_dict:Dict):
    assert 'feature_names' in process_dict, 'Specify feature names, e.g:["buildings", "text", "trees"]'
    assert 'color_thresholds' in process_dict, 'Specify color thresholds, e.g:[77, 106, 167]'
    assert 'label_file_extension' in process_dict, 'Specify label file extension, e.g: .jpg'
    assert 'path_to_hand_segmented_features' in process_dict, 'Specify path to segmented features, e.g:               hand_segmented_features'
    assert 'epsilon' in process_dict, 'Specify epsilon float value for approxPolyDP'
    assert 'area_detection_threshold' in process_dict, 'Specify aread detection threshold int value for false positive detection'

def check_ml_segmentation(process_dict:Dict):
    assert 'feature_names' in process_dict, 'Specify feature names arguments, e.g:["buildings", "text", ...]'
    assert 'city_overwrite' in process_dict, 'Specify if the process overwrites the potentially existing city results (boolean argument)'
    assert 'tile_overwrite' in process_dict, 'Specify if the process overwrites the potentially existing tile results (boolean argument)'
    ### Check how data is fed to the model
    assert 'data_feeding_dict' in process_dict, 'Specify data feeding dictionnary'
    assert 'batch_size' in process_dict['data_feeding_dict'], 'Specify batch size (int)'
    assert 'shuffle' in process_dict['data_feeding_dict'], 'Specify shuffle dataset (bool)'
    assert 'num_workers' in process_dict['data_feeding_dict'], 'Specify num_workers (int)'
    assert 'tile_transform' in process_dict['data_feeding_dict'], 'Specify tile transform (str)'
    assert 'thumbnail_transforms' in process_dict['data_feeding_dict'], 'Specify thumbnail transforms (List[str])'
    ### Check how the model is defined
    assert 'model_dict' in process_dict, 'Specify model definition dictionnary'
    assert 'name' in process_dict['model_dict'], 'Specify model name'
    assert 'device' in process_dict['model_dict'], 'Specify device name'
    if process_dict['model_dict']['name'] == 'maphis_segmentation':
        pass


    else:
        print(f'The checkers for model {process_dict["model_dict"]["name"]} are not implemented, passing...')


def check_extract_features(operation_dict:Dict):
    assert 'essentials' in operation_dict, 'Specify essential vriables of extract features procedure'
    assert 'geo_data_file_extension' in operation_dict['essentials'], 'Specify geodata file extenstion'
    assert 'geo_data_file_extension_driver' in operation_dict['essentials'], 'Specify geodata file extension driver'

    assert 'processes'in operation_dict, 'Specify the processes part of feature extraction operation'
    for process_dict in operation_dict['processes']:
        assert 'process_name' in process_dict, 'Specify process name'
        if process_dict['process_name'] == 'template_matching':
            check_template_matching(process_dict)
        elif process_dict['process_name'] == 'ml_segmentation':
            check_ml_segmentation(process_dict)
        elif process_dict['process_name'] == 'hand_labelled_feature_extraction':
            check_hand_labelled_feature_extraction(process_dict)
        else:
            raise NotImplementedError

def check_make_training_data(operation_dict:Dict):
    assert 'training_data_path' in operation_dict, 'Specify the path of the folder to save training data to'
    assert 'feature_names' in operation_dict, 'Specify the features names (List) for making a batch of training data '
    assert 'training_data_file_extension' in operation_dict, 'Specify the training data file extension (str)'
    assert 'record_file_name' in operation_dict, 'Specify the record file name (str)'
    assert 'record_file_extension' in operation_dict, 'Specify the record file extension (str)'
    assert 'geo_data_file_extension' in operation_dict, 'Specify geodata file extenstion (str)'

def check_operations(config_dict:Dict):
    assert 'operations' in config_dict, 'Specify the operations information'
    for operation_name, operation_dict in config_dict['operations'].items():
        if operation_name == 'pre_processing':
            check_pre_processing(operation_dict)
        elif operation_name == 'display_city':
            check_display_city(operation_dict)
        elif operation_name == 'pre_processing':
            check_pre_processing(operation_dict)
        elif operation_name == 'extract_features':
            check_extract_features(operation_dict)
        elif operation_name == 'compute_features_density':
            check_compute_feature_density(operation_dict)
        elif operation_name == 'display_features_density':
            check_display_features_density(operation_dict)
        elif operation_name == 'display_features':
            check_display_features(operation_dict)
        elif operation_name == 'make_training_data':
            check_make_training_data(operation_dict)
        elif operation_name == 'train_segmentation':
            pass
        else:
            raise NotImplementedError(f'{operation_name} not implemented')



def check_configuration_file_consistency(config_dict:Dict):
    print('Checking the provided project configuration file...')
    ### Check paths
    check_paths(config_dict)
    ### Check the project data
    check_project_data(config_dict)
    ### Check operations
    check_operations(config_dict)


    print(u'The provided project configuration file is consistent \u2713')
