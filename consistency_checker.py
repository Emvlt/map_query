from typing import List, Dict

def check_configuration_file_consistency(config_dict:Dict):
    print('Checking the provided project configuration file...')

    assert 'paths' in config_dict.keys(), 'Specify the projects paths'
    assert 'raw' in config_dict['paths'].keys(), 'Specify the raw data path'
    assert 'processed' in config_dict['paths'].keys(), 'Specify the raw data path'

    print(u'The provided project configuration file is consistent \u2713')
