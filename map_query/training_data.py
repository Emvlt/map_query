from typing import  Dict, List
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely import Polygon, Point, box
import cv2

from map_query.utils.image_transforms import load_tile_as_array

def polygon_to_list(polygon:Polygon):
    xx, yy = polygon.exterior.coords.xy
    return np.asarray([[np.int32(x), np.int32(y)] for x,y in zip(xx, yy)]).reshape(len(xx), 1, 2)

def make_training_data(
    paths:Dict[str, Path],
    city_name:str,
    operation_dict):
    ### Unpack arguments
    # Unpack paths arguments
    processed_city_path = paths['processed'].joinpath(city_name)
    assert processed_city_path.is_dir(), 'The city has not been pre-processed'
    ### Unpacking operation_dict
    feature_names = operation_dict['feature_names']
    training_data_path = operation_dict['training_data_path']
    training_data_file_extension = operation_dict['training_data_file_extension']
    record_file_name = operation_dict['record_file_name']
    record_file_extension = operation_dict['record_file_extension']
    geo_data_file_extension = operation_dict['geo_data_file_extension']

    ### Specifying the path to the training data folder
    training_data_path = Path(training_data_path)
    training_data_path.mkdir(exist_ok=True, parents=True)
    project_name = training_data_path.stem
    ### Specifying the path to the file tracking the processed file
    ### <!> By default, we store this temporary file in the PARENT folder of the training data folder
    ### <!> We recommend a path like 'datasets/training_data/project_name
    path_to_records = training_data_path.parent.joinpath(f'{record_file_name}{record_file_extension}')
    if path_to_records.is_file():
        print(f'Loading records file from {path_to_records}')
        if record_file_extension == '.csv':
            records = pd.read_csv(path_to_records)
        else:
            raise NotImplementedError('Record file format not accepted')
    else:
        print(f'No records file found at {path_to_records}, creating an empty record file')
        records = {
            'project_name' : [],
            'city_name' : [],
            'tile_name' : [],
            'features' : [],
            'date' : [],
            'n_samples':[],
            'first_index':[],
            'cumulated_samples':[]

        }
        records = pd.DataFrame.from_dict(records)

    print('Printing records file')
    print(records)

    ### Load city dataframe
    processed_city_path = paths['processed'].joinpath(city_name)
    processed_city_path.mkdir(exist_ok=True, parents=True)
    city_dataframe_path = processed_city_path.joinpath(f'{city_name}{geo_data_file_extension}')
    assert city_dataframe_path.is_file(), f'File not found at {city_dataframe_path}, Make sure to pre_process the city first'
    city_dataframe:gpd.GeoDataFrame = gpd.GeoDataFrame.from_file(city_dataframe_path)

    for row_index, tile_information in city_dataframe.iterrows():
        tile_information:pd.Series
        tile_name = tile_information['tile_name']
        print(f'Processing tile {tile_name}')
        ### Add checkers to not overwrite accidently features
        if (
            (records['project_name'] == project_name) &
            (records['city_name'] == city_name) &
            (records['tile_name'] == int(tile_name))
            ).any():

            print(f'A record exists:')
            print(records[(
            (records['project_name'] == project_name) &
            (records['city_name'] == city_name) &
            (records['tile_name'] == int(tile_name))
            )])
            print('Passing...')
        else:

            # Load the feature dataframe
            tile_dataframe_path = processed_city_path.joinpath(f'{tile_name}/hand_labelled_feature_extraction_features{geo_data_file_extension}')
            tile_dataframe:gpd.GeoDataFrame = gpd.GeoDataFrame.from_file(tile_dataframe_path)

            ### Check if all 3 features are available
            ## <!> This is temporary, due to poor data quality
            ## <!> This will break on the first tile without the three features
            for feature_name in feature_names:
                assert (tile_dataframe['feature'] == feature_name).any(), f'There is no reference data for the feature {feature_name} the tile {tile_name}, passing'

            ### Make the actual dataframe -> trianing data pipeline
            # Load the reference image
            tile = load_tile_as_array(tile_information['tile_path'])
            # Tiling the image to accomodate for features located at the border
            padding = 256
            window_size = 512
            half_window_size = int(window_size/2)
            tile = np.pad(tile, pad_width=((padding,padding),(padding,padding)), constant_values=255)
            mask = np.zeros((1+len(feature_names), np.shape(tile)[0], np.shape(tile)[1]), dtype=np.uint8)

            # Set the geometry to be the contour column
            col_geom = gpd.GeoSeries.from_wkt(tile_dataframe['contour'])
            tile_dataframe = tile_dataframe.set_geometry(col_geom) #type:ignore
            # Draw the shapes on the tile
            for feature_index, feature_name in enumerate(feature_names):
                feature_contours = tile_dataframe.loc[tile_dataframe['feature'] == feature_name]['geometry']
                ## This is necessary to parse the polygon object to the ArrayOfArray type expected by cv2.drawContours
                feature_contours = feature_contours.apply(polygon_to_list)
                feature_contours = feature_contours.to_numpy()
                mask[1+feature_index] = cv2.drawContours(mask[1+feature_index], feature_contours, -1, 255,thickness=cv2.FILLED, offset=(padding,padding))
                mask[0] = cv2.bitwise_or(mask[0], mask[1+feature_index])
            mask[0] = np.invert(mask[0], dtype=np.uint8)
            n_samples  = len(tile_dataframe)
            print(f'Found {n_samples} contours in the dataframe, processing...')
            ### Get index
            if records.empty:
                first_index = 0
                cumulated_samples = n_samples
            else:
                ### Get the rows of records with matching features for the other tiles
                existing_records = records[
                    (records['project_name'] == project_name) &
                    (records['city_name'] == city_name)
                    ]
                first_index = existing_records['n_samples'].sum()
                cumulated_samples = first_index + n_samples

            for row_index, row in tqdm(tile_dataframe.iterrows()):
                current_index =  first_index + int(row_index) #type:ignore
                main_contour:Polygon = row['geometry']
                centroid:Point = main_contour.centroid
                thumbnail_input  = tile[
                    padding + int(centroid.y) - half_window_size: padding + int(centroid.y) + half_window_size,
                    padding + int(centroid.x) - half_window_size: padding + int(centroid.x) + half_window_size]
                thumbnail_input = np.expand_dims(thumbnail_input, axis=0)
                thumbnail_target = mask[
                    :,
                    padding + int(centroid.y) - half_window_size: padding + int(centroid.y) + half_window_size,
                    padding + int(centroid.x) - half_window_size: padding + int(centroid.x) + half_window_size]
                ### Save input thumbnail and target thumbnail
                input_file_path = training_data_path.joinpath(f'input_{current_index}{training_data_file_extension}')
                target_file_path = training_data_path.joinpath(f'target_{current_index}{training_data_file_extension}')
                if training_data_file_extension == '.npy':
                    assert np.shape(thumbnail_input)  == (1,window_size,window_size), f'{np.shape(thumbnail_input)}, {centroid}'
                    assert np.shape(thumbnail_target) == (1+len(feature_names),window_size,window_size), f'{np.shape(thumbnail_target)}, {centroid}'

                    np.save(input_file_path, thumbnail_input)
                    np.save(target_file_path, thumbnail_target)
                else:
                    raise NotImplementedError
            tile_record = {
                'project_name' : [project_name],
                'city_name' : [city_name],
                'tile_name' : [tile_name],
                'features' : [feature_names],
                'date' : [pd.Timestamp(datetime.datetime.now())],
                'n_samples':[n_samples],
                'first_index':[first_index],
                'cumulated_samples':[cumulated_samples]
            }

            tile_record = pd.DataFrame.from_dict(tile_record)

            if records.empty:
                records = tile_record
            else:
                records = pd.concat([records,tile_record])
            if record_file_extension == '.csv':
                records.to_csv(path_to_records, index=False)
            else:
                raise NotImplementedError('Record file format not accepted')
