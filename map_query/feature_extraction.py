from typing import Dict, Tuple, List
from pathlib import Path
import uuid
import re

from torch.utils.data import DataLoader
import cv2
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box
import matplotlib.pyplot as plt
from tqdm import tqdm

from map_query.machine_learning.datasets import Thumbnails
from map_query.machine_learning.models import load_model
from map_query.machine_learning.transforms import parse_transformations_list
from map_query.utils.image_transforms import load_and_preprocess, load_tile_as_array


HIGH_TO_LOW = {
    'buildings':['buildings'],
    'text':['labels', 'stamps_large_font', 'stamps_small_font'],
    'vegetation':['trees']
}

def coordinate_affine_transform(pixel_position, pixel_offset, pixel_scaling):
    return pixel_offset + pixel_position*pixel_scaling

def contour_to_projected(contour, west_boundary, north_boundary, width_scaling, height_scaling):
    return (
        (
            coordinate_affine_transform(contour[0][0], west_boundary, width_scaling),
            coordinate_affine_transform(contour[0][1], north_boundary, height_scaling)
        )
        )

def contour_to_polygon_geojson(contour, west_boundary, north_boundary, width_scaling, height_scaling):
    lines = []
    lines.append(contour_to_projected(contour[0], west_boundary, north_boundary, width_scaling, height_scaling))
    for c in reversed(contour):
        lines.append(contour_to_projected(c, west_boundary, north_boundary, width_scaling, height_scaling))
    return lines

def create_geodataframe(tile_name:str, feature_name:str, contour:Polygon, lattitude:float, longitude:float, bounding_box:Polygon, crs) -> gpd.GeoDataFrame:
    entry_dict = {
        'id':[uuid.uuid1()],
        'tile_name':[tile_name],
        'feature':[feature_name],
        'contour':[contour],
        'lattitude' : [lattitude],
        'longitude': [longitude],
        'geometry': [bounding_box]
            }
    return  gpd.GeoDataFrame(entry_dict, crs=crs) #type:ignore

def process_contour_any_shape(
    contour,
    west_boundary, north_boundary,
    width_scaling, height_scaling):
    moments = cv2.moments(contour)
    ## Define center of detected shape
    center_width  = int(moments["m10"] / moments["m00"])
    center_height = int(moments["m01"] / moments["m00"])

    lattitude_start = coordinate_affine_transform(center_width, west_boundary, width_scaling)
    longitude_start = coordinate_affine_transform(center_height, north_boundary, height_scaling)

    polygon_shape = Polygon(contour_to_polygon_geojson(
        contour,
        west_boundary, north_boundary,
        width_scaling, height_scaling))

    return lattitude_start, longitude_start, polygon_shape

def process_contour_template_matching(
    contour,
    west_boundary, north_boundary,
    width_scaling, height_scaling,
    template_height, template_width, pixel_tolerance):
    moments = cv2.moments(contour)
    ## Define center of detected shape
    center_width  = int(moments["m10"] / moments["m00"])
    center_height = int(moments["m01"] / moments["m00"])

    lattitude_start = west_boundary+center_width*width_scaling
    lattitude_stop  = west_boundary+(center_width+template_width)*width_scaling
    longitude_start = north_boundary+center_height*height_scaling
    longitude_stop  = north_boundary+(center_height+template_height)*height_scaling

    lattitude_start_tol = west_boundary+(center_width-pixel_tolerance)*width_scaling
    lattitude_stop_tol  = west_boundary+(center_width+template_width+pixel_tolerance)*width_scaling
    longitude_start_tol = north_boundary+(center_height-pixel_tolerance)*height_scaling
    longitude_stop_tol  = north_boundary+(center_height+template_height+pixel_tolerance)*height_scaling

    bounding_box = Polygon((
            (lattitude_start, longitude_start),
            (lattitude_stop,  longitude_start),
            (lattitude_stop,  longitude_stop),
            (lattitude_start, longitude_stop),
            ))

    browse_box = box(
        lattitude_start_tol,
        longitude_start_tol,
        lattitude_stop_tol,
        longitude_stop_tol)
    return lattitude_start, longitude_start, bounding_box, browse_box

def extract_contours_from_segmented_tile(segmented_tile:np.uint8, process_dict:Dict, tile_information:pd.Series, feature_name:str, tile_feature_dataframe:gpd.GeoDataFrame, crs:str, **kwargs):
    ### Unpack process_dict arguments
    process_name = process_dict["process_name"]

    ### Unpack tile_information
    tile_name = tile_information['tile_name']
    tile_width = tile_information['width']
    tile_height = tile_information['height']
    west_boundary  = tile_information['west']
    north_boundary = tile_information['north']
    lattitude_length = tile_information['lat_length']
    longitude_length = tile_information['lon_length']
    width_scaling  = lattitude_length / tile_width
    height_scaling = longitude_length / tile_height

    contours, hierarchy = cv2.findContours(segmented_tile, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0 :
        print( '\t' + f'{len(contours)} {feature_name} contours found, passing...')
    else:
        print( '\t' + f'{len(contours)} {feature_name} contours found')
        for contour in tqdm(contours):
            if process_name == 'template_matching':
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    lattitude_start, longitude_start, bounding_box, browse_box = process_contour_template_matching(
                        contour, west_boundary, north_boundary, width_scaling, height_scaling,
                        template_height = kwargs['template_height'], template_width = kwargs['template_width'], pixel_tolerance = kwargs['pixel_tolerance']
                        )
                    inter =tile_feature_dataframe.loc[tile_feature_dataframe['feature'] == feature_name ].sindex.intersection(browse_box.bounds) #type:ignore

                    if len(inter) == 0:
                        contour_polygon = Polygon(contour_to_polygon_geojson(contour,0,0,1,1))
                        tile_feature_dataframe = pd.concat([tile_feature_dataframe, create_geodataframe(tile_name, feature_name, contour_polygon, lattitude_start, longitude_start, bounding_box, crs)]) #type:ignore
                    else:
                        print(f'A point already exists in the dataframe position ({lattitude_start}, {longitude_start})')

            elif process_name == 'hand_labelled_feature_extraction' or process_name == 'ml_segmentation':
                moments = cv2.moments(contour)
                if (moments["m00"]) != 0 and  ( kwargs['area_detection_threshold'] < cv2.contourArea(contour)):
                    contour = cv2.approxPolyDP(contour, kwargs['epsilon'], True)

                    lattitude_start, longitude_start, polygon_shape = process_contour_any_shape(
                        contour, west_boundary, north_boundary, width_scaling, height_scaling
                        )
                    contour_polygon = Polygon(contour_to_polygon_geojson(contour,0,0,1,1))
                    tile_feature_dataframe = pd.concat([tile_feature_dataframe, create_geodataframe(tile_name, feature_name, contour_polygon, lattitude_start, longitude_start, polygon_shape, crs)]) #type:ignore

            else:
                raise NotImplementedError
    return tile_feature_dataframe

def ml_segmentation(tile_information:pd.Series, process_dict:Dict, tile_feature_dataframe:gpd.GeoDataFrame, crs:str) -> gpd.GeoDataFrame:
    tile_name = tile_information['tile_name']
    print(f'Performing Machine Learning segmentation on tile {tile_name}')
    ### Unpacking process_dict
    feature_names = process_dict['feature_names']
    data_feeding_dict = process_dict['data_feeding_dict']
    model_dict = process_dict['model_dict']

    #Define the tile_dataloader on which we will iterate to get the 512*512 thumbnails
    print('\t Loading dataset...')
    tile_dataset = Thumbnails(
        tile_information,
        tile_transform=data_feeding_dict['tile_transform'],
        thumbnail_transforms=parse_transformations_list(data_feeding_dict['thumbnail_transforms'])
        )
    tile_dataloader = DataLoader(
        dataset = tile_dataset,
        batch_size = data_feeding_dict['batch_size'],
        shuffle = data_feeding_dict['shuffle'],
        num_workers = data_feeding_dict['num_workers'],
        drop_last=False)

    device = torch.device(model_dict['device'])

    segmentation_model = load_model(model_dict, device)
    segmentation_model.to(device)
    segmentation_model.eval()

    segmented_tile = torch.zeros((1+len(feature_names), tile_dataset.padded_tile_shape[0], tile_dataset.padded_tile_shape[1] ))
    tile_height = tile_dataset.tile_shape[0]
    tile_width  = tile_dataset.tile_shape[1]
    height_padding = tile_dataset.tile_dict['height_padding']
    width_padding  = tile_dataset.tile_dict['width_padding']

    print('\t Processing the tile...')
    with torch.no_grad():
        for batched_thumbnails, batch_indices in tqdm(tile_dataloader):
            batched_thumbnails = batched_thumbnails.to(device)
            segmented_thumbnails = segmentation_model(batched_thumbnails)
            for index, thumbnail_index in enumerate(batch_indices):
                coords = tile_dataset.thumbnail_coordinates[int(thumbnail_index)]
                segmented_tile[:,coords['h_low']:coords['h_high'],coords['w_low']:coords['w_high']] += segmented_thumbnails[index].detach().cpu()

    unpadded_tile = segmented_tile[
        :,
        height_padding : height_padding + tile_height,
        width_padding  : width_padding  + tile_width
        ].numpy()

    binary_tile = np.nan_to_num(unpadded_tile)
    binary_tile = np.where(0.95 < binary_tile, 1,0)*255
    binary_tile = binary_tile.astype(np.uint8)

    for feature_index, feature_name in enumerate(feature_names):
        tile_feature_dataframe = extract_contours_from_segmented_tile(
            binary_tile[1+feature_index],
            process_dict,
            tile_information,
            feature_name,
            tile_feature_dataframe,
            crs,
            epsilon=process_dict['epsilon'],
            area_detection_threshold=process_dict['area_detection_threshold'])

    return tile_feature_dataframe


def template_matching(tile_information:pd.Series, process_dict:Dict, tile_feature_dataframe:gpd.GeoDataFrame,
                      crs:str) -> gpd.GeoDataFrame:
    tile_name = tile_information['tile_name']
    print(f'Performing Template Matching on tile {tile_name}')
    ### Unpack process_dict arguments
    pixel_tolerance = process_dict['pixel_tolerance']
    detection_threshold = process_dict["detection_threshold"]
    templates_path = Path(process_dict['templates_path'])
    template_names = process_dict['template_names']
    template_file_extension = process_dict['template_file_extension']

    tile_path = Path(tile_information['tile_path'])
    tile = load_and_preprocess(tile_path)

    ### We start by iterating on the template_names list:
    for template_name in template_names:
        # 1) For each template name, we iterate through the list of available templates
        template_path = templates_path.joinpath(template_name)
        for template_file_path in list(template_path.glob(f'*{template_file_extension}')):
            # Load template
            template  = load_and_preprocess(template_file_path)
            # Get template info
            template_height = template.shape[0]
            template_width  = template.shape[1]
            # Perform template matching
            result:np.ndarray = cv2.matchTemplate(tile, template, cv2.TM_CCOEFF_NORMED) #type:ignore
            # Threshold the probability map
            segmented_tile = np.uint8(cv2.threshold(result, thresh=detection_threshold, maxval=1, type=cv2.THRESH_BINARY)[1]*255)
            # Compute the feature_dataframe for the given template
            tile_feature_dataframe = extract_contours_from_segmented_tile(
                segmented_tile, process_dict, tile_information, template_name, tile_feature_dataframe,
                crs, pixel_tolerance = pixel_tolerance, template_width = template_width, template_height=template_height,
                )

    return tile_feature_dataframe

def hand_labelled_feature_extraction(
    tile_information:pd.Series, process_dict:Dict,
    tile_feature_dataframe:gpd.GeoDataFrame, crs:str) -> gpd.GeoDataFrame:
    tile_name = tile_information['tile_name']
    print(f'Performing Feature Extraction on tile {tile_name}')
    tile_path = Path(tile_information['tile_path']) #type:ignore
    city_name = tile_path.parent.stem
    project_name = tile_path.parent.parent.stem
    ### Unpack process_dict arguments
    feature_names = process_dict['feature_names']
    color_thresholds = process_dict['color_thresholds']
    label_file_extension = process_dict['label_file_extension']
    path_to_hand_segmented_features = Path(process_dict['path_to_hand_segmented_features']).joinpath(f'{project_name}/{city_name}/{tile_name}')
    for high_level_feature_name, color_threshold in zip(feature_names, color_thresholds):
        for sub_feature_name in HIGH_TO_LOW[high_level_feature_name]:
            print(f'Processing sub feature {sub_feature_name}')
            path_to_feature_tile = path_to_hand_segmented_features.joinpath(f'{sub_feature_name}{label_file_extension}')
            if path_to_feature_tile.is_file():
                feature_tile = load_tile_as_array(path_to_feature_tile)
                segmented_tile = np.uint8(cv2.threshold(feature_tile, thresh=color_threshold, maxval=1, type=cv2.THRESH_BINARY_INV)[1]*255)
                element =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                segmented_tile = cv2.morphologyEx(segmented_tile, cv2.MORPH_CLOSE, element)
                tile_feature_dataframe = extract_contours_from_segmented_tile(
                        segmented_tile, process_dict, tile_information,
                        high_level_feature_name, tile_feature_dataframe, crs, epsilon=process_dict['epsilon'], area_detection_threshold=process_dict['area_detection_threshold'])
            else:
                print(f'No {sub_feature_name}{label_file_extension} file found in folder {path_to_hand_segmented_features}, passing')

    return tile_feature_dataframe

def load_city_feature_dataframe(processed_city_path:Path, city_name:str, process_name:str, geo_data_file_extension:str, columns:List, crs:str) ->Tuple[Path, gpd.GeoDataFrame]:
    ### Load feature dataframe
    city_feature_dataframe_path = processed_city_path.joinpath(f"{city_name}_{process_name}_features{geo_data_file_extension}")
    if not city_feature_dataframe_path.is_file():
        city_feature_dataframe = gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs) #type:ignore
    else:
        city_feature_dataframe = gpd.GeoDataFrame.from_file(city_feature_dataframe_path)
    return city_feature_dataframe_path, city_feature_dataframe

def extract_features(
    paths:Dict[str,Path],
    city_name:str,
    operation_dict:Dict):
    ### Unpack arguments
    # Unpack paths arguments
    processed_city_path = paths['processed'].joinpath(city_name)
    processed_city_path.mkdir(exist_ok=True, parents=True)

    # Unpack operation essential arguments
    geo_data_file_extension = operation_dict['essentials']['geo_data_file_extension']
    geo_data_file_extension_driver = operation_dict['essentials']['geo_data_file_extension_driver']

    ### Load city dataframe
    city_dataframe_path = processed_city_path.joinpath(f'{city_name}{geo_data_file_extension}')
    assert city_dataframe_path.is_file(), f'File not found at {city_dataframe_path}, Make sure to pre_process the city first'
    city_dataframe:gpd.GeoDataFrame = gpd.GeoDataFrame.from_file(city_dataframe_path)

    crs = city_dataframe.crs
    columns = ["id", "tile_name", "feature", "lattitude", "longitude", "geometry"]

    for process_dict in operation_dict['processes']:
        process_name = process_dict['process_name']
        city_feature_dataframe_path, city_feature_dataframe = load_city_feature_dataframe(
            processed_city_path,
            city_name,
            process_name,
            geo_data_file_extension,
            columns,
            crs)

        if process_dict['city_overwrite']:
            for row_index, tile_information in city_dataframe.iterrows():
                tile_information:pd.Series
                tile_name = tile_information['tile_name']
                print(f'Processing tile {tile_name}')
                # 1) Make tile folder
                tile_folder = processed_city_path / str(tile_name)
                tile_folder.mkdir(exist_ok=True, parents=True)
                # 2) Define the feature dataframe path
                tile_feature_dataframe_path = tile_folder.joinpath(f'{process_name}_features{geo_data_file_extension}')

                # 3) Execute the feature_extraction core function
                if process_dict['tile_overwrite']:
                    tile_feature_dataframe = gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs)#type:ignore
                    if process_name == 'template_matching':
                        tile_feature_dataframe = template_matching(tile_information, process_dict, tile_feature_dataframe, crs)
                    elif process_name == 'hand_labelled_feature_extraction':
                        tile_feature_dataframe = hand_labelled_feature_extraction(tile_information, process_dict, tile_feature_dataframe, crs)
                    elif process_name == 'ml_segmentation':
                        tile_feature_dataframe = ml_segmentation(tile_information, process_dict, tile_feature_dataframe, crs)

                    else:
                        raise NotImplementedError
                    # 4) Save the tile feature dataframe
                    tile_feature_dataframe.to_file(tile_feature_dataframe_path, driver=geo_data_file_extension_driver)
                    tile_feature_dataframe.plot()
                    plt.show()
                else:
                    tile_feature_dataframe = gpd.GeoDataFrame.from_file(tile_feature_dataframe_path)

                # 5) Perform a spatial join to remove duplicates
                city_feature_dataframe = pd.concat([city_feature_dataframe, tile_feature_dataframe], ignore_index=True) #type:ignore
                # city_feature_dataframe.sjoin(tile_feature_dataframe)

            # Conversion to correct datafile format
            if 'index' in city_feature_dataframe.columns:
                city_feature_dataframe.drop(columns=['index'])
            #city_feature_dataframe = gpd.GeoDataFrame(city_feature_dataframe, geometry="geometry") #type:ignore

            # Once the loop over the rows of the city dataframe is over, we can save the city_features_dataframe
            city_feature_dataframe:gpd.GeoDataFrame
            city_feature_dataframe.to_file(city_feature_dataframe_path, driver=geo_data_file_extension_driver)
