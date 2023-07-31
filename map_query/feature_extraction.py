from typing import Dict, Tuple, List
from pathlib import Path
import uuid

import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box

from map_query.image_transforms import load_and_preprocess


def create_geodataframe(tile_name:str, feature_name:str,lattitude:float, longitude:float, bounding_box:Polygon, crs) -> gpd.GeoDataFrame:
    entry_dict = {
        'id':[uuid.uuid1()],
        'tile_name':[tile_name],
        'feature_name':[feature_name],
        'lattitude' : [lattitude],
        'longitude': [longitude],
        'geometry': [bounding_box]
            }
    return  gpd.GeoDataFrame(entry_dict, crs=crs) #type:ignore

def process_contour_template_matching(
    moments,
    west_boundary, north_boundary,
    width_scaling, height_scaling,
    template_height, template_width, pixel_tolerance):
    ## Define center of detected shape
    center_width  = int(moments["m10"] / moments["m00"])
    center_height = int(moments["m01"] / moments["m00"])
    ## Check if a shape already exists in the dataset, that has the same tag and
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
    tile_width = tile_information['tile_width']
    tile_height = tile_information['tile_height']
    west_boundary  = tile_information['west_boundary']
    north_boundary = tile_information['north_boundary']
    lattitude_length = tile_information['lattitude_length']
    longitude_length = tile_information['longitude_length']
    width_scaling  = lattitude_length / tile_width
    height_scaling = longitude_length / tile_height

    contours, hierarchy = cv2.findContours(segmented_tile, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print( '\t' + f'{len(contours)} Occurences found')
    for contour in contours:
        moments = cv2.moments(contour)
        if process_name == 'template_matching':
            if moments["m00"] != 0:
                lattitude_start, longitude_start, bounding_box, browse_box = process_contour_template_matching(
                    moments, west_boundary, north_boundary, width_scaling, height_scaling,
                    template_height = kwargs['template_height'], template_width = kwargs['template_width'], pixel_tolerance = kwargs['pixel_tolerance'])
                inter =tile_feature_dataframe.loc[tile_feature_dataframe['feature_name'] == feature_name ].sindex.intersection(browse_box.bounds) #type:ignore
                if len(inter) == 0:
                    tile_feature_dataframe = pd.concat([tile_feature_dataframe, create_geodataframe(tile_name, feature_name, lattitude_start, longitude_start, bounding_box, crs)]) #type:ignore
                else:
                    print(f'A point already exists in the dataframe position ({lattitude_start}, {longitude_start})')
        else:
            raise NotImplementedError

    return tile_feature_dataframe

def template_matching(tile:np.ndarray, tile_information:pd.Series, process_dict:Dict, tile_feature_dataframe:gpd.GeoDataFrame,
                      crs:str) -> gpd.GeoDataFrame:
    tile_name = tile_information['tile_name']
    print(f'Performing Template Matching on tile {tile_name}')
    ### Unpack process_dict arguments
    pixel_tolerance = process_dict['pixel_tolerance']
    detection_threshold = process_dict["detection_threshold"]
    templates_path = Path(process_dict['templates_path'])
    template_names = process_dict['template_names']
    template_file_extension = process_dict['template_file_extension']

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
            result:np.ndarray = cv2.matchTemplate(tile, template, cv2.TM_CCOEFF_NORMED)
            # Threshold the probability map
            segmented_tile = np.uint8(cv2.threshold(result, thresh=detection_threshold, maxval=1, type=cv2.THRESH_BINARY)[1]*255)
            # Compute the feature_dataframe for the given template
            tile_feature_dataframe = extract_contours_from_segmented_tile(
                segmented_tile, process_dict, tile_information, template_name, tile_feature_dataframe,
                crs, pixel_tolerance = pixel_tolerance, template_width = template_width, template_height=template_height,
                )

    return tile_feature_dataframe


def extract_features(
    paths:Dict[str,Path],
    city_name:str,
    operation_dict:Dict) -> Tuple[Path, gpd.GeoDataFrame]:
    ### Unpack arguments
    # Unpack paths arguments
    processed_city_path = paths['processed'].joinpath(city_name)
    processed_city_path.mkdir(exist_ok=True, parents=True)

    # Unpack operation essential arguments
    crs = operation_dict['essentials']['crs']
    geo_data_file_extension = operation_dict['essentials']['geo_data_file_extension']
    geo_data_file_extension_driver = operation_dict['essentials']['geo_data_file_extension_driver']

    ### Load city dataframe
    city_dataframe_path = processed_city_path.joinpath(f'{city_name}{geo_data_file_extension}')
    assert city_dataframe_path.is_file(), f'File not found at {city_dataframe_path}, Make sure to pre_process the city first'
    city_dataframe:gpd.GeoDataFrame = gpd.GeoDataFrame.from_file(city_dataframe_path)

    ### Load feature dataframe
    columns = ["id", "tile_name", "feature_name", "lattitude", "longitude", "geometry"]
    city_feature_dataframe_path = processed_city_path.joinpath(f'{city_name}_features{geo_data_file_extension}')
    if not city_feature_dataframe_path.is_file():
        city_feature_dataframe = gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs)#type:ignore
    else:
        city_feature_dataframe = gpd.GeoDataFrame.from_file(city_feature_dataframe_path)


    process_dict = operation_dict['process_dependent']

    if process_dict['city_overwrite']:
        for row_index, tile_information in city_dataframe.iterrows():
            tile_information:pd.Series
            tile_name = tile_information['tile_name']
            print(f'Processing tile {tile_name}')
            # 1) Make tile folder
            tile_folder = processed_city_path / str(tile_name)
            tile_folder.mkdir(exist_ok=True, parents=True)
            # 2) Load real tile image
            tile = load_and_preprocess(Path(tile_information['tile_path']))
            # 3) Create or load the feature dataframe
            tile_feature_dataframe_path = tile_folder.joinpath(f'features{geo_data_file_extension}')
            if not tile_feature_dataframe_path.is_file():
                tile_feature_dataframe = gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs)#type:ignore
            else:
                tile_feature_dataframe = gpd.GeoDataFrame.from_file(tile_feature_dataframe_path)
            # 4) Execute the feature_extraction core function
            if process_dict['tile_overwrite']:
                if process_dict['process_name'] == 'template_matching':
                    tile_feature_dataframe:gpd.GeoDataFrame = template_matching(tile, tile_information, process_dict, tile_feature_dataframe, crs)
                elif process_dict['process_name'] == 'hand_labeled_feature':
                    raise NotImplementedError

                else:
                    raise NotImplementedError
            # 5) Save the tile feature dataframe
            tile_feature_dataframe.to_file(tile_feature_dataframe_path, driver=geo_data_file_extension_driver)
            # 6) Perform a spatial join to remove duplicates
            city_feature_dataframe.sjoin(tile_feature_dataframe)

        # Once the loop over the rows of the city dataframe is over, we can save the city_features_dataframe
        city_feature_dataframe.to_file(city_feature_dataframe_path, driver=geo_data_file_extension_driver)


    return city_feature_dataframe_path, city_feature_dataframe