from typing import Dict, Tuple, List
from pathlib import Path
import uuid

import matplotlib.pyplot as plt
import numpy as np
import cv2
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
import shapely.wkt as wkt
import pandas as pd

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

def tile_pattern_matching(
        tile_information:gpd.GeoSeries,
        tile_path:Path,
        template_name:str,
        columns:List,
        crs:str,
        template_feature_path:Path,
        template_file_extension:str,
        pixel_tolerance:int,
        detection_threshold:float) -> gpd.GeoDataFrame:
    print( '\t' + f'Processing pattern {template_name}')
    ### Load tile
    tile = load_and_preprocess(tile_path)
    tile_height = tile.shape[0]
    tile_width  = tile.shape[1]
    # +------------------------+
    # |(0,0)          (0,11400)|
    # |                        |
    # |                        |
    # |                        |
    # |(7590,0)    (7590,11400)|
    # +------------------------+
    ### Unpack tile_information
    tile_name = tile_information['tile_name']
    west_boundary  = tile_information['west_boundary']
    north_boundary = tile_information['north_boundary']
    lattitude_length = tile_information['lattitude_length']
    longitude_length = tile_information['longitude_length']
    width_scaling  = lattitude_length / tile_width
    height_scaling = longitude_length / tile_height
    ### create tile_feature geo_dataframe
    tile_geo_dataframe = gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs)#type:ignore
    ### Iterate through the available templates
    for template_file_path in list(template_feature_path.glob(f'*{template_file_extension}')):
        # Load template
        template  = load_and_preprocess(template_file_path)
        template_height = template.shape[0]
        template_width  = template.shape[1]
        result:np.ndarray = cv2.matchTemplate(tile, template, cv2.TM_CCOEFF_NORMED)
        thresholded = np.uint8(cv2.threshold(result, thresh=detection_threshold, maxval=1, type=cv2.THRESH_BINARY)[1]*255)
        contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print( '\t' + f'{len(contours)} Occurences found')
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                ## Define center of detected shape
                center_width  = int(M["m10"] / M["m00"])
                center_height = int(M["m01"] / M["m00"])
                ## Check if a shape already exists in the dataset, that has the same tag and
                feature_center_tile_reference = center_height, center_width

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

                if tile_geo_dataframe.empty:
                    print('The dataframe is currently empty, creating...')
                    tile_geo_dataframe = create_geodataframe(tile_name, template_name, lattitude_start, longitude_start, bounding_box, crs) #type:ignore
                else:
                    # Now, we check if there is a point in the vicinity
                    inter = tile_geo_dataframe.sindex.intersection(browse_box.bounds)
                    if len(inter) == 0:
                        tile_geo_dataframe:gpd.GeoDataFrame = pd.concat([tile_geo_dataframe, create_geodataframe(tile_name, template_name, lattitude_start, longitude_start, bounding_box, crs)]) #type:ignore
                    else:
                        print(f'Point ({lattitude_start, longitude_start}) already exists, passing...')

    return tile_geo_dataframe

def template_matching(
    paths:Dict[str,Path],
    city_name:str,
    pre_process_dict:Dict) -> Tuple[Path, gpd.GeoDataFrame]:

    map_image_file_extension = pre_process_dict['map_image_file_extension']
    template_file_extension = pre_process_dict['template_file_extension']

    crs = pre_process_dict['crs']
    geo_data_file_extension = pre_process_dict['geo_data_file_extension']
    geo_data_file_extension_driver = pre_process_dict['geo_data_file_extension_driver']

    templates:List = pre_process_dict['templates']

    pixel_tolerance = pre_process_dict['pixel_tolerance']
    detection_threshold = pre_process_dict['detection_threshold']

    template_folder_path:Path = Path(pre_process_dict['templates_path'])

    raw_city_path = paths['raw'].joinpath(city_name)
    processed_city_path = paths['processed'].joinpath(city_name)
    processed_city_path.mkdir(exist_ok=True, parents=True)
    city_dataframe_path = processed_city_path.joinpath(f'{city_name}{geo_data_file_extension}')

    assert city_dataframe_path.is_file()
    city_dataframe:gpd.GeoDataFrame = gpd.GeoDataFrame.from_file(city_dataframe_path)

    columns = ["id", "tile_name", "feature_name", "lattitude", "longitude", "geometry"]


    city_feature_dataframe_path = processed_city_path.joinpath(f'{city_name}_features{geo_data_file_extension}')

    if not city_feature_dataframe_path.is_file():
        city_feature_dataframe = gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs)#type:ignore

        for row_index, tile_information in  city_dataframe.iterrows():
            tile_name = tile_information['tile_name']
            print(f'Processing tile {tile_name}')
            tile_feature_dataframe_folder = processed_city_path / str(tile_name)
            tile_feature_dataframe_folder.mkdir(exist_ok=True, parents=True)
            tile_path = raw_city_path.joinpath(f'{tile_name}{map_image_file_extension}')
            for template_name in templates:
                tile_feature_dataframe_save_path = tile_feature_dataframe_folder.joinpath(f'{template_name}{geo_data_file_extension}')
                template_feature_path = template_folder_path.joinpath(f'{template_name}')
                if not tile_feature_dataframe_save_path.is_file():
                    tile_feature_dataframe:gpd.GeoDataFrame = tile_pattern_matching(
                        tile_information, tile_path, template_name, columns, crs, template_feature_path, template_file_extension, pixel_tolerance,detection_threshold #type:ignore
                        )
                    tile_feature_dataframe.to_file(tile_feature_dataframe_save_path, driver=geo_data_file_extension_driver)

                else:
                    print(f'{tile_feature_dataframe_save_path} already exists, loading from file')
                    tile_feature_dataframe:gpd.GeoDataFrame = gpd.GeoDataFrame.from_file(tile_feature_dataframe_save_path)

                city_feature_dataframe = city_feature_dataframe.append(tile_feature_dataframe) #type:ignore

        city_feature_dataframe.to_file(city_feature_dataframe_path, driver=geo_data_file_extension_driver)

    city_feature_dataframe = gpd.GeoDataFrame.from_file(city_feature_dataframe_path)

    return city_feature_dataframe_path, city_feature_dataframe
