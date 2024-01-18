from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path_to_clean_dataframe = Path(f'datasets/processed/MAPHIS/Luton/Luton_hand_labelled_feature_extraction_features.shp')
    path_to_segmented_dataframe = Path(f'datasets/processed/MAPHIS/Luton/Luton_ml_segmentation_features.shp')

    clean_df = gpd.GeoDataFrame.from_file(path_to_clean_dataframe)
    segmented_df = gpd.GeoDataFrame.from_file(path_to_segmented_dataframe)

    clean_df:gpd.GeoDataFrame = clean_df[clean_df['feature'] == 'buildings'] # type:ignore
    segmented_df:gpd.GeoDataFrame = segmented_df[segmented_df['feature'] == 'buildings'] # type:ignore

    invalid_seg = segmented_df[~segmented_df.is_valid]

    invalid_clean = clean_df[~clean_df.is_valid]

    valid_seg = segmented_df[segmented_df.is_valid]

    valid_clean = clean_df[clean_df.is_valid]

    assert valid_clean.crs == valid_seg.crs

    joined_tile = gpd.sjoin(valid_seg, valid_clean, how='inner', predicate='intersects', lsuffix='left', rsuffix='right')

    for i in joined_tile.index:
        joined_shapes = joined_tile.loc[[i]]

        segmented = joined_shapes['geometry']
        clean = valid_clean[valid_clean['id'] == joined_shapes['id_right'].values[0]]

        segmented_poly = segmented.geometry.values[0]
        clean_poly = clean.geometry.values[0]

        xx, yy = segmented_poly.exterior.coords.xy
        print(xx.tolist())
        print(yy.tolist())

        f, ax = plt.subplots()
        segmented.plot(ax=ax, color='b')
        clean.boundary.plot(ax=ax, color='r')

        plt.show()



    '''print(len(invalid_seg))
    print(len(invalid_clean))

    for i in invalid_clean.index:
        shape = invalid_clean.loc[[i]]
        boundary = shape.boundary
        boundary.plot()
        plt.show()'''

    '''# <!> One has to iterate on df.index and then use the .loc[[]] method
    # <!> Otherwise, methods such as iterrows or iloc return pd.Series
    # <!> When gpd objects are sought after. It's nasty but still being fixed
    # <!> by the gpd team https://github.com/geopandas/geopandas/issues/136
    for i in segmented_df.index:
        shape = segmented_df.loc[[i]]
        print(shape.is_valid)
        boundary = shape.boundary
        boundary.plot()
        plt.show()
'''