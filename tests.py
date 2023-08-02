#### Test to read shapefile crs (projection)
from pathlib import Path

import geopandas as gpd

path_to_shape_file = Path(r'C:\Users\hx21262\map_query\datasets\processed\MAPHIS\Sheerness\Sheerness.shp')

df:gpd.GeoDataFrame = gpd.GeoDataFrame.from_file(path_to_shape_file)

print(df.crs)

path_to_prj = Path('datasets/raw/MAPHIS/Luton/0105033010241.prj')

prj_string = open(path_to_prj, 'r').read()

print(prj_string)

luton_df = luton_df.to_crs(prj_string)

print(luton_df.crs)