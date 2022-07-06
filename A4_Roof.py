import geopandas as gpd
import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "This script calculates some metrics of the roof polygons.")
parser.add_argument('-i', '--input_data', type = str, metavar = '', required = True, help = 'Input shapaefile of the roofs.')
parser.add_argument('-o', '--output_data', type = str, metavar = '', required = True, help = 'Output shapefile of the roofs')
args = parser.parse_args()

print ('Message: Parameters valid, starting the script.')

in_file = args.input_data
out_file = args.output_data

# reading the input
gdf = gpd.read_file(in_file)
print(gdf)
# to calculate the roof area, it must be converted to some equivalent crs
gdf= gdf.to_crs({'init': 'epsg:32633'})
gdf["area_roof"] = gdf['geometry'].area
gdf= gdf.to_crs({'init': 'epsg:4326'})
crs = gdf.crs
# calculate the height of the roof
gdf["roof_height"] = gdf['max_poly_h']-gdf['min_poly_h']
print(gdf)

gdf = gdf.drop(columns=['max_poly_h'])

gdf['geometry'] = gdf.buffer(0.01)

gdf_right_geometry = gdf.dissolve('OBJECTID')
gdf_right_geometry = gdf_right_geometry.drop(columns=['angle'])
gdf_right_geometry = gdf_right_geometry.drop(columns=['min_poly_h'])
gdf_right_geometry = gdf_right_geometry.drop(columns=['type'])
gdf_right_geometry = gdf_right_geometry.drop(columns=['building_h'])
gdf_right_geometry = gdf_right_geometry.drop(columns=['area_roof'])
gdf_right_geometry = gdf_right_geometry.drop(columns=['roof_height'])
gdf_right_geometry = gdf_right_geometry.drop(columns=['FUNCTION'])



gdf_right_att = gpd.GeoDataFrame(gdf.groupby('OBJECTID').agg({'geometry': 'first',
                                                'angle':'mean', 
                                                'FUNCTION': "first",
                                               'type': "first", 
                                               'building_h': "mean",
                                                'roof_height': 'mean',
                                               'area_roof':'sum' }).reset_index(),
                                               geometry="geometry",
                                               crs = gdf.crs
                              )
gdf_right_att = gdf_right_att.drop(columns=['geometry'])





gdf_merged = gdf_right_att.merge(gdf_right_geometry, on='OBJECTID')
gdf_merged = gdf_merged.drop(columns=['type'])

gdf_merged.to_file(out_file)

print ('Message: All features calculated.')