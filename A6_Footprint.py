import geopandas as gpd
import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "Final merge of footprints, roofs and centroids.")
parser.add_argument('-f', '--footprints', type = str, metavar = '', required = True, help = 'Input footprints.')
parser.add_argument('-c', '--centroids', type = str, metavar = '', required = True, help = 'Input centroids.')
parser.add_argument('-r', '--roofs', type = str, metavar = '', required = True, help = 'Input roofs.')
parser.add_argument('-o', '--output_data', type = str, metavar = '', required = True, help = 'Output shapefile.')
args = parser.parse_args()

print ('Message: Parameters valid, starting the script.')

in_file = args.input_data
out_file = args.output_data

in_file = args.footprints
in_centroids = args.centroids
in_roofs = args.roofs
out_file = args.output_data


gdf = gpd.read_file(in_file)
gdf = gdf.drop(columns=['max_poly_h'])
gdf = gdf.drop(columns=['min_poly_h'])
gdf = gdf.drop(columns=['angle'])
gdf = gdf.drop(columns=['type_1'])

gdf['geometry'] = gdf.buffer(0)
gdf = gdf.dissolve('OBJECTID')

gdf_strechy = gpd.read_file(in_roofs)
gdf_strechy = gdf_strechy.drop(columns=['geometry'])
gdf_strechy = gdf_strechy.drop(columns=['building_h'])

gdf_merged = gdf.merge(gdf_strechy, on='OBJECTID')
print(gdf_merged)

gdf_centroidy = gpd.read_file(in_centroids)
print(gdf_centroidy)
gdf_centroidy = gdf_centroidy.drop(columns=['x'])
gdf_centroidy = gdf_centroidy.drop(columns=['y'])
gdf_centroidy = gdf_centroidy.drop(columns=['geometry'])

gdf_merged2 = gdf_merged.merge(gdf_centroidy, on='OBJECTID')
print(gdf_merged2)

gdf_merged2.to_file(out_file)   

print ('Message: All features calculated.')