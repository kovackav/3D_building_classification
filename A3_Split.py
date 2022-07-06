import geopandas as gpd
import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "This script splites the decomposed building into shapefile based on polygon type.")
parser.add_argument('-i', '--input_data', type = str, metavar = '', required = True, help = 'Input shapefile.')
parser.add_argument('-f', '--output_foot', type = str, metavar = '', required = True, help = 'Output name of the footprints shapefile.')
parser.add_argument('-r', '--output_roof', type = str, metavar = '', required = True, help = 'Output name of the walls shapefile.')
parser.add_argument('-w', '--output_wall', type = str, metavar = '', required = True, help = 'Output name of the roofs shapefile.')
args = parser.parse_args()

print ('Message: Parameters valid, starting the script.')

in_file = args.input_data
out_foot = args.output_foot
out_wall = args.output_roof
out_roof = args.output_wall 

gdf = gpd.read_file(in_file)
print(gdf)
gdf_strechy = gdf[gdf['type']=='strecha'].to_file(out_roof)
gdf_foot = gdf[gdf['type']=='footprint'].to_file(out_foot)
gdf_stena = gdf[gdf['type']=='stena'].to_file(out_wall)

print ('Message: All features calculated.')