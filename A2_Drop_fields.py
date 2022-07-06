import geopandas as gpd
import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "This script just drops columns which you do not need.")
parser.add_argument('-i', '--input_data', type = str, metavar = '', required = True, help = 'Input shapefile.')
parser.add_argument('-c', '--list_columns', nargs="+", required = True, help = 'List of column which shoul be dropped.')
parser.add_argument('-o', '--output_data', type = str, metavar = '', required = True, help = 'Output name of the decomposed PolygonZ.')
args = parser.parse_args()

print ('Message: Parameters valid, starting the script.')

in_file = args.input_data
out_file = args.output_data
list_columns = args.list_columns

gdf = gpd.read_file(in_file)
print(gdf)
for column in list_columns:
    gdf = gdf.drop(columns=[column])

gdf.to_file(out_file)

print(gdf)

print ('Message: All features calculated.')
