import json
import geopandas as gpd
import numpy as np
import pandas as pd
import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "This script calculates the coordinates of centroidZ.")
parser.add_argument('-i', '--input_data', type = str, metavar = '', required = True, help = 'Input multipatch shapefile.')
parser.add_argument('-o', '--output_data', type = str, metavar = '', required = True, help = 'Output shapefile, where the centroids should be stored.')
args = parser.parse_args()

print ('Message: Parameters valid, starting the script.')

in_file = args.input_data
out_file = args.output_data

# rading the input
gdf = gpd.read_file(in_file)
# converting to json
gdf_json = json.loads(gdf.to_json())

# crs definition
crs = gdf.crs
feature_list = []

i = 0
res = False

count = 1
splitted_feature_list = []
meanXlist = []
meanYlist = []
meanZlist = []
ID_BUDlist = []
Zminlist = []

# feature => multipatch of one single building
for feature in gdf_json['features']:
    Xlist = []
    Ylist = []
    Zlist = []
    skipvalue = False
    count = count + 1
    # building attribute extraction
    properties = feature['properties']['OBJECTID']
    print(properties)
    if feature['geometry'] == None:
        print("noník")
    else:
        # creates a list of the polygon Z of which the multipatch consists
        polygon_list = [p for mp in feature['geometry']['coordinates'] for p in mp]
        # reading of polygon heights
        for i in polygon_list:
            if len(i) == 3:
                    skipvalue = True
            if skipvalue == False:
                for n in i:
                    Xlist.append(n[0])
                    Ylist.append(n[1])
                    Zlist.append(n[2])
        if skipvalue == False:
            meanXlist.append(np.mean(Xlist))
            meanYlist.append(np.mean(Ylist))
            meanZlist.append(np.mean(Zlist))
            Zminlist.append(np.min(Zlist))
            ID_BUDlist.append(properties)
meanZlist = np.subtract(meanZlist,Zminlist)
#print(meanZlist)
df = pd.DataFrame({'x': meanXlist, 'y': meanYlist, 'z_centroid': meanZlist, 'OBJECTID': ID_BUDlist})
#print(df)

geometry = gpd.points_from_xy(df['x'], df['y'], df['z_centroid'])
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df['x'], df['y'], df['z_centroid']))
print(gdf)
gdf.to_file(out_file)

print ('Message: All features calculated.')