import json
from pickle import TRUE
from unittest import skip
import geopandas as gpd
from shapely.geometry import Polygon 
import math
import numpy as np
import math
import pandas as pd
import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "This script decomposes the input multipatch of an building and classifies the footprint, roof and the walls. Input should be in S-JTSK (EPSG 5514).")
parser.add_argument('-i', '--input_data', type = str, metavar = '', required = True, help = 'Input multipatch shapefile.')
parser.add_argument('-o', '--output_data', type = str, metavar = '', required = True, help = 'Output name of the decomposed PolygonZ.')
args = parser.parse_args()

print ('Message: Parameters valid, starting the script.')

in_file = args.input_data
out_file = args.output_data

# reading of multipatch
gdf = gpd.read_file(in_file)
# converting to json
gdf_json = json.loads(gdf.to_json())
# crs definition
crs = gdf.crs

feature_list = []
splitted_feature_list = []

# reading of json
with open((in_file), 'w') as f:
    json.dump(gdf_json, f, indent=4, sort_keys=True)
count = 1

# feature => multipatch of one single building
for feature in gdf_json['features']:
        print(count)
        Xlist = []
        Ylist = []
        Zlist = []
        skipvalue = False
        # counting processed buildings
        count = count + 1
        # saving building properties into an variable
        properties = feature['properties']
        # skiping feature when it has no geometry
        if feature['geometry'] == None:
            skip
        else:
        # creating the list of all polygons of a feature
            polygon_list = [p for mp in feature['geometry']['coordinates'] for p in mp]
            # reading of heights
            for i in polygon_list:
                if len(i) == 3:
                    skipvalue = True
                if skipvalue == False:
                    for n in i:
                        for element in n:
                            if type(element) == float and pd.isna(element):
                                print('nan')
                        Xlist.append(n[0])
                        Ylist.append(n[1])
                        Zlist.append(n[2])
                        print(Xlist)
            if skipvalue == False:
                # searching for a poly with the minimum Z, which should be a footprint
                minZ = min(Zlist)
                # searching for a max Z
                maxZ = max(Zlist)
                Z_difference = maxZ-minZ
                # now we are on the level of a single polygon
                for polygon in polygon_list:
                    height_list_polygon = []
                    latlon_list_polygon = []
                    latlon_list_polygon_rounded = []
                    # creating of an empty feature with properties of old feature
                    new_feature = properties.copy()
                    # filling list of coordinates
                    for i in polygon:
                        latlon_list_polygon.extend([[i[0],i[1]]])
                        height_list_polygon.append(i[2])
                    min_height = min(height_list_polygon)
                    max_height = max(height_list_polygon)
                    poly_height = max_height - min_height
                    seen = []
                    # calculating angle of an polygon
                    for i in polygon:
                        if (i[2]) == min_height:
                            minzx = (i[0])
                            minzy = (i[1])
                        if (i[2]) == max_height:   
                            maxzx = (i[0])
                            maxzy = (i[1])
                    # calculating distance between polygon vertices
                    distanceXY = (math.sqrt((maxzx-minzx)**2 + (maxzy-minzy)**2))
                    coords_1 = (minzx, minzy)
                    coords_2 = (maxzx, maxzy)
                    # calculating angle of an polygon
                    if distanceXY > 0:
                        angle = np.rad2deg(math.atan(Z_difference / distanceXY))
                    else:
                        angle = 0
                    # small rounding of coordinates, because the walls are not always completely straight, but not completely sloping
                    for xy in latlon_list_polygon:
                        round_latlon = [round(element, 7) for element in xy]
                        latlon_list_polygon_rounded.append(round_latlon)
                    # here we are looking for if any XY repeats, with the fact that Z may be different. If it repeats => the wall, the points are on top of each other
                    for xy in latlon_list_polygon_rounded:
                        if xy in seen:
                            pass
                        else:
                                seen.append(xy)
                    # if the number of points + 1 (start point = end point) is different from the seen, it is a wall, because some points have the same XY
                    if len(seen)+1 != len(latlon_list_polygon_rounded):
                        new_feature['min_poly_height'] = min_height
                        new_feature['max_poly_height'] = max_height
                        new_feature['angle'] = 0
                        new_feature['type'] = "stena"
                        new_feature['building_height'] = Z_difference
                        new_feature['geometry'] = Polygon([vertex[:3] for vertex in polygon])   
                        splitted_feature_list.append(new_feature)
                    # if the number of points + 1 (start point = end point) is the same as seen, then no point is repeated and it is the roof
                    if  len(seen)+1 == len(latlon_list_polygon_rounded):
                        new_feature['min_poly_height'] = min_height
                        new_feature['max_poly_height'] = max_height
                        new_feature['angle'] = angle
                        new_feature['type'] = "strecha"
                        new_feature['building_height'] = Z_difference
                        new_feature['geometry'] = Polygon([vertex[:3] for vertex in polygon])   
                        splitted_feature_list.append(new_feature)
                    # if the number of points + 1 (start point = end points) is the same as seen, then no point is repeated and the height is minimal, it is a footprint
                    if  len(seen)+1 == len(latlon_list_polygon_rounded) and min_height == minZ:
                        new_feature['min_poly_height'] = min_height
                        new_feature['max_poly_height'] = max_height
                        new_feature['angle'] = 0
                        new_feature['type'] = "footprint"
                        new_feature['building_height'] = Z_difference
                        new_feature['geometry'] = Polygon([vertex[:3] for vertex in polygon])  
                        splitted_feature_list.append(new_feature)
            
                        
feature_list = splitted_feature_list

# production of a geodataframe from a new polygon sheet
new_gdf_gj = gpd.GeoDataFrame(feature_list, index=range(len(feature_list)), crs=crs)

# export to new shapefile
new_gdf_gj.to_file(out_file)

print ('Message: All features calculated.')