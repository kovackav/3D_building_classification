import geopandas as gpd
import math
import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "Calculating some metrics on final footprints.")
parser.add_argument('-i', '--input_data', type = str, metavar = '', required = True, help = 'Input footprints.')
parser.add_argument('-o', '--output_data', type = str, metavar = '', required = True, help = 'Output footprints with metrics calculated.')
args = parser.parse_args()

print ('Message: Parameters valid, starting the script.')

in_file = args.input_data
out_file = args.output_data

gdf = gpd.read_file(in_file)
print(gdf)

gdf = gdf.drop(columns=['FUNCTION_y'])
gdf = gdf.drop(columns=['min_poly_h'])
gdf = gdf.drop(columns=['max_poly_h'])
gdf['h_rel'] = gdf['building_h']
gdf['3D_cent_z'] = gdf['z_centroid']
gdf = gdf.drop(columns=['z_centroid'])
gdf['3D_h'] = gdf['building_h']
gdf = gdf.drop(columns=['building_h'])
gdf['3D_angle'] = gdf['angle']
gdf = gdf.drop(columns=['angle'])
gdf = gdf.drop(columns=['type'])

print(gdf)

# footprint area and perimeter
gdf= gdf.to_crs({'init': 'epsg:32633'})
bounds = gdf.bounds
gdf["w"] = gdf.bounds["maxy"] - gdf.bounds["miny"]
gdf["l"] = gdf.bounds["maxx"] - gdf.bounds["minx"]
gdf["area_foot"] = gdf['geometry'].area
gdf["roof_peri"] = gdf['geometry'].length
gdf= gdf.to_crs({'init': 'epsg:4326'})

# calculate the SI_2D square approximation
list_SI_2D = []
for index, row in gdf.iterrows():
    area = row["area_foot"]
    A = area/4
    SI_2D = round(area/(4*math.sqrt(A)),3)
    list_SI_2D.append(SI_2D)
gdf["SI_2D"] = list_SI_2D

# calculate the BI_2D rectangular approximation
list_BI_2D = []
for index, row in gdf.iterrows():
    area = row["area_foot"]
    w = row["w"]
    l = row["l"]
    BI_2D = round(area/(2*(l+w)),3)
    list_BI_2D.append(BI_2D)
gdf["BI_2D"] = list_BI_2D

# comp_2D rectangular approximation calculation
list_comp_2D = []
for index, row in gdf.iterrows():
    area = row["area_foot"]
    A = area/4
    w = row["w"]
    l = row["l"]
    comp_2D = round(((l*w)/A),3)
    list_comp_2D.append(comp_2D)
gdf["comp_2D"] = list_comp_2D

# simple volume
list_V = []
for index, row in gdf.iterrows():
    area = row["area_foot"]
    height = row["h_rel"]
    V = round((area * height),3)
    if V > 0:
        list_V.append(V)
    else:
        list_V.append(0.0001)
gdf["V"] = list_V

# calculate SI_3D cubic approximation
list_SI_3D = []
for index, row in gdf.iterrows():
    area = row["area_foot"]
    V = row["V"]
    SI_3D = round((area/(4* (V ** (1. / 3)))),3)
    list_SI_3D.append(SI_3D)
gdf["SI_3D"] = list_SI_3D

# drop unnecessary width and length
gdf = gdf.drop(columns=['w'])
gdf = gdf.drop(columns=['l'])


print(gdf)
gdf.to_file(out_file)

print ('Message: All features calculated.')