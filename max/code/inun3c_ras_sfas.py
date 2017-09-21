#%%
import time
start = time.time()

#import gdal, ogr, osr
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
import numpy as np
import scipy.ndimage as ndimage
import pandas as pd
from subprocess import call
from itertools import compress
import skfmm
import stateplane
#import pylab as p
#%matplotlib inline

googleapikey = 'AIzaSyCQwLDREXVBG2lSf8NMjOABEAYhDNv74Jc'

with open('job_info.txt', 'r') as j:
    Location = j.readline().strip('\n')
    water_elev = float(j.readline().strip('\n'))

water_elev_str = str(int(water_elev*100))
title_str = 'inun_'+Location
DIR = '../'+title_str+'/'
dir_tmp = DIR+'tmp/'
dir_2D = DIR+'2D/'
dir_3D = DIR+'3D/'
call(['mkdir', '-p', dir_tmp])
call(['mkdir', '-p', dir_2D])
call(['mkdir', '-p', dir_3D])

file_DEM = '../dem/dem_'+Location+'.tif'
#file_bool0 = dir_tmp+'inun_bool0.tif'
file_bool1 = dir_tmp+'inun_bool1.tif'
file_poly1 = dir_tmp+'inun_poly1'
file_kml_domain = dir_2D+'domain.kml'

file_DEM_SP = dir_tmp+'dem_sp.tif'

file_bool3 = dir_tmp+'inun_bool3.tif'
file_poly3 = dir_tmp+'inun_poly3'
file_kml_poly3 = dir_tmp+'inun_poly3.kml'
file_wkt_poly3 = dir_tmp+'inun_poly3.wkt'

file_bool2 = dir_tmp+'inun_bool2.tif'
file_poly2 = dir_tmp+'inun_poly2'
file_wl = dir_2D+'wl.kml'
file_wl_noaa = dir_3D+'wl_noaa.kml'
URL = 'http://hudson.dl.stevens-tech.edu/njdemo/sandy/'

file_land_SP = dir_tmp+'land_SP.tif'
file_inun_SP = dir_tmp+'inun_SP.tif'
file_land = dir_tmp+'land.tif'
file_inun = dir_tmp+'inun.tif'

#%%
#==============================================================================
# print('Reading DEM...')
# 
# # Open the dataset
# bandNum1 = 1
# DEM = gdal.Open(file_DEM, GA_ReadOnly )
# band1 = DEM.GetRasterBand(bandNum1)
# driver = gdal.GetDriverByName("GTiff")
# 
# geotransform = DEM.GetGeoTransform()
# x_ul = geotransform[0]
# y_ul = geotransform[3]
# x_size = geotransform[1]
# y_size = geotransform[5]
# 
# data_raw = BandReadAsArray(band1)
# (y_cell,x_cell) = data_raw.shape
# #xv, yv = meshgrid(range(x_cell), range(y_cell), indexing='xy')
# #x_coor = xv * x_size + x_ul + (x_size*.5)
# #y_coor = yv * y_size + y_ul + (y_size*.5)
# #x_cent = (x_coor[0][0]+x_coor[-1][-1])*.5
# #y_cent = (y_coor[0][0]+y_coor[-1][-1])*.5
# x_cent = int(x_cell*.5)*x_size+x_ul+(x_size*.5)
# y_cent = int(y_cell*.5)*y_size+y_ul+(y_size*.5)
#==============================================================================

#==============================================================================
# sp_code = stateplane.identify(x_cent,y_cent)
# sp_str = stateplane.identify(x_cent,y_cent,'short')
# 
# nodata = np.nanmin(data_raw)
# mask_domain = ~(data_raw==nodata)
# 
# 
# #%% Domain 
# print('Generating Domain...')
# 
# data1 = np.copy(data_raw)
# data1[~mask_domain] = 0
# data1[mask_domain] = 1
# data1=data1.astype(int)
# 
# dsOut = driver.Create(file_bool1, DEM.RasterXSize, DEM.RasterYSize, 
#                       bandNum1, band1.DataType)
# CopyDatasetInfo(DEM,dsOut)
# bandOut=dsOut.GetRasterBand(bandNum1)
# BandWriteArray(bandOut, data1)
#==============================================================================

# Close the datasets
#==============================================================================
# band1 = None
# DEM = None
# bandOut = None
# dsOut = None
#==============================================================================

#==============================================================================
# 
# #%% 
# print('Reprojecting to SP...')
# 
# #call(['gdalbuildvrt', file_tile_i1, file_inun_cna])
# call(['gdalwarp', '-s_srs', 'WGS84', '-t_srs', 'EPSG:'+sp_code, \
#       '-r', 'bilinear', '-srcnodata', '{:.0f}'.format(nodata), '-dstnodata', '-99999', \
#       file_DEM, file_DEM_SP])
# 
# 
# #%%
# print('Reading DEM SP...')
# 
# # Open the dataset
# bandNum1 = 1
# DEM_SP = gdal.Open(file_DEM_SP, GA_ReadOnly )
# band1_SP = DEM_SP.GetRasterBand(bandNum1)
# driver = gdal.GetDriverByName("GTiff")
# 
# geotransform_SP = DEM_SP.GetGeoTransform()
# x_ul_SP = geotransform_SP[0]
# y_ul_SP = geotransform_SP[3]
# x_size_SP = geotransform_SP[1]
# y_size_SP = geotransform_SP[5]
# 
# data_raw_SP = BandReadAsArray(band1_SP)
# (y_cell_SP,x_cell_SP) = data_raw_SP.shape
# xm_SP, ym_SP = meshgrid(range(x_cell_SP), range(y_cell_SP), indexing='xy')
# x_coor_SP = xm_SP * x_size_SP + x_ul_SP + (x_size_SP*.5)
# y_coor_SP = ym_SP * y_size_SP + y_ul_SP + (y_size_SP*.5)
# #x_cent = (x_coor[0][0]+x_coor[-1][-1])*.5
# #y_cent = (y_coor[0][0]+y_coor[-1][-1])*.5
# #x_cent = int(x_cell*.5)*x_size+x_ul+(x_size*.5)
# #y_cent = int(y_cell*.5)*y_size+y_ul+(y_size*.5)
# 
# nodata_SP = np.nanmin(data_raw_SP)
# mask_domain_SP = ~(data_raw_SP==nodata_SP)
# 
# 
# #%% Interp points to nearest grid
# print('Interpolation points to SP...')
# 
# I = pd.read_csv('job_interp.csv')
# 
# multipoint = ogr.Geometry(ogr.wkbMultiPoint)
# for i in range(len(I)):
#     x,y = I.loc[i,['lon','lat']]
#     point = ogr.Geometry(ogr.wkbPoint)
#     point.AddPoint(x,y)
#     multipoint.AddGeometry(point)
# 
# source = osr.SpatialReference()
# source.SetWellKnownGeogCS('WGS84') # 'WGS84' = 'EPSG:4326'
# 
# target = osr.SpatialReference()
# target.ImportFromEPSG(int(sp_code))
# 
# transform = osr.CoordinateTransformation(source, target)
# 
# multipoint.Transform(transform)
# multipoint_wkt = multipoint.ExportToWkt()
# 
# I_xy_sp_raw = []
# for item in multipoint_wkt.split(")"):
#     if "(" in item:
#         I_xy_sp_raw.append(item [ item.find("(")+
#                                 len("(") : ])
# 
# I_xy_sp = np.array([v3.split(' ') for v2 in [v.split(',') for v in I_xy_sp_raw] 
#          for v3 in v2])[:,:-1].astype(float)
# I['x'] = I_xy_sp[:,0]
# I['y'] = I_xy_sp[:,1]
# 
# 
# #%%
# def nearest_ixy(x,y,xgrid,ygrid):
#     diff = (xgrid-x)**2+(ygrid-y)**2
#     diff_min = np.min(diff)
#     [ix,iy] = np.nonzero(diff==diff_min)
#     return int(ix),int(iy)
# 
# ixy = []
# for i in range(len(I)):
#     ixy.append(nearest_ixy(float(I.loc[i,'x']),float(I.loc[i,'y']),x_coor_SP,y_coor_SP))
# 
# I['ixy'] = ixy
# 
# 
# #%% Interp domain
# print('Constructing interpolation domain...')
# 
# interp_max = float(I.loc[:,'elev'].max()+1)
# 
# data_raw_SP[~mask_domain_SP] = np.nan
# 
# data_bool_SP = (interp_max-data_raw_SP)>=0
# data_bool_SP[~mask_domain_SP] = 0
# #data_val_SP = interp_max-data_raw_SP
#==============================================================================

#==============================================================================
# # Zoning
# def zoning(data_bool,data_raw):
#     print('Processing hydraulic connectivity...')
#     
#     current_output, num_ids = ndimage.label(data_bool)
#     
#     # Plot outputs
#     #plt.imshow(data_bool, cmap="spectral", interpolation='nearest')
#     #plt.imshow(current_output, cmap="spectral", interpolation='nearest')
#     
#     zone_id = range(num_ids)
#     zone_count = []
#     zone_mean = []
#     zone_min = []
#     for i in zone_id:
#         zone = current_output==i
#         zone_count.append(np.sum(zone))
#         zone_mean.append(np.nanmean(data_raw[zone]))
#         zone_min.append(np.nanmin(data_raw[zone]))
#     
#     Z = pd.DataFrame({'Count':zone_count,
#                       'Mean':zone_mean,
#                       'Min':zone_min},
#                      index=zone_id)
#     
#     inun_id = Z.sort_values('Min').index[0]
#     inun_zone = current_output==inun_id
#     inun = np.zeros(data_raw.shape)
#     inun[inun_zone] = 1
#     inun = inun.astype(bool)
#     return inun
#==============================================================================

#==============================================================================
# inun_SP = zoning(data_bool_SP, data_raw_SP)
# 
# #%%
# print('Constructing distance maps...')
# 
# data_SP = np.copy(data_raw_SP)
# data_SP[inun_SP] = 0
# data_SP[~inun_SP] = 1
# data_SP = data_SP.astype(bool)
# 
# SP_shape = data_raw_SP.shape
# #%%
# dist_map = []
# for i in range(len(I)):
#     interp_grid = np.ones(SP_shape,dtype='int')
#     interp_grid[I.loc[i,'ixy']] = 0
#     interp_case = np.ma.masked_array(interp_grid, data_SP)
#     dist_map.append(skfmm.distance(interp_case))
# #==============================================================================
# # #%% Distance maps plots
# # # Turn interactive plotting off
# # p.ioff()
# # for i in range(len(dist_map)):
# #     p.imshow(dist_map[i], vmin=0, vmax=np.max(dist_map))
# #     p.colorbar()
# #     p.savefig('dist_map_'+str(i)+'.png')
# #     p.close()
# #==============================================================================
# 
# #%%
# print('Interpolating by IDW...')
# 
# idw_bot = np.zeros(dist_map[0].shape)
# idw_top = np.copy(idw_bot)
# for i in range(len(I)):
#     id2 = 1/(dist_map[i].filled(np.nan)**2)
#     idw_bot += id2
#     idw_top += float(I.loc[:,'elev'][i])*id2
# dist_map = None
# idw = idw_top/idw_bot
# #==============================================================================
# # #%% IDW surface plot
# # p.ioff()
# # p.imshow(idw)
# # p.colorbar()
# # p.savefig('idw.png')
# # p.close()
# #==============================================================================
# 
# 
# #%% Inun raster in SP
# print('Constructing land & inundation rasters in SP...')
# 
# data2_SP = idw-data_raw_SP
# data2_SP[np.isnan(data2_SP)] = np.nan
# 
# 
# dsOut = driver.Create(file_land_SP, DEM_SP.RasterXSize, DEM_SP.RasterYSize, 
#                       bandNum1, band1_SP.DataType)
# CopyDatasetInfo(DEM_SP,dsOut)
# bandOut=dsOut.GetRasterBand(bandNum1)
# BandWriteArray(bandOut, data_raw_SP)
# 
# 
# dsOut = driver.Create(file_inun_SP, DEM_SP.RasterXSize, DEM_SP.RasterYSize, 
#                       bandNum1, band1_SP.DataType)
# CopyDatasetInfo(DEM_SP,dsOut)
# bandOut=dsOut.GetRasterBand(bandNum1)
# BandWriteArray(bandOut, data2_SP)
# 
# # Close the datasets
# band1_SP = None
# DEM_SP = None
# bandOut = None
# dsOut = None
# 
# 
# #%% 
# print('Reprojecting to WGS84...')
# 
# call(['gdalwarp', '-s_srs', 'EPSG:'+sp_code, '-t_srs', 'WGS84', 
#       '-r', 'bilinear', '-srcnodata', '-99999', '-dstnodata', '-99999', 
#       file_land_SP, file_land])
# 
# call(['gdalwarp', '-s_srs', 'EPSG:'+sp_code, '-t_srs', 'WGS84', 
#       '-r', 'bilinear', '-srcnodata', '-99999', '-dstnodata', '-99999', 
#       file_inun_SP, file_inun])
# 
#==============================================================================


#==============================================================================
# #%% Creating inundation map
# print('Classifying inundation in WGS84...')
# 
# # Read land
# DEM = gdal.Open(file_land, GA_ReadOnly )
# band1 = DEM.GetRasterBand(bandNum1)
# driver = gdal.GetDriverByName("GTiff")
# 
# # Read the data into numpy arrays
# land = BandReadAsArray(band1)
# mask_domain_land = ~(land==np.nanmin(land))
# land[~mask_domain_land] = np.nan
# 
# # Close the datasets
# band1 = None
# DEM = None
# bandOut = None
# dsOut = None
# 
# 
# #%% Read inun
# DEM = gdal.Open(file_inun, GA_ReadOnly )
# band1 = DEM.GetRasterBand(bandNum1)
# driver = gdal.GetDriverByName("GTiff")
# 
# # Read the data into numpy arrays
# data_raw = BandReadAsArray(band1)
# mask_domain = ~(data_raw==np.nanmin(data_raw))
# 
# data2_raw = np.copy(data_raw)
# data2_raw[~mask_domain] = np.nan
# data2_bool = data2_raw>=0
# data2_bool[~mask_domain] = 0
# data2_val = np.copy(data2_raw)
# 
# inun = zoning(data2_bool, land)
# 
# data2_val[~inun] = np.nan
# 
# data2 = np.copy(data2_val)
# data2[data2_val<=1] = 1
# data2[(data2_val>1) & (data2_val<=3)] = 2
# data2[(data2_val>3) & (data2_val<=6)] = 3
# data2[(data2_val>6) & (data2_val<=9)] = 4
# data2[data2_val>9] = 5
# data2[~inun] = 0
# data2[land<0] = 0
# data2 = data2.astype(int)
# 
# dsOut = driver.Create(file_bool2, DEM.RasterXSize, DEM.RasterYSize, 
#                       bandNum1, band1.DataType)
# CopyDatasetInfo(DEM,dsOut)
# bandOut=dsOut.GetRasterBand(bandNum1)
# BandWriteArray(bandOut, data2)
# 
# # Close the datasets
# band1 = None
# DEM = None
# bandOut = None
# dsOut = None
# 
# #%%
# call(['gdal_polygonize.py', '-nomask', file_bool1, '-b', '1', '-q', file_poly1])
#==============================================================================
#call(['gdal_polygonize.py', '-nomask', file_bool2, '-b', '1', '-q', file_poly2])


#==============================================================================
# #%% Polygon of Domain
# print('Constructing domain polygon...')
# 
# with open(file_poly1,'r') as f_poly:
#     text_all = f_poly.read().replace('\n', '')
# 
# dn = []
# for item in text_all.split("</ogr:DN>"):
#     if "<ogr:DN>" in item:
#         dn.append(item [ item.find("<ogr:DN>")+len("<ogr:DN>") : ])
# dn = [int(v) for v in dn[:]]
# 
# outer_block = []
# for item in text_all.split("</gml:coordinates></gml:LinearRing></gml:outerBoundaryIs>"):
#     if "<gml:outerBoundaryIs><gml:LinearRing><gml:coordinates>" in item:
#         outer_block.append(item [ item.find("<gml:outerBoundaryIs><gml:LinearRing><gml:coordinates>")+
#                                 len("<gml:outerBoundaryIs><gml:LinearRing><gml:coordinates>") : ])
# outer = [[[float(v6) for v6 in v5] for v5 in v4] for v4 in 
#         [[v3.split(',') for v3 in v2] for v2 in 
#          [v.split(' ') for v in outer_block]]]
# 
# fm = []
# for item in text_all.split("</gml:featureMember>"):
#     if "<gml:featureMember>" in item:
#         fm.append(item [ item.find("<gml:featureMember>")+len("<gml:featureMember>") : ])
# 
# inner = []
# inner_count = []
# for i in range(len(fm)):
#     inner_block = []
#     for item in fm[i].split("</gml:coordinates></gml:LinearRing></gml:innerBoundaryIs>"):
#         if "<gml:innerBoundaryIs><gml:LinearRing><gml:coordinates>" in item:
#             inner_block.append(item [ item.find("<gml:innerBoundaryIs><gml:LinearRing><gml:coordinates>")+
#                                     len("<gml:innerBoundaryIs><gml:LinearRing><gml:coordinates>") : ])
#     if not inner_block:
#         inner.append([])
#         inner_count.append(0)
#     else:
#         inner.append([[[float(v6) for v6 in v5] for v5 in v4] for v4 in 
#                  [[v3.split(',') for v3 in v2] for v2 in 
#                   [v.split(' ') for v in inner_block]]])
#         inner_count.append(len(inner[-1]))
# 
# dn1 = [v==1 for v in dn]
# outer1 = list(compress(outer, dn1))
# inner1 = list(compress(inner, dn1))
# inner_count1 = list(compress(inner_count, dn1))
# 
# 
# #%% Domain KML
# print('Creating domain kml...')
# 
# c_domain = 'AB0000FF'
# c_empty = '00000000'
# 
# 
# s = []
# s = """<?xml version="1.0" encoding="UTF-8"?>
# <kml xmlns="http://www.opengis.net/kml/2.2">
# <Document>
# <name>{title}</name>
# <Style id="domain">
# <LineStyle>
# <color>{c1}</color>
# <width>1</width>
# </LineStyle>
# <PolyStyle>
# <color>{c0}</color>
# </PolyStyle>
# </Style>""".format(title=title_str, c1=c_domain,c0=c_empty)
# 
# for i in range(len(outer1)):
#     s += """
# <Placemark>
# <name>{id:d}</name>
# <visibility>1</visibility>
# <styleUrl>#domain</styleUrl>
# <Polygon>
# <extrude>0</extrude>
# <tessellate>1</tessellate>
# <altitudeMode>clampToGround</altitudeMode>
# <outerBoundaryIs>
# <LinearRing>
# <coordinates>""".format(id=i)
#     
#     for ii in range(len(outer1[i])):
#         s += """
# {lon:.15f},{lat:.15f}""".format(lon=outer1[i][ii][0],lat=outer1[i][ii][1])
#     
#     s += """
# </coordinates>
# </LinearRing>
# </outerBoundaryIs>"""
#     
#     if inner_count1[i]>0:
#         for ii in range(inner_count1[i]):
#             s += """
# <innerBoundaryIs>
# <LinearRing>
# <coordinates>"""
#             for iii in range(len(inner1[i][ii])):
#                 s += """
# {lon:.15f},{lat:.15f}""".format(lon=inner1[i][ii][iii][0],lat=inner1[i][ii][iii][1])
#             
#             s += """
# </coordinates>
# </LinearRing>
# </innerBoundaryIs>"""
#     
#     s += """
# </Polygon>
# </Placemark>"""
# 
# s += """
# </Document>
# </kml>"""
# 
# with open(file_kml_domain,'w') as f_kml:
#     f_kml.writelines(s)
# 
#==============================================================================



#%% Polygon of wl
print('Constructing inundation polygons...')

with open(file_poly2,'r') as f_poly:
    text_all = f_poly.read().replace('\n', '')

dn = []
for item in text_all.split("</ogr:DN>"):
    if "<ogr:DN>" in item:
        dn.append(item [ item.find("<ogr:DN>")+len("<ogr:DN>") : ])
dn = [int(v) for v in dn[:]]

outer_block = []
for item in text_all.split("</gml:coordinates></gml:LinearRing></gml:outerBoundaryIs>"):
    if "<gml:outerBoundaryIs><gml:LinearRing><gml:coordinates>" in item:
        outer_block.append(item [ item.find("<gml:outerBoundaryIs><gml:LinearRing><gml:coordinates>")+
                                len("<gml:outerBoundaryIs><gml:LinearRing><gml:coordinates>") : ])
outer = [[[float(v6) for v6 in v5] for v5 in v4] for v4 in 
        [[v3.split(',') for v3 in v2] for v2 in 
         [v.split(' ') for v in outer_block]]]

fm = []
for item in text_all.split("</gml:featureMember>"):
    if "<gml:featureMember>" in item:
        fm.append(item [ item.find("<gml:featureMember>")+len("<gml:featureMember>") : ])

inner = []
inner_count = []
for i in range(len(fm)):
    inner_block = []
    for item in fm[i].split("</gml:coordinates></gml:LinearRing></gml:innerBoundaryIs>"):
        if "<gml:innerBoundaryIs><gml:LinearRing><gml:coordinates>" in item:
            inner_block.append(item [ item.find("<gml:innerBoundaryIs><gml:LinearRing><gml:coordinates>")+
                                    len("<gml:innerBoundaryIs><gml:LinearRing><gml:coordinates>") : ])
    if not inner_block:
        inner.append([])
        inner_count.append(0)
    else:
        inner.append([[[float(v6) for v6 in v5] for v5 in v4] for v4 in 
                 [[v3.split(',') for v3 in v2] for v2 in 
                  [v.split(' ') for v in inner_block]]])
        inner_count.append(len(inner[-1]))

dn1 = [v==1 for v in dn]
outer1 = list(compress(outer, dn1))
inner1 = list(compress(inner, dn1))
inner_count1 = list(compress(inner_count, dn1))

dn2 = [v==2 for v in dn]
outer2 = list(compress(outer, dn2))
inner2 = list(compress(inner, dn2))
inner_count2 = list(compress(inner_count, dn2))

dn3 = [v==3 for v in dn]
outer3 = list(compress(outer, dn3))
inner3 = list(compress(inner, dn3))
inner_count3 = list(compress(inner_count, dn3))

dn3 = [v==3 for v in dn]
outer3 = list(compress(outer, dn3))
inner3 = list(compress(inner, dn3))
inner_count3 = list(compress(inner_count, dn3))

dn4 = [v==4 for v in dn]
outer4 = list(compress(outer, dn4))
inner4 = list(compress(inner, dn4))
inner_count4 = list(compress(inner_count, dn4))

dn5 = [v==5 for v in dn]
outer5 = list(compress(outer, dn5))
inner5 = list(compress(inner, dn5))
inner_count5 = list(compress(inner_count, dn5))

# SFAS color
c_empty = '00000000'
c_1 = 'AB00FF00'
c_2 = 'AB00FFFF'
c_3 = 'AB0080FF'
c_4 = 'AB0000FF'
c_5 = 'ABCC00CC'

## NOAA color
#c_empty = '00000000'
#c_1 = 'ABFF7F00'
#c_2 = 'ABFF7F00'
#c_3 = 'AB00FFFF'
#c_4 = 'AB00AAFF'
#c_5 = 'AB0000FF'

s = []
s = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>{title}</name>""".format(title=title_str)

s += """
<Style id="s_1">
<LineStyle>
<color>{c0}</color>
<width>0</width>
</LineStyle>
<PolyStyle>
<color>{c}</color>
</PolyStyle>
</Style>""".format(c=c_1,c0=c_empty)

s += """
<Style id="s_2">
<LineStyle>
<color>{c0}</color>
<width>0</width>
</LineStyle>
<PolyStyle>
<color>{c}</color>
</PolyStyle>
</Style>""".format(c=c_2,c0=c_empty)

s += """
<Style id="s_3">
<LineStyle>
<color>{c0}</color>
<width>0</width>
</LineStyle>
<PolyStyle>
<color>{c}</color>
</PolyStyle>
</Style>""".format(c=c_3,c0=c_empty)

s += """
<Style id="s_4">
<LineStyle>
<color>{c0}</color>
<width>0</width>
</LineStyle>
<PolyStyle>
<color>{c}</color>
</PolyStyle>
</Style>""".format(c=c_4,c0=c_empty)

s += """
<Style id="s_5">
<LineStyle>
<color>{c0}</color>
<width>0</width>
</LineStyle>
<PolyStyle>
<color>{c}</color>
</PolyStyle>
</Style>""".format(c=c_5,c0=c_empty)


for i in range(len(outer1)):
    s += """
<Placemark>
<name>{id:d}</name>
<visibility>1</visibility>
<styleUrl>#s_1</styleUrl>
<Polygon>
<extrude>0</extrude>
<tessellate>1</tessellate>
<altitudeMode>clampToGround</altitudeMode>
<outerBoundaryIs>
<LinearRing>
<coordinates>""".format(id=i)
    
    for ii in range(len(outer1[i])):
        s += """
{lon:.15f},{lat:.15f}""".format(lon=outer1[i][ii][0],lat=outer1[i][ii][1])
    
    s += """
</coordinates>
</LinearRing>
</outerBoundaryIs>"""
    
    if inner_count1[i]>0:
        for ii in range(inner_count1[i]):
            s += """
<innerBoundaryIs>
<LinearRing>
<coordinates>"""
            for iii in range(len(inner1[i][ii])):
                s += """
{lon:.15f},{lat:.15f}""".format(lon=inner1[i][ii][iii][0],lat=inner1[i][ii][iii][1])
            
            s += """
</coordinates>
</LinearRing>
</innerBoundaryIs>"""
    
    s += """
</Polygon>
</Placemark>"""


for i in range(len(outer2)):
    s += """
<Placemark>
<name>{id:d}</name>
<visibility>1</visibility>
<styleUrl>#s_2</styleUrl>
<Polygon>
<extrude>0</extrude>
<tessellate>1</tessellate>
<altitudeMode>clampToGround</altitudeMode>
<outerBoundaryIs>
<LinearRing>
<coordinates>""".format(id=i)
    
    for ii in range(len(outer2[i])):
        s += """
{lon:.15f},{lat:.15f}""".format(lon=outer2[i][ii][0],lat=outer2[i][ii][1])
    
    s += """
</coordinates>
</LinearRing>
</outerBoundaryIs>"""
    
    if inner_count2[i]>0:
        for ii in range(inner_count2[i]):
            s += """
<innerBoundaryIs>
<LinearRing>
<coordinates>"""
            for iii in range(len(inner2[i][ii])):
                s += """
{lon:.15f},{lat:.15f}""".format(lon=inner2[i][ii][iii][0],lat=inner2[i][ii][iii][1])
            
            s += """
</coordinates>
</LinearRing>
</innerBoundaryIs>"""
    
    s += """
</Polygon>
</Placemark>"""


for i in range(len(outer3)):
    s += """
<Placemark>
<name>{id:d}</name>
<visibility>1</visibility>
<styleUrl>#s_3</styleUrl>
<Polygon>
<extrude>0</extrude>
<tessellate>1</tessellate>
<altitudeMode>clampToGround</altitudeMode>
<outerBoundaryIs>
<LinearRing>
<coordinates>""".format(id=i)
    
    for ii in range(len(outer3[i])):
        s += """
{lon:.15f},{lat:.15f}""".format(lon=outer3[i][ii][0],lat=outer3[i][ii][1])
    
    s += """
</coordinates>
</LinearRing>
</outerBoundaryIs>"""
    
    if inner_count3[i]>0:
        for ii in range(inner_count3[i]):
            s += """
<innerBoundaryIs>
<LinearRing>
<coordinates>"""
            for iii in range(len(inner3[i][ii])):
                s += """
{lon:.15f},{lat:.15f}""".format(lon=inner3[i][ii][iii][0],lat=inner3[i][ii][iii][1])
            
            s += """
</coordinates>
</LinearRing>
</innerBoundaryIs>"""
    
    s += """
</Polygon>
</Placemark>"""


for i in range(len(outer4)):
    s += """
<Placemark>
<name>{id:d}</name>
<visibility>1</visibility>
<styleUrl>#s_4</styleUrl>
<Polygon>
<extrude>0</extrude>
<tessellate>1</tessellate>
<altitudeMode>clampToGround</altitudeMode>
<outerBoundaryIs>
<LinearRing>
<coordinates>""".format(id=i)
    
    for ii in range(len(outer4[i])):
        s += """
{lon:.15f},{lat:.15f}""".format(lon=outer4[i][ii][0],lat=outer4[i][ii][1])
    
    s += """
</coordinates>
</LinearRing>
</outerBoundaryIs>"""
    
    if inner_count4[i]>0:
        for ii in range(inner_count4[i]):
            s += """
<innerBoundaryIs>
<LinearRing>
<coordinates>"""
            for iii in range(len(inner4[i][ii])):
                s += """
{lon:.15f},{lat:.15f}""".format(lon=inner4[i][ii][iii][0],lat=inner4[i][ii][iii][1])
            
            s += """
</coordinates>
</LinearRing>
</innerBoundaryIs>"""
    
    s += """
</Polygon>
</Placemark>"""


for i in range(len(outer5)):
    s += """
<Placemark>
<name>{id:d}</name>
<visibility>1</visibility>
<styleUrl>#s_5</styleUrl>
<Polygon>
<extrude>0</extrude>
<tessellate>1</tessellate>
<altitudeMode>clampToGround</altitudeMode>
<outerBoundaryIs>
<LinearRing>
<coordinates>""".format(id=i)
    
    for ii in range(len(outer5[i])):
        s += """
{lon:.15f},{lat:.15f}""".format(lon=outer5[i][ii][0],lat=outer5[i][ii][1])
    
    s += """
</coordinates>
</LinearRing>
</outerBoundaryIs>"""
    
    if inner_count5[i]>0:
        for ii in range(inner_count5[i]):
            s += """
<innerBoundaryIs>
<LinearRing>
<coordinates>"""
            for iii in range(len(inner5[i][ii])):
                s += """
{lon:.15f},{lat:.15f}""".format(lon=inner5[i][ii][iii][0],lat=inner5[i][ii][iii][1])
            
            s += """
</coordinates>
</LinearRing>
</innerBoundaryIs>"""
    
    s += """
</Polygon>
</Placemark>"""


s += """
</Document>
</kml>"""

with open(file_wl_noaa,'w') as f_kml:
    f_kml.writelines(s)

#==============================================================================
# all_line = s.splitlines(keepends=True)
# 
# lookup = '<Placemark>\n'
# pm_line = []
# for num, line in enumerate(all_line):
#     if lookup in line:
#         pm_line.append(num)
# lookup = '</Document>\n'
# for num, line in reversed(list(enumerate(all_line))):
#     if lookup in line:
#         pm_line.append(num)
#         break
# s_beg = all_line[:pm_line[0]]
# s_end = all_line[pm_line[-1]:]
# 
# s_kml = []
# ikml = 0
# file_wl_part = 'wl_{:d}.kml'.format(ikml)
# for ipm in range(len(pm_line)-1):
#     s_kml += all_line[pm_line[ipm]:pm_line[ipm+1]]
#     if ((ipm == len(pm_line)-2) or 
#         ((len(s_kml) <= 80000) and 
#          (len(s_kml)+len(all_line[pm_line[ipm+1]:pm_line[ipm+2]]) > 80000))):
#         with open (dir_2D+file_wl_part, 'w') as f:
#             f.writelines(s_beg)
#             f.writelines(s_kml)
#             f.writelines(s_end)
#         s_kml = []
#         ikml += 1
#         file_wl_part = 'wl_{:d}.kml'.format(ikml)
#==============================================================================


#==============================================================================
# #%% HTML/JAVASCRIPT
# print('Writing HTML/JAVASCRIPT...')
# 
# s = []
# s += """<!DOCTYPE html>
# <html>
#   <head>
#     <meta name="viewport" content="initial-scale=1.0">
#     <meta charset="utf-8">
#     <title>{title:s}</title>
#     <style>
#       html, body {{
#         height: 100%;
#         margin: 0;
#         padding: 0;
#       }}
#       #map {{
#         height: 100%;
#       }}
#     </style>
#   </head>
#   <body>
#     <div id="map"></div>
#     <script>
# 
#       function initMap() {{
#         var map = new google.maps.Map(document.getElementById('map'), {{
#           zoom: 12,
#           center: {{lat: {lat:.15f}, lng: {lon:.15f}}}
#         }});""".format(title=title_str,lat=y_cent,lon=x_cent)
# 
# for i in range(ikml):
#     s += """
#         var wlLayer = new google.maps.KmlLayer({{
#           url: '{kml:s}',
#           map: map
#         }});""".format(kml=URL+'wl_{:d}.kml'.format(i))
# 
# s += """
#         var domainLayer = new google.maps.KmlLayer({{
#           url: '{kml:s}',
#           map: map
#         }});""".format(kml=URL+'domain.kml')
# 
# s += """
#       }}
#     </script>
#     <script async defer
#     src="https://maps.googleapis.com/maps/api/js?key={key:s}&signed_in=true&callback=initMap">
#     </script>
#   </body>
# </html>""".format(key=googleapikey)
# 
# with open(dir_2D+'index.shtml', 'w') as f:
#     f.writelines(s)
#==============================================================================

#%%
#==============================================================================
# #%%
# print('Reading DEM...')
# 
# # Open the dataset
# bandNum1 = 1
# DEM = gdal.Open(dir_tmp+'dist_map_'+'wgs84_'+str(0), GA_ReadOnly )
# band1 = DEM.GetRasterBand(bandNum1)
# driver = gdal.GetDriverByName("GTiff")
# 
# geotransform = DEM.GetGeoTransform()
# x_ul_dist1 = geotransform[0]
# y_ul_dist1 = geotransform[3]
# x_size_dist1 = geotransform[1]
# y_size_dist1 = geotransform[5]
# 
# data_raw_dist1 = BandReadAsArray(band1)
# (y_cell_dist1,x_cell_dist1) = data_raw_dist1.shape
# #xv, yv = meshgrid(range(x_cell), range(y_cell), indexing='xy')
# #x_coor = xv * x_size + x_ul + (x_size*.5)
# #y_coor = yv * y_size + y_ul + (y_size*.5)
# #x_cent = (x_coor[0][0]+x_coor[-1][-1])*.5
# #y_cent = (y_coor[0][0]+y_coor[-1][-1])*.5
# #==============================================================================
# # x_cent = int(x_cell*.5)*x_size+x_ul+(x_size*.5)
# # y_cent = int(y_cell*.5)*y_size+y_ul+(y_size*.5)
# # 
# # sp_code = stateplane.identify(x_cent,y_cent)
# # sp_str = stateplane.identify(x_cent,y_cent,'short')
# # 
# # nodata = np.nanmin(data_raw)
# # mask_domain = ~(data_raw==nodata)
# #==============================================================================
#==============================================================================

#%%
end = time.time()
print(end - start)
