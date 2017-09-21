#%%
import time
start = time.time()

#import gdal, ogr, osr
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo.gdalnumeric import *
from osgeo.gdalconst import *
#import numpy as np
#import scipy.ndimage as ndimage
#import pandas as pd
from subprocess import call
from itertools import compress
#import skfmm
#import stateplane
#import pylab as p
#%matplotlib inline

file_bool2 = '../inun_nj/inun_bool2.tif'
file_poly2 = '../inun_nj/inun_poly2'
file_wl = '../inun_nj/wl.kml'



#%%
call(['gdal_polygonize.py', '-nomask', file_bool2, '-b', '1', '-q', file_poly2])


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

c_empty = '00000000'
c_1 = 'AB00FF00'
c_2 = 'AB00FFFF'
c_3 = 'AB0080FF'
c_4 = 'AB0000FF'
c_5 = 'ABCC00CC'


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

with open(file_wl,'w') as f_kml:
    f_kml.writelines(s)

#%%
end = time.time()
print(end - start)
