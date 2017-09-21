#%%
import time
start = time.time()

#import gdal, ogr, osr
#from osgeo import gdal
#from osgeo import ogr
#from osgeo import osr
#from osgeo.gdalnumeric import *
#from osgeo.gdalconst import *
#import numpy as np
#import scipy.ndimage as ndimage
#import pandas as pd
from subprocess import call
#from itertools import compress
#import skfmm
#import stateplane
#import pylab as p
#%matplotlib inline

#import os
#os.environ["PATH"] += os.pathsep + '/mnt/cstor01/home/lyin1/njdemo/'

#import sys
#sys.path.append('/mnt/cstor01/home/lyin1/njdemo')
#sys.path.append("/mnt/cstor01/home/lyin1/njdemo/gdal2tiles_larry.py")
#sys.path.insert(1,'/mnt/cstor01/home/lyin1/njdemo')
#print(sys.path)


print('Creating tiles for Google Map...')


call(['/mnt/cstor01/home/lyin1/njdemo/gdal2tiles_larry.py', '-p', 'mercator', '-z', '8-20', \
      '-n', '-w', 'google', \
      '-t', 'Inun', \
      '-c', 'Stevens Institute of Technology', \
      '-g', 'AIzaSyCQwLDREXVBG2lSf8NMjOABEAYhDNv74Jc', \
      '../inun_nj/tmp/_tile2.vrt', '../inun_nj/2D/'])


#%%
end = time.time()
print(end - start)
