#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make STATIONS file
convert from lon/lat/depth to REF_ENU
"""
import sys
import warnings
#
import numpy as np
from scipy import interpolate
#
from netCDF4 import Dataset
import pyproj
#
#import matplotlib
#matplotlib.use("pdf")
#import matplotlib.pyplot as plt

import importlib

#====== parameters

mesh_par_file = str(sys.argv[1])
topo_grd = str(sys.argv[2])
station_file = str(sys.argv[3]) # STATIONS in net sta lat lon ele depth
out_file = str(sys.argv[4])

##--- mesh parameter
#mesh_par_file = "mesh_par.py"
##--- station file
#station_file = "STATIONS"
#out_file = "STATIONS"
##--- GRD file for interface topography (should be smoothed to match the SEM mesh resolutin, i.e. smoothed over the length of the element size)
##topo_grd = "SETibet_smooth.grd"
#topo_txt = "topo/ETibet_smooth.txt"

#--- load mesh parameter file
if sys.version_info < (3, ):
  raise Exception("need python3")
elif sys.version_info < (3, 5):
  spec =importlib.machinery.SourceFileLoader("mesh_par", mesh_par_file)
  par = spec.load_module()
else:
  spec = importlib.util.spec_from_file_location("mesh_par", mesh_par_file)
  par = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(par)

#--- reference point
ref_lon   =  par.mesh_ref_lon   
ref_lat   =  par.mesh_ref_lat   
ref_alt   =  par.mesh_ref_alt   
ref_ellps =  par.mesh_ref_ellps 

#====== read in station coordinates
#with open(channel_file, 'r') as f:
#  lines = [ x.replace('\n','').split('|') for x in f.readlines() if not(x.startswith('#')) ]
#
## only use vertical component
#lines = [ x for x in lines if x[3][2] == 'Z' ]
#
#netwk = [x[0] for x in lines]
#stnm = [x[1] for x in lines]
#stch = [x[2] for x in lines]
#stla = np.array([float(x[4]) for x in lines])
#stlo = np.array([float(x[5]) for x in lines])
#stdp = np.array([float(x[7]) for x in lines])

with open(station_file, 'r') as f:
  lines = [ x.replace('\n','').split() for x in f.readlines() if not(x.startswith('#')) ]

netwk = [x[0] for x in lines]
stnm = [x[1] for x in lines]
stla = np.array([float(x[2]) for x in lines])
stlo = np.array([float(x[3]) for x in lines])
stdp = np.array([float(x[5]) for x in lines])

#====== interpolate topography from geodesic GRD file to the stations

#--- get alt(lon,lat) grid from GRD file
fh = Dataset(topo_grd, mode='r')
grd_lon1 = fh.variables['lon'][:]
grd_lat1 = fh.variables['lat'][:]
#grd_lon2, grd_lat2 = np.meshgrid(grd_lon1, grd_lat1, indexing='xy')
grd_alt2 = fh.variables['z'][:] # z(lat,lon)

#with open(topo_txt, 'r') as f:
#  lines = [ l.split() for l in f.readlines() ]
#
#grd_lons = np.array([ float(l[0]) for l in lines])
#grd_lats = np.array([ float(l[1]) for l in lines])
#grd_alts = np.array([ float(l[2]) for l in lines])

#--- interpolate surface altitude at each station
#stalt = interpolate.griddata((np.ravel(grd_lon2), np.ravel(grd_lat2)), np.ravel(grd_alt2), (stlo, stla), method='linear')
xi = np.zeros((len(stla),2))
xi[:,0] = stla[:]
xi[:,1] = stlo[:]
stalt = interpolate.interpn((grd_lat1, grd_lon1), grd_alt2, xi, method='linear', bounds_error=False)

#--- convert from geodetic to ECEF coordinates
ecef = pyproj.Proj(proj='geocent', ellps=ref_ellps)
lla = pyproj.Proj(proj='latlong', ellps=ref_ellps)

x0, y0, z0 = pyproj.transform(lla, ecef, ref_lon, ref_lat, ref_alt)
xx, yy, zz = pyproj.transform(lla, ecef, stlo, stla, stalt-stdp)

#--- conver from ECEF to local ENU
cosla = np.cos(np.deg2rad(ref_lat))
sinla = np.sin(np.deg2rad(ref_lat))
coslo = np.cos(np.deg2rad(ref_lon))
sinlo = np.sin(np.deg2rad(ref_lon))

#RotM = np.zeros((3,3))
#RotM[0,:] = [       -sinlo,        coslo,   0.0 ]
#RotM[1,:] = [ -sinla*coslo, -sinla*sinlo, cosla ]
#RotM[2,:] = [  cosla*coslo,  cosla*sinlo, sinla ]

ee =       -sinlo*(xx-x0) +       coslo*(yy-y0)
nn = -sinla*coslo*(xx-x0) - sinla*sinlo*(yy-y0) + cosla*(zz-z0)
uu =  cosla*coslo*(xx-x0) + cosla*sinlo*(yy-y0) + sinla*(zz-z0)

#====== save local ENU coordinates of all stations  
with open(out_file, "w") as f:
  for i in range(len(stlo)):
    # network, stnm, stla, stlo, stel, stbur
    # must use USE_SOURCES_RECEIVERS_Z = .true. in setup/constants.h.in, so stbur is taken as the z value) 
    f.write("%-10s  %-10s  %+15.5e  %+15.5e  0.0  %+15.5e\n"%(netwk[i], stnm[i], nn[i], ee[i], uu[i]) )
