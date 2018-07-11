#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make STATIONS file, convert from lon/lat/depth to regional XYZ
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
##matplotlib.use("pdf")
#import matplotlib.pyplot as plt

#------ utility
def rotmat_enu_to_ecef(lon,lat):
  """ rotation matrix from ENU to ECEF coordinate basises
  """
  coslat = np.cos(np.deg2rad(lat))
  sinlat = np.sin(np.deg2rad(lat))
  coslon = np.cos(np.deg2rad(lon))
  sinlon = np.sin(np.deg2rad(lon))
  
  rotmat = np.zeros((3,3))
  rotmat[0,:] = [ -sinlon, -sinlat*coslon, coslat*coslon ]
  rotmat[1,:] = [  coslon, -sinlat*sinlon, coslat*sinlon ]
  rotmat[2,:] = [     0.0,  coslat,        sinlat        ]

  return rotmat

#======
channel_file = 'YP.channel.txt'
out_dir = 'source'
topo_grd = 'NEChina.grd'

#====== parameters
#--- reference point (WGS84 ellipsoid)
lon0 = 125
lat0 = 45
alt0 = 0.0

#====== read in channel list
with open(channel_file, 'r') as f:
  lines = [x.replace('\n','').split('|') for x in f.readlines() if not(x.startswith('#'))]
 
netwk = [x[0] for x in lines]
stnm = [x[1] for x in lines]
loc = [x[2] for x in lines]
chan = [x[3] for x in lines]

stla = np.array([float(x[4]) for x in lines])
stlo = np.array([float(x[5]) for x in lines])
#stel = np.array([float(x[6]) for x in lines])
stdp = np.array([float(x[7]) for x in lines])
cmpaz = np.array([float(x[8]) for x in lines])
cmpdip = np.array([float(x[9]) for x in lines])

#====== get station location in ENU0 coordinates

#--- 0. get topo on each station
fh = Dataset(topo_grd, mode='r')
grd_lon1 = fh.variables['x'][:]
grd_lat1 = fh.variables['y'][:]
grd_z2 = fh.variables['z'][:]

#plt.figure()
#plt.imshow(grd_z2)
#plt.gca().invert_yaxis()
#plt.colorbar()
#plt.show()
#sys.exit()

grd_lon2, grd_lat2 = np.meshgrid(grd_lon1, grd_lat1, indexing='xy')
stalt = interpolate.griddata((grd_lon2.flatten(), grd_lat2.flatten()), grd_z2.flatten(), (stlo, stla), method='cubic')

#--- 1. convert to ECEF
GPS_ELLPS = 'WGS84'
ecef = pyproj.Proj(proj='geocent', ellps=GPS_ELLPS)
lla = pyproj.Proj(proj='latlong', ellps=GPS_ELLPS)

x0, y0, z0 = pyproj.transform(lla, ecef, lon0, lat0, alt0)
xx, yy, zz = pyproj.transform(lla, ecef, stlo, stla, stalt-stdp)

#--- 2. transform from ECEF to ENU0
R0 = rotmat_enu_to_ecef(lon0,lat0)

nx = len(stlo)
xyz = np.zeros((3,nx))
xyz[0,:] = xx-x0
xyz[1,:] = yy-y0
xyz[2,:] = zz-z0

enu0 = np.dot(np.transpose(R0), xyz)

#======  output FORCESOLUTION for each channle
for i in range(nx):
  #--- get channel direction in ENU0 basis  
  R1 = rotmat_enu_to_ecef(stlo[i], stla[i])
  R = np.dot(np.transpose(R0), R1)
  dip = np.deg2rad(cmpdip[i])
  az = np.deg2rad(cmpaz[i])
  # cmpvec in station ENU
  cmpvec = np.array([ np.cos(dip)*np.sin(az), np.cos(dip)*np.cos(az), -1*np.sin(dip)])
  # cmpvec in ENU0
  cmpvec = np.dot(R, cmpvec)
  #--- output FORCESOLUTION
  seed_id = "%s.%s.%s.%s"%(netwk[i],stnm[i],loc[i],chan[i])
  out_file = "%s/%s.forcesolution"%(out_dir,seed_id)
  with open(out_file, "w") as f:
    f.write("FORCE                              %s\n"%(seed_id))
    f.write("time shift:                        0.0\n")
    f.write("f0:                                0.5\n")
    f.write("latorUTM:                          %f\n"%(enu0[1,i]))
    f.write("longorUTM:                         %f\n"%(enu0[0,i]))
    f.write("depth:                             %f\n"%(enu0[2,i]))
    f.write("factor force source:               1.d15\n")
    f.write("component dir vect source E:       %f\n"%(cmpvec[0]))
    f.write("component dir vect source N:       %f\n"%(cmpvec[1]))
    f.write("component dir vect source Z_UP:    %f\n"%(cmpvec[2]))
