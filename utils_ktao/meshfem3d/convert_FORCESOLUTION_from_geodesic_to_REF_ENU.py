#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""convert FORCESOLUTION from geodesic coordinate to the REF_ENU coodinate
1. convert location from lat,lon,dep to REF_ENU
2. convert vector basis from local E,N,Z to X,Y,Z in REF_ENU
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


#====== parameters
mesh_par_file = str(sys.argv[1])
topo_grd = str(sys.argv[2])
cmt_file = str(sys.argv[3])
out_file = str(sys.argv[4])

#--- GRD file for interface topography (should be smoothed to match the SEM mesh resolutin, i.e. smoothed over the length of the element size)
#topo_grd = "SETibet_smooth.grd"
#topo_txt = "topo/ETibet_smooth.txt"

##--- reference point (WGS84 ellipsoid)
#ref_lon = 103
#ref_lat = 27
#ref_alt = 0.0
#ref_ellps = 'WGS84'

#--- load mesh parameter file
if sys.version_info < (3, ):
  raise Exception("need python3")
elif sys.version_info < (3, 5):
  import importlib
  spec =importlib.machinery.SourceFileLoader("mesh_par", mesh_par_file)
  par = spec.load_module()
else:
  import importlib.util
  spec = importlib.util.spec_from_file_location("mesh_par", mesh_par_file)
  par = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(par)

#--- reference point
ref_lon   =  par.mesh_ref_lon   
ref_lat   =  par.mesh_ref_lat   
ref_alt   =  par.mesh_ref_alt   
ref_ellps =  par.mesh_ref_ellps 

#====== read in forcesolution file
with open(cmt_file, 'r') as f:
  lines = [ x.replace('\n','') for x in f.readlines() if not(x.startswith('#')) ]

header = lines[0]

lines = [x.split(":") for x in lines]
time_shift = float(lines[1][1])

f0 = float(lines[2][1])
evla = float(lines[3][1])
evlo = float(lines[4][1])
evdp = float(lines[5][1])

# point force magnitude and direction
mag = float(lines[6][1])
ve = float(lines[7][1])
vn = float(lines[8][1])
vu = float(lines[9][1])

#======

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

#--- interpolate surface altitude at the event
#evalt = interpolate.griddata((np.ravel(grd_lon2), np.ravel(grd_lat2)), np.ravel(grd_alt2), (evlo, evla), method='cubic')
evalt = interpolate.interpn((grd_lat1, grd_lon1), grd_alt2, np.array([evla, evlo]), method='linear', bounds_error=False)

#--- convert from geodetic to ECEF coordinates
ecef = pyproj.Proj(proj='geocent', ellps=ref_ellps)
lla = pyproj.Proj(proj='latlong', ellps=ref_ellps)

x0, y0, z0 = pyproj.transform(lla, ecef, ref_lon, ref_lat, ref_alt)
xx, yy, zz = pyproj.transform(lla, ecef, evlo, evla, evalt - evdp*1000.0)

#--- convert from ECEF to local ENU
cosla_ref = np.cos(np.deg2rad(ref_lat))
sinla_ref = np.sin(np.deg2rad(ref_lat))
coslo_ref = np.cos(np.deg2rad(ref_lon))
sinlo_ref = np.sin(np.deg2rad(ref_lon))

#RotM = np.zeros((3,3))
#RotM[0,:] = [       -sinlo,        coslo,   0.0 ]
#RotM[1,:] = [ -sinla*coslo, -sinla*sinlo, cosla ]
#RotM[2,:] = [  cosla*coslo,  cosla*sinlo, sinla ]

ee =           -sinlo_ref*(xx-x0) +           coslo_ref*(yy-y0)
nn = -sinla_ref*coslo_ref*(xx-x0) - sinla_ref*sinlo_ref*(yy-y0) + cosla_ref*(zz-z0)
uu =  cosla_ref*coslo_ref*(xx-x0) + cosla_ref*sinlo_ref*(yy-y0) + sinla_ref*(zz-z0)

#--- convert moment tensor from station local ENU to reference ENU

#--- in ECEF coordinate
# vector basis in local ENU
cosla = np.cos(np.deg2rad(evla))
sinla = np.sin(np.deg2rad(evla))
coslo = np.cos(np.deg2rad(evlo))
sinlo = np.sin(np.deg2rad(evlo))

vx = np.array([ -sinlo, coslo, 0.0])
vy = np.array([ -sinla*coslo, -sinla*sinlo, cosla])
vz = np.array([ cosla*coslo, cosla*sinlo, sinla])

# vector basis in REF ENU
vx_ref = np.array([ -sinlo_ref, coslo_ref, 0.0])
vy_ref = np.array([ -sinla_ref*coslo_ref, -sinla_ref*sinlo_ref, cosla_ref])
vz_ref = np.array([ cosla_ref*coslo_ref, cosla_ref*sinlo_ref, sinla_ref])

trans_mat = np.array(
    [ [ np.dot(vx_ref,vx), np.dot(vx_ref, vy), np.dot(vx_ref, vz) ],
      [ np.dot(vy_ref,vx), np.dot(vy_ref, vy), np.dot(vy_ref, vz) ],
      [ np.dot(vz_ref,vx), np.dot(vz_ref, vy), np.dot(vz_ref, vz) ] ])

v_local = np.array([ve,vn,vu])

v_ref = np.dot(trans_mat, v_local)

#====== save local ENU coordinates of all stations  
with open(out_file, 'w') as fp:
  fp.write('%s\n' % (header))
  fp.write('%-35s %+15.8E\n' % ('time shift:',  time_shift))
  fp.write('%-35s %+15.8E\n' % ('f0:',          f0))
  fp.write('%-35s %+15.8E\n' % ('latorUTM:',    nn))
  fp.write('%-35s %+15.8E\n' % ('lonorUTM:',    ee))
  fp.write('%-35s %+15.8E\n' % ('depth:',       uu))
  fp.write('%-35s %+15.8E\n' % ('factor force source:', mag))
  fp.write('%-35s %+15.8E\n' % ('component dir vect source E:', v_ref[0]))
  fp.write('%-35s %+15.8E\n' % ('component dir vect source N:', v_ref[1]))
  fp.write('%-35s %+15.8E\n' % ('component dir vect source Z_UP:', v_ref[2]))
