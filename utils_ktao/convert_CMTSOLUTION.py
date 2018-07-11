#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""convert CMTSOLUTION from geodesic coordinate to the local ENU coodinate
1. convert location from lat,lon,dep to local ENU
2. convert vector basis from r,theta,phi to X,Y,Z
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
from obspy import UTCDateTime

import matplotlib
#matplotlib.use("pdf")
import matplotlib.pyplot as plt

#====== parameters
cmt_file = sys.argv[1]
out_file = sys.argv[2]

#--- GRD file for interface topography (should be smoothed to match the SEM mesh resolutin, i.e. smoothed over the length of the element size)
topo_grd = "SETibet_smooth.grd"

#--- reference point (WGS84 ellipsoid)
ref_lon = 103
ref_lat = 27
ref_alt = 0.0
ref_ellps = 'WGS84'

#====== read in gCMT file
with open(cmt_file, 'r') as f:
  lines = [ x for x in f.readlines() if not(x.startswith('#')) ]

header = lines[0].split()
year   = header[1]
month  = header[2]
day    = header[3]
hour   = header[4]
minute = header[5]
second = header[6]

lines = [x.split(":") for x in lines]
event_id = lines[1][1].strip()
time_shift = float(lines[2][1])

tau = float(lines[3][1])
lat = float(lines[4][1])
lon = float(lines[5][1])
dep = float(lines[6][1])

# centroid time: t0
isotime = '{:s}-{:s}-{:s}T{:s}:{:s}:{:s}Z'.format(
    year, month, day, hour, minute, second)
t0 = UTCDateTime(isotime) + time_shift
# modify origin time in header line to have centroid time 
header[1] = "{:04d}".format(t0.year)
header[2] = "{:02d}".format(t0.month)
header[3] = "{:02d}".format(t0.day)
header[4] = "{:02d}".format(t0.hour)
header[5] = "{:02d}".format(t0.minute)
header[6] = "{:07.4f}".format(t0.second + 1.0e-6*t0.microsecond)

# moment tensor in the station ENU basis
# z',-y',x' -> r,theta,phi
mzz = float(lines[7][1])
myy = float(lines[8][1])
mxx = float(lines[9][1])
myz = -1*float(lines[10][1])
mxz = float(lines[11][1])
mxy = -1*float(lines[12][1])
mt = np.array([
  [mxx, mxy, mxz], 
  [mxy, myy, myz], 
  [mxz, myz, mzz]])

#======

#--- get alt(lon,lat) grid from GRD file
fh = Dataset(topo_grd, mode='r')
grd_lon1 = fh.variables['x'][:]
grd_lat1 = fh.variables['y'][:]
grd_alt2 = fh.variables['z'][:]

#--- interpolate surface altitude at each station
f = interpolate.interp2d(grd_lon1, grd_lat1, grd_alt2, kind='cubic', fill_value=np.nan)
alt = f(lon, lat)

#--- convert from geodetic to ECEF coordinates
ecef = pyproj.Proj(proj='geocent', ellps=ref_ellps)
lla = pyproj.Proj(proj='latlong', ellps=ref_ellps)

x0, y0, z0 = pyproj.transform(lla, ecef, ref_lon, ref_lat, ref_alt)
xx, yy, zz = pyproj.transform(lla, ecef, lon, lat, alt - dep*1000.0)

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
# vector basis in station ENU
cosla = np.cos(np.deg2rad(lat))
sinla = np.sin(np.deg2rad(lat))
coslo = np.cos(np.deg2rad(lon))
sinlo = np.sin(np.deg2rad(lon))
vx = np.array([ -sinlo, coslo, 0.0])
vy = np.array([ -sinla*coslo, -sinla*sinlo, cosla])
vz = np.array([ cosla*coslo, cosla*sinlo, sinla])

# vector basis in REF ENU
vx_ref = np.array([ -sinlo_ref, coslo_ref, 0.0])
vy_ref = np.array([ -sinla_ref*coslo_ref, -sinla_ref*sinlo_ref, cosla_ref])
vz_ref = np.array([ cosla_ref*coslo_ref, cosla_ref*sinlo_ref, sinla_ref])

a = np.array(
    [ [ np.dot(vx_ref,vx), np.dot(vx_ref, vy), np.dot(vx_ref, vz) ],
      [ np.dot(vy_ref,vx), np.dot(vy_ref, vy), np.dot(vy_ref, vz) ],
      [ np.dot(vz_ref,vx), np.dot(vz_ref, vy), np.dot(vz_ref, vz) ] ])

mt_ref = np.dot(np.dot(a, mt), np.transpose(a))

#====== save local ENU coordinates of all stations  
with open(out_file, 'w') as fp:
  fp.write('%s\n' % (' '.join(header)))
  fp.write('%-18s %s\n' % ('event name:', event_id))
  fp.write('%-18s %+15.8E\n' % ('time shift:',    0.0))
  fp.write('%-18s %+15.8E\n' % ('half duration:', tau))
  fp.write('%-18s %+15.8E\n' % ('latorUTM:',    nn))
  fp.write('%-18s %+15.8E\n' % ('lonorUTM:',    ee))
  fp.write('%-18s %+15.8E\n' % ('depthorZ:',    uu))
  fp.write('%-18s %+15.8E\n' % ('Mrr(dyn*cm):', mt_ref[2,2]))
  fp.write('%-18s %+15.8E\n' % ('Mtt(dyn*cm):', mt_ref[1,1]))
  fp.write('%-18s %+15.8E\n' % ('Mpp(dyn*cm):', mt_ref[0,0]))
  fp.write('%-18s %+15.8E\n' % ('Mrt(dyn*cm):', -1*mt_ref[1,2]))
  fp.write('%-18s %+15.8E\n' % ('Mrp(dyn*cm):', mt_ref[0,2]))
  fp.write('%-18s %+15.8E\n' % ('Mtp(dyn*cm):', -1*mt_ref[0,1]))
