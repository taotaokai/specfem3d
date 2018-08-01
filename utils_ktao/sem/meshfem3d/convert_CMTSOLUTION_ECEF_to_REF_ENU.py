#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""convert CMTSOLUTION from ECEF coordinate to ENU coodinate
1. convert location from XYZ to ENU
2. convert vector basis from XYZ to ENU
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
#
from meshfem3d_utils import rotmat_enu_to_ecef

#import matplotlib
##matplotlib.use("pdf")
#import matplotlib.pyplot as plt

import importlib

#====== parameters
mesh_par_file = str(sys.argv[1])
cmt_file = str(sys.argv[2])
out_file = str(sys.argv[3])

##--- reference point (WGS84 ellipsoid)
#ref_lon = 103
#ref_lat = 27
#ref_alt = 0.0
#ref_ellps = 'WGS84'

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

hdur = float(lines[3][1])
src_x = float(lines[4][1])
src_y = float(lines[5][1])
src_z = float(lines[6][1])

# centroid time: t0
isotime = '{:s}-{:s}-{:s}T{:s}:{:s}:{:s}Z'.format(year, month, day, hour, minute, second)
t0 = UTCDateTime(isotime) + time_shift
# modify origin time in header line to have centroid time 
header[1] = "{:04d}".format(t0.year)
header[2] = "{:02d}".format(t0.month)
header[3] = "{:02d}".format(t0.day)
header[4] = "{:02d}".format(t0.hour)
header[5] = "{:02d}".format(t0.minute)
header[6] = "{:07.4f}".format(t0.second + 1.0e-6*t0.microsecond)

# moment tensor in ECEF
mxx = float(lines[7][1])
myy = float(lines[8][1])
mzz = float(lines[9][1])
mxy = float(lines[10][1])
mxz = float(lines[11][1])
myz = float(lines[12][1])

mt = np.array([
  [mxx, mxy, mxz], 
  [mxy, myy, myz], 
  [mxz, myz, mzz]])

#====== transform

ecef = pyproj.Proj(proj='geocent', ellps=ref_ellps)
lla = pyproj.Proj(proj='latlong', ellps=ref_ellps)

# ref ecef
x0, y0, z0 = pyproj.transform(lla, ecef, ref_lon, ref_lat, ref_alt)

# source location in ref_enu
dxyz = np.array([src_x, src_y, src_z]) - np.array([x0, y0, z0])

rotmat = rotmat_enu_to_ecef(ref_lon, ref_lat)

enu = np.dot(np.transpose(rotmat), dxyz)

# moment tensor in ref_enu
mt_ref_enu = np.dot(np.dot(np.transpose(rotmat), mt), rotmat)
mt_ref_enu = mt_ref_enu * 1.0e+7 # from N*m to dyn*cm

#====== save local ENU coordinates of all stations  
# radial -> z, theta -> -y, phi -> x
with open(out_file, 'w') as fp:
  fp.write('%s\n' % (' '.join(header)))
  fp.write('%-18s %s\n' % ('event name:', event_id))
  fp.write('%-18s %+15.8E\n' % ('time shift:',    0.0))
  fp.write('%-18s %+15.8E\n' % ('half duration:', hdur))
  fp.write('%-18s %+15.8E\n' % ('latorUTM:',    enu[1]))
  fp.write('%-18s %+15.8E\n' % ('lonorUTM:',    enu[0]))
  fp.write('%-18s %+15.8E\n' % ('depthorZ:',    enu[2]))
  fp.write('%-18s %+15.8E\n' % ('Mrr(dyn*cm):',    mt_ref_enu[2,2]))
  fp.write('%-18s %+15.8E\n' % ('Mtt(dyn*cm):',    mt_ref_enu[1,1]))
  fp.write('%-18s %+15.8E\n' % ('Mpp(dyn*cm):',    mt_ref_enu[0,0]))
  fp.write('%-18s %+15.8E\n' % ('Mrt(dyn*cm):', -1*mt_ref_enu[1,2]))
  fp.write('%-18s %+15.8E\n' % ('Mrp(dyn*cm):',    mt_ref_enu[0,2]))
  fp.write('%-18s %+15.8E\n' % ('Mtp(dyn*cm):', -1*mt_ref_enu[0,1]))
