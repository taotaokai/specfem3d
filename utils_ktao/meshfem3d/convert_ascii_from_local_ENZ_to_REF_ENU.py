#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import numpy as np
#
import pyproj
#


from obspy import UTCDateTime
from obspy.io.sac import SACTrace

from meshfem3d_utils import rotmat_enu_to_ecef

#header = {'kstnm': 'ANMO', 'kcmpnm': 'BHZ', 'stla': 40.5, 'stlo': -108.23,
#...           'evla': -15.123, 'evlo': 123, 'evdp': 50, 'nzyear': 2012,
#...           'nzjday': 123, 'nzhour': 13, 'nzmin': 43, 'nzsec': 17,
#...           'nzmsec': 100, 'delta': 1.0/40}
#>>> sac = SACTrace(data=np.random.random(100), **header)
#>>> sac.write(filename, byteorder='little')
# band_code = 'BX?.semd'

#====== read user inputs
mesh_par_file = str(sys.argv[1])
forcesolution_file = str(sys.argv[2]) # need source origin time and lat/lon
station_file = str(sys.argv[3]) # need station latitude and longitude
band_code = str(sys.argv[4])
input_dir = str(sys.argv[5])
out_dir = str(sys.argv[6])

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
#ref_alt   =  par.mesh_ref_alt   
ref_ellps =  par.mesh_ref_ellps 

# origin time (for noise correlation use the same one as you set in those correlation sac files)
#origin_time = UTCDateTime(origin_time)

#====== read FORCESOLUTION
with open(forcesolution_file, 'r') as f:
  lines = [ x.replace('\n','') for x in f.readlines() if not(x.startswith('#')) ]

header = lines[0].split()
evnm = header[0]
evyear = header[1] 
evmonth = header[2] 
evday = header[3] 
evhour = header[4] 
evmin = header[5] 
evsec = header[6] 
origin_time = UTCDateTime("%s-%s-%sT%s:%s:%s"%(evyear,evmonth,evday,evhour,evmin,evsec))

lines = [x.split(":") for x in lines]
time_shift = float(lines[1][1])
f0 = float(lines[2][1])
evla = float(lines[3][1])
evlo = float(lines[4][1])
evdp = float(lines[5][1])

#====== read STATIONS
with open(station_file, 'r') as f:
  lines = [ x.replace('\n','').split() for x in f.readlines() if not(x.startswith('#')) ]

netwk = [x[0] for x in lines]
stnm = [x[1] for x in lines]
stla = np.array([float(x[2]) for x in lines])
stlo = np.array([float(x[3]) for x in lines])
stdp = np.array([float(x[5]) for x in lines])

#====== read in ascii files

# convert from geodetic to ECEF coordinates
ecef = pyproj.Proj(proj='geocent', ellps=ref_ellps)
lla = pyproj.Proj(proj='latlong', ellps=ref_ellps)

nstn = len(stnm)

for i in range(nstn):

  #--- read in seismogram (local ENZ) 

  # E-component
  cmpnm = band_code[0:2] + 'E' + band_code[3:]
  ascii_file = "%s/%s.%s.%s" % (input_dir,netwk[i],stnm[i],cmpnm)
  try:
    with open(ascii_file, 'r') as f:
      lines = [ x.split() for x in f.readlines() if not(x.startswith('#')) ]
  except:
    print('cannot open', ascii_file)
    continue
  times = np.array([float(x[0]) for x in lines])
  data = np.array([float(x[1]) for x in lines])

  nt = len(times)
  seis = np.zeros((3,nt))
  seis[0,:] = data[:]

  # N-component
  cmpnm = band_code[0:2] + 'N' + band_code[3:]
  ascii_file = "%s/%s.%s.%s" % (input_dir,netwk[i],stnm[i],cmpnm)
  with open(ascii_file, 'r') as f:
    lines = [ x.split() for x in f.readlines() if not(x.startswith('#')) ]
  data = np.array([float(x[1]) for x in lines])
  seis[1,:] = data[:]

  # Z-component
  cmpnm = band_code[0:2] + 'Z' + band_code[3:]
  ascii_file = "%s/%s.%s.%s" % (input_dir,netwk[i],stnm[i],cmpnm)
  with open(ascii_file, 'r') as f:
    lines = [ x.split() for x in f.readlines() if not(x.startswith('#')) ]
  data = np.array([float(x[1]) for x in lines])
  seis[2,:] = data[:]

  #--- rotate from local_ENZ to REF_ENU
  rotmat_ref = rotmat_enu_to_ecef(ref_lon, ref_lat)
  rotmat_local = rotmat_enu_to_ecef(stlo[i], stla[i])

  # local_ENZ -> ECEF -> REF_ENU
  seis = np.dot(np.transpose(rotmat_ref), np.dot(rotmat_local, seis))

  #--- write out ascii files 
  data = np.zeros((nt,2))
  data[:,0] = times
  cmp_list = ['X','Y','Z']
  for icmp in range(3):
    cmpnm = band_code[0:2] + cmp_list[icmp] + band_code[3:]
    ascii_file = "%s/%s.%s.%s" % (out_dir,netwk[i],stnm[i],cmpnm)
    data[:,1] = seis[icmp,:]
    np.savetxt(ascii_file, data, fmt="%+16.9e", delimiter="  ")