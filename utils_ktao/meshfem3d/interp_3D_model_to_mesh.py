#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Interpolate 3D model onto mesh grids

The 3D model file: lat lon dep vp vs rho

"""
import sys

import numpy as np
from scipy import interpolate
from scipy.io import FortranFile

import pyproj

#====== parameters

mesh_par_file = str(sys.argv[1])
model_file = str(sys.argv[2])
mesh_dir = str(sys.argv[3])
nproc = int(sys.argv[4])
out_dir = str(sys.argv[5])

#mesh_par_file = "sem_config/DATA/mesh_par.py"
#model_file = "1d_ref/Hua2018.txt"
#mesh_dir = "mesh/DATABASES_MPI"
#out_dir = "model"
#nproc = 110

field_list = [
  ('nspec','i4'), ('nglob','i4'), ('nspec_irregular','i4'), ('ibool','i4'),
  ('x','f4'), ('y','f4'), ('z','f4'),
# ('dxi_dx','f4'),    ('dxi_dy','f4'),    ('dxi_dz','f4'),
# ('deta_dx','f4'),   ('deta_dy','f4'),   ('deta_dz','f4'),
# ('dgamma_dx','f4'), ('dgamma_dy','f4'), ('dgamma_dz','f4'),
# ('jacobian','f4'),
# ('kappa','f4'), ('mu','f4'),
# # logical varaible is read in as integer*4, true=1, false=0
# ('ispec_is_acoustic','i4'), ('ispec_is_elastic','i4'), ('ispec_is_poroelastic','i4'), 
  ]

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

#====== read in 1-D reference model
with open(model_file, 'r') as f:
  lines = [ l.split() for l in f.readlines() if not(l.startswith('#')) ]

model_lat = np.array([float(x[0]) for x in lines])
model_lon = np.array([float(x[1]) for x in lines])
model_dep = np.array([float(x[2]) for x in lines])

npoints = len(model_dep)
model_values = np.zeros((npoints, 3))

model_values[:,0]= np.array([float(x[3]) for x in lines])*1000 # Vp, m/s
model_values[:,1]= np.array([float(x[4]) for x in lines])*1000 # Vs, m/s
model_values[:,2]= np.array([float(x[5]) for x in lines])*1000 # Rho, kg/m^3

#====== interpolate each SEM slice

#--- convert (lon,lat,alt) to ECEF
ecef = pyproj.Proj(proj='geocent', ellps=ref_ellps)
lla = pyproj.Proj(proj='latlong', ellps=ref_ellps)
x0, y0, z0 = pyproj.transform(lla, ecef, ref_lon, ref_lat, ref_alt)

#--- transform from ECEF to REF_ENU
cosla = np.cos(np.deg2rad(ref_lat))
sinla = np.sin(np.deg2rad(ref_lat))
coslo = np.cos(np.deg2rad(ref_lon))
sinlo = np.sin(np.deg2rad(ref_lon))

# vector basis of REF_ENU [Ve; Vn; Vu] in ECEF 
#RotM = np.zeros((3,3))
#RotM[0,:] = [ -sinlo, -sinla*coslo, cosla*coslo ]
#RotM[1,:] = [  coslo, -sinla*sinlo, cosla*sinlo ]
#RotM[2,:] = [      0,        cosla,       sinla ]

for iproc in range(nproc):

  print("====== ", iproc)

  input_file = "%s/proc%06d_external_mesh.bin"%(mesh_dir, iproc)

  #--- read in SEM mesh
  mesh = {}
  
  with FortranFile(input_file, 'r') as f:
    for field in field_list:
      field_name = field[0]
      data_type = field[1]
      mesh[field_name] = f.read_ints(dtype=data_type)
  
  xgll = mesh['x'][mesh['ibool']-1]
  ygll = mesh['y'][mesh['ibool']-1]
  zgll = mesh['z'][mesh['ibool']-1]

  xx = x0 + (-sinlo)*xgll + (-sinla*coslo)*ygll + (cosla*coslo)*zgll
  yy = y0 +  (coslo)*xgll + (-sinla*sinlo)*ygll + (cosla*sinlo)*zgll
  zz = z0                 +        (cosla)*ygll +       (sinla)*zgll

  #--- get lla
  gll_lon, gll_lat, gll_alt = pyproj.transform(ecef, lla, xx, yy, zz)
  gll_depth_km = -1 * gll_alt/1000.0
  
  #--- interpolate
  gll_values = interpolate.griddata((model_lon, model_lat, model_dep), model_values, (gll_lon, gll_lat, gll_depth_km), method='linear')

  #--- output model gll file
  out_file = "%s/proc%06d_vp.bin"%(out_dir, iproc)
  with FortranFile(out_file, 'w') as f:
    f.write_record(np.array(gll_values[:,0], dtype='f4'))

  out_file = "%s/proc%06d_vs.bin"%(out_dir, iproc)
  with FortranFile(out_file, 'w') as f:
    f.write_record(np.array(gll_values[:,1], dtype='f4'))

  out_file = "%s/proc%06d_rho.bin"%(out_dir, iproc)
  with FortranFile(out_file, 'w') as f:
    f.write_record(np.array(gll_values[:,2], dtype='f4'))
