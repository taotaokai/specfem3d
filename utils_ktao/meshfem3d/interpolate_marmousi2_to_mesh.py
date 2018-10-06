#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import numpy as np
from scipy import interpolate
from scipy.io import FortranFile

from obspy.io.segy.segy import _read_segy

mesh_dir = "DATABASES_MPI"
out_dir = "model"
nproc = 255

field_list = [
  ('nspec','i4'), ('nglob','i4'), ('ibool','i4'),
  ('x','f4'), ('y','f4'), ('z','f4'),
# ('dxi_dx','f4'),    ('dxi_dy','f4'),    ('dxi_dz','f4'),
# ('deta_dx','f4'),   ('deta_dy','f4'),   ('deta_dz','f4'),
# ('dgamma_dx','f4'), ('dgamma_dy','f4'), ('dgamma_dz','f4'),
# ('jacobian','f4'),
# ('kappa','f4'), ('mu','f4'),
# # logical varaible is read in as integer*4, true=1, false=0
# ('ispec_is_acoustic','i4'), ('ispec_is_elastic','i4'), ('ispec_is_poroelastic','i4'), 
  ]

#====== read in Marmousi2 model
segy = _read_segy("vp_marmousi-ii.segy")

nx = len(segy.traces)
tr = segy.traces[0]
nz = tr.npts
dx = 1.25
dz = 1.25

vp = np.zeros((nx,nz))
for i in range(nx):
  vp[i,:] = segy.traces[i].data

x = np.arange(0, nx)*dx
z = np.arange(0, nz)*dz

#--- make interpolation function
interp_vp = interpolate.RectBivariateSpline(x, z, vp)

#====== interpolate each SEM slice
for iproc in range(nproc):

  print("====== ", iproc)

  input_file = "%s/proc%06d_external_mesh.bin"%(mesh_dir, iproc)
  out_file = "%s/proc%06d_vp.bin"%(out_dir, iproc)

  #--- read in SEM mesh
  mesh = {}
  
  with FortranFile(input_file, 'r') as f:
    for field in field_list:
      field_name = field[0]
      data_type = field[1]
      mesh[field_name] = f.read_ints(dtype=data_type)
  
  #--- interpolate
  vp_gll = np.zeros(mesh['ibool'].shape)
  
  xgll = mesh['x'][mesh['ibool']-1]
  zgll = mesh['z'][mesh['ibool']-1]
  
  vp_gll = interp_vp(xgll, 0.0-zgll, grid=False)
  
  #--- output model gll file
  with FortranFile(out_file, 'w') as f:
    f.write_record(np.array(vp_gll, dtype='f4'))
