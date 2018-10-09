#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sum up event kernels
"""
import sys
import time

import numpy as np
from scipy.io import FortranFile

import pyproj
from meshfem3d_utils import rotmat_enu_to_ecef, sem_mesh_read

ts0 = time.time()

#====== parameters
mesh_par_file = str(sys.argv[1])
nproc = int(sys.argv[2])
mesh_dir = str(sys.argv[3]) # <mesh_dir>/proc******_external_mesh.bin
mask_fname = str(sys.argv[4]) # a list consists of x,y,z,sigma_H,sigma_V
out_dir = str(sys.argv[5])

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

ref_rotmat = rotmat_enu_to_ecef(ref_lon,ref_lat)

#--- transform from ECEF to REF_ENU
ecef = pyproj.Proj(proj='geocent', ellps=ref_ellps)
lla = pyproj.Proj(proj='latlong', ellps=ref_ellps)
xyz_ref = np.zeros(3)
xyz_ref[0], xyz_ref[1], xyz_ref[2] = pyproj.transform(lla, ecef, ref_lon, ref_lat, ref_alt)

#--- read in mask_list
with open(mask_fname, 'r') as f:
  lines = [l.split() for l in f.readlines() if not l.startswith('#')]
nmask = len(lines)
xyz_mask = np.zeros((3,nmask))
xyz_mask[0,:] = np.array([float(l[0]) for l in lines])
xyz_mask[1,:] = np.array([float(l[1]) for l in lines])
xyz_mask[2,:] = np.array([float(l[2]) for l in lines])
sigmaH_mask = np.array([float(l[3]) for l in lines])
sigmaV_mask = np.array([float(l[4]) for l in lines])

# convert to ECEF frame
xyz_mask = xyz_ref.reshape((3,1)) + np.dot(ref_rotmat, xyz_mask)
r_mask = (np.sum(xyz_mask**2, axis=0))**0.5
xyz_mask /= r_mask
sigma2_theta = (sigmaH_mask/r_mask)**2

#--- loop over each mesh slice
for iproc in range(nproc):

  print("====== proc# ", iproc)
  sys.stdout.flush()

  #--- read in mesh
  mesh_file = "%s/proc%06d_external_mesh.bin"%(mesh_dir, iproc)
  meshdb = sem_mesh_read(mesh_file)

  nspec = meshdb['nspec']
  nglob = meshdb['nglob']
  gll_dims = meshdb['gll_dims']

  xyz_glob = xyz_ref.reshape((3,1)) + np.dot(ref_rotmat, meshdb['xyz_glob'])
  ibool = meshdb['ibool'].ravel(order='F') - 1 # flatten in Fortran column-major
  ngll = len(ibool)
  xyz_gll = xyz_glob[:,ibool]
  r_gll = np.sum(xyz_gll**2, axis=0)**0.5
  xyz_gll /= r_gll

  mask = np.ones(ibool.shape)

  for imask in range(nmask):
    dist2_radial = (r_gll - r_mask[imask])**2 # radial distance
    iprod = np.sum(xyz_mask[:,imask].reshape((3,1))*xyz_gll, axis=0)
    iprod[iprod>1] = 1.0
    dist2_theta = np.arccos(iprod)**2 # angle distance
    expterm = 0.5*(dist2_radial/sigmaV_mask[imask]**2 + dist2_theta/sigma2_theta[imask])
    weight = np.ones(ibool.shape)
    idx = expterm <= 1e-5
    weight[idx] = 0
    idx = (expterm>1e-5) & (expterm<1e1)
    weight[idx] = 1 - np.exp(-1*expterm[idx])
    mask *= weight

  #--- output mask file
  out_file = "%s/proc%06d_mask.bin"%(out_dir, iproc)
  with FortranFile(out_file, 'w') as f:
    f.write_record(np.array(mask, dtype='f4'))

ts1 = time.time()
print("time elapse: ", ts1-ts0)

