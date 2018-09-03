#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Interpolate 3D model onto mesh grids

The 3D model file: lat lon dep vp vs

"""
import sys
import time

import numpy as np
from scipy.io import FortranFile
from scipy import spatial

from mpi4py import MPI

import pyproj
import importlib

from meshfem3d_utils import rotmat_enu_to_ecef 
from meshfem3d_utils import sem_mesh_read, sem_mesh_get_vol_gll
from meshfem3d_utils import NGLLX,NGLLY,NGLLZ,MIDX,MIDY,MIDZ

from gll_library import zwgljd
from smooth_gauss_cap import smooth_gauss_cap 

#====== parameters

mesh_par_file = str(sys.argv[1])
nproc = int(sys.argv[2])
mesh_dir = str(sys.argv[3]) # <mesh_dir>/proc******_external_mesh.bin
model_dir = str(sys.argv[4]) # <model_dir>/proc******_<model_name>.bin
model_names = str(sys.argv[5]) # comma delimited e.g. vp,vs,rho,qmu,qkappa
sigma_h = float(sys.argv[6]) # e.g. 10000 meters, horizontal smoothing length
sigma_r = float(sys.argv[7]) # e.g. 5000 meters, radial smoothing length
out_dir = str(sys.argv[8])

#--- Gaussian smoothing kernel 
sigma2_h = sigma_h**2
sigma2_r = sigma_r**2
search_radius = 3.0*max(sigma_h, sigma_r) # search neighboring points

#--- model names
model_names = model_names.split(',')
nmodel = len(model_names)

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

ref_rotmat = rotmat_enu_to_ecef(ref_lon,ref_lat)

#--- transform from ECEF to REF_ENU
ecef = pyproj.Proj(proj='geocent', ellps=ref_ellps)
lla = pyproj.Proj(proj='latlong', ellps=ref_ellps)
xyz_ref = np.zeros(3)
xyz_ref[0], xyz_ref[1], xyz_ref[2] = pyproj.transform(lla, ecef, ref_lon, ref_lat, ref_alt)

#====== smooth each target mesh slice
comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()

for iproc_target in range(mpi_rank,nproc,mpi_size):

  print("====== target proc# ", iproc_target)

  #--- read in target SEM mesh
  mesh_file = "%s/proc%06d_external_mesh.bin"%(mesh_dir, iproc_target)
  mesh_target = sem_mesh_read(mesh_file)

  nspec_target = mesh_target['nspec']
  gll_dims_target = mesh_target['gll_dims']

  xyz_elem_target = xyz_ref.reshape((3,1)) + np.dot(ref_rotmat,mesh_target['xyz_elem'])
  xyz_glob_target = xyz_ref.reshape((3,1)) + np.dot(ref_rotmat,mesh_target['xyz_glob'])

  weight_model_gll_target = np.zeros((nmodel,)+gll_dims_target)
  weight_gll_target = np.zeros(gll_dims_target)

  #====== loop over each contribution mesh slice
  for iproc_contrib in range(nproc):

    print("--- contrib proc# ", iproc_contrib)
    sys.stdout.flush()
    #start = time.clock()

    #--- read in contributing SEM mesh
    mesh_file = "%s/proc%06d_external_mesh.bin"%(mesh_dir, iproc_contrib)
    mesh_contrib = sem_mesh_read(mesh_file)

    nspec_contrib = mesh_contrib['nspec']
    gll_dims_contrib = mesh_contrib['gll_dims']

    xyz_elem_contrib = xyz_ref.reshape((3,1)) + np.dot(ref_rotmat,mesh_contrib['xyz_elem'])
    xyz_glob_contrib = xyz_ref.reshape((3,1)) + np.dot(ref_rotmat,mesh_contrib['xyz_glob'])

    vol_gll_contrib = sem_mesh_get_vol_gll(mesh_contrib)

    #--- get neighboring contrib elements
    points_target = np.c_[xyz_elem_target[0,:].ravel(), xyz_elem_target[1,:].ravel(), xyz_elem_target[2,:].ravel()]
    tree_target = spatial.cKDTree(points_target)

    points_contrib = np.c_[xyz_elem_contrib[0,:].ravel(), xyz_elem_contrib[1,:].ravel(), xyz_elem_contrib[2,:].ravel()]
    tree_contrib = spatial.cKDTree(points_contrib)

    neighbor_lists = tree_target.query_ball_tree(tree_contrib, search_radius)

    #skip mesh slice that is too faraway from the target mesh slice
    if not any(neighbor_lists[:]):
      #print("... too faraway, skip this mesh slice")
      continue

    #--- read model values of the contributing mesh slice
    #print("... read model")
    model_gll_contrib = np.zeros((nmodel,)+gll_dims_contrib)
    for imodel in range(nmodel):
      model_tag = model_names[imodel]
      model_file = "%s/proc%06d_%s.bin"%(model_dir, iproc_contrib, model_tag)
      with FortranFile(model_file, 'r') as f:
        model_gll_contrib[imodel,:,:,:,:] = np.reshape(f.read_ints(dtype='f4'), 
                                                       gll_dims_contrib)

    #--- gather contributions for each target gll point
    #print("nspec_target(%d):            "%(nspec_target) )
    #sys.stdout.write("nspec_target(%d):            "%(nspec_target))
    #sys.stdout.flush()

    ispec_target_list = [ ispec for ispec in range(nspec_target) if neighbor_lists[ispec] ]
    #for ispec_target in range(nspec_target):
    for ispec_target in ispec_target_list:

      #print("\b\b\b\b\b\b\b\b\b%09d"%(ispec_target), end="")
      #sys.stdout.write("\b\b\b\b\b\b\b\b\b%09d"%(ispec_target))
      #print("... ispec_target: ", ispec_target)
      #sys.stdout.flush()

      iglob_target = mesh_target['ibool'][:,:,:,ispec_target].ravel() - 1
      ngll_target = len(iglob_target)
      xyz_gll_target = xyz_glob_target[:,iglob_target]

      ielem_contrib = neighbor_lists[ispec_target]
      #print("nelem_contrib: ", len(ielem_contrib))

      iglob_contrib = mesh_contrib['ibool'][:,:,:,ielem_contrib].ravel() - 1
      ngll_contrib = len(iglob_contrib)
      xyz_gll_contrib = xyz_glob_contrib[:,iglob_contrib]

      #DEBUG
      #max_dist = np.max(np.sum((xyz_gll_contrib - xyz_elem_target[:,ispec_target].reshape((3,1)))**2,axis=0)**0.5)
      #print('max_dist, search_radius: ', max_dist, search_radius)
      #max_dist = np.max(np.sum((xyz_elem_contrib[:,ielem_contrib] - xyz_elem_target[:,ispec_target].reshape((3,1)))**2,axis=0)**0.5)
      #print('max_dist, search_radius: ', max_dist, search_radius)

      model_gll = model_gll_contrib[:,:,:,:,ielem_contrib].reshape((nmodel,-1))
      vol_gll = vol_gll_contrib[:,:,:,ielem_contrib].ravel()

      #print(ngll_target, ngll_contrib)
      weight_model_val = np.empty((nmodel,NGLLX,NGLLY,NGLLZ))
      weight_val = np.empty((NGLLX,NGLLY,NGLLZ))
      weight_model_val, weight_val = smooth_gauss_cap(xyz_gll_target, xyz_gll_contrib, 
                                                      model_gll, vol_gll, 
                                                      sigma2_h, sigma2_r)

      weight_model_gll_target[:,:,:,:,ispec_target] += weight_model_val.reshape((nmodel,NGLLX,NGLLY,NGLLZ))
      weight_gll_target[:,:,:,ispec_target] += weight_val.reshape((NGLLX,NGLLY,NGLLZ))

      #r_gll_target = np.sum(xyz_gll_target**2, axis=0)**0.5
      #ngll_target = len(r_gll_target)
      #nelem_contrib = len(ielem_contrib)
      #r_gll_contrib = (np.sum(xyz_gll_contrib**2, axis=0))**0.5
      #ngll_contrib = len(r_gll_contrib)
      #sigma2_theta = (sigma2_h/r_gll_target**2).reshape((ngll_target,1))
      #dist2_radial = (r_gll_contrib.reshape((1,ngll_contrib)) - r_gll_target.reshape((ngll_target,1)))**2
      #xyz_gll_target /= r_gll_target
      #xyz_gll_contrib /= r_gll_contrib
      #iprod = np.sum(xyz_gll_contrib.reshape((3,1,ngll_contrib)) * xyz_gll_target.reshape((3,ngll_target,1)), axis=0)
      #iprod[iprod>1] = 1.0
      #dist2_theta = np.arccos(iprod)
      #weight = np.exp(-0.5*dist2_radial/sigma2_r) * np.exp(-0.5*dist2_theta/sigma2_theta) * vol_gll.reshape((1,ngll_contrib))
      #tmp = np.sum(weight*model_gll.reshape((nmodel,1,ngll_contrib)), axis=2)
      #model_gll_target[:,:,:,:,ispec_target] += tmp.reshape((nmodel,NGLLX,NGLLY,NGLLZ))
      #weight_gll_target[:,:,:,ispec_target] += np.sum(weight,axis=1).reshape((NGLLX,NGLLY,NGLLZ))

    #END--- gather contributions for each target gll point  
    #print("")
    #print("time used(sec): ", time.clock() - start)

  #END====== loop over each contributing mesh slice

  #--- weighted average model values on each target gll point
  weight_model_gll_target /= weight_gll_target
 
  #--- output model gll file
  for imodel in range(nmodel):
    model_tag = model_names[imodel]
    out_file = "%s/proc%06d_%s.bin"%(out_dir, iproc_target, model_tag)
    with FortranFile(out_file, 'w') as f:
      out_data = np.ravel(weight_model_gll_target[imodel,:], order='F') # Fortran column-major
      f.write_record(np.array(out_data, dtype='f4'))
