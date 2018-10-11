#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sum up event kernels
"""
import sys

import numpy as np
from scipy.io import FortranFile

#from mpi4py import MPI

from meshfem3d_utils import sem_mesh_read
#from meshfem3d_utils import NGLLX,NGLLY,NGLLZ,MIDX,MIDY,MIDZ

#====== parameters
nproc = int(sys.argv[1])
mesh_dir = str(sys.argv[2]) # <mesh_dir>/proc******_external_mesh.bin
event_dir_list = str(sys.argv[3]) # a list of <kernel_dir> under which proc******_<model_name>.bin exist
mask_fname = str(sys.argv[4]) # a list under <kernel_dir> consists of x,y,z,sigma_H,sigma_V
model_names = str(sys.argv[5]) # comma delimited e.g. vp,vs,rho,qmu,qkappa
out_dir = str(sys.argv[6])

#--- model names
model_names = model_names.split(',')
nmodel = len(model_names)

#======  loop over each mesh slice
#comm = MPI.COMM_WORLD
#mpi_size = comm.Get_size()
#mpi_rank = comm.Get_rank()

# read in model_dir_list
with open(event_dir_list, 'r') as f:
  event_dirs = [l.split()[0] for l in f.readlines() if not l.startswith('#')]
nevent = len(event_dirs)

#for iproc in range(mpi_rank,nproc,mpi_size):
for iproc in range(nproc):

  print("====== proc# ", iproc)
  sys.stdout.flush()

  #--- read in mesh
  mesh_file = "%s/proc%06d_external_mesh.bin"%(mesh_dir, iproc)
  meshdb = sem_mesh_read(mesh_file)

  nspec = meshdb['nspec']
  gll_dims = meshdb['gll_dims']

# #--- read in model_dir_list
# if mpi_rank == 0:
#   with open(event_dir_list, 'r') as f:
#     event_dirs = [l.split()[0] for l in f.readlines() if not l.startswith('#')]
# else:
#   event_dirs = None
# event_dirs = comm.bcast(event_dirs, root=0)
# nevent = len(event_dirs)
# print(event_dirs)
# print(nevent)

  #--- sum over each model_dirs
  model_gll_sum = np.zeros((nmodel,)+gll_dims)
  for ievent in range(nevent):

    print("--- event dir: %s"%(event_dirs[ievent]))
    sys.stdout.flush()

    #--- read in mask_list
    with open("%s/%s"%(event_dirs[ievent],mask_fname), 'r') as f:
      lines = [l.split() for l in f.readlines() if not l.startswith('#')]
    npts_mask = len(lines)
    mask_xyz = np.zeros((npts_mask,3))
    mask_xyz[:,0] = np.array([float(l[0]) for l in lines])
    mask_xyz[:,1] = np.array([float(l[1]) for l in lines])
    mask_xyz[:,2] = np.array([float(l[2]) for l in lines])
    mask_sigmaH = np.array([float(l[3]) for l in lines])
    mask_sigmaV = np.array([float(l[4]) for l in lines])

    mesh_xyz = meshdb['xyz_glob']
    nglob = meshdb['nglob']
    dxyz = mesh_xyz.reshape((nglob,1,3)) - mask_xyz.reshape((1,npts_mask,3))

    mask = np.exp()

    #--- read model values of the contributing event_dir
    for imodel in range(nmodel):
      model_tag = model_names[imodel]
      model_file = "%s/proc%06d_%s.bin"%(event_dirs[ievent], iproc, model_tag)
      with FortranFile(model_file, 'r') as f:
        # note: must use fortran convention when reshape to N-D array!!! 
        model_gll_sum[imodel,:,:,:,:] += np.reshape(f.read_ints(dtype='f4'), gll_dims, order='F')

  #--- output summed model gll file
  for imodel in range(nmodel):
    model_tag = model_names[imodel]
    out_file = "%s/proc%06d_%s.bin"%(out_dir, iproc, model_tag)
    with FortranFile(out_file, 'w') as f:
      out_data = np.ravel(model_gll_sum[imodel,:], order='F') # Fortran column-major
      f.write_record(np.array(out_data, dtype='f4'))
