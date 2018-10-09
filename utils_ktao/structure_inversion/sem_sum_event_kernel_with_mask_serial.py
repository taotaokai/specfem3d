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
event_dir_list = str(sys.argv[3]) # a list of <event_dir> under which proc******_<model_name>.bin exist
model_names = str(sys.argv[4]) # comma delimited e.g. vp,vs,rho,qmu,qkappa
mask_tag = str(sys.argv[5]) # mask file tag: <event_dir>/proc******_<mask_tag>.bin
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

  #--- sum over each model_dirs
  ngll = np.prod(gll_dims)
  model_gll_sum = np.zeros((nmodel,ngll))
  for ievent in range(nevent):

    print("--- event dir: %s"%(event_dirs[ievent]))
    sys.stdout.flush()

    # read mask
    mask_file = "%s/proc%06d_%s.bin"%(event_dirs[ievent], iproc, mask_tag)
    with FortranFile(mask_file, 'r') as f:
      mask = f.read_ints(dtype='f4')

    #--- read model values of the contributing event_dir
    for imodel in range(nmodel):
      model_tag = model_names[imodel]
      model_file = "%s/proc%06d_%s.bin"%(event_dirs[ievent], iproc, model_tag)
      with FortranFile(model_file, 'r') as f:
        model_gll_sum[imodel,:] += mask*f.read_ints(dtype='f4')

  #--- output summed model gll file
  for imodel in range(nmodel):
    model_tag = model_names[imodel]
    out_file = "%s/proc%06d_%s.bin"%(out_dir, iproc, model_tag)
    with FortranFile(out_file, 'w') as f:
      f.write_record(np.array(model_gll_sum[imodel,:], dtype='f4'))
