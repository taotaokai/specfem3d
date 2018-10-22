#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" oLBFGS
"""
import sys

import numpy as np
from scipy.io import FortranFile
from mpi4py import MPI

from meshfem3d_utils import sem_mesh_read
               
#====== user input
nproc = int(sys.argv[1])
mesh_dir = str(sys.argv[2])
kernel_dir = str(sys.argv[3])
kernel_tags = str(sys.argv[4]) # e.g. alpha_kernel,beta_kernel,rhop_kernel
maxamp_dmodel = float(sys.argv[5]) # maximum amplitude of dmodel 
out_dir = str(sys.argv[6])
dmodel_tags = str(sys.argv[7]) # e.g. alpha_dmodel,beta_dmodel,rhop_dmodel

kernel_tags = kernel_tags.split(',')
dmodel_tags = dmodel_tags.split(',')
nmodel = len(dmodel_tags)
if len(kernel_tags) != nmodel:
  print("ERROR: kernel_tags, dmodel_tags must have the same length")
  sys.exit(-1)

#--- initialize MPI comm
comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()

#--- read in mesh
mesh_file = "%s/proc%06d_external_mesh.bin"%(mesh_dir, mpi_rank)
meshdb = sem_mesh_read(mesh_file)
gll_dims = meshdb['gll_dims']
ngll = np.prod(gll_dims)

#--- read gradient
kernel = np.empty((nmodel,ngll))
for imodel in range(nmodel):
  kernel_file = "%s/proc%06d_%s.bin"%(kernel_dir, mpi_rank, kernel_tags[imodel])
  with FortranFile(kernel_file, 'r') as f:
    kernel[imodel,:] = f.read_ints(dtype='f4')

max_val = np.max(np.abs(kernel))
max_val = comm.allreduce(max_val, op=MPI.MAX)
scale_factor = maxamp_dmodel/max_val

if mpi_rank == 0:
  print("scale_factor:", scale_factor)

#====== output dmodel
for imodel in range(nmodel):
  out_file = "%s/proc%06d_%s.bin"%(out_dir, mpi_rank, dmodel_tags[imodel])
  with FortranFile(out_file, 'w') as f:
    f.write_record(np.array(scale_factor*kernel[imodel,:], dtype='f4'))
