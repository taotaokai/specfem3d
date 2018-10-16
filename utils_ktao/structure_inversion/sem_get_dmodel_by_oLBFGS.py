#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" oLBFGS
"""
import sys

import numpy as np
from scipy.io import FortranFile
from mpi4py import MPI

from meshfem3d_utils import sem_mesh_read, sem_mesh_get_vol_gll
               
#====== user input
nproc = int(sys.argv[1])
mesh_dir = str(sys.argv[2])
kernel_dir = str(sys.argv[3])
kernel_tags = str(sys.argv[4]) # e.g. alpha_dkernel,beta_dkernel,rhop_dkernel
dmodel_dkernel_dir_list = str(sys.argv[5]) # list of dmodel_dir dkernel_dir from oldest to newest iterations
dmodel_tags = str(sys.argv[6]) # e.g. alpha_dmodel,beta_dmodel,rhop_dmodel
dkernel_tags = str(sys.argv[7]) # e.g. alpha_dkernel,beta_dkernel,rhop_dkernel
out_dir = str(sys.argv[8])

kernel_tags = kernel_tags.split(',')
dmodel_tags = dmodel_tags.split(',')
dkernel_tags = dkernel_tags.split(',')
nmodel = len(dmodel_tags)
if len(dkernel_tags) != nmodel or len(kernel_tags) != nmodel:
  print("ERROR: kernel_tags, dmodel_tags and dkernel_tags must have the same length")
  sys.exit(-1)

# read dmodel_dkernel_dir_list
with open(dmodel_dkernel_dir_list, 'r') as f:
  lines = [ l.split() for l in f.readlines() if not l.startswith('#') ]
dmodel_dirs = [ l[0] for l in lines ]
dkernel_dirs = [ l[1] for l in lines ]
nstep = len(dmodel_dirs)

#--- initialize MPI comm
comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()

#--- read in mesh
mesh_file = "%s/proc%06d_external_mesh.bin"%(mesh_dir, mpi_rank)
meshdb = sem_mesh_read(mesh_file)
nspec = meshdb['nspec']
gll_dims = meshdb['gll_dims']
vol_gll = sem_mesh_get_vol_gll(meshdb)

#--- read current gradient
kernel = np.empty((nmodel,) + gll_dims)
for imodel in range(nmodel):
  kernel_file = "%s/proc%06d_%s.bin"%(kernel_dir, mpi_rank, kernel_tags[imodel])
  with FortranFile(kernel_file, 'r') as f:
    kernel[imodel,:,:,:,:] = np.reshape(f.read_ints(dtype='f4'), gll_dims, order='F')

#--- two-loop LBFGS
dmodel_dkernel = np.zeros(nstep)
dkernel_dkernel = np.zeros(nstep)
alpha = np.zeros(nstep)

dmodel = np.empty((nmodel,) + gll_dims)
dkernel = np.empty((nmodel,) + gll_dims)
# loop-1
for istep in range(nstep-1,-1,-1):
  for imodel in range(nmodel):
    dmodel_file = "%s/proc%06d_%s.bin"%(dmodel_dirs[istep], mpi_rank, dmodel_tags[imodel])
    with FortranFile(dmodel_file, 'r') as f:
      dmodel[imodel,:,:,:,:] = np.reshape(f.read_ints(dtype='f4'), gll_dims, order='F')
    dkernel_file = "%s/proc%06d_%s.bin"%(dkernel_dirs[istep], mpi_rank, dkernel_tags[imodel])
    with FortranFile(dkernel_file, 'r') as f:
      dkernel[imodel,:,:,:,:] = -1*np.reshape(f.read_ints(dtype='f4'), gll_dims, order='F')

  dmodel_kernel_local = np.sum(dmodel * kernel * vol_gll)
  dmodel_dkernel_local = np.sum(dmodel * dkernel * vol_gll)
  dkernel_dkernel_local = np.sum(dkernel**2 * vol_gll)
  dmodel_dmodel_local = np.sum(dmodel**2 * vol_gll)
 
  dmodel_kernel = comm.allreduce(dmodel_kernel_local, op=MPI.SUM)
  dmodel_dkernel[istep] = comm.allreduce(dmodel_dkernel_local, op=MPI.SUM)
  dkernel_dkernel[istep] = comm.allreduce(dkernel_dkernel_local, op=MPI.SUM)
  dmodel_dmodel = comm.allreduce(dmodel_dmodel_local, op=MPI.SUM)

  alpha[istep] = dmodel_kernel/dmodel_dkernel[istep]
  kernel = kernel - alpha[istep]*dkernel

  if mpi_rank == 0:
    print("istep, dmodel_dkernel:", istep, dmodel_dkernel[istep]/dmodel_dmodel**0.5/dkernel_dkernel[istep]**0.5)

# apply step-length
step_length = np.sum(dmodel_dkernel/dkernel_dkernel)/nstep
kernel = step_length*kernel

# loop-2
for istep in range(nstep):
  for imodel in range(nmodel):
    dmodel_file = "%s/proc%06d_%s.bin"%(dmodel_dirs[istep], mpi_rank, dmodel_tags[imodel])
    with FortranFile(dmodel_file, 'r') as f:
      dmodel[imodel,:,:,:,:] = np.reshape(f.read_ints(dtype='f4'), gll_dims, order='F')
    dkernel_file = "%s/proc%06d_%s.bin"%(dkernel_dirs[istep], mpi_rank, dkernel_tags[imodel])
    with FortranFile(dkernel_file, 'r') as f:
      dkernel[imodel,:,:,:,:] = -1*np.reshape(f.read_ints(dtype='f4'), gll_dims, order='F')

  dkernel_kernel_local = np.sum(dkernel * kernel * vol_gll)
  dkernel_kernel = comm.allreduce(dkernel_kernel_local, op=MPI.SUM)

  beta =  dkernel_kernel/dmodel_dkernel[istep]
  kernel = kernel + (alpha[istep] - beta)*dmodel

#====== output dmodel
for imodel in range(nmodel):
  out_file = "%s/proc%06d_%s.bin"%(out_dir, mpi_rank, dmodel_tags[imodel])
  with FortranFile(out_file, 'w') as f:
    f.write_record(np.array(np.ravel(kernel[imodel,:,:,:,:], order='F'), dtype='f4'))
  #INFO
  kernel_max_local = np.max(kernel[imodel,:,:,:,:])
  kernel_min_local = np.min(kernel[imodel,:,:,:,:])
  kernel_max = comm.reduce(kernel_max_local,op=MPI.MAX,root=0)
  kernel_min = comm.reduce(kernel_min_local,op=MPI.MIN,root=0)
  if mpi_rank == 0:
    print("dmodel_tag, min/max:", dmodel_tags[imodel], kernel_min, kernel_max)
