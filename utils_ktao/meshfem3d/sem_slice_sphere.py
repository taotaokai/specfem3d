#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" create horizontal slice of SEM model at a given depth
"""
import sys
import time

import numpy as np
from scipy.io import FortranFile

from mpi4py import MPI

import pyproj
from netCDF4 import Dataset

from meshfem3d_utils import rotmat_enu_to_ecef 
from meshfem3d_utils import sem_mesh_read, sem_locate_points_hex8 

#====== parameters

mesh_par_file = str(sys.argv[1])
nproc = int(sys.argv[2])
mesh_dir = str(sys.argv[3]) # <mesh_dir>/proc******_external_mesh.bin
model_dir = str(sys.argv[4]) # <model_dir>/proc******_<model_name>.bin
model_names = str(sys.argv[5]) # comma delimited e.g. vp,vs,rho,qmu,qkappa
model_units = str(sys.argv[6]) # comma delimited e.g. km.s-1,km.s-1,kg.m-3,count,count
min_lat = float(sys.argv[7])
max_lat = float(sys.argv[8])
nlat = int(sys.argv[9])
min_lon = float(sys.argv[10])
max_lon = float(sys.argv[11])
nlon = int(sys.argv[12])
depth_km = float(sys.argv[13])
out_file = str(sys.argv[14])

# model names
model_names = model_names.split(',')
model_units = model_units.split(',')
nmodel = len(model_names)

# load mesh parameter file
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

#--- create slice grid
grd_lon1 = np.linspace(min_lon, max_lon, nlon)
grd_lat1 = np.linspace(min_lat, max_lat, nlat)
grd_lon2, grd_lat2 = np.meshgrid(grd_lon1, grd_lat1)
grd_alt2 = np.ones(grd_lon2.shape) * -1000.0 * depth_km
# convert (lon,lat,alt) to ECEF
ecef = pyproj.Proj(proj='geocent', ellps=ref_ellps)
lla = pyproj.Proj(proj='latlong', ellps=ref_ellps)
x0, y0, z0 = pyproj.transform(lla, ecef, ref_lon, ref_lat, ref_alt)
xx, yy, zz = pyproj.transform(lla, ecef, grd_lon2, grd_lat2, grd_alt2)
# transform from ECEF to REF_ENU
dxyz = np.vstack((xx.ravel()-x0, yy.ravel()-y0, zz.ravel()-z0))
grd_enu = np.dot(np.transpose(ref_rotmat), dxyz)
npoints = grd_enu.shape[1]

#====== interpolate
comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()

misloc = np.zeros(npoints)
model_interp = np.zeros((nmodel,npoints))

misloc[:] = np.inf
model_interp[:] = np.nan

#--- loop over each SEM mesh
for iproc in range(mpi_rank,nproc,mpi_size):

  print("====== proc# ", iproc)
  sys.stdout.flush()

  #--- read in target SEM mesh
  mesh_file = "%s/proc%06d_external_mesh.bin"%(mesh_dir, iproc)
  mesh_data = sem_mesh_read(mesh_file)

  #--- read model values of the contributing mesh slice
  gll_dims = mesh_data['gll_dims']
  model_gll = np.zeros((nmodel,)+gll_dims)
  for imodel in range(nmodel):
    model_tag = model_names[imodel]
    model_file = "%s/proc%06d_%s.bin"%(model_dir, iproc, model_tag)
    with FortranFile(model_file, 'r') as f:
      # note: must use fortran convention when reshape to N-D array!!!
      model_gll[imodel,:,:,:,:] = np.reshape(f.read_ints(dtype='f4'), gll_dims, order='F')

  # locate each points
  loc_data = sem_locate_points_hex8(mesh_data, grd_enu)
  for ipoint in range(npoints):
    loc = loc_data[ipoint]
    # interpolate model at points located inside an element
    # or being closer to a target point
    if loc['is_inside'] or loc['misloc'] < misloc[ipoint]:
      misloc[ipoint] = loc['misloc']
      ispec = loc['ispec']
      model_interp[:,ipoint] = np.sum(model_gll[:,:,:,:,ispec]*loc['lagrange'], axis=(1,2,3))

#--- synchronize all processes
comm.Barrier()

#--- gather info to the root process 
MPI_TAG_misloc = 10
MPI_TAG_model = 11

if mpi_rank != 0:
  comm.Send(misloc, dest=0, tag=MPI_TAG_misloc)
  comm.Send(model_interp, dest=0, tag=MPI_TAG_model)

else:
  misloc_copy = np.empty(npoints)
  model_interp_copy = np.empty((nmodel,npoints))
  for iproc in range(1,mpi_size):
    comm.Recv(misloc_copy, source=iproc, tag=MPI_TAG_misloc) 
    comm.Recv(model_interp_copy, source=iproc, tag=MPI_TAG_model) 
    for ipoint in range(npoints):
      if misloc_copy[ipoint] < misloc[ipoint]:
        misloc[ipoint] = misloc_copy[ipoint]
        model_interp[:,ipoint] = model_interp_copy[:,ipoint]

  #- output interpolated model
  dataset = Dataset(out_file ,'w',format='NETCDF4_CLASSIC')
  dataset.description = "sem_slice_sphere"
  #
  dataset.createDimension('longitude',nlon)
  dataset.createDimension('latitude',nlat)
  #
  longitudes = dataset.createVariable('longitude', np.float32, ('longitude',))
  latitudes = dataset.createVariable('latitude', np.float32, ('latitude',))
  longitudes.units = 'degree_east'
  latitudes.units = 'degree_north'
  longitudes[:] = grd_lon1
  latitudes[:] = grd_lat1
  # model values
  for imodel in range(nmodel):
    model = dataset.createVariable(model_names[imodel], np.float32, ('longitude','latitude',))
    model.units = model_units[imodel]
    #model.long_name = model_names[imodel]
    model[:,:] = model_interp[imodel,].reshape((nlon,nlat))
  # location misfit of interpolation points 
  misloc_data = dataset.createVariable('misloc', np.float32, ('longitude','latitude',))
  misloc_data.units = 'meter'
  misloc_data[:,:] = misloc.reshape((nlon,nlat))
  #
  dataset.close()
