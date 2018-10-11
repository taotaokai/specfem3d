#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

#///////////////////////////////////////////////
# constants.h
NGLLX = 5
NGLLY = NGLLX
NGLLZ = NGLLX

MIDX = int((NGLLX-1)/2)
MIDY = int((NGLLY-1)/2)
MIDZ = int((NGLLZ-1)/2)

GAUSSALPHA = 0
GAUSSBETA = 0

field_list = [                       
  ('nspec','i4')                       ,
  ('nglob','i4')                       ,
  ('nspec_irregular','i4')             ,
  ('ibool','i4')                       ,
  ('x','f4')                           ,
  ('y','f4')                           ,
  ('z','f4')                           ,
  ('irregular_element_number','i4')    ,
  ('dxi_dx_regular','f4')              ,
  ('jacobian_regular','f4')            ,
  ('dxi_dx','f4')                      ,
  ('dxi_dy','f4')                      ,
  ('dxi_dz','f4')                      ,
  ('deta_dx','f4')                     ,
  ('deta_dy','f4')                     ,
  ('deta_dz','f4')                     ,
  ('dgamma_dx','f4')                   ,
  ('dgamma_dy','f4')                   ,
  ('dgamma_dz','f4')                   ,
  ('jacobian','f4')                    ,
# ('kappa','f4')                       ,
# ('mu','f4')                          , 
# # logical varaible is read in as integer*4, true=1, false=0
# ('ispec_is_acoustic','i4'), 
# ('ispec_is_elastic','i4'), 
# ('ispec_is_poroelastic','i4'), 
  ]

#//////////////////////////////////////////////
def rotmat_enu_to_ecef(lon,lat):
  """ rotation matrix from local ENU (lon,lat,alt) to ECEF coordinate basises
  rotmat[:,0] = Ve # column vector is the Easting direction in ECEF coordinate
  rotmat[:,1] = Vn
  rotmat[:,2] = Vu
  
  xyz_ecef = xyz0_ecef + rotmat * enu
  enu = transpose(rotmat) * (xyz_ecef - xyz0_ecef)

  , where xyz0_ecef is the reference point at (lon,lat,alt).
  """
  coslat = np.cos(np.deg2rad(lat))
  sinlat = np.sin(np.deg2rad(lat))
  coslon = np.cos(np.deg2rad(lon))
  sinlon = np.sin(np.deg2rad(lon))
  
  rotmat = np.zeros((3,3))
  rotmat[0,:] = [ -sinlon, -sinlat*coslon, coslat*coslon ]
  rotmat[1,:] = [  coslon, -sinlat*sinlon, coslat*sinlon ]
  rotmat[2,:] = [     0.0,  coslat,        sinlat        ]

  return rotmat


#///////////////////////////////////////////////////
def sem_mesh_read(mesh_file):
  """ read in SEM mesh slice
  """
  from scipy.io import FortranFile

  mesh_data = {}

  with FortranFile(mesh_file, 'r') as f:
    for field in field_list:
      field_name = field[0]
      data_type = field[1]
      mesh_data[field_name] = f.read_ints(dtype=data_type)
  
  mesh_data['nspec'] = mesh_data['nspec'][0]
  mesh_data['nglob'] = mesh_data['nglob'][0]

  # GLL dims
  gll_dims = (NGLLX,NGLLY,NGLLZ,mesh_data['nspec'])
  mesh_data['gll_dims'] = gll_dims

  # reshape
  for field_name in ['ibool', 'jacobian']:
    mesh_data[field_name] = np.reshape(mesh_data[field_name], gll_dims, order='F')
    #NB: binary files are written in Fortran !!!

  # use xyz_glob
  nglob = mesh_data['nglob']
  x = mesh_data['x'].reshape((1,nglob))
  y = mesh_data['y'].reshape((1,nglob))
  z = mesh_data['z'].reshape((1,nglob))
  mesh_data['xyz_glob'] = np.r_[x,y,z]

  del mesh_data['x']
  del mesh_data['y']
  del mesh_data['z']

  # add xyz_elem
  iglob_elem = mesh_data['ibool'][MIDX,MIDY,MIDZ,:] - 1
  mesh_data['xyz_elem'] = mesh_data['xyz_glob'][:,iglob_elem]

  return mesh_data


#///////////////////////////////////////////////////
def sem_mesh_get_vol_gll(mesh_data):
  """ get xyz and volumen facotr of each gll point
  """

  from gll_library import zwgljd

  #--- quadrature weights on GLL points
  zx, wx = zwgljd(NGLLX,GAUSSALPHA,GAUSSBETA)
  zy, wy = zwgljd(NGLLX,GAUSSALPHA,GAUSSBETA)
  zz, wz = zwgljd(NGLLX,GAUSSALPHA,GAUSSBETA)

  wgll_cube = wx.reshape((NGLLX,1,1))*wy.reshape((1,NGLLY,1))*wx.reshape((1,1,NGLLZ))

  #--- jacobian * gll_quad_weights
  vol_gll = mesh_data['jacobian']*wgll_cube.reshape((NGLLX,NGLLY,NGLLZ,1))

  return vol_gll

# nspec = int(mesh_data['nspec'])
#  for ispec in range(nspec):
#    for i in range(NGLLX):
#      for j in range(NGLLY):
#        for k in range(NGLLZ):
#          iglob = mesh_data['ibool'][i,j,k,ispec] - 1
#          xyz_gll[0,i,j,k,ispec] = mesh_data['x'][iglob]
#          xyz_gll[1,i,j,k,ispec] = mesh_data['y'][iglob]
#          xyz_gll[2,i,j,k,ispec] = mesh_data['z'][iglob]
#
#
#  xyz_gll = np.zeros((3,NGLLX,NGLLY,NGLLZ,nspec))