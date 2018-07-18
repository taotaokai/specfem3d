#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

def rotmat_enu_to_ecef(lon,lat):
  """ rotation matrix from local ENU (lon,lat,alt) to ECEF coordinate basises
  rotmat[:,0] = Ve
  rotmat[:,1] = Vn
  rotmat[:,1] = Vu
  
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
