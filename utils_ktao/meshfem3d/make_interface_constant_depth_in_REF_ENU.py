#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import warnings
#
import numpy as np
from scipy import interpolate, ndimage
#
from netCDF4 import Dataset
import pyproj
#
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

#====== parameters
mesh_par_file = str(sys.argv[1])

min_lon = float(sys.argv[2])
max_lon = float(sys.argv[3])
dlon = float(sys.argv[4])
min_lat = float(sys.argv[5])
max_lat = float(sys.argv[6])
dlat = float(sys.argv[7])

min_x = float(sys.argv[8])
max_x = float(sys.argv[9])
dx = float(sys.argv[10])
min_y = float(sys.argv[11])
max_y = float(sys.argv[12])
dy = float(sys.argv[13])

interface_depth_km = float(sys.argv[14])

out_file = str(sys.argv[15])

##--- mesh parameter
#mesh_par_file = "mesh_par.py"
#
##--- local ENU (x,y) mesh for interface data
## mesh dimensiosn must be consistent with meshfem3D_files/interfaces.dat !
## mesh size (meter)
#min_x = -1200000
#max_x =  1200000
#min_y = -1200000
#max_y =  1200000
## grid interval (should be smaller than the actual SEM mesh element size)
#dx = 1000
#dy = 1000
#
#out_file = '100km.dat'

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

#====== interpolate Moho topography from geodesic GRD file to the local ENU grid

#--- local ENU grid (x,y)
x1 = np.arange(min_x, max_x, dx)
y1 = np.arange(min_y, max_y, dy)
x2, y2 = np.meshgrid(x1, y1)

#--- set an interface at constant depth
grd_lon1 = np.arange(min_lon, max_lon, dlon)
grd_lat1 = np.arange(min_lat, max_lat, dlat)
grd_lon2, grd_lat2 = np.meshgrid(grd_lon1, grd_lat1)
grd_alt2 = np.ones(grd_lon2.shape) * -1000.0 * interface_depth_km

#--- convert (lon,lat,alt) to ECEF
ecef = pyproj.Proj(proj='geocent', ellps=ref_ellps)
lla = pyproj.Proj(proj='latlong', ellps=ref_ellps)

x0, y0, z0 = pyproj.transform(lla, ecef, ref_lon, ref_lat, ref_alt)
xx, yy, zz = pyproj.transform(lla, ecef, grd_lon2, grd_lat2, grd_alt2)

#--- transform from ECEF to local ENU
coslat = np.cos(np.deg2rad(ref_lat))
sinlat = np.sin(np.deg2rad(ref_lat))
coslon = np.cos(np.deg2rad(ref_lon))
sinlon = np.sin(np.deg2rad(ref_lon))

ee =        -sinlon*(xx-x0) +        coslon*(yy-y0)
nn = -sinlat*coslon*(xx-x0) - sinlat*sinlon*(yy-y0) + coslat*(zz-z0)
uu =  coslat*coslon*(xx-x0) + coslat*sinlon*(yy-y0) + sinlat*(zz-z0)

#--- interpolate uu(ee,nn) to local ENU grid (x,y)
z2 = interpolate.griddata((ee.flatten(), nn.flatten()), uu.flatten(), (x2, y2), method='cubic')

plt.figure()
plt.imshow(z2, extent=(min_x,max_x,max_y,min_y))
plt.colorbar()
plt.gca().invert_yaxis()
#plt.show()
plt.savefig(out_file+'.pdf', format='pdf')

#====== save interface data on the local ENU grid

#specfem3d/src/meshfem3D/create_interfaces_mesh.f90
#241     ! reads in interface points
#242     do iy = 1,npy_interface_top
#243       do ix = 1,npx_interface_top
#244         ! reading without trailing/triming...
#245         read(45,*,iostat=ier) elevation
#246         !in case data file would have comment lines...
#247         !call read_value_dble_precision_mesh(45,DONT_IGNORE_JUNK,elevation,'Z_INTERFACE_TOP',ier)
#248         if (ier /= 0) stop 'Error reading interface value'
#249 
#250         ! stores values for interpolation
#251         interface_top(ix,iy) = elevation
#252 

print("mesh dimensions (nx,ny): ", len(x1), len(y1))
with open(out_file, "w") as f:
  for iy in range(len(y1)):
    for ix in range(len(x1)):
       f.write("%+12.5E\n" %(z2[iy,ix]))
