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
topo_grd = str(sys.argv[2])
min_x = float(sys.argv[3])
max_x = float(sys.argv[4])
dx = float(sys.argv[5])
min_y = float(sys.argv[6])
max_y = float(sys.argv[7])
dy = float(sys.argv[8])
out_file = str(sys.argv[9])

##--- mesh parameter
#mesh_par_file = "mesh_par.py"
#
##--- GRD file for interface topography (should be smoothed to match the SEM mesh resolutin, i.e. smoothed over the length of the element size)
#topo_grd = "SETibet_smooth.grd"
##topo_txt = "topo/ETibet_smooth.txt"
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
#out_file = 'topo.dat'

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


#====== interpolate topography from geodetic GRD file to the local ENU grid

#--- local ENU grid (x,y)
x1 = np.arange(min_x, max_x, dx)
y1 = np.arange(min_y, max_y, dy)
x2, y2 = np.meshgrid(x1, y1)

#--- get alt(lon,lat) grid from GRD file
fh = Dataset(topo_grd, mode='r')
grd_lon1 = fh.variables['lon'][:]
grd_lat1 = fh.variables['lat'][:]
grd_lon2, grd_lat2 = np.meshgrid(grd_lon1, grd_lat1, indexing='xy')
grd_alt2 = fh.variables['z'][:] # z(lat,lon)

#with open(topo_txt, 'r') as f:
#  lines = [ l.split() for l in f.readlines() ]
#grd_lons = np.array([ float(l[0]) for l in lines])
#grd_lats = np.array([ float(l[1]) for l in lines])
#grd_alts = np.array([ float(l[2]) for l in lines])

#--- convert (lon,lat,alt) to ECEF
ecef = pyproj.Proj(proj='geocent', ellps=ref_ellps)
lla = pyproj.Proj(proj='latlong', ellps=ref_ellps)

x0, y0, z0 = pyproj.transform(lla, ecef, ref_lon, ref_lat, ref_alt)
#xx, yy, zz = pyproj.transform(lla, ecef, grd_lons, grd_lats, grd_alts)
xx, yy, zz = pyproj.transform(lla, ecef, grd_lon2, grd_lat2, grd_alt2)

#--- transform from ECEF to REF_ENU
cosla = np.cos(np.deg2rad(ref_lat))
sinla = np.sin(np.deg2rad(ref_lat))
coslo = np.cos(np.deg2rad(ref_lon))
sinlo = np.sin(np.deg2rad(ref_lon))

#RotM = np.zeros((3,3))
#RotM[0,:] = [       -sinlo,        coslo,   0.0 ]
#RotM[1,:] = [ -sinla*coslo, -sinla*sinlo, cosla ]
#RotM[2,:] = [  cosla*coslo,  cosla*sinlo, sinla ]

grd_ee =       -sinlo*(xx-x0) +       coslo*(yy-y0)
grd_nn = -sinla*coslo*(xx-x0) - sinla*sinlo*(yy-y0) + cosla*(zz-z0)
grd_uu =  cosla*coslo*(xx-x0) + cosla*sinlo*(yy-y0) + sinla*(zz-z0)

#--- interpolate uu(ee,nn) to local ENU grid (x,y)
z2 = interpolate.griddata((np.ravel(grd_ee), np.ravel(grd_nn)), np.ravel(grd_uu), (x2, y2), method='cubic')

plt.figure()
plt.imshow(z2, extent=(min_x,max_x,max_y,min_y))
plt.colorbar()
plt.gca().invert_yaxis()
#plt.show()
plt.savefig('%s.pdf'%(out_file), format='pdf')
#sys.exit(-1)

#====== save interface data on the REF_ENU grid
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
