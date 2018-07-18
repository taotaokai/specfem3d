#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import numpy as np
from scipy.io import FortranFile

#====== read in externe_mesh file
input_file = "DATABASES_MPI/proc000000_external_mesh.bin"


field_list = [
  ('nspec','i4'), ('nglob','i4'), ('ibool','i4'),
  ('x','f4'), ('y','f4'), ('z','f4'),
  ('dxi_dx','f4'),    ('dxi_dy','f4'),    ('dxi_dz','f4'),
  ('deta_dx','f4'),   ('deta_dy','f4'),   ('deta_dz','f4'),
  ('dgamma_dx','f4'), ('dgamma_dy','f4'), ('dgamma_dz','f4'),
  ('jacobian','f4'),
  ('kappa','f4'), ('mu','f4'),
  # logical varaible is read in as integer*4, true=1, false=0
  ('ispec_is_acoustic','i4'), ('ispec_is_elastic','i4'), ('ispec_is_poroelastic','i4'), 
  ]

#--- this does not work, why?
#record = f.read_record([('nspec','i4'), ('nglob','i4'), ('ibool','i4')])
#print(record)

mesh = {}

with FortranFile(input_file, 'r') as f:
  for field in field_list:
    field_name = field[0]
    data_type = field[1]
    mesh[field_name] = f.read_ints(dtype=data_type)

print(mesh['x'])
print(np.min(mesh['x']), np.max(mesh['x']))

#====== test output
out_file = "proc000000_x.bin"

with FortranFile(out_file, 'w') as f:
  f.write_record(mesh['x'])
