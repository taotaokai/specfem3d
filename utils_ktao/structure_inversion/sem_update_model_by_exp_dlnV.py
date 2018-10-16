#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ln(m_{i+1}/m_i) = dlnv
"""
import sys

import numpy as np
from scipy import interpolate
from scipy.io import FortranFile
               
#====== user input
nproc = int(sys.argv[1])
model_dir = str(sys.argv[2])
model_names = str(sys.argv[3]) # vp,vs,rho
dmodel_dir = str(sys.argv[4])
dmodel_names = str(sys.argv[5]) # alpha_dmodel,beta_dmodel,rhop_dmodel
out_dir = str(sys.argv[6])
out_names = str(sys.argv[7]) # vp,vs,rho

model_names = model_names.split(',')
dmodel_names = dmodel_names.split(',')
out_names = out_names.split(',')
nmodel = len(model_names)
if len(out_names) != nmodel or len(dmodel_names) != nmodel:
  print("ERROR: dmodel_names and out_names should be the same length of model_names")
  sys.exit(-1)

#====== scale amplitude
for iproc in range(nproc): 
  for imodel in range(nmodel):
    model_file = "%s/proc%06d_%s.bin"%(model_dir, iproc, model_names[imodel])
    with FortranFile(model_file, 'r') as f:
      V = f.read_reals(dtype='f4')
    dmodel_file = "%s/proc%06d_%s.bin"%(dmodel_dir, iproc, dmodel_names[imodel])
    with FortranFile(dmodel_file, 'r') as f:
      dlnV = f.read_reals(dtype='f4')
    out_file = "%s/proc%06d_%s.bin"%(out_dir, iproc, out_names[imodel])
    with FortranFile(out_file, 'w') as f:
      f.write_record(np.array(V*np.exp(dlnV), dtype='f4'))
