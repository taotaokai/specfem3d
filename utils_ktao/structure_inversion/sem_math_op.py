#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate expression between two sets of model gll files
"""
import sys

import numpy as np
from scipy import interpolate
from scipy.io import FortranFile
               
#====== user input
nproc = int(sys.argv[1])
model1_dir = str(sys.argv[2])
model1_names = str(sys.argv[3]) # vp,vs,rho -> V1
model2_dir = str(sys.argv[4])
model2_names = str(sys.argv[5]) # alpha_model2,beta_model2,rhop_model2 -> V2
math_expr = str(sys.argv[6]) # math expression, e.g. V1+V2,V1-V2,V1*np.exp(V2)
out_dir = str(sys.argv[7])
out_names = str(sys.argv[8]) # vp,vs,rho

model1_names = model1_names.split(',')
model2_names = model2_names.split(',')
out_names = out_names.split(',')
nmodel = len(model1_names)
if len(out_names) != nmodel or len(model2_names) != nmodel:
  print("ERROR: model2_names and out_names should be the same length of model1_names")
  sys.exit(-1)

#====== scale amplitude
for iproc in range(nproc): 
  for imodel in range(nmodel):
    model1_file = "%s/proc%06d_%s.bin"%(model1_dir, iproc, model1_names[imodel])
    with FortranFile(model1_file, 'r') as f:
      V1 = f.read_reals(dtype='f4')
    model2_file = "%s/proc%06d_%s.bin"%(model2_dir, iproc, model2_names[imodel])
    with FortranFile(model2_file, 'r') as f:
      V2 = f.read_reals(dtype='f4')
    V3 = eval(math_expr)
    out_file = "%s/proc%06d_%s.bin"%(out_dir, iproc, out_names[imodel])
    with FortranFile(out_file, 'w') as f:
      f.write_record(np.array(V3, dtype='f4'))
