#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import numpy as np
from scipy import interpolate
from scipy.io import FortranFile
               
#import pyproj
#import matplotlib.pyplot as plt

#====== user input
procnum = int(sys.argv[1])
model_dir = str(sys.argv[2])
model_name = str(sys.argv[3])

#====== read in gll file
input_file = "%s/proc%06d_%s.bin"%(model_dir, procnum, model_name)

with FortranFile(input_file, 'r') as f:
  gll = f.read_reals(dtype='f4')

print("n= %d"%(gll.size))
print("min/max= %f/%f"%(np.min(gll), np.max(gll)))
