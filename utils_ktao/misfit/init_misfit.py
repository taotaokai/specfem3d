#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Process misfit
"""
import sys
import os
import importlib.util
from misfit_semlocal import Misfit

# read command line args
par_file = str(sys.argv[1])
misfit_file = str(sys.argv[2])
cmt_file = str(sys.argv[3])
channel_file = str(sys.argv[4])
topo_grd = str(sys.argv[5])

# load parameter file
if sys.version_info < (3, ):
  raise Exception("need python3")
elif sys.version_info < (3, 5):
  spec =importlib.machinery.SourceFileLoader("misfit_par", par_file)
  par = spec.load_module()
else:
  spec = importlib.util.spec_from_file_location("misfit_par", par_file)
  par = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(par)

print("\n====== initialize\n")
misfit = Misfit()

print("\n====== setup REF_ENU\n")
misfit.setup_REF_ENU(par.REF_lon, par.REF_lat, par.REF_alt, par.REF_ellps)
print(misfit.data['REF_ENU'])

print("\n====== setup event\n")
misfit.setup_event_from_FORCESOLUTION(cmt_file)
print(misfit.data['event'])

print("\n====== setup station\n")
misfit.setup_station(channel_file, topo_grd, band_code=par.obs_band_code,three_channels=True)
print(misfit.data['station'])

print("\n====== save data\n")
misfit.save(misfit_file)