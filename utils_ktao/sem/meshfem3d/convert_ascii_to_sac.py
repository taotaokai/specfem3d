#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import numpy as np

from obspy.io.sac import SACTrace

#header = {'kstnm': 'ANMO', 'kcmpnm': 'BHZ', 'stla': 40.5, 'stlo': -108.23,
#...           'evla': -15.123, 'evlo': 123, 'evdp': 50, 'nzyear': 2012,
#...           'nzjday': 123, 'nzhour': 13, 'nzmin': 43, 'nzsec': 17,
#...           'nzmsec': 100, 'delta': 1.0/40}
#>>> sac = SACTrace(data=np.random.random(100), **header)
#>>> sac.write(filename, byteorder='little')

#cmpnm_list = ['BXX.semd', 'BXY.semd', 'BXZ.semd', 'BXX.semv', 'BXY.semv', 'BXZ.semv']
cmpnm_list = ['BXX.semd', 'BXY.semd', 'BXZ.semd']

#====== read user inputs
station_file = str(sys.argv[1])
input_dir = str(sys.argv[2])
out_dir = str(sys.argv[3])

#====== read STATIONS
with open(station_file, 'r') as f:
  lines = [ x.split() for x in f.readlines() if not(x.startswith('#')) ]

netwk = [x[0] for x in lines]
stnm = [x[1] for x in lines]


#====== read in ascii files
nstn = len(stnm)

for i in range(nstn):
  for cmpnm in cmpnm_list:
    ascii_file = "%s/%s.%s.%s" % (input_dir,netwk[i],stnm[i],cmpnm)

    with open(ascii_file, 'r') as f:
      lines = [ x.split() for x in f.readlines() if not(x.startswith('#')) ]
    times = np.array([float(x[0]) for x in lines])
    data = np.array([float(x[1]) for x in lines])

    dt = np.mean(np.diff(times))
    header = {'o':0.0, 'b':times[0], 'kstnm':stnm[i], 'kcmpnm':cmpnm, 'delta':dt}

    sac = SACTrace(data=data, **header)

    out_file = "%s/%s.%s.%s.sac" % (out_dir,netwk[i],stnm[i],cmpnm)
    sac.write(out_file, byteorder='little')

## output sac file
#sacfn_t = "%s/%s.%s.BXT.sem.sac"%(out_dir,sta[0],sta[1])
#tr_t = tr_e.copy()
#tr_t.meta.channel = 'BXT'
#tr_t.data = seis_t
#tr_t.write(sacfn_t,format="SAC")