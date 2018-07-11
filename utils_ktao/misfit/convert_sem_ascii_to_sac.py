#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import numpy as np

from obspy.io.sac import SACTrace
from obspy import UTCDateTime

#header = {'kstnm': 'ANMO', 'kcmpnm': 'BHZ', 'stla': 40.5, 'stlo': -108.23,
#...           'evla': -15.123, 'evlo': 123, 'evdp': 50, 'nzyear': 2012,
#...           'nzjday': 123, 'nzhour': 13, 'nzmin': 43, 'nzsec': 17,
#...           'nzmsec': 100, 'delta': 1.0/40}
#>>> sac = SACTrace(data=np.random.random(100), **header)
#>>> sac.write(filename, byteorder='little')

cmpnm_list = ['BXX', 'BXY', 'BXZ']
suffix = "semd"

#====== read user inputs
station_file = str(sys.argv[1])
forcesolution_file = str(sys.argv[2]) # FORCESOLUTION with first line of event_ID ref_time
input_dir = str(sys.argv[3])
out_dir = str(sys.argv[4])

#====== read STATIONS
with open(station_file, 'r') as f:
  lines = [ x.split() for x in f.readlines() if not(x.startswith('#')) ]

stnm = [x[0] for x in lines]
netwk = [x[1] for x in lines]

#====== read FORCESOLUTION
with open(forcesolution_file, 'r') as f:
  lines = [ l.split() for l in f.readlines() if not(l.startswith('#')) ]

ref_time = UTCDateTime(lines[0][1])

#====== read in ascii files
nstn = len(stnm)

for i in range(nstn):
  for cmpnm in cmpnm_list:
    ascii_file = "%s/%s.%s.%s.%s" % (input_dir,netwk[i],stnm[i],cmpnm,suffix)

    with open(ascii_file, 'r') as f:
      lines = [ x.split() for x in f.readlines() if not(x.startswith('#')) ]
    times = np.array([float(x[0]) for x in lines])
    data = np.array([float(x[1]) for x in lines])

    dt = np.mean(np.diff(times))
    header = {'kstnm':stnm[i], 'kcmpnm':cmpnm, 'delta':dt}

    sac = SACTrace(data=data, **header)
    sac.reftime = ref_time
    sac.o = 0.0
    sac.b = times[0]

    out_file = "%s/%s.%s.%s.%s.sac" % (out_dir,netwk[i],stnm[i],cmpnm,suffix)
    sac.write(out_file, byteorder='little')