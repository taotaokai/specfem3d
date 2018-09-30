#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt

# input STATIONS file
fn_in = str(sys.argv[1]) #e.g. STATIONS.REF_ENU
min_dist = float(sys.argv[2]) #e.g. 300000
fn_out = str(sys.argv[3]) #e.g. STATIONS.REF_ENU.select
outfig = str(sys.argv[4]) #e.g. STATIONS.REF_ENU.select.pdf

# read in points
with open(fn_in, 'r') as f:
  lines = [ l for l in f.readlines() if not l.startswith('#') ]

split_lines = [ l.split() for l in lines ]
net = [ l[0] for l in split_lines ]
sta = [ l[1] for l in split_lines ]
y = np.array([ float(l[2]) for l in split_lines ])
x = np.array([ float(l[3]) for l in split_lines ])

# randomly select points that are apart from each other by a minimum distance
npoint = len(x)
points = np.zeros((npoint,2))
points[:,0] = x
points[:,1] = y

randsize = npoint
randint = np.random.randint(0,npoint,randsize)

pts_select = np.zeros(points.shape)
idx_select = np.zeros(npoint, dtype='int')

idx = randint[0]
pts_select[0,:] = points[idx,:]
idx_select[0] = idx
n_select = 1

for i in range(1,randsize):
  if n_select >= npoint: break  # in case there is only one data point
  idx = randint[i]
  test_point = points[idx,:]
  d = np.min(np.sum((pts_select[0:n_select,:] - test_point)**2, axis=1)**0.5)
  if d < min_dist: continue
  pts_select[n_select,:] = test_point
  idx_select[n_select] = idx
  n_select += 1
  if n_select >= npoint: break 

# write out selected STATIONS  
with open(fn_out, 'w') as f:
  for i in range(n_select):
    idx = idx_select[i]
    f.write(lines[idx])

# plot the selected data point
plt.figure()
plt.plot(points[:,0], points[:,1], '.')
plt.plot(pts_select[0:n_select,0], pts_select[0:n_select,1], 'ro')
plt.savefig(outfig, format='pdf')
#plt.show()