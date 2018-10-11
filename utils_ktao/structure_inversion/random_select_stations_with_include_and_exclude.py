#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt

# input STATIONS file
fn_all = str(sys.argv[1]) #e.g. STATIONS.REF_ENU.all
fn_include = str(sys.argv[2]) #e.g. STATIONS.REF_ENU.include
fn_exclude = str(sys.argv[3]) #e.g. STATIONS.REF_ENU.exclude
min_dist = float(sys.argv[4]) #e.g. 300000
fn_out = str(sys.argv[5]) #e.g. STATIONS.REF_ENU.select
outfig = str(sys.argv[6]) #e.g. STATIONS.REF_ENU.select.pdf

# read in all points
with open(fn_all, 'r') as f:
  lines_all = [ l for l in f.readlines() if not l.startswith('#') ]
split_lines = [ l.split() for l in lines_all ]
net_sta_all = [ l[0]+"."+l[1] for l in split_lines ]
# read points to exclude
with open(fn_exclude, 'r') as f:
  lines = [ l for l in f.readlines() if not l.startswith('#') ]
split_lines = [ l.split() for l in lines ]
net_sta_exclude = [ l[0]+"."+l[1] for l in split_lines ]
# remove excluded points
idx_exclude = [ net_sta_all.index(s) for s in net_sta_exclude if s in net_sta_all ]
lines_all = [ l for ii, l in enumerate(lines_all) if ii not in idx_exclude ]

#VERY BAD IDEA to remove elements at multiple indexs in this way!
# Since the list changes at each del, this gives you undesired result.
#for ii in idx_exclude:
#  del lines_all[ii]

split_lines = [ l.split() for l in lines_all ]
net_sta_all = [ l[0]+"."+l[1] for l in split_lines ]
y_all = np.array([ float(l[2]) for l in split_lines ])
x_all = np.array([ float(l[3]) for l in split_lines ])
npoint = len(x_all)
points = np.zeros((npoint,2))
points[:,0] = x_all
points[:,1] = y_all

# read points to include
with open(fn_include, 'r') as f:
  lines = [ l for l in f.readlines() if not l.startswith('#') ]
split_lines = [ l.split() for l in lines ]
net_sta_include = [ l[0]+"."+l[1] for l in split_lines ]
y_include = np.array([ float(l[2]) for l in split_lines ])
x_include = np.array([ float(l[3]) for l in split_lines ])

# first, add included points to the list of selected points
pts_select = np.zeros(points.shape)
idx_select = np.zeros(npoint, dtype='int')

idx_include = [ net_sta_all.index(s) for s in net_sta_include if s in net_sta_all ] # exclude over rules include
n_include = len(idx_include)
pts_select[0:n_include,:] = points[idx_include,:]
idx_select[0:n_include] = idx_include

# then randomly select new points that are apart from each other and the included points by a minimum distance
randsize = 5*npoint
randint = np.random.randint(0,npoint,randsize)

n_select = n_include
for i in range(randsize):
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
    f.write(lines_all[idx])

# plot the selected data point
plt.figure()
plt.plot(points[:,0], points[:,1], '.')
plt.plot(pts_select[0:n_select,0], pts_select[0:n_select,1], 'ro')
plt.savefig(outfig, format='pdf')
#plt.show()