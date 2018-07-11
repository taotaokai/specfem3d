#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import warnings
#import os.path
import re
import datetime
#
import numpy as np
import scipy.signal as signal
from scipy import interpolate
#
import pickle
#
from obspy import UTCDateTime, read, Trace
#from obspy.geodetics import gps2dist_azimuth, kilometer2degrees
from obspy.taup import TauPyModel
from obspy.imaging.beachball import beach
#
import pyproj
#
from lanczos_interp1 import lanczos_interp1
#
import matplotlib
matplotlib.use("pdf")
from matplotlib import colors, ticker, cm
from matplotlib.backends.backend_pdf import PdfPages 
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

#NOTE
# 1. spectrum relation between DFT and FT
#   x(n*dt): DFT[x]*dt ~ FT[x], IDFT[FT[x]]/dt ~ x

#====== utility functions
def is_equal(lst):
  return not lst or [lst[0]]*len(lst) == lst

def stf_gauss_spectrum(f, tau):
  """ 
  spectrum of the Gaussian STF of unit area: 
    stf(t;t0,tau) = 1/sqrt(PI)/tau * exp(-((t-t0)/tau)^2)
    here, we take t0 = 0, then
    F_stf = exp(- pi^2 * f^2 * tau^2)
  """
  F_src = np.exp(-np.pi**2 * f**2 * tau**2)
# F_ds_dt0 = -2.0j * np.pi * f * F_src
# F_ds_dtau = -2.0 * (np.pi*f)**2 * tau * F_src
# return F_src, F_ds_dt0, F_ds_dtau
  return F_src

def stf_gauss_spectrum_der(f, tau):
  """ 
  spectrum of the derivatives of Gaussian STF
  """
  F_ds_dt0 = -2.0j * np.pi * f * F_src
  F_ds_dtau = -2.0 * (np.pi*f)**2 * tau * F_src
  return F_ds_dt0, F_ds_dtau

def prime_factor(n):
  """
  Prime factorization
  """
  factor = []
  d = 2
  while True:
    while n%d == 0:
      factor.append(d)
      n //= d
    if n == 1:
      return factor
    d += 1
    if d**2 > n:
      factor.append(n)
      return factor

def optimal_fft_size(n, d):
  """
  Return closest upper integer of n whose largest-prime-factor is smaller than d
  """
  if d < 2:
    raise ValueError("d should >= 2, while %d is used" % (d))
  while True:
    factor = prime_factor(n)
    if max(factor) > d:
      n += 1
    else:
      return n

#def stf_gauss(n, dt, tau):
#  """ Gaussian source time function
#    stf(t,tau) = 1/sqrt(PI)/tau * exp(-(t/tau)^2)
#    t = dt * [0:(n+1)/2, -n/2:-1]
#  """
#  n = int(n)
#  idx = np.arange(n)
#  idx[(n+1)/2:] -= n
#  t = idx * dt / tau
#  stf = np.exp(-t**2)/tau/np.pi**0.5
#  ds_dt0 = 2.0/tau * t * stf
#  ds_dtau = 2.0/tau * (t**2 - 0.5) * stf 
#  return stf, ds_dt0, ds_dtau


def cosine_taper(x, xc):
    """cosine taper at two ends stop,pass, pass,stop
        xc: (array-like)
            stop,pass[,pass,stop]
        x: scalar or array-like
            sample points
    """
    nc = len(xc)
    if np.isscalar(x):
        x = np.array([x, ])
    else:
        x = np.array(x)
    y = np.ones(len(x))

    if nc == 2: # one-side taper
        l = xc[1] - xc[0] # taper width
        if l == 0:
            raise ValueError('pass and stop values cannot be the same.')
        elif l > 0: # tapered at left side
            idx = x < xc[0]; y[idx] = 0.0
            idx = (xc[0] <= x) & (x <= xc[1])
            y[idx] = 0.5 - 0.5*np.cos(np.pi*(x[idx] - xc[0])/l)
            idx = x > xc[1]; y[idx] = 1.0
        else: # tapered at right side
            idx = x > xc[0]; y[idx] = 0.0
            idx = (xc[1] <= x) & (x <= xc[0])
            y[idx] = 0.5 + 0.5*np.cos(np.pi*(x[idx] - xc[1])/l)
            idx = x < xc[1]; y[idx] = 1.0
    elif nc == 4: # two side taper
        if not (xc[0]<xc[1]<xc[2]<xc[3]):
            raise ValueError('4 cutoff values must be in increasing order.')
        else:
            idx = x <= xc[0]; y[idx] = 0.0

            idx = (xc[0] < x) & (x < xc[1])
            y[idx] = 0.5 - 0.5*np.cos(np.pi*(x[idx] - xc[0])/(xc[1]-xc[0]))
            
            idx = (xc[1] <= x) & (x <= xc[2]); y[idx] = 1.0

            idx = (xc[2] < x) & (x < xc[3])
            y[idx] = 0.5 + 0.5*np.cos(np.pi*(x[idx] - xc[2])/(xc[3]-xc[2]))

            idx = x >= xc[3]; y[idx] = 0.0
    else:
        raise ValueError('number of cutoff values must be either 2 or 4.')

    # restore return value to scalar when input x is a scalar
    if len(y) == 1: y = y[0]

    return y
# enddef cosine_taper(x, xc):


def rotmat_enu_to_ecef(lon, lat):
  """ rotation matrix from ENU to ECEF coordinate basises
  """
  coslat = np.cos(np.deg2rad(lat))
  sinlat = np.sin(np.deg2rad(lat))
  coslon = np.cos(np.deg2rad(lon))
  sinlon = np.sin(np.deg2rad(lon))
  
  rotmat = np.zeros((3,3))
  rotmat[0,:] = [ -sinlon, -sinlat*coslon, coslat*coslon ]
  rotmat[1,:] = [  coslon, -sinlat*sinlon, coslat*sinlon ]
  rotmat[2,:] = [     0.0,  coslat,        sinlat        ]

  return rotmat
#enddef rotmat_enu_to_ecef(lon,lat):

#======
class Misfit(object):
  """Class managing all misfit windows

  Unit: kg,m,s,degree

  Coordinate: reference ENU

  Data:
  {
    'REF_ENU':{'ellipsoid':"WGS84" or "sphere", 
               'longitude':(degree), 'latitude':(degree), 'altitude':(meter)}

    'event': {
        'stat':{'code':, 'msg':},
        'id':, # event ID
        'type':"point_force" or "moment_tensor"
        't0':[utcdatetime], # centroid time
        'tau':, # parametric source time function: gaussian width exp(-(t-t0)^2/tau^2)/tau/pi^0.5
        'xyz': [x, y, z], # source location in REF_ENU coordinate
        'force_vector': [fx, fy, fz], # point force vector in REF_ENU coordinate (unit: Newton)
        #'moment_tensor':, # point moment tensor in REF_ENU coordinate (unit: Newton*meter)
        #'mt_rtp': # moment tensor in spherical coord
    }

    'src_frechet': { #derivative of misfit function w.r.t. source param.
        'stat':{'code':, 'msg':},
        't0':, 'dau':, 'xs':, 'mt':},

    'src_perturb': {'t0':1.0, 'tau':1.0, 'xs':, 'mt':, },

    'chi': # weighted average of normalized zero-lag CC

    'station': {

        <station_id> : {
            'stat': {code:, msg:}, #0: created; 1: read_obs_syn;
            'meta': {
                latitude:, longitude:, elevation:, depth:,
                xyz: # in REF_ENU coordinates
                rotmat_local_to_ref_ENU: #(3,3) rotation matrix from local ENU to REF_ENU
                azimuth:, back_azimuth:, dist_degree:,
                channel: [ {code,az,dip,...}, ...], # az,dip: local ENU
                ttime: # traveltime info
            },

            #### all waveforms are converted into local ENU frame
            'waveform': { 
                'time_sample': {starttime:, delta:, nt:, nl:, nr},
                'obs': (3,nt), # observed seismograms
                'syn': (3,nt), # synthetic seismograms (use stf in simulation)
                'grn': (3,nt), # synthetic seismograms (use delta stf in simulation)
            },

            'waveform_der': { # Du/Dm, u: synthetics, m: model
               'xs': {'dm':array(3), 'du':, }, # source point location
               'mt': {'dm':array([3,3]), 'du':, }, #moment-tensor
            },

            'window': {
                <window_id>: {
                    'stat': {code:, msg:}, 
                    #bad:<0, ok:0, measured adj:1, and hessian_src:2

                    'filter': {'type':,'order':,'freqlim':,'a':,'b':},

                    'taper': {
                        'starttime':, 'endtime':,
                        'type':, 'ratio':, 'win':array(nt) }, 

                    'polarity': { # in local ENU frame
                        'component':, 'azimuth':, 'dip':,
                        'proj_matrix':array([3,3]) },

                    'quality': {
                        'Amax_obs':, 'Amax_noise':, 'Amax_syn':, 'SNR':, },

                    'cc': {'time':, 'cc':, 'cc_tshift': 
                        'CC0':, 'CCmax':, 'AR0':, 'ARmax':},
                    'weight':,

                    #'dchi_du':, # adjoint source from this window
                    #'dchi_dg':,

                    'hessian_src':{ # approximated Hessian of chi to source
                        ('dt0','dt0'):,
                        ('dtau','dtau'):,
                        ('dxs','dxs'):,
                        ...
                    },
                },
                <window_id>...
            },

            # adjoint source as a sum from all windows
            # in local ENU 
            'dchi_du': (3,nt) array, chi is the objective function

        },

        <station_id>...
    },
  }

  Methods:
    setup_event
    setup_station
    read_obs_syn
    delete_window
    add_window_body_wave
    add_window_surface_wave
    measure_adj
    window_quality_control (determine bad/OK, and window weight)
    output_adj
    make_cmt_dxs/dmt
    waveform_der_dxs/dmt
    cc_perturbed_seisomgram
    grid_cc

  NOTE:
    1. 1D Earth model: ak135

  """
#
#======================================================
#

  def __init__(self):
    """ initialize
    """
    self.data = {
        'event':{},
        'src_frechet':{},
        'src_perturb':{'t0':1.0, 'tau':1.0},
        'station':{},
        'chi':0.0,
        }
#
#======================================================
#

  def save(self, filename):
    """Save data
    """
    # use highest protocol available 
    with open(filename, 'wb') as f:
      pickle.dump(self.data, f)

#
#======================================================
#

  def load(self, filename):
    """Load data
    """
    with open(filename, 'rb') as f:
      self.data = pickle.load(f)

#
#======================================================
#

  def setup_REF_ENU(self, longitude, latitude, altitude, ellipsoid="WGS84"):
    """ setup reference ENU coordinate
    """
    REF_ENU = {
        'longitude':longitude, 
        'latitude':latitude, 
        'altitude':altitude,
        'ellipsoid':ellipsoid,
        'stat': {'code':0, 'msg':"created on "+UTCDateTime.now().isoformat()}
        }

    self.data['REF_ENU'] = REF_ENU

  #enddef setup_ENU0

##
##======================================================
##
#
#  def setup_event(self, event_file, topo_grd):
#    """read in event info
#
#    topo_grd: GRD file of topography
#    """
#
#    #====== read in event file
#    with open(event_file, 'r') as f:
#      lines = [ x.replace('\n','').split('|') for x in f.readlines() if not(x.startswith('#')) ]
# 
#    nsrc = len(lines)
#    #for i in range(nsrc):
#    i = 0
#
#    event_id = str(lines[i][0])
#    t0 = UTCDateTime(lines[i][1])
#    latitude = float(lines[i][2])
#    longitude = float(lines[i][3])
#    elevation = float(lines[i][4])
#    depth = float(lines[i][5])
#
#    source_type = str(lines[i][6])
#    if source_type == "point_force":
#      force_azimuth = float(lines[i][7])
#      force_dip = float(lines[i][8])
#      force_scale = float(lines[i][9])
#      n = 10
#    #elif source_type == "moment_tensor": 
#    #  mrr = float(lines[i][7])
#    #  mtt = float(lines[i][8])
#    #  mpp = float(lines[i][9])
#    #  mrt = float(lines[i][10])
#    #  mrp = float(lines[i][11])
#    #  mtp = float(lines[i][12])
#    #  n = 13
#
#    stf_type = str(lines[i][n]); n += 1
#    if stf_type == "gauss":
#      stf_tau = float(lines[i][n]); n += 1
#    #elif stf_type == "user_input":
#    #  stf_file = str(lines[i][n]); n += 1
#
#    #====== convert to ENU0 coordinates
#    ENU0 = self.data['ENU0']
#    lat0 = ENU0['latitude']
#    lon0 = ENU0['longitude']
#    alt0 = ENU0['altitude']
#    ellps = ENU0['ellipsoid']
#
#    #--- 0. get topo on each station
#    if ellps == "WGS84":
#      fh = Dataset(topo_grd, mode='r')
#      grd_lon1 = fh.variables['x'][:]
#      grd_lat1 = fh.variables['y'][:]
#      grd_z2 = fh.variables['z'][:]
#      grd_lon2, grd_lat2 = np.meshgrid(grd_lon1, grd_lat1, indexing='xy')
#      topo = interpolate.griddata((grd_lon2.flatten(), grd_lat2.flatten()), grd_z2.flatten(), (longitude, latitude), method='cubic')
#    elif ellps == "sphere":
#      topo = 0.0
#    else:
#      raise Exception("unknown reference ellipsoid: %s"%(ellps))
#
#    #--- 1. convert to ECEF
#    ecef = pyproj.Proj(proj='geocent', ellps=ellps)
#    lla = pyproj.Proj(proj='latlong', ellps=ellps)
#    x0, y0, z0 = pyproj.transform(lla, ecef, lon0, lat0, alt0)
#    xx, yy, zz = pyproj.transform(lla, ecef, longitude, latitude, topo-depth)
#
#    #--- 2. transform location from ECEF to ENU0
#    R0 = rotmat_enu_to_ecef(lon0, lat0)
#    xyz_enu0 = np.dot(np.transpose(R0), np.array([xx,yy,zz]))
#
#    #--- 3. transform force vector from local ENU to ENU0
#    ve = np.cos(np.deg2rad(force_dip))*np.sin(np.deg2rad(force_azimuth))
#    vn = np.cos(np.deg2rad(force_dip))*np.cos(np.deg2rad(force_azimuth))
#    vz = -1*np.sin(np.deg2rad(force_dip))
#
#    R1 = rotmat_enu_to_ecef(longitude, latitude)
#    vec_enu0 = np.dot(np.transpose(R0), np.dot(R1, np.array([ve,vn,vz])))
#
#    # add event
#    event = {
#        'id':event_id,
#        't0':t0, 
#        'latitude':latitude, 'longitude':longitude, 'elevation':elevation, 'depth':depth,
#        'source_type':source_type, 
#        'force_azimuth':force_azimuth, 'force_dip':force_dip, 'force_scale':force_scale,
#        #'mrr':mrr,
#        'stf_type':stf_type, 'stf_tau':stf_tau,
#        'xyz':xyz_enu0, 'force_vec':vec_enu0,
#        #'moment_tensor':
#        'stat': {'code':0, 'msg':"created on "+UTCDateTime.now().isoformat()}
#        }
#
#    self.data['event'] = event
#
#  #enddef setup_event

#
#======================================================
#

  def setup_event_from_FORCESOLUTION(self, input_file):
    """read in FORCESOLUTION
    """
    # REF_ENU
    REF_ENU = self.data['REF_ENU']
    REF_lat = REF_ENU['latitude']
    REF_lon = REF_ENU['longitude']
    REF_alt = REF_ENU['altitude']
    REF_ellps = REF_ENU['ellipsoid']

    # initiate Geod
    geod = pyproj.Geod(ellps=REF_ellps)
    ecef = pyproj.Proj(proj='geocent', ellps=REF_ellps)
    lla = pyproj.Proj(proj='latlong', ellps=REF_ellps)

    # read in event file
    with open(input_file, 'r') as f:
      lines = [ x.replace('\n','').split(":") for x in f.readlines() if not(x.startswith('#')) ]
 
    event_id = str(lines[0][1])
    t0 = float(lines[1][1])
    tau = float(lines[2][1])
    x = float(lines[3][1])
    y = float(lines[4][1])
    z = float(lines[5][1])
    f_scale = float(lines[6][1])
    fx = float(lines[7][1])
    fy = float(lines[8][1])
    fz = float(lines[9][1])

    # convert from REF_ENU to ECEF
    x0, y0, z0 = pyproj.transform(lla, ecef, REF_lon, REF_lat, REF_alt)
    R0 = rotmat_enu_to_ecef(REF_lon, REF_lat)
    xyz = np.dot(R0, np.array([x,y,z])) + np.array([x0,y0,z0])
    lon, lat, alt = pyproj.transform(ecef, lla, xyz[0], xyz[1], xyz[2])

    # add event
    event = {
        'source_type':"point_force", 
        'id':event_id,
        'latitude':lat, 'longitude':lon, 'altitude':alt,
        'xyz':np.array([x,y,z]),
        't0':t0, 'tau':tau,
        'force_scale':f_scale, 'force_vector':np.array([fx,fy,fz]),
        'stat': {'code':0, 'msg':"created on "+UTCDateTime.now().isoformat()} 
        }

    self.data['event'] = event

  #enddef setup_event

#
#======================================================
#

  def setup_station(self,
      channel_file,
      topo_grd,
      band_code=None,
      three_channels=True,
      ):
    """ Setup station metadata.

      channel_file (str):
        FDSN-station text format file at channel level
      topo_grd (str):
        GRD file of topography, used to get local topography
      band_code (str):
        instrument/band code [default: None]
      three_channels (bool):
        check for completeness of 3 channels [default: False]

      Note:
      1) only use stations which have the same lat/lon/ele/depth 
        in all the available channels.
      2) event info must be set first.
    """
    # REF_ENU
    REF_ENU = self.data['REF_ENU']
    REF_lat = REF_ENU['latitude']
    REF_lon = REF_ENU['longitude']
    REF_alt = REF_ENU['altitude']
    REF_ellps = REF_ENU['ellipsoid']

    # initiate Geod
    geod = pyproj.Geod(ellps=REF_ellps)
    ecef = pyproj.Proj(proj='geocent', ellps=REF_ellps)
    lla = pyproj.Proj(proj='latlong', ellps=REF_ellps)

    # rotation matrix from ECEF of REF_ENU
    x0, y0, z0 = pyproj.transform(lla, ecef, REF_lon, REF_lat, REF_alt)
    R0 = rotmat_enu_to_ecef(REF_lon, REF_lat)

    # initiate taup
    taup_model = TauPyModel(model="ak135")

    # read station file
    with open(channel_file, 'r') as f:
      lines = [x.replace('\n','').split('|')  \
          for x in f.readlines() if not(x.startswith('#'))]
    
    #--- get topo on each station
    lats = np.array([ float(l[4]) for l in lines])
    lons = np.array([ float(l[5]) for l in lines])

    if REF_ellps == "WGS84":
      fh = Dataset(topo_grd, mode='r')
      grd_lon1 = fh.variables['x'][:]
      grd_lat1 = fh.variables['y'][:]
      grd_z2 = fh.variables['z'][:]
      grd_lon2, grd_lat2 = np.meshgrid(grd_lon1, grd_lat1, indexing='xy')
      topo = interpolate.griddata((grd_lon2.flatten(), grd_lat2.flatten()), grd_z2.flatten(), (lons, lats), method='cubic')
    elif REF_ellps == "sphere":
      topo = np.zeros(lats.shape)
    else:
      raise Exception("unknown reference ellipsoid: %s"%(REF_ellps))

    #--- get all station metadata
    metadata = {}
    nlines = len(lines)
    for i in range(nlines):
      x = lines[i]
      net_sta_loc = (x[0], x[1], x[2])
      date1 = [ int(a) for a in re.sub("\D", " ", x[15]).split() ]
      date2 = [ int(a) for a in re.sub("\D", " ", x[16]).split() ]
      t1 = UTCDateTime(date1[0], date1[1], date1[2]) \
          + 60.0*(60.0*date1[3] + date1[4]) + date1[5]
      t2 = UTCDateTime(date2[0], date2[1], date2[2]) \
          + 60.0*(60.0*date2[3] + date2[4]) + date2[5]
      channel = {
          'code':    x[3],
          'latitude':  float(x[4]),
          'longitude':   float(x[5]),
          #'elevation':   float(x[6]), # ignore the topo in channel file
          'elevation':   topo[i], # use the surface topo extracted from GRD file (to be compatible with mesh)
          'depth':     float(x[7]),
          'azimuth':   float(x[8]),
          'dip':     float(x[9]),
          'starttime':   t1,
          'endtime':   t2
          }

      # get ENU0 coordinates for the station
      #--- 1. convert to ECEF
      xx, yy, zz = pyproj.transform(lla, ecef,
        channel['longitude'],
        channel['latitude'],
        channel['elevation'] - channel['depth']
        )
      #--- 2. transform location from ECEF to REF_ENU
      xyz = np.dot(np.transpose(R0), np.array([xx-x0,yy-y0,zz-z0]))
      ##--- 3. transform force vector from local ENU to ENU0
      #ve = np.cos(np.deg2rad(channel['dip']))*np.sin(np.deg2rad(channel['azimuth']))
      #vn = np.cos(np.deg2rad(channel['dip']))*np.cos(np.deg2rad(channel['azimuth']))
      #vz = -1*np.sin(np.deg2rad(channel['dip']))
      #R1 = rotmat_enu_to_ecef(channel['longitude'], channel['latitude'])
      #vec_enu0 = np.dot(np.transpose(R0), np.dot(R1, np.array([ve,vn,vz])))

      channel['xyz'] = xyz
      #channel['cmpvec'] = vec_enu0

      if net_sta_loc not in metadata:
        metadata[net_sta_loc] = []
      metadata[net_sta_loc].append(channel)

    # check if event info is set 
    data = self.data
    if ('event' not in data) or (data['event']['stat']['code'] < 0):
      raise Exception('Event info not set.')
    event = data['event']

    # initialize station dict
    data['station'] = {}
    station = data['station']

    # station active time is set to centroid time
    # used to filter channel list
    #active_time = event['t0']

    for net_sta_loc in metadata:
      # station_id: net.sta.loc
      station_id = '.'.join(net_sta_loc)

      # skip existing stations if not update
      if (station_id in station) and (not update):
        raise Exception('station %s already exist' % (station_id))

      # select channels which are active at the specified time 
      #channel = [ x for x in metadata[net_sta_loc] 
      #    if x['starttime'] < active_time and x['endtime'] > active_time ]

      if not channel:
        print("[WARNING] %s not active." % (station_id))
        continue

      # select band code (e.g. BH )
      if band_code:
        n = len(band_code)
        channel = [ x for x in channel if x['code'][0:n]==band_code ]
        if not channel:
          print("[WARNING] %s has no specified band code %s." % (station_id, band_code))
          continue

      # check if all selected channels have the same location
      lats = [ x['latitude'] for x in channel ]
      lons = [ x['longitude'] for x in channel ]
      eles = [ x['elevation'] for x in channel ]
      deps = [ x['depth'] for x in channel ]
      if lats.count(lats[0])!=len(lats) or \
          lons.count(lons[0])!=len(lons) or \
          eles.count(eles[0])!=len(eles) or \
          deps.count(deps[0])!=len(deps):
        print("[WARNING] %s: " \
          "channels do NOT have the same coordinates, SKIP." \
          % (station_id))
        continue

      # check completeness for 3-components
      if three_channels:
        if len(channel) != 3:
          print('[WARNING] %s: not exactly 3 components found, '\
            'SKIP.' % (station_id))
          continue
        # check channel orientations
        Z_comp = [ (x['code'], x['azimuth'], x['dip'])
            for x in channel if x['code'][2] == 'Z']
        if len(Z_comp) != 1 or abs(Z_comp[0][2]) != 90.0: 
          print('[WARNING] %s: problematic Z channel, SKIP' \
              % (station_id))
          print('      channel: ', H_comp)
          continue
        H_comp = [ (x['code'], x['azimuth'], x['dip']) \
            for x in channel if x['code'][2] != 'Z']
        if len(H_comp) != 2 or \
            abs(H_comp[0][2]) != 0.0 or \
            abs(H_comp[1][2]) != 0.0 or \
            abs(np.cos(np.deg2rad(
              H_comp[0][1] - H_comp[1][1]))) > 0.1: 
          print('[WARNING] %s: problematic horizontal channels, SKIP'\
              % (station_id))
          print('      channel: ', H_comp)
          continue

      # geodetic and ak135 traveltimes
      #dist, az, baz = gps2dist_azimuth(
      #    event['latitude'], event['longitude'],
      #    channel[0]['latitude'], channel[0]['longitude'])
      az, baz, dist = geod.inv(event['longitude'], event['latitude'], 
                               channel[0]['longitude'], channel[0]['latitude'])
      #dist_degree = kilometer2degrees(dist/1000.0)
      dist_degree = np.rad2deg(dist/6371000.0)

#      evdp_km = event['depth']
#      if evdp_km < 0.0:
#        evdp_km = 0.0
#      arrivals = taup_model.get_travel_times(
#          source_depth_in_km=evdp_km,
#          distance_in_degree=dist_degree,
#          phase_list=['ttp'],
#          )
 

      # make station metadata 
      meta = { #TODO remove dumplicated info in channels?
          'latitude': channel[0]['latitude'], 
          'longitude': channel[0]['longitude'],
          'elevation': channel[0]['elevation'],
          'depth': channel[0]['depth'],
          'xyz': channel[0]['xyz'],
          'channel': channel,
          'azimuth': az,
          'back_azimuth': baz,
          'dist_degree': dist_degree,
          'ttime': arrivals}

      # rotation matrix from local ENU to REF_ENU
      R1 = rotmat_enu_to_ecef(meta['longitude'], meta['latitude'])
      rotmat_local_to_ref = np.dot(np.transpose(R0), R1)
      meta['rotmat_local_to_ref'] = rotmat_local_to_ref

      # add station info
      station[station_id] = {
          'meta': meta,
          'waveform':{},
          'window':{},
          'waveform_der':{},
          'stat': {
            'code': 0,
            'msg': "created on "+UTCDateTime.now().isoformat()} 
          }

    #endfor net_sta_loc in stations_all:
  #enddef setup_stations_from_channel_file

#
#======================================================
#

  def output_STATIONS(self, out_file):
    """ Output STATIONS file for SEM_cartesian simulation
    """
    station_dict = self.data['station']

    with open(out_file, 'w') as f:
      for station_id in station_dict:
        station = station_dict[station_id]

        tmp = station_id.split(".")
        netwk = tmp[0]
        stnm = tmp[1]
        loc = tmp[2]

        meta = station['meta']
        xyz = meta['xyz']

        f.write("%s.%s  %s  %f  %f  0.0  %f\n"%(
                stnm,loc, netwk, xyz[1],xyz[0],xyz[2]))

#
#======================================================
#

  def read_obs_syn(self,
      obs_dir='obs',
      syn_dir='syn', syn_band_code='BX', syn_suffix='.semd.sac', syn_is_grn=False,
      left_pad=100, right_pad=0, obs_preevent=100):
    """ 
    Read in observed and synthetic seismograms.

    Parameters
    ----------
    left_pad: 
      time length to pad before synthetics 

    obs_pretime: 
      pre-event time length of obs (for noise assessment)

    syn_is_grn:
      flag if synthetic seismograms are computed with zero half duration;
      if true, ignore absolute time in syn files, use event['t0'] as the origin time

    Note
    ----
    sac file naming rule: NET.STA.LOC.CHA
      , where CHA consists of band_code(e.g. BH or MX) + orientation[E|N|Z]

    syn should be sac files and have sac header b and o set correctly.
    """
    syn_orientation_codes = ['X', 'Y', 'Z'] #in REF_ENU frame

    event = self.data['event']
    station_dict = self.data['station']

    if left_pad < 0:
      warnings.warn("[arg] left_pad must g.e. 0, set to 0.0")
      left_pad = 0
    if right_pad < 0:
      warnings.warn("[arg] right_pad must g.e. 0, set to 0.0")
      right_pad = 0
    if obs_preevent < 0:
      obs_preevent = 50.0
      warnings.warn("[arg] obs_preevent must g.e. 0, set to 50")

    for station_id in station_dict:
      station = station_dict[station_id]
      meta = station['meta']
      channel = meta['channel']
 
      #------ get file paths of obs, syn seismograms
      obs_files = [ '{:s}/{:s}.{:s}'.format(
        obs_dir, station_id, x['code']) for x in channel ]
      syn_files = [ '{:s}/{:s}.{:2s}{:1s}{:s}'.format(
        syn_dir, station_id, syn_band_code, x, syn_suffix)
        for x in syn_orientation_codes ]

      #------ read in obs, syn seismograms
      try:
        obs_st  = read(obs_files[0])
        obs_st += read(obs_files[1])
        obs_st += read(obs_files[2])
        syn_st  = read(syn_files[0])
        syn_st += read(syn_files[1])
        syn_st += read(syn_files[2])
      except Exception as e:
        warn_str = "failed to read in sac files for %s, SKIP (%s)" \
            % (station_id, sys.exc_info()[0])
        warnings.warn(warn_str, UserWarning)
        print(e)
        station['stat']['code'] = -1
        station['stat']['msg'] = "failed to read in sac files [%s]" \
            % UTCDateTime.now().isoformat()
        continue

      #------ get time samples of syn seismograms
      if not is_equal( [ (tr.stats.starttime, tr.stats.delta, tr.stats.npts) \
          for tr in syn_st ] ):
        raise Exception('%s not equal time samples in'\
            ' synthetic seismograms.' % (station_id))
      tr = syn_st[0]
      syn_delta = tr.stats.delta
      syn_npts = tr.stats.npts
      # left padding
      nl = int(left_pad/syn_delta)
      # right padding
      nr = int(right_pad/syn_delta)
      # right-padding zeros to optimal fft size
      # by making the largest-prime-factor samller than 10
      nt = optimal_fft_size(syn_npts+nl+nr, 10)
      nr = nt - (syn_npts + nl)
      # ENZ_syn
      # check if the origin time in sac files are consistent with event['t0']
      syn_origin_time = tr.stats.starttime + \
          (tr.stats.sac['o'] - tr.stats.sac['b'])
      if abs(syn_origin_time - event['t0']) > 1.0e-6:
        err = "Inconsistent origin time between sac headers (%s) and event['t0'] (%s)" \
            % (syn_origin_time, event['t0'])
        raise Exception(err)
      syn_starttime = tr.stats.starttime - nl*syn_delta
      syn_times = np.arange(nt) * syn_delta

      # read synthetic seismograms
      syn_ENZ = np.zeros((3, nt))
      for i in range(3):
        syn_ENZ[i,nl:(nl+syn_npts)] = syn_st[i].data

      #------ interpolate obs into the same time samples of syn
      obs_ENZ = np.zeros((3, nt))
      syn_nyq = 0.5/syn_delta
      flag = True
      # use the minimum obs endtime
      obs_endtime = min(
          obs_st[0].stats.endtime,
          obs_st[1].stats.endtime,
          obs_st[2].stats.endtime,
          )
      for i in range(3):
        tr = obs_st[i]
        obs_npts = tr.stats.npts
        obs_delta = tr.stats.delta
        obs_starttime = tr.stats.starttime
        #obs_endtime = tr.stats.endtime
        # check if obs has enough pre-event records
        first_arrtime = event['t0'] + meta['ttime'][0].time
        # required obs begin/end times
        obs_t1 = first_arrtime - obs_preevent
        #obs_t2 = syn_starttime + (nt-nr)*syn_delta
        #if obs_starttime > obs_t1 or obs_endtime < obs_t2:
        if obs_starttime > obs_t1:
          flag = False
          warn_str = "%s: record not long enough, SKIP %s" \
              % (obs_files[i], station_id)
          warnings.warn(warn_str)
          print("obs_starttime = ", obs_starttime)
          print("required starttime = ", obs_t1)
          station['stat']['code'] = -1
          station['stat']['msg'] = "%s [%s]" \
              % (warn_str, UTCDateTime.now().isoformat())
          break

        #DEBUG
        #print(station_id, obs_files[i]
        #obs_times = (obs_starttime-syn_starttime) + np.arange(obs_npts)*obs_delta
        #plt.plot(obs_times, tr.data, 'k')

        # lowpass below the nyquist frequency of synthetics
        # repeat twice to avoid numerical inaccuries
        tr.detrend(type='linear')
        tr.detrend(type='linear')
        # repeat lowpass filter twice to make sharper edge
        tr.filter('lowpass', freq=0.8*syn_nyq, corners=10, zerophase=True)
        tr.filter('lowpass', freq=0.8*syn_nyq, corners=10, zerophase=True)
        # interpolation: windowed sinc reconstruction
        obs_ENZ[i,:] = lanczos_interp1(tr.data, obs_delta,
            syn_times+(syn_starttime-obs_starttime), na=40)

        #DEBUG
        #plt.plot(obs_times, tr.data, 'r')
        ##plt.plot(syn_times, obs_ENZ[i,:], 'r')
        #plt.show()

      # if bad data, skip this station
      if not flag:
        continue

      #------ rotate obs to local ENU
      # projection matrix: obs = proj * ENZ => ENZ = inv(proj) * obs
      proj_matrix = np.zeros((3, 3))
      for i in range(3):
        chan = channel[i]
        sin_az = np.sin(np.deg2rad(chan['azimuth']))
        cos_az = np.cos(np.deg2rad(chan['azimuth']))
        sin_dip = np.sin(np.deg2rad(chan['dip']))
        cos_dip = np.cos(np.deg2rad(chan['dip']))
        # column vector = obs channel polarization 
        proj_matrix[i,0] = cos_dip*sin_az # proj to E
        proj_matrix[i,1] = cos_dip*cos_az # proj to N
        proj_matrix[i,2] = -sin_dip     # proj to U
      # inverse projection matrix: ENZ = inv(proj) * obs
      inv_proj = np.linalg.inv(proj_matrix)
      obs_ENZ = np.dot(inv_proj, obs_ENZ)

      #------ rotate syn to local ENU
      # SEM outputs in REF_ENU
      rotmat = meta['rotmat_local_to_ref']
      # from REF_ENU to local ENU
      syn_ENZ = np.dot(np.transpose(rotmat), syn_ENZ)

      #------ record data 
      if 'waveform' not in station:
        station['waveform'] = {}
      waveform = station['waveform']
      waveform['time_sample'] = {
          'starttime': syn_starttime, 'delta': syn_delta,
          'nt': nt, 'nl': nl, 'nr': nr,
          'obs_endtime':obs_endtime,
          }
      waveform['obs'] = obs_ENZ

      if syn_is_grn:
        waveform['grn'] = syn_ENZ
      else:
        waveform['syn'] = syn_ENZ

      station['stat']['code'] = 1
      station['stat']['msg'] = "read_obs_syn OK [%s]" \
          % (UTCDateTime.now().isoformat())

    #endfor station_id in station_dict:
  #enddef read_obs_syn

#
#==============================================================================
#

  def delete_window(self, ):
    """ 
    Delete all misfit windows
    
    Parameters
    ----------

    Notes
    -----
    """
    station_dict = self.data['station']

    #------ loop each station
    for station_id in station_dict:
      station = station_dict[station_id]

      # skip station not processed by read_obs_syn()
      if station['stat']['code'] < 1:
        continue

      # delete element 'window' from station
      if 'window' in station:
        del station['window']

#
#==============================================================================
#

  def add_window_body_wave(self,
      component='Z',
      phase='p,P',
      taper_percentage=0.1,
      begin_time=-50,
      end_time=50,
      filter_type='butter',
      filter_order=2,
      min_frequency=0.01,
      max_frequency=0.1,
      min_dist=0.0,
      max_dist=180.0,
      pre_weight=1.0,
      ):
    """ 
    Add misfit window for all stations
    
    Parameters
    ----------

    Notes
    -----
      taper_type only supports cosine window now.
      
      all phases in phase_list are contained in one time window.

    """
    # filter/taper parameters
    filter_dict = {
        'type': filter_type, 
        'order': filter_order, 
        'freqlim': np.array([min_frequency, max_frequency])}

    # half maximum period used to limit time window
    half_period = 0.5/np.min(filter_dict['freqlim'])

    if not 0.0 < taper_percentage < 0.5:
      raise ValueError("taper_percentage must be between 0 and 0.5.")

    event = self.data['event']
    station_dict = self.data['station']

    # initiate taup
    taup_model = TauPyModel(model="ak135")

    #------ loop each station
    for station_id in station_dict:
      station = station_dict[station_id]

      # skip station not processed by read_obs_syn()
      if station['stat']['code'] < 1:
        continue

      meta = station['meta']
      baz = meta['back_azimuth']
      gcarc = meta['dist_degree']

      #--- skip if outside the required distance range
      if gcarc < min_dist or gcarc > max_dist:
        continue

      #--- get valid time range of waveforms
      time_sample = station['waveform']['time_sample']
      syn_starttime = time_sample['starttime']
      syn_delta = time_sample['delta']
      syn_nt = time_sample['nt']
      # left/right zero padding length
      syn_nl = time_sample['nl']
      syn_nr = time_sample['nr']
      # valid data time range
      data_starttime = syn_starttime + syn_nl*syn_delta
      data_endtime = syn_starttime + (syn_nt-syn_nr)*syn_delta
      obs_endtime = time_sample['obs_endtime']
      if obs_endtime < data_endtime:
        data_endtime = obs_endtime

      # get ak135 traveltimes
      phase_list = phase.split(',')
      evdp_km = event['depth']
      if evdp_km < 0.0:
        evdp_km = 0.0
      arrivals = taup_model.get_travel_times(
          source_depth_in_km=evdp_km,
          distance_in_degree=gcarc,
          phase_list=phase_list,
          )

      # initialize window dict
      if 'window' not in station:
        station['window'] = {}
      window = station['window']

      window_id = "%s_%s" % (phase, component)

      # get time window
      if arrivals:
        # if more than one phase specified,
        # use a time window extended from the first to last arrivals
        # with addition to begin and end length
        ttime = np.array([arr.time for arr in arrivals])
        min_ttime = np.min(ttime)
        max_ttime = np.max(ttime)
      else:
        warn = "phase %s not found (dist=%f, evdp=%f), window not created" \
            % (phase, meta['dist_degree'], event['depth'] )
        warnings.warn(warn)
        continue
      win_starttime = event['t0'] + min_ttime + begin_time
      win_endtime = event['t0'] + max_ttime + end_time

      # check if time window lies out side of valid data time range
      if win_endtime < (data_starttime + half_period) \
          or win_starttime > (data_endtime - half_period):
        warn = "%s %s: window lies outside of the data time window" % (station_id, window_id)
        warnings.warn(warn)
        continue
      if win_starttime < data_starttime:
        warn = "%s %s has a starttime(%s) smaller than data starttime(%s)" \
            ", limited to data" % (station_id, window_id, win_starttime, data_starttime)
        warnings.warn(warn)
        win_starttime = data_starttime
      if win_endtime > (data_endtime - half_period):
        warn = "%s %s has an endtime(%s) larger than data endtime-half_period(%s - %f)" \
            ", limited to data" % (station_id, window_id, win_endtime, data_endtime, half_period)
        warnings.warn(warn)
        win_endtime = data_endtime - half_period
      if (win_endtime - win_starttime) < half_period/2.0:
        warn = "%s %s has an win_endtime(%s) smaller than win_starttime+half_period/2(%s + %f/2)" \
            ", skip" % (station_id, window_id, win_endtime, win_starttime, half_period)
        warnings.warn(warn)
        continue

      # window taper
      taper_dict = { 'type':'cosine', 'ratio':taper_percentage,
          'starttime':win_starttime, 'endtime':win_endtime}

      # window component 
      if component == 'Z': # vertcal component
        cmpaz = 0.0
        cmpdip = -90.0
      elif component == 'R': # radial component
        cmpaz = (baz + 180.0)%360.0
        cmpdip = 0.0
      elif component == 'T': # tangential component (TRZ: right-hand convention)
        cmpaz = (baz - 90.0)%360.0
        cmpdip = 0.0
      elif component == 'H': # horizontal particle motion 
        cmpaz = float('nan')
        cmpdip = 0.0
      elif component == 'F': # 3-d particle motion 
        cmpaz = float('nan')
        cmpdip = float('nan')
      else:
        print("[WARN] %s: unrecognized component, SKIP." % (comp))
        continue
      polarity_dict = {'component':component, 'azimuth': cmpaz, 'dip': cmpdip }

      # add window
      window[window_id] = {
        'stat': {
          'code': 0,
          'msg': "created on "+UTCDateTime.now().isoformat() },
        'filter': filter_dict,
        'taper': taper_dict,
        'polarity': polarity_dict,
        'pre_weight': pre_weight,
        }

    #endfor station_id, station in station_dict.iteritems():
  #enddef add_window_body_wave

#
#==============================================================================
#

  def add_window_surface_wave(self,
      component='T',
      phase='Love',
      min_slowness=25,
      max_slowness=40,
      taper_percentage=0.1,
      begin_time=-50,
      end_time=50,
      filter_type='butter',
      filter_order=2,
      min_frequency=0.01,
      max_frequency=0.05,
      pre_weight=1.0,
      ):
    """ 
    Add misfit window for all stations
    
    Parameters
    ----------
    phase: surface wave name

    min_/max_slowness: in s/deg
        e.g. Rayleigh: 25 ~ 40, Love: 25 ~ 35

    Notes
    -----
      taper_type only supports cosine window now.
      
    """
    # filter/taper parameters
    filter_dict = {
        'type': filter_type, 
        'order': filter_order, 
        'freqlim': np.array([min_frequency, max_frequency])}

    # half maximum period used to limit time window
    half_period = 0.5/np.min(filter_dict['freqlim'])

    if not 0.0 < taper_percentage < 0.5:
      raise ValueError("taper_percentage must be between 0 and 0.5.")

    event = self.data['event']
    station_dict = self.data['station']

    #------ loop each station
    for station_id in station_dict:
      station = station_dict[station_id]

      # skip station not processed by read_obs_syn()
      if station['stat']['code'] < 1:
        continue

      meta = station['meta']
      baz = meta['back_azimuth']
      gcarc = meta['dist_degree']

      #--- get valid time range of waveforms
      time_sample = station['waveform']['time_sample']
      syn_starttime = time_sample['starttime']
      syn_delta = time_sample['delta']
      syn_nt = time_sample['nt']
      # left/right zero padding length
      syn_nl = time_sample['nl']
      syn_nr = time_sample['nr']
      # valid data time range
      data_starttime = syn_starttime + syn_nl*syn_delta
      data_endtime = syn_starttime + (syn_nt-syn_nr)*syn_delta
      obs_endtime = time_sample['obs_endtime']
      if obs_endtime < data_endtime:
        data_endtime = obs_endtime

      # initialize window dict
      if 'window' not in station:
        station['window'] = {}
      window = station['window']

      window_id = "%s_%s" % (phase, component)

      # get time window
      win_starttime = event['t0'] + gcarc*min_slowness + begin_time
      win_endtime = event['t0'] + gcarc*max_slowness + end_time

      # check if time window lies out side of valid data time range
      if win_endtime < (data_starttime + half_period) \
          or win_starttime > (data_endtime - half_period):
        warn = "%s %s lies outside of the data time window" % (station_id, window_id)
        warnings.warn(warn)
        continue
      if win_starttime < data_starttime:
        warn = "%s %s has a starttime(%s) smaller than data starttime(%s)" \
            ", limited to data" % (station_id, window_id, win_starttime, data_starttime)
        warnings.warn(warn)
        win_starttime = data_starttime
      if win_endtime > (data_endtime - half_period):
        warn = "%s %s has an endtime(%s) larger than data endtime-half_period(%s - %f)" \
            ", limited to data" % (station_id, window_id, win_endtime, data_endtime, half_period)
        warnings.warn(warn)
        win_endtime = data_endtime - half_period
      if (win_endtime - win_starttime) < half_period/2.0:
        warn = "%s %s has an win_endtime(%s) smaller than win_starttime+half_period/2(%s + %f/2)" \
            ", skip" % (station_id, window_id, win_endtime, win_starttime, half_period)
        warnings.warn(warn)
        continue

      # window taper
      taper_dict = { 'type':'cosine', 'ratio':taper_percentage,
          'starttime':win_starttime, 'endtime':win_endtime}

      # window component 
      if component == 'Z': # vertcal component
        cmpaz = 0.0
        cmpdip = -90.0
      elif component == 'R': # radial component
        cmpaz = (baz + 180.0)%360.0
        cmpdip = 0.0
      elif component == 'T': # tangential component (TRZ: right-hand convention)
        cmpaz = (baz - 90.0)%360.0
        cmpdip = 0.0
      elif component == 'H': # horizontal particle motion 
        cmpaz = float('nan')
        cmpdip = 0.0
      elif component == 'F': # 3-d particle motion 
        cmpaz = float('nan')
        cmpdip = float('nan')
      else:
        print("[WARN] %s: unrecognized component, SKIP." % (comp))
        continue
      polarity_dict = {'component':component, 'azimuth': cmpaz, 'dip': cmpdip }

      # add window
      window[window_id] = {
        'stat': {
          'code': 0,
          'msg': "created on "+UTCDateTime.now().isoformat() },
        'filter': filter_dict,
        'taper': taper_dict,
        'polarity': polarity_dict,
        'pre_weight': pre_weight,
        }

    #endfor station_id, station in station_dict.iteritems():
  #enddef add_window_surface_wave

#
#======================================================
#

  def measure_adj(self,
      plot=False,
      cc_delta=0.001, 
      misfit_type='cc0',
      weight_param={
        'cc_tshift':[-10,-8, 8,10],
        'SNR':[10,15], 
        'CC0':[0.5,0.7],
        'CCmax':None,
        'dist':None}
      ):
    """ 
    calculate adjoint sources (dchi_du)

    Parameters
    ----------
    misfit_type:
      cc0: zero-lag cross-correlation misfit
      ccdt: cross-correlation traveltime
    weight_param : SNR, CCmax, CC0, cc_tshift

    Notes
    -----
    chi : misfit functional (normalized zero-lag correlation coef.)
    u : synthetic waveform

    """
    #------
    event = self.data['event']
    station_dict = self.data['station']

    #------ check weight_param
    if 'cc_tshift' in weight_param:
      if len(weight_param['cc_tshift']) != 4:
        raise Exception("weight_param['cc_tshift'] must have 4 corner values e.g. [-10,-8,8,10]!")

    # loop each station
    for station_id in station_dict:
      station = station_dict[station_id]
      # skip rejected statations
      if station['stat']['code'] < 1:
        continue

      meta = station['meta']
      window_dict = station['window']

      waveform = station['waveform']
      time_sample = waveform['time_sample']
      syn_starttime = time_sample['starttime']
      syn_delta = time_sample['delta']
      syn_nyq = 0.5/syn_delta
      syn_nt = time_sample['nt']
      syn_nl = time_sample['nl']
      syn_nr = time_sample['nr']
      syn_times = syn_delta * np.arange(syn_nt)

      obs = waveform['obs']
      syn_freq = np.fft.rfftfreq(syn_nt, d=syn_delta)
      if 'syn' in waveform:
        syn = waveform['syn']
      elif 'grn' in waveform:
        grn = waveform['grn']
        # source spectrum (moment-rate function)
        F_src = stf_gauss_spectrum(syn_freq, event['tau'])
        syn = np.fft.irfft(F_src*np.fft.rfft(grn), syn_nt)
      else:
        error_str = "%s: No synthetics found" % (station_id)
        raise Exception(error_str)

      #------ loop each window
      # sum of adjoint sources from all windows
      dchi_du = np.zeros((3, syn_nt))

      for window_id in window_dict:
        # window parameters
        window = window_dict[window_id]
        # skip bad windows
        if window['stat']['code'] < 0:
          continue

        #------ filter
        filter_dict = window['filter']
        filter_type = filter_dict['type']
        filter_order = filter_dict['order']
        filter_freqlim = filter_dict['freqlim']
        # filter design 
        if filter_type == 'butter':
          filter_b, filter_a = signal.butter(filter_order,
            np.array(filter_freqlim)/syn_nyq, btype='band')
        else:
          raise Exception("filter_type %s not recognized" % filter_type)
        # record filter coeff.
        filter_dict['a'] = filter_a
        filter_dict['b'] = filter_b

        #------ taper
        taper = window['taper']
        taper_type = taper['type']
        taper_ratio = taper['ratio']
        # time range
        window_starttime = taper['starttime']
        window_endtime = taper['endtime']
        window_len = window_endtime - window_starttime
        # taper design
        win_b = window_starttime - syn_starttime
        win_e = window_endtime - syn_starttime
        taper_width = window_len * min(taper_ratio, 0.5)
        win_c = [win_b, win_b+taper_width, win_e-taper_width, win_e]
        if taper_type == "cosine":
          win_func = cosine_taper(syn_times, win_c)
        else:
          raise Exception("taper_type not recognized.")
        taper['win'] = win_func

        #------ polarity 
        polarity = window['polarity']
        comp = polarity['component']
        cmpaz = polarity['azimuth']
        cmpdip = polarity['dip']
        if comp in ['Z', 'R', 'T']:
          sin_az = np.sin(np.deg2rad(cmpaz))
          cos_az = np.cos(np.deg2rad(cmpaz))
          sin_dip = np.sin(np.deg2rad(cmpdip))
          cos_dip = np.cos(np.deg2rad(cmpdip))
          n = np.array([ [cos_dip * sin_az], # cos(E, comp)
                 [cos_dip * cos_az], # N, comp
                 [-sin_dip] ])     # Z, comp
          proj_matrix = np.dot(n, n.transpose())
        elif comp == 'H': # horizontal vector 2d
          proj_matrix = np.identity(3)
          proj_matrix[2,2] = 0.0 # zero Z component
        elif comp == 'F': # full 3d vector
          proj_matrix = np.identity(3)
        else:
          print('[WARNING] %s:%s unrecognized component code, SKIP' \
              % (station_id, window_id))
          continue
        polarity['proj_matrix'] = proj_matrix

        #------ filter obs, syn
        #NOTE: use lfilter (causal filter) to avoid contamination from the right
        # end of the signal, but with asymmetric response and 
        # peak shift ~ 1/4 min. period (e.g. 0.01-0.1Hz -> 2.5s peak shift)
        # , however the duration of the filter response is determined by the
        # max. period (e.g. 0.01-0.1Hz -> ~50s). So the time window chosen 
        # should not be affected by the relatively small peak shift.
        #-- F * d
        obs_filt = signal.filtfilt(filter_b, filter_a, obs)
        #-- F * u (u = S*grn)
        syn_filt = signal.filtfilt(filter_b, filter_a, syn)
        #syn_filt = np.fft.irfft(F_src*np.fft.rfft(syn_filt), syn_nt)
        #DEBUG
        #for i in range(3):
        #  plt.subplot(311+i)
        #  plt.plot(syn_times, obs[i,:], 'k', syn_times, obs_filt[i,:], 'r')
        #plt.show()
        #-- noise: use signals 30s before first arrival time on obs
        first_arrtime = event['t0'] + meta['ttime'][0].time
        #FIXME: better choice of the time length before first arrival? 
        tnoise = (first_arrtime - 30.0) - syn_starttime
        noise_idx = syn_times < tnoise
        #t = syn_times[noise_idx]
        #b = t[0]
        #e = t[-1]
        #taper_width = (e-b) * 0.1
        #win_c = [b, b+taper_width, e-taper_width, e]
        #taper = cosine_taper(t, win_c)
        # F * noise
        noise_filt = obs_filt[:,noise_idx]

        #------ apply window taper and polarity projection
        # obs = w * F * d
        obs_filt_win = np.dot(proj_matrix, obs_filt) * win_func 
        # syn = w * F * u (u = S*g)
        syn_filt_win = np.dot(proj_matrix, syn_filt) * win_func 
        # noise (only projection)
        noise_filt_win = np.dot(proj_matrix, noise_filt)
        #DEBUG
        #diff = obs_ENZ_win - syn_ENZ_win
        #for i in range(3):
        #  plt.subplot(311+i)
        #  plt.plot(syn_times, obs_ENZ_win[i,:], 'k')
        #  plt.plot(syn_times, syn_ENZ_win[i,:], 'r')
        #  plt.plot(syn_times, diff[i,:], 'c')
        #plt.show()

        #------ measure SNR (based on maximum amplitude)
        Amax_obs = np.sqrt(np.max(np.sum(obs_filt_win**2, axis=0)))
        Amax_syn = np.sqrt(np.max(np.sum(syn_filt_win**2, axis=0)))
        Amax_noise =  np.sqrt(np.max(np.sum(noise_filt_win**2, axis=0)))
        if Amax_obs == 0: # bad record
          print('[WARN] %s:%s empty obs trace, SKIP.' % (
            station_id, window_id))
          window['stat']['code'] = -1
          window['stat']['msg'] = "Amax_obs=0"
          continue
        if Amax_noise == 0: # could occure when the data begin time is too close to the first arrival
          print('[WARN] %s:%s empty noise trace, SKIP.' % (
            station_id, window_id))
          window['stat']['code'] = -1
          window['stat']['msg'] = "Amax_noise=0"
          continue
        snr = 20.0 * np.log10(Amax_obs/Amax_noise)
 
        #------ measure CC time shift (between w*F*d and w*F*u)
        obs_norm = np.sqrt(np.sum(obs_filt_win**2))
        syn_norm = np.sqrt(np.sum(syn_filt_win**2))
        # window normalization factor (without dt)
        Nw = obs_norm * syn_norm
        # NOTE the order (obs,syn) is important. The positive time on 
        # CC means shifting syn in the positive time direction to match
        # the observed obs, and vice verser.
        # [-(nt-1), nt) * dt
        cc = np.zeros(2*syn_nt-1)
        for i in range(3):
          cc += signal.fftconvolve(
              obs_filt_win[i,:], syn_filt_win[i,::-1], 'full')
        cc /= Nw
        #DEBUG
        #print(window_id
        #print(cc[syn_nt-2] - np.sum(obs_ENZ_win * syn_ENZ_win)
        #print(cc[syn_nt-1] - np.sum(obs_ENZ_win * syn_ENZ_win)
        #print(cc[syn_nt] - np.sum(obs_ENZ_win * syn_ENZ_win)
        #print(cc[syn_nt+1] - np.sum(obs_ENZ_win * syn_ENZ_win)
        #print(cc[syn_nt+2] - np.sum(obs_ENZ_win * syn_ENZ_win)
        #-- zero-lag cc coeff.
        CC0 = cc[syn_nt-1] #the n-th point corresponds to zero lag time 
        AR0 = CC0 * syn_norm / obs_norm # amplitude ratio syn/obs 
        #DEBUG
        #print(CC0 - np.sum(obs_ENZ_win * syn_ENZ_win)/obs_norm/syn_norm
        #-- interpolate cc to finer time samples
        CC_shift_range = window_len/2.0 #TODO: more reasonable choice?
        ncc = int(CC_shift_range / cc_delta)
        cc_times = np.arange(-ncc,ncc+1) * cc_delta
        if syn_delta < cc_delta:
          warnings.warn("syn_delta(%f) < cc_time_step(%f)" \
              % (syn_delta, cc_delta))
        ti = cc_times + (syn_nt-1)*syn_delta  # -(npts-1)*dt: begin time in cc
        cci = lanczos_interp1(cc, syn_delta, ti, na=40)
        # time shift at the maximum correlation
        imax = np.argmax(cci)
        CC_time_shift = cc_times[imax]
        CCmax = cci[imax]
        ARmax = CCmax * syn_norm / obs_norm # amplitude ratio: syn/obs

        #------ window weighting based on SNR and misfit
        weight = window['pre_weight']
        if 'SNR' in weight_param:
          weight *= cosine_taper(snr, weight_param['SNR'])
        if 'CCmax' in weight_param:
          weight *= cosine_taper(CCmax, weight_param['CCmax'])
        if 'CC0' in weight_param:
          weight *= cosine_taper(CC0, weight_param['CC0'])
        if 'dist' in weight_param:
          dist = meta['dist_degree']
          dist_range = np.array(weight_param['dist'])
          weight *= cosine_taper(dist, dist_range)
        if 'cc_tshift' in weight_param:
          weight *= cosine_taper(CC_time_shift, weight_param['cc_tshift'])

        #------ measure adjoint source
        Aw = CC0 * obs_norm / syn_norm # window amplitude raito
        if misfit_type == 'cc0':
          # misfit: zero-lag cross-correlation
          # adjoint source: dchiw_du (misfit functional: zero-lag cc coef.)
          # dchiw_du = conj(F * [S]) * w * [ w * F * d - A * w * F * S * g] / N, 
          # , where A = CC0(un-normalized) / norm(u)**2, N = norm(d)*norm(u)
          #-- dchiw_du
          #NOTE: *dt is put back to Nw
          dchiw_du1 = win_func * (obs_filt_win - Aw*syn_filt_win) / Nw / syn_delta
          # apply conj(F), equivalent to conj(F*conj(adj))
          # for two-pass filter (zero phase) conj(F) = F
          dchiw_du = signal.filtfilt(filter_b, filter_a, dchiw_du1[:,::-1])
          dchiw_du = dchiw_du[:,::-1]
        elif misfit_type == 'ccdt':
          # misfit: cross-correlation traveltime difference -1*|ccdtau|^2
          # adjoint source: dchiw_du = ccdt * dtau/du
          # dchiw_du = conj(F)*w*d/dt(w*F*u)/N,
          # , where N = norm(d/dt(w*F*u))
          #-- dchiw_du
          #NOTE: *dt is put back to Nw
          dwFu_dt = np.fft.irfft(np.fft.rfft(syn_filt_win)*2.0j*np.pi*syn_freq, n=syn_nt)
          w_dwFu_dt = win_func * dwFu_dt
          # apply conj(F), for two-pass filter (zero phase) conj(F) = F
          conjF_w_dwFu_dt = signal.filtfilt(filter_b, filter_a, w_dwFu_dt[:,::-1])
          conjF_w_dwFu_dt = conjF_w_dwFu_dt[:,::-1]
          # 
          dchiw_du = -1*CC_time_shift*conjF_w_dwFu_dt/np.sum(dwFu_dt**2)
          #DEBUG
          #print(CC_time_shift)
          #plt.subplot(411)
          #plt.plot(syn_times, syn_filt_win[1,:])
          #plt.subplot(412)
          #plt.plot(syn_times, dwFu_dt[1,:])
          #plt.subplot(413)
          #plt.plot(syn_times, conjF_w_dwFu_dt[1,:])
          #plt.subplot(414)
          #plt.plot(syn_times, dchiw_du[1,:])
          #plt.show()
        else:
          error_str = "%s:%s: unknown misfit type (%s)" % (station_id, window_id, misfit_type)
          raise Exception(error_str)

        #DEBUG
        #for i in range(3):
        #  plt.subplot(311+i)
        #  plt.plot(syn_times, dchiw_du1[i,:], 'k')
        #  plt.plot(syn_times, dchiw_du[i,:], 'r')
        #plt.show()
        # add into total dchi_du
        dchi_du += weight * dchiw_du
        #-- dchiw_dg = conj(S) * dchiw_du
        #dchiw_dg = np.fft.irfft(np.conjugate(F_src) * np.fft.rfft(dchiw_du), syn_nt)
        # add into total dchi_dg
        #dchi_dg += weight * dchiw_dg 
        #DEBUG
        #for i in range(3):
        #  plt.subplot(311+i)
        #  plt.plot(syn_times, dchiw_du[i,:], 'k')
        #  plt.plot(syn_times, dchiw_dg[i,:], 'r')
        #plt.show()

        #------ record results
        quality_dict = {
            'Amax_obs': Amax_obs, 'Amax_syn': Amax_syn, 
            'Amax_noise': Amax_noise, 'SNR': snr}
        cc_dict = {
            #'time': cc_times, 'cc': cci,
            'cc_tshift': CC_time_shift,
            'CC0': CC0, 'CCmax': CCmax,
            'AR0': AR0, 'ARmax': ARmax,
            'Nw':Nw, 'Aw':Aw }
        window['quality'] = quality_dict
        window['cc'] = cc_dict
        window['weight'] = weight
        window['stat'] = {'code': 1, 
            'msg': "measure adj on "+UTCDateTime.now().isoformat()}

        #------ plot measure window and results 
        if plot:
          syn_npts = syn_nt - syn_nl - syn_nr
          syn_orientation_codes = ['E', 'N', 'Z']
          adj = dchiw_du
          Amax_adj = np.sqrt(np.max(np.sum(adj**2, axis=0)))
          t = syn_times
          for i in range(3):
            plt.subplot(411+i)
            if i == 0:
              plt.title('%s dt %.2f CCmax %.3f ARmax %.3f CC0 %.3f '
                  'AR0 %.3f \nAobs %g Anoise %g SNR %.1f weight %.3f'
                  % (station_id, CC_time_shift, CCmax, ARmax, 
                    CC0, AR0, Amax_obs, Amax_noise, snr, weight) )
            idx_plt = range(syn_nl,(syn_nl+syn_npts))
            plt.plot(t[idx_plt], obs_filt[i,idx_plt]/Amax_obs, 'k', linewidth=0.2)
            plt.plot(t[idx_plt], syn_filt[i,idx_plt]/Amax_obs*Aw, 'r', linewidth=0.2)
            #plt.plot(t[noise_idx], noise_filt[i,:]/Amax_obs, 'b', linewidth=1.0)
            idx = (win_b <= syn_times) & (syn_times <= win_e)
            plt.plot(t[idx], obs_filt_win[i,idx]/Amax_obs, 'k', linewidth=1.0)
            plt.plot(t[idx], syn_filt_win[i,idx]/Amax_obs*Aw, 'r', linewidth=1.0)
            plt.plot(t[idx_plt], adj[i,idx_plt]/Amax_adj, 'c', linewidth=1.0)
            plt.ylim((-1.5, 1.5))
            #plt.xlim((min(t), max(t)))
            plt.xlim((t[syn_nl], t[syn_nl+syn_npts-1]))
            plt.ylabel(syn_orientation_codes[i])
          plt.subplot(414)
          plt.plot(cc_times, cci, 'k-')
          plt.xlim((min(cc_times), max(cc_times)))
          plt.ylabel(window_id)
          plt.show()
      #====== end for window_id in windows:

      #------ store adjoint source for this station
      station['dchi_du'] = dchi_du

      #DEBUG
      #for i in range(3):
      #  plt.subplot(311+i)
      #  plt.plot(syn_times, dchi_du[i,:], 'k')
      #  plt.plot(syn_times, dchi_dg[i,:], 'r')
      #plt.show()

    #endfor station_id in station_dict:
  #enddef measure_windows_for_one_station(self,

#
#======================================================
#

  def output_misfit(self, out_file):
    """
    Output misfit measurements

    """
    event = self.data['event']
    station_dict = self.data['station']

    fp = open(out_file, 'w')
    fp.write("#station window weight CC0 CCmax dt_cc SNR AR0 ARmax\n")

    for station_id in station_dict:
      station = station_dict[station_id]
      # skip rejected stations
      if station['stat']['code'] < 1:
        continue

      window_dict = station['window']
      for window_id in window_dict:
        window = window_dict[window_id]
        # skip rejected/un-measured windows 
        if window['stat']['code'] < 1:
          continue

        # skip bad windows 
        if window['weight'] < 1.0e-3:
          continue

        cc = window['cc']
        quality = window['quality']

        fp.write("{:15s} {:15s} {:7.5f}  {:7.5f}  {:7.5f}  {:+7.3f}" \
            "  {:7.3f}  {:12.5e}  {:12.5e}\n".format(
          station_id, window_id, window['weight'], 
          cc['CC0'], cc['CCmax'], cc['cc_tshift'], 
          quality['SNR'], cc['AR0'], cc['ARmax']))

    fp.close()

#
#======================================================
#

  def plot_histogram(self, 
      max_dt=10,
      nbins=100,
      outfig='cc_tshift_histogram.pdf',
      ):
    """
    Plot histograms of measured cc time shifts for each windows

    Parameters
    ---------
    (string) window_type: type of windows to plot
        only surface or body are supported

    (float) max_dt: maximum abs(dt) to use
    """
    event = self.data['event']
    station_dict = self.data['station']

    # get measurement results
    data = {}
    for station_id in station_dict:
      station = station_dict[station_id]
      # skip rejected stations
      if station['stat']['code'] < 1:
        continue
      window_dict = station['window']
      # loop all windows
      for window_id in window_dict:
        window = window_dict[window_id]
        # skip rejected/un-measured windows 
        if window['stat']['code'] < 1:
          continue
        # skip bad windows
        if window['weight'] < 1.0e-3:
          continue
        # append info to list
        if window_id not in data:
          data[window_id] = []
        cc_dict = window['cc']
        data[window_id].append(cc_dict['cc_tshift'])

    #------ plot
    with PdfPages(outfig) as pdf:
      # individual windows
      for window_id in data:
        dt_cc = np.array(data[window_id])
        idx = np.abs(dt_cc)<=max_dt
        plt.hist(dt_cc[idx], nbins, histtype='step')
        title_str = "%s %s %f$\pm$%f" % (event['id'], window_id, np.mean(dt_cc[idx]), np.std(dt_cc[idx]))
        plt.title(title_str)
        plt.xlabel("dt_cc [obs-syn] (second)")
        plt.ylabel("Window number")
        pdf.savefig()
        plt.close()
      # body-wave windows
      dt_cc = []
      for window_id in data:
        if 'surface_' not in window_id:
          dt_cc += data[window_id]
      dt_cc = np.array(dt_cc)
      idx = np.abs(dt_cc)<=max_dt
      plt.hist(dt_cc[idx], nbins, histtype='step')
      title_str = "%s body-wave %f$\pm$%f" % (event['id'], np.mean(dt_cc[idx]), np.std(dt_cc[idx]))
      plt.title(title_str)
      plt.xlabel("dt_cc [obs-syn] (second)")
      plt.ylabel("Window number")
      pdf.savefig()
      plt.close()
      # surface_RZ 
      dt_cc = []
      for window_id in data:
        if ('surface_R' in window_id) or ('surface_Z' in window_id):
          dt_cc += data[window_id]
      dt_cc = np.array(dt_cc)
      idx = np.abs(dt_cc)<=max_dt
      plt.hist(dt_cc[idx], nbins, histtype='step')
      title_str = "%s surface_RZ %f$\pm$%f" % (event['id'], np.mean(dt_cc[idx]), np.std(dt_cc[idx]))
      plt.title(title_str)
      plt.xlabel("dt_cc [obs-syn] (second)")
      plt.ylabel("Window number")
      pdf.savefig()
      plt.close()

#
#======================================================
#

  def plot_histogram_combine(self,
      window_type='surface',
      max_dt=10,
      nbins=100,
      out_file='./histogram_dt_cc.pdf',
      ):
    """
    Plot histograms of measured cc time shifts for combined window type

    Parameters
    ---------
    (string) window_type: type of windows to plot
        only surface or body are supported

    (float) max_dt: maximum abs(dt) to use

    """
    event = self.data['event']
    station_dict = self.data['station']

    # get measurement results
    weight = []
    cc0 = []
    dt_cc = []
    for station_id in station_dict:
      station = station_dict[station_id]
      # skip rejected stations
      if station['stat']['code'] < 1:
        continue
      window_dict = station['window']
      # loop all windows
      for window_id in window_dict:
        window = window_dict[window_id]
        # skip rejected/un-measured windows 
        if window['stat']['code'] < 1:
          continue
        # check window type
        if window_type == 'surface':
          if 'surface' not in window_id:
            continue
        elif window_type == 'body':
          if 'surface' in window_id:
            continue
        else:
          raise Exception('window_type should be either surface or body !')
        # skip bad windows
        if window['weight'] < 1.0e-3:
          continue
        # append info to list
        weight = window['weight']
        cc_dict = window['cc']
        cc0.append(cc_dict['CC0'])
        dt_cc.append(cc_dict['cc_tshift'])

    # plot
    weight = np.array(weight)
    cc0 = np.array(cc0)
    dt_cc = np.array(dt_cc)

    idx = np.abs(dt_cc)<=max_dt
    
    dt_mean = np.mean(dt_cc)
    dt_std = np.std(dt_cc)

    plt.hist(dt_cc[idx], nbins=nbins)
    title_str = "%s %s_wave mean(std)=%f(%f)" % (event['id'], window_type, dt_mean, dt_std)
    plt.title(title_str)
    plt.xlabel("dt_cc [obs-syn] (second)")
    plt.ylabel("Event number")
    plt.savefig(out_file, format='pdf')


#
#======================================================
#

  def output_adj_REF_ENU(self, 
      out_dir='adj',
      syn_band_code='BX'):
    """
    Output adjoint sources

    """
    syn_orientation_codes = ['X', 'Y', 'Z']
    event = self.data['event']
    station_dict = self.data['station']

    tr = Trace()
    for station_id in station_dict:
      station = station_dict[station_id]
      meta = station['meta']
      # skip rejected statations
      if station['stat']['code'] < 1:
        continue
      # time samples
      waveform = station['waveform']
      time_sample = waveform['time_sample']
      syn_starttime = time_sample['starttime']
      syn_delta = time_sample['delta']
      syn_nt = time_sample['nt']
      syn_nl = time_sample['nl']
      syn_nr = time_sample['nr']
      # without padding
      npts = syn_nt - syn_nl - syn_nr
      starttime = syn_starttime + syn_nl*syn_delta
      # time samples for ascii output, referred to origin time
      syn_times = np.arange(npts)*syn_delta
      b = starttime - event['t0']
      syn_times += b

      # adjoint source
      adj = station['dchi_du']

      # now adj is in local ENU coordinate, we need to conver it to REF_ENU
      rotmat = meta['rotmat_local_to_ref']
      # from REF_ENU to local ENU
      adj = np.dot(rotmat, adj)

      # loop ENZ
      for i in range(3):
        tr.data = adj[i, syn_nl:(syn_nl+npts)]
        tr.stats.starttime = starttime
        tr.stats.delta = syn_delta

        out_file = '{:s}/{:s}.{:2s}{:1s}'.format(
            out_dir, station_id, syn_band_code,
            syn_orientation_codes[i])

        # sac format
        tr.write(out_file + '.adj.sac', 'sac')

        # ascii format (needed by SEM)
        # time is relative to event origin time: t0
        with open(out_file+'.adj','w') as fp:
          for j in range(npts):
            fp.write("{:16.9e}  {:16.9e}\n".format(
              syn_times[j], adj[i,syn_nl+j]))

      #endfor i in range(3):
    #endfor station_id in station_dict:
  #enddef output_adjoint_source

#
#======================================================
#

  def read_srcfrechet(self, filename=None):
    """ Read in source derivative of misfit function
        Dchi/Dxs, Dchi/Dmt
    """
    with open(filename, 'r') as f:
      lines = [ x for x in f.readlines() if not(x.startswith('#')) ]

    lines = [x.split() for x in lines]

    t0  = float(lines[0][0]);  dchi_dt0  = float(lines[0][1])
    tau = float(lines[1][0]);  dchi_dtau = float(lines[1][1])
    x   = float(lines[2][0]);  dchi_dx   = float(lines[2][1])
    y   = float(lines[3][0]);  dchi_dy   = float(lines[3][1])
    z   = float(lines[4][0]);  dchi_dz   = float(lines[4][1])
    mxx = float(lines[5][0]);  dchi_dmxx = float(lines[5][1])
    myy = float(lines[6][0]);  dchi_dmyy = float(lines[6][1])
    mzz = float(lines[7][0]);  dchi_dmzz = float(lines[7][1])
    mxy = float(lines[8][0]);  dchi_dmxy = float(lines[8][1])
    mxz = float(lines[9][0]);  dchi_dmxz = float(lines[9][1])
    myz = float(lines[10][0]); dchi_dmyz = float(lines[10][1])

    dchi_dxs = np.array([dchi_dx, dchi_dy, dchi_dz])

    dchi_dmt = np.zeros((3,3))
    dchi_dmt[0,0] = dchi_dmxx
    dchi_dmt[1,1] = dchi_dmyy
    dchi_dmt[2,2] = dchi_dmzz
    dchi_dmt[0,1] = dchi_dmxy
    dchi_dmt[1,0] = dchi_dmxy
    dchi_dmt[0,2] = dchi_dmxz
    dchi_dmt[2,0] = dchi_dmxz
    dchi_dmt[1,2] = dchi_dmyz
    dchi_dmt[2,1] = dchi_dmyz

    # check if the same as event info
    data = self.data
    event = data['event']
    #...

    # record 
    src_frechet = {
        't0':dchi_dt0,
        'tau':dchi_dtau,
        'xs':dchi_dxs,
        'mt':dchi_dmt,
        'stat': {'code':0, 'msg':"created on "+UTCDateTime.now().isoformat()}
        }
    data['src_frechet'] = src_frechet

#
#======================================================
#

  def make_cmt_perturb(self,
      srcfrechet_file,
      max_dxs_ratio=0.001, 
      out_cmt_file="CMTSOLUTION",
      out_perturb_file="CMTSOLUTION.perturb"
      ):
    """Make perturbed CMTSOLUTION based on source gradient
    """
    # get source parameters
    event = self.data['event']

    # check parameters
    if max_dxs_ratio <= 0.0:
      error_str = "max_dxs_ratio must > 0"
      raise ValueError(error_str)

    #====== read source gradient
    with open(srcfrechet_file, 'r') as f:
      lines = [ x for x in f.readlines() if not(x.startswith('#')) ]
    
    lines = [x.split() for x in lines]
    
    t0  = float(lines[0][0]);  dchi_dt0  = float(lines[0][1])
    tau = float(lines[1][0]);  dchi_dtau = float(lines[1][1])
    
    xs = np.zeros(3); dchi_dxs = np.zeros(3)
    xs[0] = float(lines[2][0]);  dchi_dxs[0] = float(lines[2][1])
    xs[1] = float(lines[3][0]);  dchi_dxs[1] = float(lines[3][1])
    xs[2] = float(lines[4][0]);  dchi_dxs[2] = float(lines[4][1])
    
    mt = np.zeros((3,3)); dchi_dmt = np.zeros((3,3));
    mt[0,0] = float(lines[5][0]);  dchi_dmt[0,0] = float(lines[5][1])
    mt[1,1] = float(lines[6][0]);  dchi_dmt[1,1] = float(lines[6][1])
    mt[2,2] = float(lines[7][0]);  dchi_dmt[2,2] = float(lines[7][1])
    
    mt[0,1] = float(lines[8][0]);  dchi_dmt[0,1] = float(lines[8][1])
    mt[1,0] = mt[0,1]; dchi_dmt[1,0] = dchi_dmt[0,1]
    
    mt[0,2] = float(lines[9][0]);  dchi_dmt[0,2] = float(lines[9][1])
    mt[2,0] = mt[0,2]; dchi_dmt[2,0] = dchi_dmt[0,2]
    
    mt[1,2] = float(lines[10][0]); dchi_dmt[1,2] = float(lines[10][1])
    mt[2,1] = mt[1,2]; dchi_dmt[2,1] = dchi_dmt[1,2]
    
    # correlation between mt and dchi_dmt
    cc = np.sum(dchi_dmt*mt)/np.sum(mt**2)**0.5/np.sum(dchi_dmt**2)**0.5
    print("cc = ", cc)
    
    #====== get gradient for xs_ratio and mt_ratio
    # xs = R_earth * xs_ratio
    dchi_dxs_ratio = R_earth * dchi_dxs

    # mt = m0 * mt_ratio
    m0 = (0.5*np.sum(mt**2))**0.5
    dchi_dmt_ratio = m0 * dchi_dmt
    
    # make dchi_dmt orthogonal to mt, i.e. keep scalar moment unchanged
    dchi_dmt_ratio_ortho = dchi_dmt_ratio - mt*np.sum(dchi_dmt_ratio*mt)/np.sum(mt**2)
    print(dchi_dmt_ratio_ortho)
    
    #====== make perturbed CMTSOLUTION
    scale_factor = max_dxs_ratio/(np.sum(dchi_dxs_ratio**2))**0.5
    
    dxs_ratio = scale_factor * dchi_dxs_ratio
    dmt_ratio = scale_factor * dchi_dmt_ratio_ortho
    
    #====== write out new CMTSOLUTION
    xs_perturb = xs + R_earth*dxs_ratio

    mt_perturb = mt + m0*dmt_ratio
    # force mt_perturb to have the same scalar moment as mt
    mt_perturb = m0 * mt_perturb/(0.5*np.sum(mt_perturb**2))**0.5
    
    # write out new CMTSOLUTION file
    with open(out_cmtfile, 'w') as fp:
      fp.write('%s\n' % event['header'])
      fp.write('%-18s %s_dmt\n' % ('event name:',event['id']))
      fp.write('%-18s %+15.8E\n' % ('t0(s):',    0.0))
      fp.write('%-18s %+15.8E\n' % ('tau(s):',   0.0))
      fp.write('%-18s %+15.8E\n' % ('x(m):',     xs_perturb[0]))
      fp.write('%-18s %+15.8E\n' % ('y(m):',     xs_perturb[1]))
      fp.write('%-18s %+15.8E\n' % ('z(m):',     xs_perturb[2]))
      fp.write('%-18s %+15.8E\n' % ('Mxx(N*m):', mt_perturb[0,0]))
      fp.write('%-18s %+15.8E\n' % ('Myy(N*m):', mt_perturb[1,1]))
      fp.write('%-18s %+15.8E\n' % ('Mzz(N*m):', mt_perturb[2,2]))
      fp.write('%-18s %+15.8E\n' % ('Mxy(N*m):', mt_perturb[0,1]))
      fp.write('%-18s %+15.8E\n' % ('Mxz(N*m):', mt_perturb[0,2]))
      fp.write('%-18s %+15.8E\n' % ('Myz(N*m):', mt_perturb[1,2]))
    
    # write out parameter changes
    with open(out_file, 'w') as fp:
      fp.write('%s\n' % event['header'])
      fp.write('%-18s %s_dmt\n' % ('event name:',event['id']))
      fp.write('%-18s %+15.8E\n' % ('t0(s):',    0.0))
      fp.write('%-18s %+15.8E\n' % ('tau(s):',   0.0))
      fp.write('%-18s %+15.8E\n' % ('dx_ratio(m):',     xs_perturb[0]))
      fp.write('%-18s %+15.8E\n' % ('dy_ratio(m):',     xs_perturb[1]))
      fp.write('%-18s %+15.8E\n' % ('dz_ratio(m):',     xs_perturb[2]))
      fp.write('%-18s %+15.8E\n' % ('dmxx_ratio(N*m):', mt_perturb[0,0]))
      fp.write('%-18s %+15.8E\n' % ('dmyy_ratio(N*m):', mt_perturb[1,1]))
      fp.write('%-18s %+15.8E\n' % ('dmzz_ratio(N*m):', mt_perturb[2,2]))
      fp.write('%-18s %+15.8E\n' % ('dmxy_ratio(N*m):', mt_perturb[0,1]))
      fp.write('%-18s %+15.8E\n' % ('dmxz_ratio(N*m):', mt_perturb[0,2]))
      fp.write('%-18s %+15.8E\n' % ('dmyz_ratio(N*m):', mt_perturb[1,2]))

#
#======================================================
#

#  def waveform_der_stf(self):
#    """ Calculate waveform derivatives for source time function (t0, tau)
#    """
#    event = self.data['event']
#    t0 = event['t0']
#    tau = event['tau']
#
#    station_dict = self.data['station']
#    for station_id in station_dict:
#      station = station_dict[station_id]
#      # skip rejected statations
#      if station['stat']['code'] < 0:
#        continue
#
#      # source time function
#      waveform = station['waveform']
#      time_sample = waveform['time_sample']
#      starttime = time_sample['starttime']
#      dt = time_sample['delta']
#      nt = time_sample['nt']
#      t = np.arange(nt) * dt + (starttime - t0) #referred to t0
#      # s(t), Ds(t)/Dt0, Ds(t)/Dtau
#      freq = np.fft.rfftfreq(nt, d=dt)
#      F_src, F_ds_dt0, F_ds_dtau = stf_gauss_spectrum(freq, tau)
#
#      #------ waveform derivative
#      # green's function
#      grn = waveform['grn']
#      # convolve Ds(t)/Dt0,tau with Green's function
#      du_dt0 = np.fft.irfft(F_ds_dt0 * np.fft.rfft(grn), nt)
#      du_dtau = np.fft.irfft(F_ds_dtau * np.fft.rfft(grn), nt)
#      # zero records before origin time (wrap around from the end)
#      idx = t < -5.0*tau
#      du_dt0[:,idx] = 0.0
#      du_dtau[:,idx] = 0.0
#
#      #------ misfit derivative
#      # adjoint source = Dchi/Du
#      dchi_du = station['dchi_du']
#      dchi_dt0 = np.sum(dchi_du * du_dt0) * dt
#      dchi_dtau = np.sum(dchi_du * du_dtau) * dt
#
#      #------ record derivatives
#      if 'waveform_der' not in station:
#        station['waveform_der'] = {}
#      station['waveform_der']['dt0'] = {
#          'dm':1.0, 'du':du_dt0, 'dchi':dchi_dt0 }
#      station['waveform_der']['dtau'] = {
#          'dm':1.0, 'du': du_dtau, 'dchi':dchi_dtau }
#
#      # DEBUG
#      #print(dchi_dt0, dchi_dtau
#      #for i in range(3):
#      #  plt.subplot(311+i)
#      #  #plt.plot(t, dchi_du[i,:], 'k')
#      #  plt.plot(t, du_dt0[i,:], 'b', t, du_dtau[i,:], 'r')
#      #plt.show()
#
#  #enddef derivative_stf(self)

#
#======================================================
#

  def make_cmt_dxs(self, out_file="CMTSOLUTION.dxs", norm=2000.0, GPS_ELLPS="WGS84"):
    """ Calculate derivative for source location along one direction
    norm: displacement in meter
    """
    norm = float(norm)
    if norm <= 0.0:
      raise Exception("norm(dxs) must be larger than 0.")

    # initialize pyproj objects
    #geod = pyproj.Geod(ellps='WGS84')
    ecef = pyproj.Proj(proj='geocent', ellps=GPS_ELLPS)
    lla = pyproj.Proj(proj='latlong', ellps=GPS_ELLPS)

    # get source parameters
    event = self.data['event']
    tau = event['tau']
    xs = event['xs']
    mt = event['mt']

    # get perturbed source location
    if 'src_frechet' not in self.data:
      raise Exception('src_frechet not set.')
    src_frechet = self.data['src_frechet']
    dxs = src_frechet['xs']
    # normalize dxs
    dxs = dxs/(np.sum(dxs**2))**0.5
    # apply given norm 
    dxs *= norm
    # get new src location
    xs1 = xs + dxs
    lon, lat, alt = pyproj.transform(ecef, lla, xs1[0], xs1[1], xs1[2])
    depth = -alt
    if depth < 0.0:
      warnings.warn("perturbed src depth %f < 0.0" % (depth/1000.0))

    # record dxs
    if 'src_perturb' not in self.data:
      self.data['src_perturb'] = {}
    self.data['src_perturb']['xs'] = dxs

    # write out new CMTSOLUTION file
    with open(out_file, 'w') as fp:
      fp.write('%s\n' % event['header'])
      fp.write('%-18s %s_dxs\n' % ('event name:',event['id']))
      fp.write('%-18s %+15.8E\n' % ('t0(s):',    0.0))
      fp.write('%-18s %+15.8E\n' % ('tau(s):',   0.0))
      fp.write('%-18s %+15.8E\n' % ('x(m):',     xs1[0]))
      fp.write('%-18s %+15.8E\n' % ('y(m):',     xs1[1]))
      fp.write('%-18s %+15.8E\n' % ('z(m):',     xs1[2]))
      fp.write('%-18s %+15.8E\n' % ('Mxx(N*m):', mt[0,0]))
      fp.write('%-18s %+15.8E\n' % ('Myy(N*m):', mt[1,1]))
      fp.write('%-18s %+15.8E\n' % ('Mzz(N*m):', mt[2,2]))
      fp.write('%-18s %+15.8E\n' % ('Mxy(N*m):', mt[0,1]))
      fp.write('%-18s %+15.8E\n' % ('Mxz(N*m):', mt[0,2]))
      fp.write('%-18s %+15.8E\n' % ('Myz(N*m):', mt[1,2]))

#
#======================================================
#

  def waveform_der_source(self,
      syn_dir='output_perturb',
      syn_band_code='MX',
      syn_suffix='.sem.sac',
      model_name='xs_mt',
      sac_dir=None):
    """ 
    Calculate waveform derivative for source parameters

    Notes
    -----
    use finite difference to get waveform derivative

    """
    syn_orientation_codes = ['E', 'N', 'Z']
    event = self.data['event']
    station_dict = self.data['station']

    for station_id in station_dict:
      station = station_dict[station_id]
      # skip rejected statations
      if station['stat']['code'] < 1:
        continue

      #------ time samples
      waveform = station['waveform']
      time_sample = waveform['time_sample']
      starttime = time_sample['starttime']
      dt = time_sample['delta']
      nt = time_sample['nt']
      nl = time_sample['nl'] # npts of left padding
      nr = time_sample['nr'] # npts of right padding
      sem_nt = nt-nl-nr # number of time sample number in SEM simulation 
      #DEBUG
      #t = np.arange(nt) * dt + (starttime - t0) #referred to t0

      #------ get file paths of syn seismograms
      syn_files = [ '{:s}/{:s}.{:2s}{:1s}{:s}'.format(
        syn_dir, station_id, syn_band_code, x, syn_suffix)
        for x in syn_orientation_codes ]

      #------ read in obs, syn seismograms
      syn_st  = read(syn_files[0])
      syn_st += read(syn_files[1])
      syn_st += read(syn_files[2])

      #------ check the same time samples as original syn
      if not is_equal( [ (tr.stats.starttime, tr.stats.delta, tr.stats.npts) \
          for tr in syn_st ] ):
        raise Exception('%s: not equal time samples in'\
            ' synthetic seismograms.' % (station_id))
      tr = syn_st[0]

      if tr.stats.delta != dt:
        raise Exception("%s: not the same dt for diff-srcloc!" % (station_id))

      tr_starttime =  tr.stats.starttime - nl*dt
      if tr_starttime != starttime:
      #if np.abs(tr_starttime - starttime) > 1.0e-5:
        raise Exception("%s: not the same starttime for diff-srcloc!" % (station_id))

      if tr.stats.npts != sem_nt:
        raise Exception("%s: not the same npts for diff-srcloc!" % (station_id))

      #------ read syn seismograms from perturbed source location
      syn_ENZ = np.zeros((3, nt))
      for i in range(3):
        syn_ENZ[i,nl:(nl+sem_nt)] = syn_st[i].data

      # waveform partial derivatives 
      g0 = waveform['grn']
      dg = syn_ENZ - g0

      #------ record derivatives
      if 'waveform_der' not in station:
        station['waveform_der'] = {}
      station['waveform_der'][model_name] = {
          'dg':dg }

      # DEBUG
      #print(dchi
      #for i in range(3):
      #  plt.subplot(311+i)
      #  #plt.plot(t,grn0[i,:],'k', t,syn_ENZ[i,:],'r', t,dg[i,:], 'b')
      #  plt.plot(t, du[i,:], 'k')
      #plt.show()
      if sac_dir:
        for i in range(3):
          tr.data = du[i,:]
          tr.stats.starttime = starttime
          tr.stats.delta = dt
          tr.stats.npts = nt
          out_file = '{:s}/{:s}.{:2s}{:1s}'.format(
              sac_dir, station_id, syn_band_code,
              syn_orientation_codes[i])
          tr.write(out_file, 'sac')

#
#======================================================
#

  def make_cmt_dmt(self,
      out_file="CMTSOLUTION.dmt",
      fix_M0=True, zerotrace=True, ratio_M0=0.01):
    """ Calculate derivative for source location along one direction
      fix_M0: project dmt orthogonal to mt to keep seismic moment M0 = sqrt(0.5*m:m) fixed
      zerotrace: zero tr(dmt)
      ratio_M0: magnitude of dmt is set to the ratio of M0
    """
    # get source parameters
    event = self.data['event']
    tau = event['tau']
    xs = event['xs']
    mt = event['mt']
    
    # check parameters
    if ratio_M0 <= 0.0:
      error_str = "ratio_M0(%f) must > 0" % ratio_M0
      raise ValueError(error_str)

    # get perturbed moment tensor
    if 'src_frechet' not in self.data:
      raise Exception('src_frechet not set.')
    src_frechet = self.data['src_frechet']
    # set dmt parallel to dchi_dmt
    dmt = src_frechet['mt']
    # project dmt perpendicular to M0 change direction
    if fix_M0:
      dmt = dmt - mt*np.sum(dmt*mt)/np.sum(mt**2)
    if zerotrace:
      dmt = dmt - np.identity(3)*np.trace(dmt)/3.0
    # normalize dmt to have unit seismic moment
    dmt = dmt/(0.5*np.sum(dmt**2))**0.5
    # use ratio_M0 as the magnitude of dmt
    m0 = (0.5*np.sum(mt**2))**0.5
    dmt *= ratio_M0 * m0

    # record dmt
    if 'src_perturb' not in self.data:
      self.data['src_perturb'] = {}
    self.data['src_perturb']['mt'] = dmt

    # write out new CMTSOLUTION file
    with open(out_file, 'w') as fp:
      fp.write('%s\n' % event['header'])
      fp.write('%-18s %s_dmt\n' % ('event name:',event['id']))
      fp.write('%-18s %+15.8E\n' % ('t0(s):',    0.0))
      fp.write('%-18s %+15.8E\n' % ('tau(s):',   0.0))
      fp.write('%-18s %+15.8E\n' % ('x(m):',     xs[0]))
      fp.write('%-18s %+15.8E\n' % ('y(m):',     xs[1]))
      fp.write('%-18s %+15.8E\n' % ('z(m):',     xs[2]))
      fp.write('%-18s %+15.8E\n' % ('Mxx(N*m):', dmt[0,0]))
      fp.write('%-18s %+15.8E\n' % ('Myy(N*m):', dmt[1,1]))
      fp.write('%-18s %+15.8E\n' % ('Mzz(N*m):', dmt[2,2]))
      fp.write('%-18s %+15.8E\n' % ('Mxy(N*m):', dmt[0,1]))
      fp.write('%-18s %+15.8E\n' % ('Mxz(N*m):', dmt[0,2]))
      fp.write('%-18s %+15.8E\n' % ('Myz(N*m):', dmt[1,2]))

#
#======================================================
#

  def waveform_der_dmt(self,
      syn_dir='output_dmt',
      syn_band_code='MX',
      syn_suffix='.sem.sac',
      sac_dir=None):
    """ Calculate derivative for moment tensor along a given direction
    NOTE:
      1) dmt: 3 by 3 symetric matrix (unit: N*m)
      2) use green's function as input (i.e. set tau to zero in simulation)
    """
    syn_orientation_codes = ['E', 'N', 'Z']
    # event
    event = self.data['event']
    tau = event['tau']
    t0 = event['t0']

    # src_perturb
    dmt = self.data['src_perturb']['mt']

    station_dict = self.data['station']
    for station_id in station_dict:
      station = station_dict[station_id]
      # skip rejected statations
      if station['stat']['code'] < 1:
        continue

      #------ time samples
      waveform = station['waveform']
      time_sample = waveform['time_sample']
      starttime = time_sample['starttime']
      dt = time_sample['delta']
      nt = time_sample['nt']
      nl = time_sample['nl'] # npts of left padding
      nr = time_sample['nr'] # npts of right padding
      sem_nt = nt-nl-nr # number of time samples in SEM simuation
      t = np.arange(nt) * dt + (starttime - t0) #referred to t0

      #------ get file paths of syn seismograms
      syn_files = [ '{:s}/{:s}.{:2s}{:1s}{:s}'.format(
        syn_dir, station_id, syn_band_code, x, syn_suffix)
        for x in syn_orientation_codes ]

      #------ read in obs, syn seismograms
      syn_st  = read(syn_files[0])
      syn_st += read(syn_files[1])
      syn_st += read(syn_files[2])

      #------ check the same time samples as original syn
      if not is_equal( [ (tr.stats.starttime, tr.stats.delta, tr.stats.npts) \
          for tr in syn_st ] ):
        raise Exception('%s: not equal time samples in'\
            ' synthetic seismograms.' % (station_id))

      tr = syn_st[0]

      if tr.stats.delta != dt:
        raise Exception("%s: not the same dt for diff-srcloc!" % (station_id))

      tr_starttime = tr.stats.starttime - nl*dt
      if tr_starttime != starttime:
        raise Exception("%s: not the same starttime for diff-srcloc!" % (station_id))

      if tr.stats.npts != sem_nt:
        raise Exception("%s: not the same npts for diff-srcloc!" % (station_id))

      #------ read syn seismograms from perturbed source location
      dg = np.zeros((3, nt))
      for i in range(3):
        dg[i,nl:(nl+sem_nt)] = syn_st[i].data

      #------ record derivatives
      if 'waveform_der' not in station:
        station['waveform_der'] = {}
      station['waveform_der']['mt'] = {
          'dm':np.array(dmt), 'dg':dg }
      #'dm':np.array(dmt), 'dg':dg, 'du':du, 'dchi':dchi }

      # DEBUG
      #print(dchi
      #for i in range(3):
      #  plt.subplot(311+i)
      #  plt.plot(t, du[i,:], 'k')
      #plt.show()
      if sac_dir:
        for i in range(3):
          tr.data = du[i,:]
          tr.stats.starttime = starttime
          tr.stats.delta = dt
          tr.stats.npts = nt
          out_file = '{:s}/{:s}.{:2s}{:1s}'.format(
              sac_dir, station_id, syn_band_code,
              syn_orientation_codes[i])
          tr.write(out_file, 'sac')

#
#======================================================
#

  def waveform_der_dmodel(self,
      syn_dir='output_perturb/sac',
      syn_band_code='MX',
      syn_suffix='.sem.sac',
      model_name='vsv',
      sac_dir=None):
    """ 
    Get partial derivative seismograms from synthetics of perturbed model 
    along the model update direction
    
    Notes
    -----
    use finite difference to get waveform derivative

    must keep all source parameter the same as used in the forward simulation. 
    
    """
    syn_orientation_codes = ['E', 'N', 'Z']

    event = self.data['event']
    t0 = event['t0']

    station_dict = self.data['station']
    for station_id in station_dict:
      station = station_dict[station_id]
      # skip rejected statations
      if station['stat']['code'] < 1:
        continue

      #------ time samples
      waveform = station['waveform']
      time_sample = waveform['time_sample']
      starttime = time_sample['starttime']
      dt = time_sample['delta']
      nt = time_sample['nt']
      nl = time_sample['nl'] # npts of left padding
      nr = time_sample['nr'] # npts of right padding
      sem_nt = nt-nl-nr # number of time sample number in SEM simulation 
      t = np.arange(nt) * dt + (starttime - t0) #referred to t0

      #------ get file paths of syn seismograms
      syn_files = [ '{:s}/{:s}.{:2s}{:1s}{:s}'.format(
        syn_dir, station_id, syn_band_code, x, syn_suffix)
        for x in syn_orientation_codes ]

      #------ read in obs, syn seismograms
      syn_st  = read(syn_files[0])
      syn_st += read(syn_files[1])
      syn_st += read(syn_files[2])

      #------ check the same time samples as original syn
      if not is_equal( [ (tr.stats.starttime, tr.stats.delta, tr.stats.npts) \
          for tr in syn_st ] ):
        raise Exception('%s: not equal time samples in'\
            ' synthetic seismograms.' % (station_id))
      tr = syn_st[0]

      if tr.stats.delta != dt:
        raise Exception("%s: not the same dt for diff-srcloc!" % (station_id))

      tr_starttime =  tr.stats.starttime - nl*dt
      if tr_starttime != starttime:
        raise Exception("%s: not the same starttime for diff-srcloc!" % (station_id))

      if tr.stats.npts != sem_nt:
        raise Exception("%s: not the same npts for diff-srcloc!" % (station_id))

      #------ read syn seismograms from perturbed source location
      syn_ENZ = np.zeros((3, nt))
      for i in range(3):
        syn_ENZ[i,nl:(nl+sem_nt)] = syn_st[i].data

      # waveform partial derivatives 
      if 'syn' not in waveform:
        raise Exception("%s: initial syn is not stored in waveform!" % (station_id))
      u0 = waveform['syn']
      du = syn_ENZ - u0

      #------ record derivatives
      if 'waveform_der' not in station:
        station['waveform_der'] = {}
      station['waveform_der'][model_name] = {
          'du':du }

      # DEBUG: check du
      #print(dchi
      #for i in range(3):
      #  plt.subplot(311+i)
      #  #plt.plot(t,grn0[i,:],'k', t,syn_ENZ[i,:],'r', t,dg[i,:], 'b')
      #  plt.plot(t, du[i,:], 'k')
      #plt.show()
      if sac_dir:
        for i in range(3):
          tr.data = du[i,:]
          tr.stats.starttime = starttime
          tr.stats.delta = dt
          tr.stats.npts = nt
          out_file = '{:s}/{:s}.{:2s}{:1s}'.format(
              sac_dir, station_id, syn_band_code,
              syn_orientation_codes[i])
          tr.write(out_file, 'sac')

#
#======================================================
#

  def measure_hessian_src(self, 
      src_param={'t0':0, 'tau':0, 'xs':0, 'mt':0},
      update=True,
      ):
    """ 
    Approximate Hessian matrix of source parameters based 
    partial derivative waveforms

    Note
    ----
    """
    event = self.data['event']
    #src_param = ('dt0','dtau','dxs','dmt')
    n_srcparam = len(src_param)

    #------ loop each station
    station_dict = self.data['station']
    for station_id in station_dict:
      station = station_dict[station_id]
      # skip rejected statations
      if station['stat']['code'] < 1:
        continue

      # waveform
      waveform = station['waveform']
      time_sample = waveform['time_sample']
      syn_starttime = time_sample['starttime']
      syn_delta = time_sample['delta']
      syn_nyq = 0.5/syn_delta
      syn_nt = time_sample['nt']
      syn_nl = time_sample['nl']
      syn_nr = time_sample['nr']
      syn_times = syn_delta * np.arange(syn_nt)
      # seismograms 
      obs = waveform['obs']
      syn = waveform['syn']

      # source spectrum (moment-rate function)
      syn_freq = np.fft.rfftfreq(syn_nt, d=syn_delta)
      F_src = stf_gauss_spectrum(syn_freq, event['tau'])
      # spectrum of source derivatives
      if 't0' in src_param or 'tau' in src_param:
        F_ds_dt0, F_ds_dtau = stf_gauss_spectrum_der(syn_freq, event['tau'])
        # convolve Ds(t)/Dt0,tau with Green's function
        du_dt0 = np.fft.irfft(F_ds_dt0 * np.fft.rfft(grn), nt)
        du_dtau = np.fft.irfft(F_ds_dtau * np.fft.rfft(grn), nt)
        # zero records before origin time (wrap around from the end)
        idx = t < -5.0*tau
        du_dt0[:,idx] = 0.0
        du_dtau[:,idx] = 0.0
 
      # waveform derivatives
      waveform_der = station['waveform_der']

      #------ loop each window
      window_dict = station['window']
      for window_id in window_dict:
        # window parameters
        window = window_dict[window_id]
        # skip bad windows
        if window['stat']['code'] < 1:
          warnings.warn("Window %s not measured for adj, SKIP" % window_id)
          continue

        #------ window parameters 
        # filter
        filter_dict = window['filter']
        filter_a = filter_dict['a']
        filter_b = filter_dict['b']
        # taper
        win_func = window['taper']['win']
        # polarity projection 
        proj_matrix = window['polarity']['proj_matrix']

        #------ filter obs, syn
        # F * d
        obs_filt = signal.filtfilt(filter_b, filter_a, obs)
        # F * u (u = S*grn)
        syn_filt = signal.filtfilt(filter_b, filter_a, grn)
        syn_filt = np.fft.irfft(F_src*np.fft.rfft(syn_filt), syn_nt)
        # apply window taper and polarity projection
        # obs = w * F * d
        wFd = np.dot(proj_matrix, obs_filt) * win_func 
        # syn = w * F * u (u = S*grn)
        wFu = np.dot(proj_matrix, syn_filt) * win_func 
        # norm
        norm_wFd = np.sqrt(np.sum(wFd**2))
        norm_wFu = np.sqrt(np.sum(wFu**2))
        # window normalization factor
        Nw = norm_wFd * norm_wFu
        # window amplitude raito
        Aw = np.sum(wFd * wFu) / norm_wFu**2

        #DEBUG
        #print("Nw: %e %e" % (Nw, window['cc']['Nw'])
        #print("Aw: %e %e" % (Aw, window['cc']['Aw'])

        #------ filter differential seismograms (w * F * du_dm)
        wFdu = {}
        for par in src_param:
          if par == 't0':
            Fdu = signal.filtfilt(filter_b, filter_a, du_dt0)
            wFdu[par] = np.dot(proj_matrix, Fdu) * win_func 
          elif par == 'tau':
            Fdu = signal.filtfilt(filter_b, filter_a, du_dtau)
            wFdu[par] = np.dot(proj_matrix, Fdu) * win_func 
          else:
            du = waveform_der[param]['du']
            Fdu = signal.filtfilt(filter_b, filter_a, du)
            wFdu[param] = np.dot(proj_matrix, Fdu) * win_func 

        #------ hessian src
        # chi: zero-lag correlation coef. between wFu and wFd
        # hessian: ddchi_dmdm
        hessian_src = {}
        for i in range(n_srcparam):
          for j in range(i, n_srcparam):
            par1 = src_param[i]
            par2 = src_param[j]
            wFdu1 = wFdu[par1]
            wFdu2 = wFdu[par2]
            wFdu1_wFdu2 = np.sum(wFdu1 * wFdu2)
            wFu_wFdu1 = np.sum(wFu * wFdu1)
            wFu_wFdu2 = np.sum(wFu * wFdu2)
            wFd_wFdu1 = np.sum(wFd * wFdu1)
            wFd_wFdu2 = np.sum(wFd * wFdu2)
            key12 = (par1, par2)
            hessian_src[key12] = ( \
                - Aw * wFdu1_wFdu2 \
                + ( 3.0 * Aw * wFu_wFdu1 * wFu_wFdu2 \
                             - wFu_wFdu1 * wFd_wFdu2 \
                             - wFu_wFdu2 * wFd_wFdu1 \
                  ) / norm_wFu**2 
                ) / Nw

        #------ record results
        if 'hessian_src' not in window:
          window['hessian_src'] = hessian_src
          window['stat'] = {'code': 2,
              'msg': "add hessian_src on "+UTCDateTime.now().isoformat()}
        elif update:
          window['hessian_src'].update(hessian_src)
          window['stat'] = {'code': 2,
              'msg': "update hessian_src on "+UTCDateTime.now().isoformat()}
        else:
          warnings.warn("hessian_src already set, nothing changed")
      # end for window_id in windows:
    # endfor station_id in station_dict:
  #enddef measure_windows_for_one_station(self,

#
#======================================================
#

# def update_source(self, src_param=):
#   """ Update source parameters based on waveform derivatives and hessian
#   """
#   event = self.data['event']
#   src_param = ('dt0','dtau','dxs','dmt')
#   n_srcparam = len(src_param)

#   dchi_dm = np.zeros(n_srcparam)
#   hessian = np.zeros([n_srcparam,n_srcparam])

#   #------ get dchi_dm and Hessian 
#   #-- loop each station
#   station_dict = self.data['station']
#   for station_id in station_dict:
#     station = station_dict[station_id]
#     # skip rejected statations
#     if station['stat']['code'] < 0:
#       continue

#     # dchi_dm
#     for i in range(n_srcparam):
#       key = src_param[i]
#       dchi_dm[i] += station['waveform_der'][key]['dchi']

#     #-- loop each window
#     window_dict = station['window']
#     for window_id in window_dict:
#       # window parameters
#       window = window_dict[window_id]
#       # skip bad windows
#       if window['stat']['code'] < 1:
#         warnings.warn("Window %s not measured for adj, SKIP" % window_id)
#         continue
#       # 
#       weight = window['weight']
#       hessian_win = window['hessian_src']
#       for i in range(n_srcparam):
#         for j in range(i, n_srcparam):
#           par1 = src_param[i]
#           par2 = src_param[j]
#           key = (par1,par2)
#           hessian[i,j] += weight * hessian_win[key]
#     #end for window_id in windows:

#   #end for station_id in station_dict:

#   for i in range(n_srcparam):
#     for j in range(i+1, n_srcparam):
#         hessian[j,i] = hessian[i,j]

#   print("dchi_dm:"
#   print(dchi_dm 

#   print("hessian:"
#   print(hessian

#   print("====== 0:4:"
#   w, v = np.linalg.eigh(hessian, UPLO='U')
#   print(w
#   print(v
#   x, residual, rank, sigval = np.linalg.lstsq(hessian, -dchi_dm)
#   print(" inv(hessian)*(-1.0 * dchi_dm): \n", x
#   print("dt0: \n", x[0]
#   print("dtau:\n", x[1]
#   print("dxs: \n", x[2]*self.data['src_perturb']['xs'] 
#   print("dmt: \n", x[3]*self.data['src_perturb']['mt'] 

#   print("====== only 0:3"
#   h3 = hessian[0:3,0:3]
#   v3 = dchi_dm[0:3]
#   w, v = np.linalg.eigh(h3, UPLO='U')
#   print(w
#   print(v
#   x, residual, rank, sigval = np.linalg.lstsq(h3, -v3)
#   print("inv(hessian)*(-1.0 * dchi_dm): \n", x
#   print("dt0: \n", x[0]
#   print("dtau:\n", x[1]
#   print("dxs: \n", x[2]*self.data['src_perturb']['xs'] 
#   #print("dmt: \n", x[3]*self.data['src_perturb']['dmt'] 

#   print("====== only 0:2"
#   h3 = hessian[0:2,0:2]
#   v3 = dchi_dm[0:2]
#   w, v = np.linalg.eigh(h3, UPLO='U')
#   print(w
#   print(v
#   x, residual, rank, sigval = np.linalg.lstsq(h3, -v3)
#   print("inv(hessian)*(-1.0 * dchi_dm): \n", x
#   print("dt0: \n", x[0]
#   print("dtau:\n", x[1]

#   print("====== only 0,2"
#   idx = [0,2]
#   hess = hessian[idx,:][:,idx]
#   kernel = dchi_dm[idx]
#   print(hess
#   eigs, eigv = np.linalg.eigh(hess, UPLO='U')
#   print(eigs
#   print(eigv
#   x, residual, rank, sigval = np.linalg.lstsq(hess, -kernel)
#   print("inv(hessian)*(-1.0 * dchi_dm): \n", x
#   print("dt0: \n", x[0]
#   print("dxs:\n", x[1]

# #enddef measure_windows_for_one_station(self,

#
#======================================================
#

  def cc_linearized_seismogram_for_source(self,
      dm={'t0':None, 'xs_mt':None},
      plot=False,
      ):
    """ 
    calculate normalized zero-lag cc for perturbed seismograms 
    by linear combination of waveform derivatives of source parameters (t0,tau,xs,M)

    dm={<model_name>:<model_step_sizes>, ...}

    return wcc_sum, weight_sum
    """
    # check model vectors in dm have the same length
    if (not dm) or len(dm) == 0:
      error_str = "dm must not be empty"
      raise Exception(error_str)
    model_num = len(dm)

    nstep = []
    for model_name in dm:
      if dm[model_name].ndim != 1:
        error_str = "dm[%s] must be vector" % model_name
        raise Exception(error_str)
      nstep.append(np.size(dm[model_name]))
    if not is_equal(nstep) or nstep[0] < 1:
      error_str = "vectors in dm must have the same non-zero length"
      raise Exception(error_str)
    nstep = nstep[0]

    # check parameters
    event = self.data['event']
    if 'tau' in dm:
      tau = dm['tau'] + event['tau']
      if any(tau <= 0):
        error_str = "dm['dtau'] has invalid values (event['tau']=%f)!" \
            % event['tau']
        raise Exception(error_str)

    #------ loop each station
    # sum of weighted normalized zero-lag cc at each model grid
    wcc_sum = np.zeros(nstep)
    # sum of all windows' weighting
    weight_sum = 0.0

    station_dict = self.data['station']
    for station_id in station_dict:
      station = station_dict[station_id]
      # skip rejected stations
      if station['stat']['code'] < 0:
        continue

      #print(station_id, flush=True)
      #print(datetime.datetime.utcnow(), flush=True)

      # check if model parameter included in waveform_der
      waveform_der = station['waveform_der']
      for model_name in dm:
        if (model_name not in ['t0', 'tau']) and \
           (model_name not in waveform_der):
          error_str = "%s not in waveform_der of %s" % (model_name, station_id)
          raise Exception(error_str)

      #---- get seismograms: obs,grn 
      waveform = station['waveform']
      obs = waveform['obs']
      grn = waveform['grn']
      # time samples
      time_sample = waveform['time_sample']
      syn_starttime = time_sample['starttime']
      syn_delta = time_sample['delta']
      syn_nt = time_sample['nt']
      syn_nl = time_sample['nl']
      syn_nr = time_sample['nr']
      syn_freq = np.fft.rfftfreq(syn_nt, d=syn_delta)

      #---- measure misfit
      window_dict = station['window']
      for window_id in window_dict:
        window = window_dict[window_id]
        # skip bad windows
        if window['stat']['code'] < 1:
          warnings.warn("Window %s not measured for adj, SKIP" % window_id)
          continue
        # window weight
        weight = window['weight']
        # skip window with zero weight
        if np.isclose(weight, 0.0):
          continue
        weight_sum += weight
        # filter
        filter_dict = window['filter']
        filter_a = filter_dict['a']
        filter_b = filter_dict['b']
        # taper
        win_func = window['taper']['win']
        win_starttime = window['taper']['starttime']
        win_endtime = window['taper']['endtime']
        if plot:
          syn_times = np.arange(syn_nt) * syn_delta
          plt.plot(syn_times, win_func)
          plt.show()
          plt.title("wind_func")
        # polarity projection 
        proj_matrix = window['polarity']['proj_matrix']
        #-- filter,project,taper obs
        # F * d
        obs_filt = signal.filtfilt(filter_b, filter_a, obs)
        # w * p * F * d (window,project,filter)
        wpFd = np.dot(proj_matrix, obs_filt) * win_func 
        norm_wpFd = np.sqrt(np.sum(wpFd**2))
        #-- filter,project grn 
        # F * g
        grn_filt = signal.filtfilt(filter_b, filter_a, grn)
        # p * F * g
        pFg = np.dot(proj_matrix, grn_filt)
        #if plot:
        #  F_src = stf_gauss_spectrum(syn_freq, event['tau'])
        #  # S * F * g
        #  syn_filt = np.fft.irfft(F_src*np.fft.rfft(grn_filt), syn_nt)
        #  # w * p * S * F * g
        #  wpFu = np.dot(proj_matrix, syn_filt) * win_func
        #-- filter,project dg: pFdg
        pFdg = {}
        for model_name in dm:
          # exclude source time function parameters
          if model_name not in ['t0', 'tau']:
            dg = waveform_der[model_name]['dg']
            dg_filt = signal.filtfilt(filter_b, filter_a, dg)
            pFdg[model_name] = np.dot(proj_matrix, dg_filt)
        #-- misfit function: zero-lag cc
        for istep in range(nstep):
          # perturbed grn: pFg1
          pFg1 = np.zeros((3,syn_nt))
          pFg1 += pFg
          for model_name in dm:
            # exclude source time function
            if model_name not in ['t0', 'tau']:
              pFg1 += dm[model_name][istep] * pFdg[model_name]
          # perturbed source time function
          dt0 = 0.0
          if 't0' in dm:
            dt0 = dm['t0'][istep]
          dtau = 0.0
          if 'tau' in dm:
            dtau = dm['tau'][istep]
          F_src = stf_gauss_spectrum(syn_freq, event['tau']+dtau)
          # perturbed syn: w * S * p * F * g1
          phase_shift = np.exp(-2.0j * np.pi * syn_freq * dt0)
          wpFu1 = np.fft.irfft(phase_shift*F_src*np.fft.rfft(pFg1), syn_nt) \
              * win_func
          norm_wpFu1 = np.sqrt(np.sum(wpFu1**2))
          Nw = norm_wpFd * norm_wpFu1
          #normalized cc between obs and perturbed syn
          cc_wpFd_wpFu1 = np.sum(wpFd*wpFu1) / Nw
          # weighted cc
          wcc_sum[istep] += weight * cc_wpFd_wpFu1
          #DEBUG
          #if plot:
          #  syn_npts = syn_nt - syn_nl - syn_nr
          #  syn_orientation_codes = ['E', 'N', 'Z']
          #  syn_times = np.arange(syn_nt) * syn_delta
          #  Amax_obs = np.max(np.sum(wpFd**2, axis=0))**0.5
          #  Amax_syn = np.max(np.sum(wpFu**2, axis=0))**0.5
          #  Amax_syn1 = np.max(np.sum(wpFu1**2, axis=0))**0.5
          #  win_b = win_starttime - syn_starttime
          #  win_e = win_endtime - syn_starttime
          #  obs_filt_proj = np.dot(proj_matrix, obs_filt)
          #  syn_filt_proj = np.dot(proj_matrix, syn_filt)
          #  for i in range(3):
          #    plt.subplot(311+i)
          #    if i == 0:
          #      title_str = "%s.%s " % (station_id, window_id)
          #      for model_name in dm:
          #        title_str += "%s:%.2f " \
          #            % (model_name, dm[model_name][idx_model])
          #      title_str += "NCCwin:%.2f" % (cc_wpFd_wpFu1)
          #      plt.title(title_str)

          #    idx_plt = range(syn_nl,(syn_nl+syn_npts))
          #    # whole trace, projected
          #    plt.plot(syn_times[idx_plt], obs_filt_proj[i,idx_plt]/Amax_obs, 
          #        'k', linewidth=0.2)
          #    plt.plot(syn_times[idx_plt], syn_filt_proj[i,idx_plt]/Amax_syn,
          #        'b', linewidth=0.2)
          #    # windowed trace
          #    idx_plt = (win_b <= syn_times) & (syn_times <= win_e)
          #    plt.plot(syn_times[idx_plt], wpFd[i,idx_plt]/Amax_obs,
          #        'k', linewidth=1.0)
          #    plt.plot(syn_times[idx_plt], wpFu[i,idx_plt]/Amax_syn,
          #        'b', linewidth=1.0)
          #    plt.plot(syn_times[idx_plt], wpFu1[i,idx_plt]/Amax_syn1,
          #        'r', linewidth=1.0)

          #    plt.xlim((syn_times[syn_nl], syn_times[syn_nl+syn_npts-1]))
          #    plt.ylim((-1.5, 1.5))
          #    plt.ylabel(syn_orientation_codes[i])
          #  plt.show()

      #end for window_id in window_dict:
    #end for station_id in station_dict:

    return wcc_sum, weight_sum

#
#======================================================
#

#  def search1d_cc_perturbed_seismogram(self,
#      dm_range = {
#        't0': [-5,5], 
#        'tau': [-5,5],
#        'xs': [-5,5],
#        'mt': [-5,5],
#        },
##     dist_min=0.0,
##     dist_max=180.0,
#      ngrid=4,
#      max_niter=5,
#      range_ratio=0.7,
#      plot_seism=False,
#      log_file="search1d_cc.log",
#      cmt_file="CMTSOLUTION.search1d",
#      ):
#    """ 
#    calculate misfit over 2D model grids based on perturbed seismograms 
#
#    Parameters
#    ----------
#    range_ratio : scalar
#      between (0, 1), used to shrink the search range to cc values above range_ratio
#      between cc_max and cc_min.
#
#    Return
#    ------
#    optimal dm
#
#    """
#    # check validity of dm_range
#    event = self.data['event']
#    tau = event['tau']
#    if 'tau' in dm_range:
#      tau_lim = dm_range['tau']
#      if (min(tau_lim) + tau) < 1.0:
#        warnings.warn("Too small dtau value, increase to avoid negative tau")
#        dm_range['tau'][0] = -0.95*event['tau']
#        print(dm_range)
#
#    # grid parameters
#    model_name = [x for x in dm_range]
#
#    range_ratio = float(range_ratio)
#    if range_ratio < 0.5 or range_ratio > 0.9:
#      raise ValueError("range_ratio should be within (0.5, 0.9)")
#
#    dm_opt = {}
#    for par in model_name:
#      dm_opt[par] = 0.0
#
#    # prepare log file
#    fid = open(log_file,"w")
#
#    niter = 0
#    while niter < max_niter:
#      fid.write("====== iteration: %d\n" % (niter))
#      fid.flush()
#
#      # loop each parameter
#      for par in model_name:
#        fid.write("------ %s\n" % (par))
#        fid.flush()
#        xlim = np.array(dm_range[par])
#
#        if xlim.size == 1:
#          dm_opt[par] = xlim[0]
#          fid.write("fixed at %.5e\n" % (dm_opt[par]))
#          fid.flush()
#          continue
#        else:
#          fid.write("range: %.5e, %.5e\n" % (np.min(xlim), np.max(xlim)))
#          fid.flush()
#          xx = np.linspace(np.min(xlim), np.max(xlim), ngrid)
#
#          # search
#          dm_grid = {}
#          dm_grid[par] = xx
#          for par_fix in model_name:
#            if par_fix != par:
#              dm_grid[par_fix] = np.ones(ngrid) * dm_opt[par_fix]
#          cc, weight = self.cc_perturbed_seismogram(
#              dm=dm_grid, 
##             dist_min=dist_min, 
##             dist_max=dist_max,
#              plot=plot_seism)
#          cc /= weight # weighted average of normalized zero-lag CC
#
#          # interpolation
#          x_i = np.linspace(np.min(xlim), np.max(xlim), 100)
#          interp = interpolate.interp1d(xx, cc, kind='cubic')
#          cc_i = interp(x_i)
#          imax = np.argmax(cc_i)
#          x_max = x_i[imax]
#          cc_max = cc_i[imax]
#
#          #change search range
#          imin = np.argmin(cc_i)
#          cc_min = cc_i[imin]
#          cc_thred = cc_min + (cc_max-cc_min)*range_ratio
#          x1 = x_i[cc_i>cc_thred]
#          dm_range[par] = [np.min(x1), np.max(x1)]
#
#          dm_opt[par] = x_max
#          fid.write("maximum at %.5e with cc %.5e\n" % (dm_opt[par], cc_max))
#          fid.flush()
#
#      # print(out optimal model
#      fid.write("------ optimal dm_alpha\n")
#      fid.flush()
#      for par in model_name:
#        fid.write("%s: %.5e\n" %(par, dm_opt[par]))
#        fid.flush()
#
#      niter += 1
#    #END while niter < 5:
#    fid.close()
#
#    # make new model
#    model = {}
#    event = self.data['event']
#    src_perturb = self.data['src_perturb']
#    for par in dm_opt:
#      m0 = event[par]
#      dm = src_perturb[par]
#      alpha = dm_opt[par]
#      model[par] = {
#          'm0':m0, 
#          'dm':dm,
#          'alpha':alpha,
#          'm1':m0+alpha*dm
#          }
#
#    # write out new CMTSOLUTION file
#    t0 = event['t0']
#    if 't0' in model: t0 = model['t0']['m1']
#    # modify origin time in header line to have centroid time 
#    header = event['header'].split()
#    header[1] = "{:04d}".format(t0.year)
#    header[2] = "{:02d}".format(t0.month)
#    header[3] = "{:02d}".format(t0.day)
#    header[4] = "{:02d}".format(t0.hour)
#    header[5] = "{:02d}".format(t0.minute)
#    header[6] = "{:07.4f}".format(t0.second + 1.0e-6*t0.microsecond)
#    
#    tau = event['tau']
#    if 'tau' in model: tau = model['tau']['m1']
#    xs = event['xs']
#    if 'xs' in model: xs = model['xs']['m1']
#    mt = event['mt']
#    m0 = (0.5*np.sum(mt**2))**0.5
#    if 'mt' in model: mt = model['mt']['m1']
#    # zero trace
#    mt = mt - np.identity(3)*np.trace(mt)/3.0
#    # scale to orignal moment
#    mt = mt/(0.5*np.sum(mt**2))**0.5
#    mt *= m0
#    
#    with open(cmt_file, 'w') as fp:
#      fp.write('%s \n' % (' '.join(header)))
#      fp.write('%-18s %s\n' % ('event name:', event['id']))
#      fp.write('%-18s %+15.8E\n' % ('t0(s):',    0.0))
#      fp.write('%-18s %+15.8E\n' % ('tau(s):',   tau))
#      fp.write('%-18s %+15.8E\n' % ('x(m):',     xs[0]))
#      fp.write('%-18s %+15.8E\n' % ('y(m):',     xs[1]))
#      fp.write('%-18s %+15.8E\n' % ('z(m):',     xs[2]))
#      fp.write('%-18s %+15.8E\n' % ('Mxx(N*m):', mt[0,0]))
#      fp.write('%-18s %+15.8E\n' % ('Myy(N*m):', mt[1,1]))
#      fp.write('%-18s %+15.8E\n' % ('Mzz(N*m):', mt[2,2]))
#      fp.write('%-18s %+15.8E\n' % ('Mxy(N*m):', mt[0,1]))
#      fp.write('%-18s %+15.8E\n' % ('Mxz(N*m):', mt[0,2]))
#      fp.write('%-18s %+15.8E\n' % ('Myz(N*m):', mt[1,2]))

#
#======================================================
#

#  def grid_cc_perturbed_seismogram(self,
#      dm = {
#        't0': np.linspace(-10,0,11), 
#        'xs': np.linspace(-5,5,11),
#        },
#      axes=[ ('t0','xs'), ('t0',), ('xs',) ],
#      outfig="grid_cc.pdf",
#      plot_seism=False,
#      cmt_file="CMTSOLUTION.grid_cc"
#      ):
#    """ calculate misfit over 2D model grids based on perturbed seismograms 
#    """
#    # grid parameters
#    model_num = len(dm)
#    model_name = [x for x in dm]
#
#    # check parameters 
#    for xy in axes:
#      if len(xy) < 1 or len(xy) > 2:
#        error_str = "axes should have one or two elements"
#        raise Exception(error_str)
#      for axis in xy:
#        if axis not in model_name:
#          error_str = "axis(%s) not in dm" % axis
#          raise Exception(error_str)
#
#    # model grid
#    x = [np.array(dm[s]) for s in model_name]
#    xx = np.meshgrid(*x, indexing='ij')
#    nn = xx[0].size
#
#    # calculate cc values over all grid points 
#    dm_grid = {}
#    for i in range(model_num):
#      par = model_name[i]
#      dm_grid[par] = xx[i].flatten()
#
#    zz, weight = self.cc_perturbed_seismogram(dm=dm_grid, plot=plot_seism)
#    zz /= weight # weighted average of normalized zero-lag CC
#    zz = zz.reshape(xx[0].shape)
#
#    # get maximum cc value 
#    imax = np.argmax(zz)
#    idx_max = np.unravel_index(imax, xx[0].shape)
#    zz_max = zz[idx_max]
#
#    zz_min = np.min(zz)
#
#    # plot out results
#    with PdfPages(outfig) as pdf:
#      for xy in axes:
#        # plot 1D section
#        if (len(xy) == 1) or (xy[0] == xy[1]):
#          ix = model_name.index(xy[0])
#          idx = [range(len(v)) for v in x]
#          for i in range(model_num):
#            if i != ix:
#              idx[i] = [idx_max[i],]
#          zz_1d = zz[np.ix_(*idx)].squeeze()
#
#          xx_1d = x[ix]
#          x_i = np.linspace(np.min(xx_1d), np.max(xx_1d), 100)
#
#          interp = interpolate.interp1d(xx_1d, zz_1d, kind='cubic')
#          zz_i = interp(x_i)
#          imax_i = np.argmax(zz_i)
#          x_i_max = x_i[imax_i]
#          zz_i_max = zz_i[imax_i]
#
#          # plot cross-sections through maximum point
#          #fig = plt.figure(figsize=(8.5,11))
#          fig = plt.figure()
#          plt.plot(xx_1d, zz_1d, 'ro', x_i, zz_i)
#
#          title_str = "Weighted average CC  (%.2e, %.2e)" % (x_i_max, zz_i_max)
#          plt.title(title_str)
#
#          xlabel_str = "alpha (* d%s)" % model_name[ix]
#          plt.xlabel(xlabel_str)
#          plt.ylabel("weighted avg. CC")
#
#          pdf.savefig()
#          plt.close()
#
#        # plot 2D cross section
#        else:
#          ix = model_name.index(xy[0])
#          iy = model_name.index(xy[1])
#          idx = [range(len(v)) for v in x]
#          for i in range(model_num):
#            if i != ix and i != iy:
#              idx[i] = [idx_max[i],]
#          idx = np.ix_(*idx)
#          zz_2d = zz[idx].squeeze()
#          xx_2d = xx[ix][idx].squeeze()
#          yy_2d = xx[iy][idx].squeeze()
#          print(xx_2d)
#          print(yy_2d)
#          print(zz_2d)
#          if ix > iy:
#            zz_2d = np.transpose(zz_2d)
#            xx_2d = np.transpose(xx_2d)
#            yy_2d = np.transpose(yy_2d)
#
#          # make interpolation grid
#          x_i = np.linspace(np.min(xx_2d), np.max(xx_2d), 100)
#          y_i = np.linspace(np.min(yy_2d), np.max(yy_2d), 100)
#          xx_i, yy_i = np.meshgrid(x_i, y_i)
#
#          interp = interpolate.interp2d(xx_2d, yy_2d, zz_2d, kind='cubic')
#          zz_i = interp(x_i, y_i)
#          imax_i = np.argmax(zz_i)
#          idx_max_i = np.unravel_index(imax_i, xx_i.shape)
#          zz_i_max = zz_i[idx_max_i]
#          x_i_max = xx_i[idx_max_i]
#          y_i_max = yy_i[idx_max_i]
#
#          # plot 2D surface through the maximum
#          #fig = plt.figure(figsize=(8.5, 11))
#          fig = plt.figure()
#
#          levels = zz_min + (zz_max-zz_min)*np.linspace(0.5, 1.0, 20)
#          #cs = plt.contour(xx_2d, yy_2d, zz_2d, levels, colors='k')
#          cs = plt.contour(xx_i, yy_i, zz_i, levels, colors='k')
#          plt.clabel(cs, fontsize=9, inline=1)
#
#          #x_max = xx[ix][idx_max]
#          #y_max = xx[iy][idx_max]
#          #text_str = "(%.1f,%.1f)" % (x_max, y_max)
#          #plt.text(x_max, y_max, text_str,
#          #    horizontalalignment="center", verticalalignment="top")
#          #plt.plot(x_max, y_max, 'ko')
#
#          #text_str = "%.2e,%.2e" % (x_i_max, y_i_max)
#          #plt.text(x_i_max, y_i_max, text_str,
#          #    horizontalalignment="center", verticalalignment="top")
#          plt.plot(x_i_max, y_i_max, 'k+')
#
#          title_str = "Weighted average CC (%.2e,%.2e)" % (x_i_max, y_i_max)
#          plt.title(title_str)
#
#          xlabel_str = "alpha (* d%s)" % model_name[ix]
#          plt.xlabel(xlabel_str)
#          ylabel_str = "alpha (* d%s)" % model_name[iy]
#          plt.ylabel(ylabel_str)
#
#          pdf.savefig()
#          plt.close()
#
#    # get best model
#    event = self.data['event']
#    src_perturb = self.data['src_perturb']
#    
#    model = {}
#    for par in dm:
#      m0 = event[par]
#      dm = src_perturb[par]
#      ix = model_name.index(par)
#      alpha = xx[ix][idx_max]
#      model[par] = {
#          'm0':m0, 
#          'dm':dm,
#          'alpha':alpha,
#          'm1':m0+alpha*dm
#          }
#
#    #------ write out new CMTSOLUTION file
#    t0 = event['t0']
#    if 't0' in model: t0 = model['t0']['m1']
#    # modify origin time in header line to have centroid time 
#    header = event['header'].split()
#    header[1] = str(t0.year)
#    header[2] = str(t0.month)
#    header[3] = str(t0.day)
#    header[4] = str(t0.hour)
#    header[5] = str(t0.minute)
#    header[6] = str(t0.second + 1.0e-6*t0.microsecond)
#
#    tau = event['tau']
#    if 'tau' in model: tau = model['tau']['m1']
#    xs = event['xs']
#    if 'xs' in model: xs = model['xs']['m1']
#    mt = event['mt']
#    if 'mt' in model: mt = model['mt']['m1']
#
#    with open(cmt_file, 'w') as fp:
#      fp.write('%s | wCCmax %f weight %f\n'
#          % (' '.join(header), zz_max, weight))
#      fp.write('%-18s %s\n' % ('event name:', "grid_cc"))
#      fp.write('%-18s %+15.8E\n' % ('t0(s):',    0.0))
#      fp.write('%-18s %+15.8E\n' % ('tau(s):',   tau))
#      fp.write('%-18s %+15.8E\n' % ('x(m):',     xs[0]))
#      fp.write('%-18s %+15.8E\n' % ('y(m):',     xs[1]))
#      fp.write('%-18s %+15.8E\n' % ('z(m):',     xs[2]))
#      fp.write('%-18s %+15.8E\n' % ('Mxx(N*m):', mt[0,0]))
#      fp.write('%-18s %+15.8E\n' % ('Myy(N*m):', mt[1,1]))
#      fp.write('%-18s %+15.8E\n' % ('Mzz(N*m):', mt[2,2]))
#      fp.write('%-18s %+15.8E\n' % ('Mxy(N*m):', mt[0,1]))
#      fp.write('%-18s %+15.8E\n' % ('Mxz(N*m):', mt[0,2]))
#      fp.write('%-18s %+15.8E\n' % ('Myz(N*m):', mt[1,2]))
#
#    return model, zz_max, weight

#
#======================================================
#

#  def relocate_1d(self, 
#      event_id,
#      window_id_list=['F.p,P', 'F.s,S'], 
#      fix_depth=False,
#      out_cmt_file=None):
#    """relocate event using ray path in reference earth model
#    """
#    # check inputs
#    events = self.data['events']
#    if event_id not in events:
#      print("[ERROR] %s does NOT exist. Exit" % (event_id))
#      sys.exit()
#
#    # select windows
#    sta_win_id_list = []
#    event = events[event_id]
#    stations = event['stations']
#    for station_id in stations:
#
#      station = stations[station_id]
#      if station['stat']['code'] < 0:
#        continue
#
#      windows = station['windows']
#      for window_id in window_id_list:
#        if window_id not in windows:
#          continue
#
#        window = windows[window_id]
#        if window['stat']['code'] != 1:
#          continue 
#
#        misfit = window['misfit']
#        #if window['quality']['SNR'] < min_SNR or \
#        #    misfit['CC0'] < min_CC0 or \
#        #    misfit['CCmax'] < min_CCmax or\
#        #    abs(misfit['CC_time_shift']) > max_CC_time_shift:
#        #  continue
#
#        sta_win_id = (station_id, window_id)
#        sta_win_id_list.append(sta_win_id)
#
#    # create sensitivity matrix G in local NED coordinate
#    # G * dm  = dt_cc
#    # G: [[-px_1, -py_1, -pz_1, 1.0], # ray1
#    #   [-px_2, -py_2, -pz_2, 1.0], # ray2
#    #   ...]
#    # dm: [dNorth(km), dEast, dDepth, dT(sec)]
#    # dt_cc: [dt1, dt2, ...]
#    n = len(sta_win_id_list)
#    G = np.zeros((n, 4))
#    dt_cc = np.zeros(n)
#    R_Earth_km = 6371.0
#    gcmt = event['gcmt']
#    evdp = gcmt['depth']
#    for i in range(n):
#      sta_win_id = sta_win_id_list[i]
#      station_id = sta_win_id[0]
#      window_id = sta_win_id[1]
#  
#      station = stations[station_id]
#      meta = station['meta']
#      window = station['windows'][window_id]
#      phase = window['phase']
#      misfit = window['misfit']
#      weight = window['weight']
#
#      azimuth = np.deg2rad(meta['azimuth'])
#      takeoff_angle = phase['takeoff_angle']
#      takeoff = np.deg2rad(takeoff_angle + 180.0*(takeoff_angle<0))
#      ray_param = phase['ray_param']
#      slowness = ray_param / (R_Earth_km - evdp) #unit: s/km
#      # local coordinate: NED
#      pd = np.cos(takeoff) * slowness
#      pn = np.cos(azimuth) * np.sin(takeoff) * slowness
#      pe = np.sin(azimuth) * np.sin(takeoff) * slowness
#      # create sensitivity matrix 
#      G[i,:] = weight * np.array([-pn, -pe, -pd, 1.0]) # -p: from receiver to source
#      dt_cc[i] = weight * misfit['CC_time_shift']
#
#    #linearized inversion (can be extended to second order using dynamic ray-tracing)
#    if fix_depth: 
#      G[:, 2] = 0.0
#    dm, residual, rank, sigval = np.linalg.lstsq(G, dt_cc)
#
#    # convert dm from NED to ECEF coordinate
#    evla = gcmt['latitude']
#    evlo = gcmt['longitude']
#
#    slat = np.sin(np.deg2rad(evla))
#    clat = np.cos(np.deg2rad(evla))
#    slon = np.sin(np.deg2rad(evlo))
#    clon = np.cos(np.deg2rad(evlo))
#
#    N = [-slat*clon, -slat*slon, clat]
#    E = [-slon, clon, 0.0]
#    D = [-clat*clon, -clat*slon, -slat]
#
#    ev_dx = N[0]*dm[0] + E[0]*dm[1] + D[0]*dm[2]
#    ev_dy = N[1]*dm[0] + E[1]*dm[1] + D[1]*dm[2]
#    ev_dz = N[2]*dm[0] + E[2]*dm[1] + D[2]*dm[2]
#    ev_dt = dm[3]
#
#    # initialize pyproj objects
#    #geod = pyproj.Geod(ellps='WGS84')
#    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
#    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
#
#    # old location in ECEF (meters)
#    evx, evy, evz = pyproj.transform(lla, ecef, evlo, evla, -1000.0*evdp)
#
#    # new location in ECEF (meters)
#    evx1 = evx + ev_dx*1000.0
#    evy1 = evy + ev_dy*1000.0
#    evz1 = evz + ev_dz*1000.0
#    # in LLA
#    evlo1, evla1, evalt1 = pyproj.transform(ecef, lla, evx1, evy1, evz1)
#    evdp1 = -evalt1/1000.0
#
#    # residuals 
#    # linearized modelling
#    dt_syn = G.dot(dm)
#    dt_res = dt_cc - dt_syn
#
#    # make results
#    new_centroid_time = UTCDateTime(gcmt['centroid_time']) + ev_dt
#    reloc_dict = {
#        'window_id_list': window_id_list,
#        'singular_value': sigval.tolist(),
#        'dm': {'dNorth':dm[0], 'dEast':dm[1], 'dDepth':dm[2], 
#          'dT':dm[3]},
#        'latitude':evla1, 
#        'longitude':evlo1, 
#        'depth':evdp1,
#        'centroid_time': str(new_centroid_time),
#        'data': {'num':n, 'mean':np.mean(dt_cc), 'std':np.std(dt_cc)},
#        'residual': {'mean':np.mean(dt_res), 'std':np.std(dt_res)} }
#
#    event['relocate'] = reloc_dict
#
#    # make new CMTSOLUTION file
#    if out_cmt_file:
#      M = gcmt['moment_tensor']
#      with open(out_cmt_file, 'w') as fp:
#        # header line: 
#        #PDE 2003 09 25 19 50 08.93  41.78  144.08  18.0 7.9 8.0 Hokkaido, Japan
#        # which is: event_id, date,origin time,latitude,longitude,depth, mb, MS, region
#        fp.write(new_centroid_time.strftime(
#          'RELOC %Y %m %d %H %M %S.%f ') + \
#          '%.4f %.4f %.1f 0.0 0.0 END\n' % (evla1,evlo1,evdp1) )
#        fp.write('event name:    %s\n'   % (event_id))
#        fp.write('time shift:    0.0\n'        ) 
#        fp.write('tau:   %.1f\n'   % (gcmt['tau']))
#        #fp.write('half duration:   0.0\n'  % (gcmt['tau']))
#        fp.write('latitude:    %.4f\n'   % (evla1)   )
#        fp.write('longitude:     %.4f\n'   % (evlo1)   )
#        fp.write('depth:       %.4f\n'   % (evdp1)   )
#        fp.write('Mrr:       %12.4e\n' % (M[0][0]) )
#        fp.write('Mtt:       %12.4e\n' % (M[1][1]) )
#        fp.write('Mpp:       %12.4e\n' % (M[2][2]) )
#        fp.write('Mrt:       %12.4e\n' % (M[0][1]) )
#        fp.write('Mrp:       %12.4e\n' % (M[0][2]) )
#        fp.write('Mtp:       %12.4e\n' % (M[1][2]) )

#
#======================================================
#

  def cc_linearized_seismogram_for_dmodel(self,
      dm={'vsv':None, 'vsh':None},
      plot=False
      ):
    """ 
    Calculate normalized zero-lag cc between observed and linearized seismograms with waveform_der_dmodel
    for different step size.

    return weighted_cc_sum[:], weight_sum
    """
    #------ validate inputs 
    # check step length arrays in dm have the same length
    if (not dm) or len(dm) == 0:
      error_str = "dm must not be empty"
      raise Exception(error_str)
    model_num = len(dm)

    nstep = []
    for model_name in dm:
      if dm[model_name].ndim != 1:
        error_str = "dm[%s] must be vector" % model_name
        raise Exception(error_str)
      nstep.append(np.size(dm[model_name]))
    if not is_equal(nstep) or nstep[0] < 1:
      error_str = "vectors in dm must have the same non-zero length"
      raise Exception(error_str)
    nstep = nstep[0]

    #------ loop each station
    event = self.data['event']
    # sum of weighted normalized zero-lag cc at each model grid
    wcc_sum = np.zeros(nstep)
    # sum of all windows' weighting
    weight_sum = 0.0

    station_dict = self.data['station']
    for station_id in station_dict:
      station = station_dict[station_id]
      # skip rejected stations
      if station['stat']['code'] < 0:
        continue

      # check if model parameter included in waveform_der
      waveform_der = station['waveform_der']
      for model_name in dm:
        if model_name not in waveform_der:
          error_str = "%s: waveform_der['%s'] does not exist." % (
              station_id, model_name)
          raise Exception(error_str)

      #---- get seismograms: obs,grn 
      waveform = station['waveform']
      obs = waveform['obs']
      syn = waveform['syn']
      # time samples
      time_sample = waveform['time_sample']
      syn_starttime = time_sample['starttime']
      syn_delta = time_sample['delta']
      syn_nt = time_sample['nt']
      syn_nl = time_sample['nl']
      syn_nr = time_sample['nr']
      syn_freq = np.fft.rfftfreq(syn_nt, d=syn_delta)

      #---- measure misfit
      window_dict = station['window']
      for window_id in window_dict:
        window = window_dict[window_id]
        # skip bad windows
        if window['stat']['code'] < 1:
          warnings.warn("Window %s not measured for adj, SKIP" % window_id)
          continue
        # window weight
        weight = window['weight']
        # skip window with zero weight
        if np.isclose(weight, 0.0):
          continue
        weight_sum += weight
        # filter
        filter_dict = window['filter']
        filter_a = filter_dict['a']
        filter_b = filter_dict['b']
        # taper
        win_func = window['taper']['win']
        win_starttime = window['taper']['starttime']
        win_endtime = window['taper']['endtime']
        if plot:
          syn_times = np.arange(syn_nt) * syn_delta
          plt.plot(syn_times, win_func)
          plt.show()
          plt.title("wind_func")
        # polarity projection 
        proj_matrix = window['polarity']['proj_matrix']
        #-- filter,project,taper obs
        # F * d
        obs_filt = signal.filtfilt(filter_b, filter_a, obs)
        # w * p * F * d (window,project,filter)
        wpFd = np.dot(proj_matrix, obs_filt) * win_func 
        norm_wpFd = np.sqrt(np.sum(wpFd**2))
        #-- filter,project,taper syn
        # F * u
        syn_filt = signal.filtfilt(filter_b, filter_a, syn)
        # w * p * F * u
        wpFu = np.dot(proj_matrix, syn_filt) * win_func 
        #-- filter,project,taper du: wpFdu
        wpFdu = {}
        for model_name in dm:
          du = waveform_der[model_name]['du']
          du_filt = signal.filtfilt(filter_b, filter_a, du)
          # w * p * F * du
          wpFdu[model_name] = np.dot(proj_matrix, du_filt) * win_func
        #-- misfit function: zero-lag cc
        for istep in range(nstep):
          wpFu1 = np.zeros((3,syn_nt))
          wpFu1 += wpFu
          for model_name in dm:
            wpFu1 += dm[model_name][istep] * wpFdu[model_name]
          norm_wpFu1 = np.sqrt(np.sum(wpFu1**2))
          Nw = norm_wpFd * norm_wpFu1
          #normalized cc between obs and perturbed syn
          cc_wpFd_wpFu1 = np.sum(wpFd*wpFu1) / Nw
          # weighted cc
          wcc_sum[istep] += weight * cc_wpFd_wpFu1
          #DEBUG
          if plot:
            syn_npts = syn_nt - syn_nl - syn_nr
            syn_orientation_codes = ['E', 'N', 'Z']
            syn_times = np.arange(syn_nt) * syn_delta
            Amax_obs = np.max(np.sum(wpFd**2, axis=0))**0.5
            Amax_syn = np.max(np.sum(wpFu**2, axis=0))**0.5
            Amax_syn1 = np.max(np.sum(wpFu1**2, axis=0))**0.5
            win_b = win_starttime - syn_starttime
            win_e = win_endtime - syn_starttime
            obs_filt_proj = np.dot(proj_matrix, obs_filt)
            syn_filt_proj = np.dot(proj_matrix, syn_filt)
            for i in range(3):
              plt.subplot(311+i)
              if i == 0:
                title_str = "%s.%s " % (station_id, window_id)
                title_str += "step_size:%.2f " % (step_size[istep])
                title_str += "NCCwin:%.2f" % (cc_wpFd_wpFu1)
                plt.title(title_str)

              idx_plt = range(syn_nl,(syn_nl+syn_npts))
              # whole trace, projected
              plt.plot(syn_times[idx_plt], obs_filt_proj[i,idx_plt]/Amax_obs, 
                  'k', linewidth=0.2)
              plt.plot(syn_times[idx_plt], syn_filt_proj[i,idx_plt]/Amax_syn,
                  'b', linewidth=0.2)
              # windowed trace
              idx_plt = (win_b <= syn_times) & (syn_times <= win_e)
              plt.plot(syn_times[idx_plt], wpFd[i,idx_plt]/Amax_obs,
                  'k', linewidth=1.0)
              plt.plot(syn_times[idx_plt], wpFu[i,idx_plt]/Amax_syn,
                  'b', linewidth=1.0)
              plt.plot(syn_times[idx_plt], wpFu1[i,idx_plt]/Amax_syn1,
                  'r', linewidth=1.0)

              plt.xlim((syn_times[syn_nl], syn_times[syn_nl+syn_npts-1]))
              plt.ylim((-1.5, 1.5))
              plt.ylabel(syn_orientation_codes[i])
            plt.show()
      #end for window_id in window_dict:
    #end for station_id in station_dict:

    return wcc_sum, weight_sum

#
#======================================================
#


  def plot_misfit(self, event_id, window_id, out_file=None):
    """Plot misfit for a certain event and window_id  
    """
    # CC0 map  | CC0 v.s. SNR (size ~ weight)
    #------------|-----------------
    # DTcc map   | avg. CC0      

    # check inputs
    events = self.data['events']
    if event_id not in events:
      print("[ERROR] %s does NOT exist. Exit" \
          % (event_id))
      sys.exit()
    event = events[event_id]
    stations = event['stations']

    # get list of station,window id
    #sta_win_id_list = []
    stla_list = []
    stlo_list = []
    cc_dt_list = []
    CC0_list = []
    CCmax_list = []
    snr_list = []
    weight_list = []
    for station_id in stations:
      station = stations[station_id]
      windows = station['windows']

      # skip bad station 
      if station['stat']['code'] < 1:
        continue

      if window_id not in windows:
        continue

      window = windows[window_id]
      if window['stat']['code'] != 1:
        continue

      meta = station['meta']
      misfit = window['misfit']
      quality = window['quality']

      #sta_win_id = (station_id, window_id)
      #sta_win_id_list.append(sta_win_id)
      stla_list.append(meta['latitude'])
      stlo_list.append(meta['longitude'])
      cc_dt_list.append(misfit['CC_time_shift'])
      CC0_list.append(misfit['CC0'])
      CCmax_list.append(misfit['CCmax'])
      snr_list.append(quality['SNR'])
      weight_list.append(window['weight'])

    # get event data
    gcmt = event['gcmt']
    evla = gcmt['latitude']
    evlo = gcmt['longitude']
    M = gcmt['moment_tensor']
    Mrr = M[0][0]
    Mtt = M[1][1]
    Mpp = M[2][2]
    Mrt = M[0][1]
    Mrp = M[0][2]
    Mtp = M[1][2]
    focmec = [ Mrr, Mtt, Mpp, Mrt, Mrp, Mtp ]

    # map range
    min_lat = min(min(stla_list), evla)
    max_lat = max(max(stla_list), evla)
    lat_range = max_lat - min_lat
    min_lat -= 0.1*lat_range
    max_lat += 0.1*lat_range
    min_lon = min(min(stlo_list), evlo)
    max_lon = max(max(stlo_list), evlo)
    lon_range = max_lon - min_lon
    min_lon -= 0.1*lon_range
    max_lon += 0.1*lon_range
    #lat_true_scale = np.mean(stla_list)
    lat_0 = np.mean(stla_list)
    lon_0 = np.mean(stlo_list)
    # 
    parallels = np.arange(0.,81,10.)
    meridians = np.arange(0.,351,10.)

    # figure size
    fig = plt.figure(figsize=(11, 8.5))
    str_title = '%s %s' % (event_id, window_id)
    fig.text(0.5, 0.95, str_title, size='x-large', 
        horizontalalignment='center')

    #------ color map CC_time_shift, symbol size ~ SNR 
    ax = fig.add_axes([0.05, 0.5, 0.4, 0.35])
    ax.set_title("DT_cc (symbol_size ~ SNR)")

    m = Basemap(projection='merc', resolution='l',
        llcrnrlat=min_lat, llcrnrlon=min_lon, 
        urcrnrlat=max_lat, urcrnrlon=max_lon,
        lat_0=lat_0, lon_0=lon_0 )
    m.drawcoastlines(linewidth=0.1)
    m.drawcountries(linewidth=0.1)
    m.drawparallels(parallels, linewidth=0.1, labels=[1,0,0,1])
    m.drawmeridians(meridians, linewidth=0.1, labels=[1,0,0,1])
    
    # CC_time_shift, SNR
    sx, sy = m(stlo_list, stla_list)
    size_list = [ 0.1 if x<0.1 else x for x in snr_list ]
    im = m.scatter(sx, sy, s=size_list, marker='o',
        c=cc_dt_list, cmap='seismic', 
        edgecolor='grey', linewidth=0.05)
    mean_amp = np.mean(cc_dt_list)
    std_amp = np.std(cc_dt_list)
    #plot_amp = abs(mean_amp)+std_amp
    plot_amp = 5.0 
    im.set_clim(-plot_amp, plot_amp)
    
    # focal mechanism
    sx, sy = m(evlo, evla)
    b = beach(focmec, xy=(sx, sy), width=200000, linewidth=0.2, 
        facecolor='k')
    ax.add_collection(b)
    
    # colorbar
    cbar_ax = fig.add_axes([0.46, 0.575, 0.005, 0.2])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar_ax.tick_params(labelsize=9)
    cbar_ax.set_xlabel('DT_cc(s)', fontsize=9)
    cbar_ax.xaxis.set_label_position('top')
   
    #------ color map CC0, symbol size ~ SNR
    ax = fig.add_axes([0.05, 0.05, 0.4, 0.35])
    ax.set_title("CC0 (symbol_size ~ SNR)")

    m = Basemap(projection='merc', resolution='l',
        llcrnrlat=min_lat, llcrnrlon=min_lon, 
        urcrnrlat=max_lat, urcrnrlon=max_lon,
        lat_0=lat_0, lon_0=lon_0 )
    m.drawcoastlines(linewidth=0.1)
    m.drawcountries(linewidth=0.1)
    m.drawparallels(parallels, linewidth=0.1, labels=[1,0,0,1])
    m.drawmeridians(meridians, linewidth=0.1, labels=[1,0,0,1])
    
    # CC0, SNR 
    sx, sy = m(stlo_list, stla_list)
    #size_list = [ 20**x for x in CCmax_list ]
    size_list = [ 0.1 if x<0.1 else x for x in snr_list ]
    im = m.scatter(sx, sy, s=size_list, marker='o',
        c=CC0_list, cmap='jet', 
        edgecolor='grey', linewidth=0.05)
    im.set_clim(0.5, 1.0)
    
    # focal mechanism
    sx, sy = m(evlo, evla)
    b = Beach(focmec, xy=(sx, sy), width=200000, linewidth=0.2, 
        facecolor='k')
    ax.add_collection(b)
 
    #add colorbar
    cbar_ax = fig.add_axes([0.46, 0.125, 0.005, 0.2])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar_ax.tick_params(labelsize=9) 
    cbar_ax.set_xlabel('CC0', fontsize=9)
    cbar_ax.xaxis.set_label_position('top')

    #------ SNR v.s. CC0, colored by cc_dt, size ~ weight
    ax = fig.add_axes([0.58, 0.65, 0.35, 0.2])
    im = ax.scatter(snr_list, CC0_list, marker='o', 
        s=10.*np.array(weight_list), 
        c=cc_dt_list, cmap='seismic',
        edgecolor='grey', linewidth=0.05)
    mean_amp = np.mean(cc_dt_list)
    std_amp = np.std(cc_dt_list)
    #plot_amp = abs(mean_amp)+std_amp
    plot_amp = 5.0
    im.set_clim(-plot_amp, plot_amp)
    #ax.set_xlim([min(snr_list), max(snr_list)])
    #ax.set_ylim([min(CCmax_list), max(CCmax_list)])
    ax.set_xlim([0, max(snr_list)])
    ax.set_ylim([0.3, 1.0])
    ax.set_xlabel("SNR")
    ax.set_ylabel("CC0")
    #add colorbar
    cbar_ax = fig.add_axes([0.95, 0.65, 0.005, 0.2])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar_ax.tick_params(labelsize=9)
    cbar_ax.set_xlabel('DT_cc(s)', fontsize=9)
    cbar_ax.xaxis.set_label_position('top')

    ##------ CC0 v.s. CCmax, colored by cc_dt
    #ax = fig.add_axes([0.58, 0.375, 0.35, 0.2])
    #im = ax.scatter(CC0_list, CCmax_list, marker='o', s=10,
    #    c=cc_dt_list, cmap='seismic',
    #    edgecolor='grey', linewidth=0.05)
    #mean_amp = np.mean(cc_dt_list)
    #std_amp = np.std(cc_dt_list)
    #plot_amp = abs(mean_amp)+std_amp
    #im.set_clim(-plot_amp, plot_amp)
    #ax.set_xlim([min(CC0_list), max(CC0_list)])
    #ax.set_ylim([min(CCmax_list), max(CCmax_list)])
    #ax.set_xlabel("CC0")
    #ax.set_ylabel("CCmax")
    ##add colorbar
    #cbar_ax = fig.add_axes([0.95, 0.375, 0.005, 0.2])
    #fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    #cbar_ax.tick_params(labelsize=9)
    #cbar_ax.set_xlabel('cc_dt(s)', fontsize=9)
    #cbar_ax.xaxis.set_label_position('top')

    ##------ cc_dt v.s. CCmax, colored by SNR
    #ax = fig.add_axes([0.58, 0.1, 0.35, 0.2])
    #im = ax.scatter(cc_dt_list, CCmax_list, marker='o', s=10, 
    #    c=snr_list, cmap='seismic',
    #    edgecolor='grey', linewidth=0.05)
    #im.set_clim(min(snr_list), max(snr_list))
    #ax.set_xlim([min(cc_dt_list), max(cc_dt_list)])
    #ax.set_ylim([min(CCmax_list), max(CCmax_list)])
    #ax.set_xlabel("cc_dt")
    #ax.set_ylabel("CCmax")
    ##add colorbar
    #cbar_ax = fig.add_axes([0.95, 0.1, 0.005, 0.2])
    #fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    #cbar_ax.tick_params(labelsize=9)
    #cbar_ax.set_xlabel('SNR(dB)', fontsize=9)
    #cbar_ax.xaxis.set_label_position('top')

    ##------ histogram of dt_cc and dt_res
    #ax1 = fig.add_axes([0.5,0.28,0.4,0.15])
    #n, bins, patches = ax1.hist(dt_cc, 50, facecolor='green', alpha=0.75)
    #amp = max(abs(dt_cc))
    #ax1.set_xlim([-amp, amp])
    #ax1.set_title('dt_cc: mean=%.2f std=%.2f' % (np.mean(dt_cc), np.std(dt_cc)))
    #ax1.tick_params(labelsize=10) 
    #
    #ax2 = fig.add_axes([0.5,0.07,0.4,0.15])
    #n, bins, patches = ax2.hist(dt_res, 50, facecolor='green', alpha=0.75)
    #amp = max(abs(dt_cc))
    #ax2.set_xlim([-amp, amp])
    #ax2.set_title('dt_res: mean=%.2f std=%.2f' % (np.mean(dt_res), np.std(dt_res)))
    #ax2.set_xlabel('dt (sec)')
    #ax2.tick_params(labelsize=10)

    #------ save figure
    if not out_file:
      out_file = '%s_%s.pdf' % (event_id, window_id)
    fig.savefig(out_file, format='pdf')
    #fig.savefig("misfit.pdf", bbox_inches='tight', format='pdf')

#
#======================================================
#

  def plot_seismogram_3comp(self,
      savefig=False, 
      out_dir='plot',
      plot_param={
        'time':[0,100], 'rayp':10., 'azbin':10, 'window_id':'F.p,P',
        'SNR':None, 'CC0':None, 'CCmax':None, 'dist':None }
      ):
    """ Plot seismograms for one event
      azbin: azimuthal bin size
      win: 
    """
    comp_name = ['R', 'T', 'Z']
    #------ selection parameters
    plot_time = plot_param['time']
    plot_azbin = plot_param['azbin']
    plot_rayp = plot_param['rayp']
    plot_window_id = plot_param['window_id']

    plot_SNR = np.array(plot_param['SNR'])
    plot_CC0 = np.array(plot_param['CC0'])
    plot_CCmax = np.array(plot_param['CCmax'])
    plot_dist = np.array(plot_param['dist'])

    #------ event info
    event = self.data['event']
    t0 = event['t0']
    tau = event['tau']
    evla = event['latitude']
    evlo = event['longitude']
    evdp = event['depth']
    mt = event['mt_rtp']
    Mrr = mt[0][0]
    Mtt = mt[1][1]
    Mpp = mt[2][2]
    Mrt = mt[0][1]
    Mrp = mt[0][2]
    Mtp = mt[1][2]
    focmec = [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]

    #------ station info
    station_dict = self.data['station']
    stla_all = []
    stlo_all = []
    dist_all = []
    for station_id in station_dict:
      station = station_dict[station_id]
      meta = station['meta']
      window_dict = station['window']
      # select data 
      if station['stat']['code'] < 1:
        continue
      if plot_window_id not in window_dict:
        continue
      stla_all.append(meta['latitude'])
      stlo_all.append(meta['longitude'])
      dist_all.append(meta['dist_degree'])

    #------ traveltime curve
    model = TauPyModel(model="ak135")
    dist_ttcurve = np.arange(0.0,max(dist_all),0.5)
    ttcurve_p = []
    ttcurve_P = []
    ttcurve_s = []
    ttcurve_S = []
    for dist in dist_ttcurve:
      arrivals = model.get_travel_times(
          source_depth_in_km=evdp, 
          distance_in_degree=dist, 
          phase_list=['p','P','s','S'])
      for arr in arrivals:
        if arr.name == 'p':
          ttcurve_p.append((arr.distance, arr.time, arr.ray_param))
        elif arr.name == 'P':
          ttcurve_P.append((arr.distance, arr.time, arr.ray_param))
        elif arr.name == 's':
          ttcurve_s.append((arr.distance, arr.time, arr.ray_param))
        elif arr.name == 'S':
          ttcurve_S.append((arr.distance, arr.time, arr.ray_param))
    # sort phases
    ttcurve_p = sorted(ttcurve_p, key=lambda x: x[2])
    ttcurve_P = sorted(ttcurve_P, key=lambda x: x[2])
    ttcurve_s = sorted(ttcurve_s, key=lambda x: x[2])
    ttcurve_S = sorted(ttcurve_S, key=lambda x: x[2])

    #------ map configuration 
    min_lat = min(min(stla_all), evla)
    max_lat = max(max(stla_all), evla)
    lat_range = max_lat - min_lat
    min_lat -= 0.1*lat_range
    max_lat += 0.1*lat_range
    min_lon = min(min(stlo_all), evlo)
    max_lon = max(max(stlo_all), evlo)
    lon_range = max_lon - min_lon
    min_lon -= 0.1*lon_range
    max_lon += 0.1*lon_range
    lat_0 = np.mean(stla_all)
    lon_0 = np.mean(stlo_all)
    # 
    parallels = np.arange(0.,81,10.)
    meridians = np.arange(0.,351,10.)

    #------ plot azimuthal bins (one figure per azbin)
    if plot_azbin <= 0.0:
      raise Exception("plot_param['azbin']=%f must > 0.0" % plot_azbin)

    for az in np.arange(0, 360, plot_azbin):
      azmin = az
      azmax = az + plot_azbin

      print(azmin, azmax)

      #---- gather data for the current azbin 
      data_azbin = {}
      for station_id in station_dict:
        # skip bad station
        station = station_dict[station_id]
        if station['stat']['code'] < 1:
          continue

        # skip un-selected station 
        meta = station['meta']
        azimuth = meta['azimuth']
        dist_degree = meta['dist_degree']
        if plot_dist.any():
          if dist_degree < np.min(plot_dist) or dist_degree > np.max(plot_dist):
            continue
        if azimuth < azmin or azimuth >= azmax:
          continue
        if plot_window_id not in window_dict:
          continue

        # skip bad window
        window_dict = station['window']
        window = window_dict[plot_window_id]
        if window['stat']['code'] <= 0:
          continue

        quality = window['quality']
        if plot_SNR and quality['SNR']<np.min(plot_SNR):
          continue

        cc = window['cc']
        if plot_CC0 and cc['CC0']<np.min(plot_CC0):
          continue
        if plot_CCmax and cc['CCmax']<np.min(plot_CCmax):
          continue

        # get seismograms: syn/obs
        waveform = station['waveform']
        time_sample = waveform['time_sample']
        syn_starttime = time_sample['starttime']
        syn_npts = time_sample['nt']
        syn_delta = time_sample['delta']
        syn_nyq = 0.5/syn_delta
        obs = waveform['obs']
        grn = waveform['grn']
        # filter parameter
        filter_param = window['filter']
        filter_a = filter_param['a']
        filter_b = filter_param['b']
        # filter seismograms 
        obs = signal.filtfilt(filter_b, filter_a, obs)
        grn = signal.filtfilt(filter_b, filter_a, grn)
        # convolve stf on grn
        syn_freq = np.fft.rfftfreq(syn_npts, d=syn_delta)
        F_src = stf_gauss_spectrum(syn_freq, event['tau'])
        syn = np.fft.irfft(F_src*np.fft.rfft(grn), syn_npts)
        # project to polarity defined by the window
        proj_matrix = window['polarity']['proj_matrix']
        obs = np.dot(proj_matrix, obs)
        syn = np.dot(proj_matrix, syn)
        # rotate EN(0:2) -> RT(0:2) (T-R-Z: right-hand convention)
        Raz = (meta['back_azimuth'] + 180.0) % 360.0
        sin_Raz = np.sin(np.deg2rad(Raz))
        cos_Raz = np.cos(np.deg2rad(Raz))
        proj_matrix = [ 
            [sin_Raz,  cos_Raz], 
            [cos_Raz, -sin_Raz] ]
        obs[0:2,:] = np.dot(proj_matrix, obs[0:2,:])
        syn[0:2,:] = np.dot(proj_matrix, syn[0:2,:])
        # append to data
        data_dict = {
            'meta': meta,
            'window': window,
            'syn': syn,
            'obs': obs
            }
        data_azbin[station_id] = data_dict
      #endfor station_id in station_dict:
  
      #---- skip empty azbin
      if not data_azbin:
        warn_str = "No station in the azbin [%f %f]." %(azmin, azmax)
        warnings.warn(warn_str)
        continue

      #---- create figure
      fig = plt.figure(figsize=(8.5, 11)) # US letter
      str_title = '{:s} (win: {:s}, az: {:04.1f}~{:04.1f})'.format(
          event['id'], plot_window_id, azmin, azmax)
      fig.text(0.5, 0.95, str_title, size='x-large', horizontalalignment='center')
      #---- plot station/event map
      ax_origin = [0.3, 0.74]
      ax_size = [0.4, 0.2]
      ax_map = fig.add_axes(ax_origin + ax_size)
      ax_bm = Basemap(projection='merc', resolution='l',
          llcrnrlat=min_lat, llcrnrlon=min_lon, 
          urcrnrlat=max_lat, urcrnrlon=max_lon,
          lat_0=lat_0, lon_0=lon_0 )
      ax_bm.drawcoastlines(linewidth=0.1)
      ax_bm.drawcountries(linewidth=0.1)
      ax_bm.drawparallels(parallels, linewidth=0.1, labels=[1,0,0,1], 
          fontsize=10, fmt='%3.0f')
      ax_bm.drawmeridians(meridians, linewidth=0.1, labels=[1,0,0,1], 
          fontsize=10, fmt='%3.0f')
      sx, sy = ax_bm(stlo_all, stla_all)
      ax_bm.scatter(sx, sy, s=10, marker='^', facecolor='blue', edgecolor='')
      # plot focal mechanism
      sx, sy = ax_bm(evlo, evla)
      bb_width = 110000.0 * np.abs(max(stlo_all)-min(stlo_all)) * 0.1
      b = Beach(focmec, xy=(sx, sy), width=bb_width, linewidth=0.2, facecolor='r')
      ax_map.add_collection(b)
      #-- plot the station location
      stla = [ x['meta']['latitude'] for x in data_azbin.itervalues() ]
      stlo = [ x['meta']['longitude'] for x in data_azbin.itervalues() ]
      sx, sy = ax_bm(stlo, stla)
      ax_bm.scatter(sx, sy, s=10, marker='^', facecolor='red', edgecolor='')

      #-- create axis for seismograms
      ax_RTZ = []
      for i in range(3):
        ax_origin = [0.07+0.3*i, 0.05]
        ax_size = [0.25, 0.65]
        ax_RTZ.append(fig.add_axes(ax_origin + ax_size))

      #-- plot traveltime curves
      for i in range(3):
        ax = ax_RTZ[i]
        ax.plot([x[1]-plot_rayp*x[0] for x in ttcurve_p], \
            [x[0] for x in ttcurve_p], 'b-', linewidth=0.2)
        ax.plot([x[1]-plot_rayp*x[0] for x in ttcurve_P], \
            [x[0] for x in ttcurve_P], 'b-', linewidth=0.2)
        ax.plot([x[1]-plot_rayp*x[0] for x in ttcurve_s], \
            [x[0] for x in ttcurve_s], 'c-', linewidth=0.2)
        ax.plot([x[1]-plot_rayp*x[0] for x in ttcurve_S], \
            [x[0] for x in ttcurve_S], 'c-', linewidth=0.2)

      #-- ylim setting
      y = [ x['meta']['dist_degree'] for x in data_azbin.itervalues() ]
      ny = len(y)
      plot_dy = 0.5*(max(y)-min(y)+1)/ny
      if plot_dist:
        plot_ymax = max(plot_dist) + 2*plot_dy
        plot_ymin = min(plot_dist) - 2*plot_dy
      else:
        plot_ymax = max(y) + 2*plot_dy
        plot_ymin = min(y) - 2*plot_dy

      #-- plot each station
      for station_id in data_azbin:
        station = data_azbin[station_id]
        meta = station['meta']
        window = station['window']
        syn = station['syn']
        obs = station['obs']

        # get plot time 
        dist_degree = meta['dist_degree']
        reduced_time = dist_degree * plot_rayp
        # time of first sample referred to centroid time 
        t0 = syn_starttime - event['t0']
        # time of samples referred to centroid time
        syn_times = syn_delta*np.arange(syn_npts) + t0
        # plot time window
        plot_t0 = min(plot_time) + reduced_time
        plot_t1 = max(plot_time) + reduced_time
        plot_idx = (syn_times > plot_t0) & (syn_times < plot_t1)
        # plot time
        t_plot = syn_times[plot_idx] - reduced_time

        #  window begin/end
        taper = window['taper']
        win_starttime = taper['starttime']
        win_endtime = taper['endtime']
        win_t0 = (win_starttime - event['t0']) - reduced_time
        win_t1 = (win_endtime - event['t0']) - reduced_time

        # plot seismograms
        Amax_obs = np.sqrt(np.max(np.sum(obs[:,plot_idx]**2, axis=0)))
        Amax_syn = np.sqrt(np.max(np.sum(syn[:,plot_idx]**2, axis=0)))
        for i in range(3):
          ax = ax_RTZ[i]
          ax.plot(t_plot, plot_dy*obs[i,plot_idx]/Amax_obs+dist_degree, \
              'k-', linewidth=0.5)
          ax.plot(t_plot, plot_dy*syn[i,plot_idx]/Amax_syn+dist_degree, \
              'r-', linewidth=0.5)
          # mark measure window range
          ax.plot(win_t0, dist_degree, 'k|', markersize=8)
          ax.plot(win_t1, dist_degree, 'k|', markersize=8)
          # annotate amplitude
          if i == 0:
            ax.text(max(plot_time), dist_degree, '%.1e ' % (Amax_obs), 
                verticalalignment='bottom', 
                horizontalalignment='right', 
                fontsize=7, color='black')
            ax.text(max(plot_time), dist_degree, '%.1e ' % (Amax_syn), 
                verticalalignment='top', 
                horizontalalignment='right', 
                fontsize=7, color='red')
          # annotate CC0
          if i == 0:
            ax.text(max(plot_time), dist_degree, ' %.3f'%(window['cc']['CC0']),
                verticalalignment='center', fontsize=7)
          # annotate window weight
          if i == 1:
            ax.text(max(plot_time), dist_degree, ' %.1f' % (window['weight']),
                verticalalignment='center', fontsize=7)
          #annotate station names 
          if i == 2:
            #str_annot = '%.3f,%.1f,%s' % (
            #    misfit['CC0'], window['weight'], station_id)
            ax.text(max(plot_time), dist_degree, ' '+station_id, \
                verticalalignment='center', fontsize=7)
      #endfor data in data_azbin:

      #-- set axes limits and lables, annotation
      for i in range(3):
        ax = ax_RTZ[i]
        ax.set_xlim(min(plot_time), max(plot_time))
        ax.set_ylim(plot_ymin, plot_ymax)
        ax.set_title(comp_name[i])
        ax.set_xlabel('t - {:.1f}*dist (s)'.format(plot_rayp))
        ax.tick_params(axis='both',labelsize=10)
        # ylabel 
        if i == 0:
          ax.set_ylabel('dist (deg)')
        else:
          ax.set_yticklabels([])

      #-- save figures
      if savefig: 
        out_file = '%s/%s_az_%03d_%03d_%s.pdf' \
            % (out_dir, event['id'], azmin, azmax, plot_window_id)
        plt.savefig(out_file, format='pdf')
      else:
        plt.show()
      plt.close(fig)
    
#
#======================================================
#

  def plot_seismogram_1comp(self,
      savefig=False, 
      out_dir='plot',
      window_id='p,P_Z',
      azbin=10,
      begin_time=0,
      end_time=0,
      clip_ratio=1.5,
      min_CC0=None,
      min_CCmax=None,
      min_SNR=None,
      dist_lim=None,
      plot_az0=0,
      ):
    """ 
    Plot record section in azimuthal bins

    Parameters
    ----------
    azbin: azimuthal bin size

    begin/end_time: time range that is added to the automatically determined
      plot time range. See below.

    clip: do not plot waveform with amplitudes larger than 
      <clip>*max_amplitude_in_select_time_window

    Notes
    -----
      The time for plot is reduced time relative to origin time + rayp*dist

      linear regression is done to find the average rayp of misfit windows
      and the plot begin/end time is found that can include all misfit windows
      in the plot.

    """
    #------ check parameters
    plot_time = np.array([begin_time, end_time])

    plot_azbin = float(azbin)
    if plot_azbin <= 0:
      raise Exception("plot_azbin(%f) should be larger than 0.0" % (plot_azbin))

    plot_window_id = window_id
    plot_SNR = np.array(min_SNR)
    plot_CC0 = np.array(min_CC0)
    plot_CCmax = np.array(min_CCmax)
    plot_dist = np.array(dist_lim)

    plot_clip = float(clip_ratio)
    if plot_clip < 1.0:
      raise Exception("clip_ratio(%f) should be larger than 1.0" % (plot_clip))

    #------ event info
    event = self.data['event']
    t0 = event['t0']
    tau = event['tau']
    evla = event['latitude']
    evlo = event['longitude']
    evdp = event['depth']
    # evdp has to be >=0 otherwise taup would crash
    if evdp < 0.0:
      evdp = 0.0

    mt = event['mt_rtp']
    Mrr = mt[0][0]
    Mtt = mt[1][1]
    Mpp = mt[2][2]
    Mrt = mt[0][1]
    Mrp = mt[0][2]
    Mtp = mt[1][2]
    focmec = [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]

    #------ get station info
    station_dict = self.data['station']
    stla_all = []
    stlo_all = []
    dist_all = []
    winb_all = []
    wine_all = []
    for station_id in station_dict:
      station = station_dict[station_id]
      meta = station['meta']
      window_dict = station['window']
      # select data 
      if station['stat']['code'] < 1:
        continue
      if plot_window_id not in window_dict:
        continue
      stla_all.append(meta['latitude'])
      stlo_all.append(meta['longitude'])
      dist_all.append(meta['dist_degree'])
      taper = window_dict[plot_window_id]['taper']
      winb_all.append(taper['starttime'] - event['t0'])
      wine_all.append(taper['endtime'] - event['t0'])
    
    if not dist_all:
      warnings.warn("No data to plot!")
      return

    # get average moveout of the window center 
    dist_all = np.array(dist_all)
    winb_all = np.array(winb_all)
    wine_all = np.array(wine_all)
    winc_all = (winb_all + wine_all)/2.0
    # linear regression tc = dist*rayp + tb
    A = np.vstack([dist_all, np.ones(len(dist_all))]).T
    plot_rayp, plot_c = np.linalg.lstsq(A, winc_all)[0]
    # round to the integer
    plot_rayp = np.round(plot_rayp)
    # get time window relative to the regressed window central time
    plot_t0 = np.min(winb_all - plot_rayp*dist_all)
    plot_t1 = np.max(wine_all - plot_rayp*dist_all)
    # modify the plot time rage
    plot_time[0] += plot_t0
    plot_time[1] += plot_t1

    #------ calculate traveltime curves (only for body wave)
    phase_names = plot_window_id.split('_')[0]
    if phase_names not in ['surface', 'Rayleigh', 'Love']:
      model = TauPyModel(model="ak135")
      # distance samples
      dist_ttcurve = np.arange(0.0,max(dist_all),0.5)
      phase_list = [x for x in phase_names.split(',')]
      ttcurve = {}
      for phase_name in phase_list:
        ttcurve[phase_name] = []
      for dist in dist_ttcurve:
        arrivals = model.get_travel_times(
            source_depth_in_km=evdp, 
            distance_in_degree=dist, 
            phase_list=phase_list)
        for arr in arrivals:
          for phase_name in phase_list:
            if arr.name == phase_name:
              ttcurve[phase_name].append((arr.distance, arr.time, arr.ray_param))
      # sort (dist, ttime, rayp) points based on ray parameter
      for phase_name in phase_list:
        ttcurve[phase_name] = sorted(ttcurve[phase_name], key=lambda x: x[2])

    #------ map configuration 
    min_lat = min(min(stla_all), evla)
    max_lat = max(max(stla_all), evla)
    lat_range = max_lat - min_lat
    min_lat -= 0.1*lat_range
    max_lat += 0.1*lat_range
    min_lon = min(min(stlo_all), evlo)
    max_lon = max(max(stlo_all), evlo)
    lon_range = max_lon - min_lon
    min_lon -= 0.1*lon_range
    max_lon += 0.1*lon_range
    lat_0 = np.mean(stla_all)
    lon_0 = np.mean(stlo_all)
    #
    parallels = np.arange(-90.,89.,10.)
    meridians = np.arange(0.,351,10.)

    #------ plot azimuthal bins (one figure per azbin)
    if plot_azbin <= 0.0:
      raise Exception("plot_param['azbin']=%f must > 0.0" % plot_azbin)

    for az in np.arange(plot_az0, plot_az0+360, plot_azbin):
      azmin = az
      azmax = az + plot_azbin

      print("Azimuthal range: ", azmin, azmax)

      #---- gather data for the current azbin 
      data_azbin = {}
      for station_id in station_dict:
        station = station_dict[station_id]
        # skip bad station
        if station['stat']['code'] < 1:
          continue
        # skip station not in the selection criteria 
        meta = station['meta']
        azimuth = meta['azimuth']
        dist_degree = meta['dist_degree']
        if plot_dist.any():
          if dist_degree < np.min(plot_dist) or dist_degree > np.max(plot_dist):
            continue
        #if azimuth < azmin or azimuth >= azmax:
        if (azimuth-azmin)%360 >= plot_azbin:
          continue

        window_dict = station['window']
        # check if required window exists
        if plot_window_id not in window_dict:
          continue
        window = window_dict[plot_window_id]
        # skip bad window
        if window['stat']['code'] <= 0:
          continue
        # skip window which does not pass the selection criteria
        quality = window['quality']
        if plot_SNR and quality['SNR']<np.min(plot_SNR):
          continue
        cc = window['cc']
        if plot_CC0 and cc['CC0']<np.min(plot_CC0):
          continue
        if plot_CCmax and cc['CCmax']<np.min(plot_CCmax):
          continue

        # get seismograms: syn/obs
        waveform = station['waveform']
        time_sample = waveform['time_sample']
        syn_starttime = time_sample['starttime']
        syn_npts = time_sample['nt']
        syn_delta = time_sample['delta']
        syn_nyq = 0.5/syn_delta
        # filter parameter
        filter_param = window['filter']
        filter_a = filter_param['a']
        filter_b = filter_param['b']
        # filter seismograms 
        obs = signal.filtfilt(filter_b, filter_a, waveform['obs'])
        if 'syn' in waveform:
          syn = signal.filtfilt(filter_b, filter_a, waveform['syn'])
        elif 'grn' in waveform:
          grn = signal.filtfilt(filter_b, filter_a, waveform['grn'])
          # convolve stf with grn
          syn_freq = np.fft.rfftfreq(syn_npts, d=syn_delta)
          F_src = stf_gauss_spectrum(syn_freq, event['tau'])
          syn = np.fft.irfft(F_src*np.fft.rfft(grn), syn_npts)
        else:
          err = "station(%s) has no syn or grn in waveform data." % (station_id)
          raise Exception(err)
        # project to polarity defined by the window
        polarity = window['polarity']
        comp = polarity['component']
        cmpaz = polarity['azimuth']
        cmpdip = polarity['dip']
        if comp in ['Z', 'R', 'T']:
          sin_az = np.sin(np.deg2rad(cmpaz))
          cos_az = np.cos(np.deg2rad(cmpaz))
          sin_dip = np.sin(np.deg2rad(cmpdip))
          cos_dip = np.cos(np.deg2rad(cmpdip))
          cmp_vec = np.array([ 
            cos_dip*sin_az, # cos(E, comp)
            cos_dip*cos_az, # N, comp
            -sin_dip] )     # Z, comp
        else:
          raise Exception("Not single component: " % (comp))
        obs = np.dot(cmp_vec, obs)
        syn = np.dot(cmp_vec, syn)
        # append to data
        data_dict = {
            'meta': meta,
            'window': window,
            'syn': syn,
            'obs': obs
            }
        data_azbin[station_id] = data_dict
      #endfor station_id in station_dict:
  
      #---- skip empty azbin
      if not data_azbin:
        warn_str = "No station in the azbin [%f %f]." %(azmin, azmax)
        warnings.warn(warn_str)
        continue

      #---- create figure
      fig = plt.figure(figsize=(11, 8.5)) # US Letter
      str_title = '{:s} ({:s} az:{:04.1f}~{:04.1f} dep:{:.1f})'.format(
          event['id'], plot_window_id, azmin, azmax, event['depth'])
      fig.text(0.5, 0.965, str_title, size='x-large', horizontalalignment='center')

      #---- plot station/event map
      ax_origin = [0.05, 0.60]
      ax_size = [0.3, 0.3]
      ax_map = fig.add_axes(ax_origin + ax_size)
      ax_bm = Basemap(projection='merc', resolution='l',
          llcrnrlat=min_lat, llcrnrlon=min_lon, 
          urcrnrlat=max_lat, urcrnrlon=max_lon,
          lat_0=lat_0, lon_0=lon_0 )
      ax_bm.drawcoastlines(linewidth=0.1)
      ax_bm.drawcountries(linewidth=0.1)
      ax_bm.drawparallels(parallels, linewidth=0.1, labels=[1,0,0,1], 
          fontsize=10, fmt='%3.0f')
      ax_bm.drawmeridians(meridians, linewidth=0.1, labels=[1,0,0,1], 
          fontsize=10, fmt='%3.0f')
      sx, sy = ax_bm(stlo_all, stla_all)
      ax_bm.scatter(sx, sy, s=10, marker='^', facecolor='blue', edgecolor='')
      # plot focal mechanism
      sx, sy = ax_bm(evlo, evla)
      bb_width = 110000.0 * np.abs(max(stlo_all)-min(stlo_all)) * 0.1
      b = beach(focmec, xy=(sx, sy), width=bb_width, linewidth=0.2, facecolor='r')
      ax_map.add_collection(b)
      #-- plot the station location
      stla = [ data_azbin[key]['meta']['latitude'] for key in data_azbin ]
      stlo = [ data_azbin[key]['meta']['longitude'] for key in data_azbin ]
      sx, sy = ax_bm(stlo, stla)
      ax_bm.scatter(sx, sy, s=10, marker='^', facecolor='red', edgecolor='')

      #-- create axis for seismograms
      ax_origin = [0.45, 0.05]
      #ax_size = [0.43, 0.90]
      ax_size = [0.2, 0.90]
      ax_1comp = fig.add_axes(ax_origin + ax_size)

      #-- ylim setting
      y = [ data_azbin[key]['meta']['dist_degree'] for key in data_azbin ]
      ny = len(y)
      plot_dy = 0.5*(max(y)-min(y)+1)/ny
      if plot_dist:
        plot_ymax = max(plot_dist) + 2*plot_dy
        plot_ymin = min(plot_dist) - 2*plot_dy
      else:
        plot_ymax = max(y) + 2*plot_dy
        plot_ymin = min(y) - 2*plot_dy

      #-- plot traveltime curves
      if phase_names not in ['surface', 'Rayleigh', 'Love']:
        for phase_name in phase_list:
          # skip if no tt curves for this phase_names
          if not ttcurve[phase_name]:
            continue
          # reduced time
          phase_times = np.array([x[1]-plot_rayp*x[0] for x in ttcurve[phase_name]])
          phase_distances = np.array([x[0] for x in ttcurve[phase_name]])
          # skip if not in plot range
          max_dist = np.max(phase_distances)
          min_dist = np.min(phase_distances)
          if max_dist < plot_ymin or min_dist > plot_ymax:
            continue
          ax_1comp.plot(phase_times, phase_distances, 'b-', linewidth=0.1)
          #ax_1comp.plot(phase_times, phase_distances, 'b.', markersize=0.5)
          # label phase names
          if max_dist < plot_ymax:
            y_str = max_dist
            x_str = max(phase_times[phase_distances==max_dist])
          else:
            y_str = plot_ymax
            max_dist = max(phase_distances[phase_distances<= plot_ymax])
            x_str = max(phase_times[phase_distances==max_dist])
          ax_1comp.text(x_str, y_str, phase_name,
              verticalalignment='top',
              horizontalalignment='center',
              fontsize=11, color='blue')

      #-- plot each station
      for station_id in data_azbin:
        station = data_azbin[station_id]
        meta = station['meta']
        window = station['window']
        syn = station['syn']
        obs = station['obs']

        # get plot time 
        dist_degree = meta['dist_degree']
        reduced_time = dist_degree * plot_rayp
        # time of first sample referred to centroid time 
        t0 = syn_starttime - event['t0']
        # time of samples referred to centroid time
        syn_times = syn_delta*np.arange(syn_npts) + t0
        # plot time window
        plot_t0 = min(plot_time) + reduced_time
        plot_t1 = max(plot_time) + reduced_time
        plot_idx = (syn_times > plot_t0) & (syn_times < plot_t1)
        # plot time (reduced time)
        t_plot = syn_times[plot_idx] - reduced_time

        #  window begin/end
        taper = window['taper']
        win_starttime = taper['starttime'] - event['t0']
        win_endtime = taper['endtime'] - event['t0']
        win_t0 = win_starttime - reduced_time
        win_t1 = win_endtime - reduced_time
        win_idx = (syn_times > win_starttime) & (syn_times < win_endtime)

        # plot seismograms
        Amax_obs = np.sqrt(np.max(obs[win_idx]**2))
        Amax_syn = np.sqrt(np.max(syn[win_idx]**2))
        
        # clip large amplitudes
        y = obs[plot_idx]/Amax_obs
        idx = abs(y) > plot_clip+1.0e-3
        y[idx] = np.nan
        ax_1comp.plot(t_plot, plot_dy*y+dist_degree, \
            'k-', linewidth=0.5)

        y = syn[plot_idx]/Amax_syn
        idx = abs(y) > plot_clip+1.0e-3
        y[idx] = np.nan
        ax_1comp.plot(t_plot, plot_dy*y+dist_degree, \
            'r-', linewidth=0.5)
        # mark measure window range
        ax_1comp.plot(win_t0, dist_degree, 'k|', markersize=8)
        ax_1comp.plot(win_t1, dist_degree, 'k|', markersize=8)
        ## annotate amplitude
        #  ax.text(max(plot_time), dist_degree, '%.1e ' % (Amax_obs), 
        #      verticalalignment='bottom', 
        #      horizontalalignment='right', 
        #      fontsize=7, color='black')
        #  ax.text(max(plot_time), dist_degree, '%.1e ' % (Amax_syn), 
        #      verticalalignment='top', 
        #      horizontalalignment='right', 
        #      fontsize=7, color='red')
        ## annotate CC0
        #  ax.text(max(plot_time), dist_degree, ' %.3f'%(window['cc']['CC0']),
        #      verticalalignment='center', fontsize=7)
        ## annotate window weight
        #if i == 1:
        #  ax.text(max(plot_time), dist_degree, ' %.1f' % (window['weight']),
        #      verticalalignment='center', fontsize=7)
        ##annotate station names 
        str_annot = ' %s (%.3f,%.1f)' % (station_id,
            window['cc']['CC0'], window['weight'])
        ax_1comp.text(max(plot_time), dist_degree, str_annot, 
            verticalalignment='center', fontsize=7)
      #endfor data in data_azbin:

      #-- set axes limits and lables, annotation
      ax_1comp.set_xlim(min(plot_time), max(plot_time))
      ax_1comp.set_ylim(plot_ymin, plot_ymax)
      ax_1comp.set_xlabel('t - {:.1f}*dist (s)'.format(plot_rayp))
      ax_1comp.tick_params(axis='both',labelsize=10)
      # ylabel 
      ax_1comp.set_ylabel('dist (deg)')

      #-- save figures
      if savefig: 
        out_file = '%s/%s_az_%03d_%03d_%s.pdf' \
            % (out_dir, event['id'], azmin, azmax, plot_window_id)
        plt.savefig(out_file, format='pdf')
      else:
        plt.show()
      plt.close(fig)

#
#======================================================
#

  def output_adj_hess_part1(self,
      out_dir='adj',
      syn_band_code='MX'):
    """
    Output adjoint sources for the estimation of approximated Hessian diagonals.

    This only output adjoint source of Part 1, that is, only the random part.

    We need run adjoint simulations for Part 1 and 2 separately.

    There are actually two more terms in the Hessian related to the recorded waveforms.
    However if the recorded and modelled waveforms are similar (cc0 close to 1), then
    the difference can be ignored (for hessian) and simplified to Part 2.

    Notes
    -----
    For the objective function as the normalized zero-lag correlation ceofficient, 
    the approximated Hessian can be seperated into two parts:

    Part 1:
        H = - CC0 * norm(wFu)^(-2) * (wFdu1, wFdu2)
          = - Nw^-1 * Aw * (wFdu1, wFdu2)
        , where CC0 > 0 is the zero-lag correlation coefficient of (wFd, wFu)
        and the corresponding adjoint source is

        adj = sqrt(CC0) * norm(wFu)^(-1) * conj(F)(transpose(w)r) 
        , where r is a radome time series such that E(r*transpose(r)) = i.d.
        , for example, standard normal distribution N(0,1)

        This follows the idea of random phase encoding method.

    Part 2:
        H(x,y) = + CC0 * norm(wFu)^(-4) * (wFu, wFdu1) * (wFu, wFdu2)
               = H1(x) * H2(y)
        H1 = sqrt(CC0) * norm(wFu)^(-2) * (wFu, wFdu1)
        , the cooresponding adjoint source is

        adj = sqrt(CC0) * norm(wFu)^(-2) * conj(F)(transpose(w)wFu)
    """
    syn_orientation_codes = ['E', 'N', 'Z']
    event = self.data['event']

    #------ loop each station
    station_dict = self.data['station']
    for station_id in station_dict:
      station = station_dict[station_id]
      # skip rejected statations
      if station['stat']['code'] < 1:
        continue

      # waveform
      waveform = station['waveform']
      time_sample = waveform['time_sample']
      syn_starttime = time_sample['starttime']
      syn_delta = time_sample['delta']
      syn_nt = time_sample['nt']
      syn_nl = time_sample['nl']
      syn_nr = time_sample['nr']

      syn = waveform['syn']

      #------ loop each window
      adj = np.zeros((3, syn_nt))
      # normal distribution
      rand = np.random.randn(3, syn_nt)

      window_dict = station['window']
      for window_id in window_dict:
        # window parameters
        window = window_dict[window_id]
        # skip bad windows
        if window['stat']['code'] < 1:
          warnings.warn("Window %s not measured for adj, SKIP" % window_id)
          continue
        if window['weight'] < 1.0e-3:
          continue

        #------ window parameters 
        # filter
        filter_dict = window['filter']
        filter_a = filter_dict['a']
        filter_b = filter_dict['b']
        # taper
        win_func = window['taper']['win']
        # polarity projection 
        proj_matrix = window['polarity']['proj_matrix']
        # CC0
        cc0 = window['cc']['CC0']

        #------ Part 1: random adjoint source
        # adj = sqrt(CC0) * norm(wFu)^(-1) * conj(F)(transpose(w)r) 
        wr = np.dot(np.transpose(proj_matrix), rand) * win_func
        Fwr = signal.filtfilt(filter_b, filter_a, wr[:,::-1])
        Fwr = Fwr[:,::-1]

        #------ Part 2: synthetic related adjoint source
        # adj = sqrt(CC0) * norm(wFu)^(-2) * conj(F)(transpose(w)wFu)
        Fu = signal.filtfilt(filter_b, filter_a, syn)
        wFu = np.dot(proj_matrix, Fu) * win_func
        norm_wFu = np.sqrt(np.sum(wFu**2))
        #wwFu = np.dot(np.transpose(proj_matrix), wFu) * win_func 
        #FwwFu = signal.filtfilt(filter_b, filter_a, wwFu[:,::-1])
        #FwwFu = FwwFu[:,::-1]

        #------ Part 1: make adjoint source for the current window
        adj_w = np.sqrt(cc0)/norm_wFu * Fwr

        #------ Part 1: add into total adjoint source
        # chi = sum(chi_w * window_weight, over all w[indow])
        adj += np.sqrt(window['weight']) * adj_w

      #endfor window_id in window_dict:

      #------ output adjoint source
      # without padding
      npts = syn_nt - syn_nl - syn_nr
      starttime = syn_starttime + syn_nl*syn_delta
      # time samples for ascii output, referred to origin time
      syn_times = np.arange(npts)*syn_delta
      b = starttime - event['t0']
      syn_times += b

      # loop ENZ
      tr = Trace()
      for i in range(3):
        tr.data = adj[i, syn_nl:(syn_nl+npts)]
        tr.stats.starttime = starttime
        tr.stats.delta = syn_delta

        out_file = '{:s}/{:s}.{:2s}{:1s}'.format(
            out_dir, station_id, syn_band_code,
            syn_orientation_codes[i])

        # sac format
        tr.write(out_file + '.adj.sac', 'sac')

        # ascii format (needed by SEM)
        # time is relative to event origin time: t0
        with open(out_file+'.adj','w') as fp:
          for j in range(npts):
            fp.write("{:16.9e}  {:16.9e}\n".format(
              syn_times[j], adj[i,syn_nl+j]))

    #endfor station_id in station_dict:

#
#======================================================
#

  def hess_diag_dmodel(self, model_name):
    """
    Output approximated Hessian diagonals for one model perturbation i.e. H(dm, dm).

    There are actually two more terms in the Hessian related to the recorded waveforms.
    However if the recorded and modelled waveforms are similar (cc0 close to 1), then
    the difference can be ignored (for hessian) and simplified to Part 2.

    Notes
    -----
    For the objective function as the normalized zero-lag correlation ceofficient, 
    the approximated Hessian can be seperated into two parts:

    Part 1:
        H = - CC0 * norm(wFu)^(-2) * (wFdu1, wFdu2)
          = - Nw^-1 * Aw * (wFdu1, wFdu2)
        , where CC0 > 0 is the zero-lag correlation coefficient of (wFd, wFu)

    Part 2:
        H(x,y) = + CC0 * norm(wFu)^(-4) * (wFu, wFdu1) * (wFu, wFdu2)
               = H1(x) * H2(y)
    """
    event = self.data['event']

    #------ loop each station
    station_dict = self.data['station']
    for station_id in station_dict:
      station = station_dict[station_id]
      # skip rejected statations
      if station['stat']['code'] < 1:
        continue

      # initial waveform
      waveform = station['waveform']
      time_sample = waveform['time_sample']
      syn_starttime = time_sample['starttime']
      syn_delta = time_sample['delta']
      syn_nt = time_sample['nt']
      syn_nl = time_sample['nl']
      syn_nr = time_sample['nr']
      syn = waveform['syn']

      # perturbed waveform  
      waveform_der = station['waveform_der']
      du = waveform_der[model_name]['du']

      #------ loop each window
      window_dict = station['window']
      for window_id in window_dict:
        # window parameters
        window = window_dict[window_id]
        # skip bad windows
        if window['stat']['code'] < 1:
          warnings.warn("Window %s not measured for adj, SKIP" % window_id)
          continue

        #------ window parameters 
        # filter
        filter_dict = window['filter']
        filter_a = filter_dict['a']
        filter_b = filter_dict['b']
        # taper
        win_func = window['taper']['win']
        # polarity projection 
        proj_matrix = window['polarity']['proj_matrix']
        # CC0
        cc0 = window['cc']['CC0']

        #------ hessian for current window
        Fu = signal.filtfilt(filter_b, filter_a, syn)
        wFu = np.dot(proj_matrix, Fu) * win_func
        norm2_wFu = np.sum(wFu**2)

        Fdu = signal.filtfilt(filter_b, filter_a, du)
        wFdu = np.dot(proj_matrix, Fdu) * win_func

        hess_win = -1.0 * cc0/norm2_wFu * (np.sum(wFdu**2) - np.sum(wFu*wFdu)**2/norm2_wFu)

        #------ record hess
        if 'hess_diag' not in window:
          window['hess_diag'] = {}
        window['hess_diag'][model_name] = hess_win

      #endfor window_id in window_dict:
    #endfor station_id in station_dict:
  #enddef hess_diag_dmodel

#
#======================================================
#

  def output_hess_diag(self, model_name, out_file='hess_diag.txt'):
    """
    Output hess diag

    """
    event = self.data['event']
    station_dict = self.data['station']

    f = open(out_file, 'w')
    f.write("#station window hess_diag(%s)\n" % (model_name))

    #------ loop each station
    for station_id in station_dict:
      station = station_dict[station_id]
      # skip rejected statations
      if station['stat']['code'] < 1:
        continue

      #------ loop each window
      window_dict = station['window']
      for window_id in window_dict:
        # window parameters
        window = window_dict[window_id]
        # skip bad windows
        if window['stat']['code'] < 1:
          warnings.warn("Window %s not measured for adj, SKIP" % window_id)
          continue
        # write out hess_diag
        hess_diag = window['hess_diag'][model_name]

        f.write("{:15s} {:15s} {:12.5e}\n".format(
          station_id, window_id, hess_diag))

    f.close()
  #enddef 

#
#======================================================
#

  def output_adj_hess_model_product(self,
      model_name='perturb',
      out_dir='adj',
      syn_band_code='MX'):
    """
    Output adjoint sources for calculating Hessian-model product H * dm.

    Notes
    -----
    For the objective function as the normalized zero-lag correlation ceofficient, 
    the approximate Hessian for one data window is (ignore second order derivative in u)

    Hw = - Nw^-1 * Aw * (wFdu1, wFdu2)
         + 3 * Nw^-1 * Aw * norm(wFu)^-2 * (wFu, wFdu1) * (wFu, wFdu2)
         - Nw^-1 * norm(wFu)^-2 * (wFu, wFdu1) * (wFd, wFdu2)
         - Nw^-1 * norm(wFu)^-2 * (wFu, wFdu2) * (wFd, wFdu1)

    , where Nw = norm(wFu)*norm(wFd), Aw = (wFu, wFd)/norm(wFu)^2
    and norm(.) = sqrt((., .)), (.,.) is inner product. Notice Aw/Nw = cc0/norm(wFu)^2

    If the recorded and modelled waveforms are close enough (cc0 close to 1), then
    the last three terms in Hw can be simplified to one term (wFd ~ Aw * wFu):  

        + Nw^-1 * Aw * norm(wFu)^-2 * (wFu, wFdu1) * (wFu, wFdu2)

    For the Hessian-model product H(., dm) let du2 = Du(m; dm) = u(m+dm) - u(m)
    then the adjoint source is (simplified)

    r = - Nw^-1 * Aw * wFdu2 + Nw^-1 * Aw * norm(wFu)^-2 * (wFu, wFdu2) * wFu

    adj_w = conj(F)(transpose(w) * r)

    The total hessian is sum(weight * Hw) and the total adjoint source is sum(weight * adj_w)

    """
    syn_orientation_codes = ['E', 'N', 'Z']
    event = self.data['event']

    #------ loop each station
    station_dict = self.data['station']
    for station_id in station_dict:
      station = station_dict[station_id]
      # skip rejected statations
      if station['stat']['code'] < 1:
        continue

      # waveform
      waveform = station['waveform']
      time_sample = waveform['time_sample']
      syn_starttime = time_sample['starttime']
      syn_delta = time_sample['delta']
      syn_nt = time_sample['nt']
      syn_nl = time_sample['nl']
      syn_nr = time_sample['nr']

      syn = waveform['syn']

      du = station['waveform_der'][model_name]['du']

      #------ loop each window
      adj = np.zeros((3, syn_nt))

      window_dict = station['window']
      for window_id in window_dict:
        # window parameters
        window = window_dict[window_id]
        # skip bad windows
        if window['stat']['code'] < 1:
          warnings.warn("Window %s not measured for adj, SKIP" % window_id)
          continue
        if window['weight'] < 1.0e-3:
          continue

        #------ window parameters 
        # filter
        filter_dict = window['filter']
        filter_a = filter_dict['a']
        filter_b = filter_dict['b']
        # taper
        win_func = window['taper']['win']
        # polarity projection 
        proj_matrix = window['polarity']['proj_matrix']
        # CC0
        cc0 = window['cc']['CC0']

        #------ make adjoint source
        # r = - Nw^-1 * Aw * wFdu + Nw^-1 * Aw * norm(wFu)^-2 * (wFu, wFdu) * wFu
        #   = - cc0/norm(wFu)^2 * (wFdu - norm(wFu)^-2*(wFu, wFdu)*wFu)
        # adj_w = conj(F)(transpose(w) * r)
        Fu = signal.filtfilt(filter_b, filter_a, syn)
        wFu = np.dot(proj_matrix, Fu) * win_func
        norm_wFu = np.sqrt(np.sum(wFu**2))

        Fdu = signal.filtfilt(filter_b, filter_a, du)
        wFdu = np.dot(proj_matrix, Fdu) * win_func

        adj_w = -cc0/norm_wFu**2 * (wFdu - wFu*np.sum(wFu*wFdu)/norm_wFu**2)
        adj_w = np.dot(np.transpose(proj_matrix), adj_w) * win_func
        adj_w = signal.filtfilt(filter_b, filter_a, adj_w[:,::-1])
        adj_w = adj_w[:,::-1]

        #------ add into total adjoint source
        adj += window['weight'] * adj_w

      #endfor window_id in window_dict:

      #------ output adjoint source
      # without padding
      npts = syn_nt - syn_nl - syn_nr
      starttime = syn_starttime + syn_nl*syn_delta
      # time samples for ascii output, referred to origin time
      syn_times = np.arange(npts)*syn_delta
      b = starttime - event['t0']
      syn_times += b

      # loop ENZ
      tr = Trace()
      for i in range(3):
        tr.data = adj[i, syn_nl:(syn_nl+npts)]
        tr.stats.starttime = starttime
        tr.stats.delta = syn_delta

        out_file = '{:s}/{:s}.{:2s}{:1s}'.format(
            out_dir, station_id, syn_band_code,
            syn_orientation_codes[i])

        # sac format
        tr.write(out_file + '.adj.sac', 'sac')

        # ascii format (needed by SEM)
        # time is relative to event origin time: t0
        with open(out_file+'.adj','w') as fp:
          for j in range(npts):
            fp.write("{:16.9e}  {:16.9e}\n".format(
              syn_times[j], adj[i,syn_nl+j]))

    #endfor station_id in station_dict:

#END class misfit
