#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime as dtm
import matplotlib.pyplot as plt
import matplotlib.dates as mdt
from matplotlib.widgets import SpanSelector
import sys
import warnings
warnings.filterwarnings('ignore')
from viresclient import SwarmRequest
from .utils import *
from .fac import *
from .j1sat import j1sat
from SwarmFACE.plot_save.single_sat_MVA import *

def get_data_mva1sat(dtime_beg, dtime_end, sat, use_filter=True):
    '''
    Prepare and plot data on the screen to interactively select the
    MVA interval

    Parameters
    ----------
    dtime_beg : str
        start time in ISO format 'YYYY-MM-DDThh:mm:ss'
    dtime_end : str
        end time in ISO format
    sat : [str]
        satellite, e.g. ['A']
    use_filter : boolean
        'True' for data filtering

    Returns
    -------
    j_df : DataFrame
        FAC density from single-satellite method
    input_df : DataFrame
        input data (includes magnetic field perturbation)
    param : dict
        parameters used in the analysis
    span_sel : dict
        new MVA start and stop time
    span : reference
        reference to SpanSelector to prevent it from being garbage collected
    '''

    Bmodel="CHAOS-all='CHAOS-Core'+'CHAOS-Static'+'CHAOS-MMA-Primary'+'CHAOS-MMA-Secondary'"
    request = SwarmRequest()    
    # identify the right half-orbit interval     
    orb = request.get_orbit_number(sat[0], dtime_beg, mission='Swarm')
    orb_beg, orb_end = request.get_times_for_orbits(orb, orb, mission='Swarm', 
                                                    spacecraft=sat[0])
    half_torb = (orb_end - orb_beg)/2.
    dtm_beg = dtm.datetime.fromisoformat(dtime_beg)
    dtm_end = dtm.datetime.fromisoformat(dtime_end)
    if dtm_beg - orb_beg < half_torb:
        large_beg, large_end = orb_beg, orb_beg + half_torb
    else:
        large_beg, large_end = orb_beg + half_torb, orb_end    
    if dtm_end > large_end:
        print('*****************************************************')        
        print('*** The interval does not fit within a half-orbit ***')
        print('*****************************************************')         
        sys.exit()
    form = '%Y-%m-%dT%H:%M:%S'
    large_beg, large_end = large_beg.strftime(form), large_end.strftime(form)
    
    # download the Swarm data
    j_df, input_df, param_j = j1sat(large_beg, large_end, sat, use_filter=True,
                                savedata=False, saveplot=False)
  
    dB_df = input_df[['dB_xgeo', 'dB_ygeo', 'dB_zgeo']]  

    def onselect(xmin, xmax):
        nrl = len(ax2.lines)
        ax2.lines[nrl-1].remove()
        ax2.lines[nrl-2].remove()
        span_sel['beg'], span_sel['end'] = xmin, xmax
        ax2.axvline(xmin, ls='--', c='r')
        ax2.axvline(xmax, ls='--', c='r')
        fig.show()       
        
    span_sel={'beg': None, 'end': None}
    tmarg = dtm.timedelta(seconds=45)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 3), sharex='all')
    ax1.plot(dB_df)
    ax1.set_xlim(xmin = dtm_beg - tmarg, xmax = dtm_end + tmarg)
    ax1.axvline(dtm_beg, ls='--', c='k')
    ax1.axvline(dtm_end, ls='--', c='k')    
    ax1.axhline(0, ls='--', c='k')
    ax1.set_ylabel('$dB_{GEO}$ sw'+sat[0], linespacing=1.3)    
    if use_filter:
        ax2.plot(j_df['FAC_flt'])
    else:
        ax2.plot(j_df['FAC'])
    ax2.axhline(0, ls='--', c='k')
    ax2.axvline(dtm_beg, ls='--', c='k')
    ax2.axvline(dtm_end, ls='--', c='k') 
    ax2.set_ylabel(r'$J_{FAC}$', linespacing=1.3)    
    ax2.xaxis.set_major_formatter(mdt.DateFormatter('%H:%M:%S'))    
    span = SpanSelector(ax1, onselect, 'horizontal', useblit=True, interactive=True,
                span_stays=True, rectprops=dict(alpha=0.2, facecolor='red'))
    fig.show()
    
    param = {'dtime_beg':dtime_beg,'dtime_end':dtime_end,'sat': sat, \
             'tmarg':tmarg,'use_filter':use_filter, 'Bmodel':Bmodel, \
             'timebads':param_j['timebads']}

    return j_df, input_df, param, span_sel, span
 
def perform_mva1sat(j_df, input_df, param, span_sel, savedata=True,
                    saveplot=True):
    '''
    Perform the MVA analysis on the data provided by get_data_mva1sat

    Parameters
    ----------
    j_df : DataFrame
        FAC density from single-satellite method
    input_df : DataFrame
        input data (includes magnetic field perturbation)
    param : dict
        parameters used in the analysis
    span_sel : dict
        new MVA start and stop time
    savedata : boolean
        'True' for saving the results in an ASCII file
    saveplot : boolean
        'True' for plotting the results

    Returns
    -------
    jcorr_df : DataFrame
        corrected (for current sheet inclination) FAC density
    dBmva_df : DataFrame
        magnetic field perturbation in MVA frame
    mva_df : DataFrame
        MVA results, FAC planarity and inclination
    param : dict
        parameters used in the analysis
    '''

    dtime_beg = param['dtime_beg']
    dtime_end = param['dtime_end'] 
    sat = param['sat']
    use_filter = param['use_filter']
    
    interval = np.array([dtime_beg, dtime_end],dtype='datetime64')
    if any([val is not None for val in span_sel.values()]):
        interval = np.array(mdt.num2date(np.array([span_sel['beg'], \
                                      span_sel['end']])),dtype='datetime64')
    tmva_int = [pd.Timestamp(interval[0]).ceil(freq = 's'), 
                pd.Timestamp(interval[1]).floor(freq = 's')]

    ti = input_df.index
    Rsph = input_df[['Latitude','Longitude','Radius']].values
    Bnec = np.stack(input_df['B_NEC'].values, axis=0)
    dB = input_df[['dB_xgeo', 'dB_ygeo', 'dB_zgeo']].values
    # transforms vectors in Gepgraphyc cartesian
    R, MATnec2geo = R_in_GEOC(Rsph)   
    B = np.matmul(MATnec2geo,Bnec[...,None]).reshape(Bnec.shape)    
    Rmid = j_df[['Rmid_x', 'Rmid_y', 'Rmid_z']].values
    V3d = R[1:,:] - R[:-1,:]    
    eV3d, eRmid = normvec(V3d), normvec(Rmid)
    eV2d = normvec(np.cross(eRmid, np.cross(eV3d, eRmid))) 

    # select quantities for MVA interval and remove NaN points
    indok = np.where((ti >= tmva_int[0]) & (ti <= tmva_int[1]) & ~np.isnan(dB[:,0]))[0]
    dB_int, B_int = dB[indok,:], B[indok,:]
    B_ave = np.average(B_int, axis = 0)
    B_unit = B_ave / np.linalg.norm(B_ave)
    # apply constrained MVA
    eigval,eigvec = mva(dB_int, cdir=B_unit)
    # select the minvar orientation according to sat. velocity
    eV3d_ave = np.average(eV3d[indok[:-1]], axis = 0)  
    mindir = eigvec[:,1] 
    if np.sum(mindir*eV3d_ave) < 0:
        mindir = -eigvec[:,1]
    maxdir = np.cross(B_unit, mindir)

    # transform magnetic perturbation in MVA frame
    geo2mva = np.stack((B_unit, mindir, maxdir), axis=1)
    dBmva_df= pd.DataFrame(np.matmul(dB, geo2mva), 
                            columns=['dB B', 'dB min', 'dB max'], index=ti)
    
    # compute the FAC inclination wrt sat. velocity in the tangential plane
    eN2d, ang = eV2d.copy(), np.zeros(len(j_df))
    eN2d[indok[:-1]] = \
        normvec(np.cross(eRmid[indok[:-1]], np.cross(mindir, eRmid[indok[:-1]])))
    
    cross_v_n = np.cross(eV2d[indok[:-1]], eN2d[indok[:-1]])
    sign_ang = np.sign(np.sum(eRmid[indok[:-1]]*cross_v_n, axis=-1))
    ang[indok[:-1]]  = \
            np.degrees(np.arcsin(sign_ang*np.linalg.norm(cross_v_n, axis=-1)))
    ang[0:indok[0]] = ang[indok[0]]
    ang[indok[-1]:] = ang[indok[-2]]
    ang_ave = np.round(np.mean(ang[indok[:-1]]), 1)
    # DataFrames with FAC inclination and density corrected for inclination
    ang_df = pd.DataFrame(ang, columns=['ang_v_n'], index=j_df.index)
    jcorr_df = singleJfac(ti, R, B, dB, alpha=ang, use_filter = True)
    
    col_mva = ['sat', 'TbegMVA', 'TendMVA', 'lmin', 'mindir', 'lmax', 'maxdir', 
               'B_unit', 'angVN']
    mva_df = pd.DataFrame([[sat[0], tmva_int[0], tmva_int[1], round(eigval[1],1), \
                np.round(mindir, 4), round(eigval[2],1), np.round(maxdir, 4),\
                np.round(maxdir, 4), round(ang_ave, 1)]], columns = col_mva)    
    
    
    print('MVA interval: ', tmva_int[0], ' - ', tmva_int[1])  
    print('mean FAC inclination in the tangential plane: ', ang_ave, ' deg.')
    print('B_unit= %10.2f'%eigval[0], np.round(B_unit, decimals=4))
    print('mindir= %10.2f'%eigval[1], np.round(mindir, decimals=4))
    print('maxdir= %10.2f'%eigval[2], np.round(maxdir, decimals=4))    

    if savedata:
        save_mva1sat(jcorr_df, dBmva_df, mva_df, param)
    if saveplot:
        plot_mva1sat(j_df, input_df, jcorr_df, dBmva_df, mva_df, param)
    
    return jcorr_df, dBmva_df, mva_df, param
