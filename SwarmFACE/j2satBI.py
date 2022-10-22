#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from viresclient import SwarmRequest
from .fac import *
from .utils import *
from SwarmFACE.plot_save.dual_sat_BI import *

def j2satBI(dtime_beg, dtime_end, sats, tshift=None, dt_along = 5,
            use_filter=True, er_db=0.5, angTHR = 30., errTHR=0.1,
            saveconf = False, savedata=True, saveplot=True):
    '''
    High-level routine to estimate the FAC density with dual-satellite 
    Boundary Integral method

    Parameters
    ----------
    dtime_beg : str
        start time in ISO format 'YYYY-MM-DDThh:mm:ss'
    dtime_end : str
        end time in ISO format
    sats : [str, str]
        satellite pair, e.g. ['A', 'C']
    tshift : [float, float]
        array of time shifts (in seconds) to be introduced in satellite data
        in order to achieve the desired quad configuration
    dt_along : int
        quad’s length in the along track direction (in seconds of
        satellite travel distance)
    use_filter : boolean
        'True' for data filtering
    er_db : float
        error in magnetic field measurements
    angTHR : float
        minimum accepted angle between the magnetic field vector and
        the quad's plane
    errTHR : float
        accepted error for the current density along the normal direction
    saveconf : boolean
        'True' for adding the quad's parameters in the results 
    savedata : boolean
        'True' for saving the results in an ASCII file
    saveplot : boolean
        'True' for plotting the results

    Returns
    -------
    j_df : DataFrame
        results
    input_df : DataFrame
        input data
    param : dict
        parameters used in the analysis
    '''

    Bmodel="CHAOS-all='CHAOS-Core'+'CHAOS-Static'+'CHAOS-MMA-Primary'+'CHAOS-MMA-Secondary'"    
    dti = pd.date_range(start = dtime_beg, end = dtime_end, freq='s', closed='left')
    ndti = len(dti)
    nsc = len(sats)
    timebads={'sc0':None,'sc1':None}
    
    Rsph, QDref, Bsw, Bmod, dBnec, R, B, dB = (np.full((ndti,nsc,3),np.nan) for i in range(8))
    
    request = SwarmRequest()
    for sc in range(nsc):
        request.set_collection("SW_OPER_MAG"+sats[sc]+"_LR_1B")
        request.set_products(measurements=["B_NEC"], models=[Bmodel],
                             auxiliaries=['QDLat','QDLon','MLT'],
                             sampling_step="PT1S")
        data = request.get_between(start_time = dtime_beg, end_time = dtime_end,
                                   asynchronous=True)   
        print('Used MAG L1B file: ', data.sources[1])
        dat_df = data.as_dataframe()
        # checks for missing and bad data points
        # sets bad B_NEC data (zero magnitude in L1b LR files) to NaN. 
        if len(dat_df) != ndti:
            print('MISSING DATA FOR Sw'+sats[sc])
            sys.exit()
        ind_badsi = np.where(np.linalg.norm(np.stack(dat_df['B_NEC'].values), \
                                           axis = 1)==0)[0]
        if len(ind_badsi):
            dat_df, tbadsi = GapsAsNaN(dat_df, ind_badsi)
            print('NR. OF BAD DATA POINTS FOR Sw'+sats[sc]+': ', len(ind_badsi))
            timebads['sc'+str(sc)] = tbadsi
            print(tbadsi.values)
            print('DATA FILTERING MIGHT NOT WORK PROPERLY')
        
        # Stores position, magnetic field and magnetic model vectors in arrays
        # Takes care of the optional time-shifts introduced between the sensors
        Rsph[:,sc,:] = dat_df[['Latitude','Longitude','Radius']].values
        QDref[:,sc,:] = dat_df[['QDLat','QDLon','MLT']].values
        Bsw[:,sc,:] = np.stack(dat_df['B_NEC'].values, axis=0)
        Bmod[:,sc,:] = np.stack(dat_df['B_NEC_CHAOS-all'].values, axis=0)  
        # Computes magnetic field perturbation in NEC
        dBnec[:,sc,:] = Bsw[:,sc,:] - Bmod[:,sc,:]    
        # Computes sats positions (R), magnetic measurements (B), and magnetic  
        # perturbations (dB) in the global geographic (Cartesian) frame. 
        R[:,sc,:], MATnec2geo_sc = R_in_GEOC(np.squeeze(Rsph[:,sc,:]))  
        B[:,sc,:] = np.matmul(MATnec2geo_sc,np.squeeze(Bsw[:,sc,:])[...,None]).reshape(-1,3)
        dB[:,sc,:] = np.matmul(MATnec2geo_sc,np.squeeze(dBnec[:,sc,:])[...,None]).reshape(-1,3)
    
    if tshift is None:
        tshift = find_tshift2sat(dti, Rsph, sats)
        
    ts = np.around(np.array(tshift) - min(tshift))
    ndt = ndti - max(ts)
    tsh2s = np.around(max(ts), decimals=3)    # keep compatibility with L2 product
    dt = dti[:ndt].shift(1000.*tsh2s,freq='ms')  # new data timeline

    # shifts data and keep only relevant points    
    list_ar = [Rsph, QDref, Bsw, Bmod, dBnec, R, B, dB]
    for ii in np.arange(len(list_ar)):
        for sc in range(nsc):
            list_ar[ii][0:ndt,sc,:] = list_ar[ii][ts[sc]:ndt + ts[sc] ,sc,:]
        list_ar[ii] = np.delete(list_ar[ii], np.s_[ndt:],0)
    Rsph, QDref, Bsw, Bmod, dBnec, R, B, dB = [list_ar[ii] for ii in np.arange(8)]

    #  collects all input data in a single DataFrame
    colRsph = pd.MultiIndex.from_product([['Rsph'],sats,['Lat','Lon','Radius']], 
                                       names=['Var','Sat','Com'])
    dfRsph = pd.DataFrame(Rsph.reshape(-1,nsc*3),columns=colRsph,index=dt)

    colQDref = pd.MultiIndex.from_product([['QDref'],sats,['QDLat','QDLon','MLT']], 
                                       names=['Vac','Sat','Com'])
    dfQDref = pd.DataFrame(QDref.reshape(-1,nsc*3),columns=colQDref,index=dt)
    
    colBswBmod = pd.MultiIndex.from_product([['Bsw','Bmod'],sats,['N','E','C']], 
                                       names=['Var','Sat','Com'])
    dfBswBmod = pd.DataFrame(np.concatenate((Bsw.reshape(-1,nsc*3), 
                    Bmod.reshape(-1,nsc*3)),axis=1), columns=colBswBmod,index=dt)

    coldBgeo = pd.MultiIndex.from_product([['dBgeo'],sats,['X','Y','Z']], 
                                       names=['Var','Sat','Com'])
    dfdBgeo = pd.DataFrame(dB.reshape(-1,nsc*3),columns=coldBgeo,index=dt)

    colRgeo = pd.MultiIndex.from_product([['Rgeo'],sats,['X','Y','Z']], 
                                       names=['Var','Sat','Com'])
    dfRgeo = pd.DataFrame(R.reshape(-1,nsc*3),columns=colRgeo,index=dt)

    input_df = pd.concat([dfRsph, dfQDref, dfBswBmod, dfdBgeo, dfRgeo], axis=1)  
    
    j_df = bi_dualJfac(dt, R, B, dB, dt_along=dt_along, er_db=er_db, angTHR=angTHR, 
                       errTHR=errTHR, use_filter=use_filter, saveconf=saveconf)

    param = {'dtime_beg':dtime_beg,'dtime_end':dtime_end,'sats': sats, \
             'tshift':ts, 'dt_along':dt_along, 'angTHR': angTHR, 'errTHR':errTHR,
             'Bmodel':Bmodel, 'use_filter':use_filter, 'timebads':timebads}

    if savedata:
        save_dual_sat_BI(j_df, param)
    if saveplot:
        plot_dual_sat_BI(j_df, input_df, param)   
    
    return j_df, input_df, param
     
