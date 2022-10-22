#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import signal
from viresclient import SwarmRequest
from .utils import *
from SwarmFACE.plot_save.three_sat_conj import *

def find_3sat_conj(dtime_beg, dtime_end, delT = 120, delN=300, delE=1200, 
             jTHR=0.05, saveplot=True):
    '''
    High-level routine to find Swarm conjunctions above the auroral oval

    Parameters
    ----------
    dtime_beg : str
        start time in ISO format 'YYYY-MM-DDThh:mm:ss'
    dtime_end : str
        end time in ISO format
    delT : float
        time window (in sec.) to find conjunctions
    delN : float
        spatial range along North-South (in km) to find conjunctions
    delE : float
        spatial range along East-West (in km) to find conjunctions
    jTHR : float
        threshold to neglect small FAC densities
    saveplot : boolean
        'True' for plotting the results

    Returns
    -------
    conj_df : DataFrame
        details on the identified conjunctions
    param : dict
        parameters used in the analysis
    '''

    Re = 6800           # approx. value for Swarm orbit radius
    fs = 1              # Sampling frequency
    fc = 1./20.         # Cut-off frequency of a 20 s low-pass filter
    w = fc / (fs / 2)   # Normalize the frequency
    butter_ord = 5
    bf, af = signal.butter(butter_ord, w, 'low')
    
    request = SwarmRequest()
    sats=['A','B','C']
    nsc = len(sats)

    # Retieses the L2 FAC data from the ESA database 
    orbs = np.full((nsc,2),np.nan)
    fac_df, tlarges = ([] for i in range(2))
    for sc in range(nsc):
#        find the orbits that contain the analysis interval
        orb1 = request.get_orbit_number(sats[sc], dtime_beg, mission='Swarm')
        orb2 = request.get_orbit_number(sats[sc], dtime_end, mission='Swarm')
        print('sat: sw'+sats[sc]+', orbit start/end: '+str(orb1)+' / '+ 
              str(orb2)+ ',   nr. orbits: '+ str(orb2 - orb1 +1))
        orbs[sc, :] = [orb1, orb2]              
        large_beg, large_end = request.get_times_for_orbits(orb1, orb2, 
                                         mission='Swarm', spacecraft=sats[sc])
        tlarges.append([large_beg, large_end])
        dti = pd.date_range(start= large_beg, end= large_end, freq='s', closed='left')
        # get L2 FAC data for Northern hemisphere
        request.set_collection('SW_OPER_FAC'+sats[sc]+'TMS_2F')
        request.set_products(measurements=["FAC"], 
                             auxiliaries=['QDLat','QDLon','MLT'],
                             sampling_step="PT1S")
        request.set_range_filter('QDLat', 45, 90)
        data = request.get_between(start_time = large_beg, 
                                   end_time = large_end,
                                   asynchronous=True)  
        print('Used L2 FAC file: ', data.sources[0])
        datN_fac = data.as_dataframe()
        request.clear_range_filter()   
        # get L2 FAC data for Southern hemisphere
        request.set_range_filter('QDLat', -90, -45)
        data = request.get_between(start_time = large_beg, 
                                   end_time = large_end,
                                   asynchronous=True)   
        print('Used L2 FAC file: ', data.sources[0])
        datS_fac= data.as_dataframe()    
        request.clear_range_filter()
        # put toghether data from both hemispheres
        dat = pd.concat([datN_fac, datS_fac]).sort_index()  
        dat['FAC_flt'] = signal.filtfilt(bf, af, dat['FAC'].values)
        dat['FAC_flt_sup'] = np.where(np.abs(dat['FAC_flt']) >= jTHR, dat['FAC_flt'], 0)
        # append data from different satellites to fac_df list
        fac_df.append(dat)  

    # For all sats, splits L2 FAC data in quarter-orbit sections. In each
    # section computes the time, QDLat and QDLon at AO central point, as well as
    # the trend and sign of QDLat. Stores these parameters in a DataFrame, 
    # together with start time, stop time, and orbit number + orbit fraction 
    qpar_df = []
    col_qpar = ['aoc_time', 'aoc_qdlat', 'aoc_qdlon', 'qdlat_trend', 
                'qdlat_sign', 'beg_time', 'end_time','orbit']
    for sc in range(nsc):
        # nr. of 1/2 orbits and 1/2 orbit duration 
        nrho = int((orbs[sc,1] - orbs[sc,0] + 1)*2)    
        dtho = (tlarges[sc][1] - tlarges[sc][0])/nrho  
        # start and stop of 1/2 orbits
        begend_hor = [tlarges[sc][0] + ii*dtho for ii in range(nrho +1)]
        # splits  in 1/2 orbit sections; get time of extreme QDLat for each
        horbs = split_into_sections(fac_df[sc], begend_hor)
        times_maxQDLat = [horbs[ii]['QDLat'].abs().idxmax().to_pydatetime() \
                        for ii in range(nrho)]
        begend_qor = sorted(times_maxQDLat + begend_hor)
        # splits FAC data in 1/4 orbits sections
        qorbs = split_into_sections(fac_df[sc], begend_qor)
        # for each 1/4 section, computes the time, QDLat and QDLon at AO central 
        # point, as well as the trend and sign of QDLat
        qparsc_df = pd.DataFrame(columns = col_qpar)  
        for jj in range(len(qorbs)):
            qpar_jj = list(find_ao_margins(qorbs[jj])[2:7])
            qpar_jj.extend([begend_qor[jj], begend_qor[jj+1], orbs[sc, 0]+jj/4.])
            if  pd.notna(qpar_jj[0]):
                qparsc_df.loc[len(qparsc_df)] = qpar_jj
        qpar_df.append(qparsc_df) 

    # finds the conjunctions and stores their parameters in a DataFrame
    col_names = ['TimeB','delTswAB', 'delTswCB', 'delNswAB', 'delNswCB', 'delEswAB', 
                 'delEswCB', 'QDLatA', 'QDLatB', 'QDLatC','QDLonA', 'QDLonB', 'QDLonC',
                 'orbitA', 'orbitB', 'orbitC', 'TimeA', 'TimeC']
    conj_df = pd.DataFrame(columns = col_names)   

    # AO central times, QDLat and QDLon
    timesA, timesB, timesC = (qpar_df[sc]['aoc_time'] for sc in range(nsc))    
    qdlatA, qdlatB, qdlatC = (qpar_df[sc]['aoc_qdlat'] for sc in range(nsc))
    qdlonA, qdlonB, qdlonC = (qpar_df[sc]['aoc_qdlon'] for sc in range(nsc))
    orbitA, orbitB, orbitC = (qpar_df[sc]['orbit'] for sc in range(nsc))
    
    for indB in range(len(timesB)):
        indA = (np.abs(timesA - timesB[indB])).argmin()
        indC = (np.abs(timesC - timesB[indB])).argmin()
        if (np.abs(timesA[indA] - timesB[indB]).total_seconds() <= delT) or \
           (np.abs(timesC[indC] - timesB[indB]).total_seconds() <= delT):
               dqlatAB = np.abs(qdlatA[indA] - qdlatB[indB])
               dqlatCB = np.abs(qdlatC[indC] - qdlatB[indB])                       
               if (dqlatAB*Re*np.pi/180 <= delN) or (dqlatCB*Re*np.pi/180 <= delN):
                   dqlonAB = np.abs(sign_dang(qdlonA[indA], qdlonB[indB]))
                   dqlonCB = np.abs(sign_dang(qdlonC[indC], qdlonB[indB]))                   
                   mqlatAB = np.abs(qdlatA[indA] + qdlatB[indB])/2
                   mqlatCB = np.abs(qdlatC[indC] + qdlatB[indB])/2
                   if (dqlonAB*Re*np.pi/180.*np.cos(mqlatAB) <= delE) or \
                      (dqlonCB*Re*np.pi/180.*np.cos(mqlatCB) <= delE):
                          conj_i = pd.DataFrame([[timesB[indB], \
                                int((timesA[indA] - timesB[indB]).total_seconds()),
                                int((timesC[indC] - timesB[indB]).total_seconds()),
                                int(dqlatAB*Re*np.pi/180),int(dqlatCB*Re*np.pi/180),
                                int(np.abs(dqlonAB*Re*np.pi/180.*np.cos(mqlatAB))), 
                                int(np.abs(dqlonCB*Re*np.pi/180.*np.cos(mqlatCB))),
                                round(qdlatA[indA],2), round(qdlatB[indB],2), 
                                round(qdlatC[indC],2), round(qdlonA[indA],2),
                                round(qdlonB[indB],2), round(qdlonC[indC],2),
                                orbitA[indA], orbitB[indB], orbitC[indC], 
                                timesA[indA], timesC[indC]]],columns=col_names) 
                          conj_df= conj_df.append(conj_i, ignore_index = True)

    param = {'dtime_beg':dtime_beg,'dtime_end':dtime_end,
             'delT': delT, 'delN': delN, 'delE': delE}

    save_three_sat_conj(conj_df, param)
    
    if saveplot:
        if len(conj_df) == 0:
            print('*******************************************************')
            print('*** No conjunctions found in the specified interval ***')
            print('*******************************************************')            
        else:
            plot_three_sat_conj(conj_df, fac_df, qpar_df, param)    
        
    return conj_df, param
