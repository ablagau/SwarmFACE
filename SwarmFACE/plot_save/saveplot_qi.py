#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.dates as mdt
from viresclient import set_token
from viresclient import SwarmRequest
import numpy as np
import pandas as pd
from SwarmFACE.utils import *
import datetime as dtm
import warnings
warnings.filterwarnings('ignore')

def save_qi(qimva_df, qicc_df, param):
    '''
    Save to ASCII file the quality indices. Input from 
    fac_qi.py
    '''          

    dtime_beg = param['dtime_beg']
    dtime_end = param['dtime_end']    
    sats = param['sats']

    str_trange = dtime_beg.replace('-','')[:8]+'_'+ dtime_beg.replace(':','')[11:15] \
        + '_'+ dtime_end.replace('-','')[:8] + '_'+ dtime_end.replace(':','')[11:15]

    # Writes the MVA related quality indices
    fname_mva = 'QI_MVA_'+ str_trange + '.dat'

    a_df = qimva_df[sats.index('A')].copy(deep=True)

    c_df = qimva_df[sats.index('C')].copy(deep=True)
    out_qimva_df = pd.concat([a_df, c_df]).sort_index(kind='mergesort')
    if len(sats) > 2:
            b_df = qimva_df[sats.index('B')].copy(deep=True)
            out_qimva_df = pd.concat([out_qimva_df,b_df])
    flt_cols = ['lmin', 'lmax', 'lrat', 'angVN']
    for cols in flt_cols:
        out_qimva_df[cols] = out_qimva_df[cols].map('{:9.1f}'.format)  
    n_cols = ['nx', 'ny', 'nz']
    for cols in n_cols:
        out_qimva_df[cols] = out_qimva_df[cols].map('{:10.3f}'.format)                             
    out_qimva_df['orbit'] = out_qimva_df['orbit'].map('{:10.2f}'.format)     
    with open(fname_mva, 'w') as file:
        file.write('# Time range for computing the MVA related quality indices: ' + dtime_beg +\
                   ' - '+ dtime_end + '\n')
        file.write('# '+ (',  '.join(out_qimva_df.columns)) + '\n')          
    out_qimva_df.to_csv(fname_mva, mode='a', sep=",", date_format='  %Y-%m-%d %H:%M:%S', header=False, index=False) 

    # Writes the corellation coefficients
    fname_cc = 'QI_CC_swAC_'+ str_trange + '.dat'
    out_qicc_df = qicc_df.copy(deep=True)
    for orb_cols in ['orbit_swA', 'orbit_swC']:
        out_qicc_df[orb_cols] = out_qicc_df[orb_cols].map('{:10.2f}'.format)                             
    out_qicc_df['cc'] = out_qicc_df['cc'].map('{:8.4f}'.format)  
    out_qicc_df['opt_lag'] = out_qicc_df['opt_lag'].map('{:6.0f}'.format)  
    with open(fname_cc, 'w') as file:
        file.write('# Time range for computing the correlation coefficients: ' + dtime_beg +\
                   ' - '+ dtime_end + '\n')
        file.write('# '+ (',  '.join(out_qicc_df.columns)) + '\n')          
    out_qicc_df.to_csv(fname_cc, mode='a', sep=",", date_format='  %Y-%m-%d %H:%M:%S', 
                     header=False, index=False) 

        
def plot_qi(qorbs_Bnec, qorbs_dB, qorbs_fac, qorbs_dBmva, qimva_df, Bcc_df, qicc_df, param):
    '''
    Plot the results from quality indices analysis. Input from 
    fac_qi.py
    '''          
    
    dtime_beg = param['dtime_beg']
    dtime_end = param['dtime_end']    
    sats = param['sats']

    str_trange = dtime_beg.replace('-','')[:8]+'_'+ dtime_beg.replace(':','')[11:15] \
        + '_'+ dtime_end.replace('-','')[:8] + '_'+ dtime_end.replace(':','')[11:15]

    iok = np.where(pd.notna(qicc_df['Trefbeg']))[0]
    for kk in range(len(iok)):
        jj = iok[kk]
        indA, indC = sats.index('A'), sats.index('C')
        tq_beg = min([qorbs_dB[indA][jj].index[0], qorbs_dB[indC][jj].index[0]])
        tq_end = max([qorbs_dB[indA][jj].index[-1], qorbs_dB[indC][jj].index[-1]])
        iref = sats.index(qicc_df['refsc'][jj])
        isec = indC if iref == indA else indA
 
        # quarter orbit time and data for the second and reference s/c
        tsec = qorbs_dBmva[isec][jj].index
        dBsec = qorbs_dBmva[isec][jj]['dB_max'].values
        
        tref = qorbs_dBmva[iref][jj].index
        dBref = qorbs_dBmva[iref][jj]['dB_max'].values

        str_trange = tq_beg.isoformat().replace('-','')[:8]+'_'+ \
             tq_beg.isoformat().replace(':','')[11:17] + \
                '_'+tq_end.isoformat().replace(':','')[11:17]
        orb_swA = int(np.floor(qicc_df['orbit_swA'][jj]))
        orb_swC = int(np.floor(qicc_df['orbit_swC'][jj]))
        qdnt_swA = int((qicc_df['orbit_swA'][jj] - orb_swA)*4)
        
        fname_fig = 'QI_swAC_'+ str_trange +'.eps'
        
        fig_size = (8.27,11.69)
        fig = plt.figure(figsize=fig_size, frameon=True)
        fig.suptitle('Quality indices for swA/swC orbit '+ \
                     str(orb_swA)+ '/'+str(orb_swC) + '  quadrant ' +str(qdnt_swA)+ '\n', 
                     size='xx-large', y=0.995)    
     
        # MVA and cc results
        text_int = 'Time interval: ' + tq_beg.isoformat()[:10]+'  ' +\
                      tq_beg.isoformat()[11:19] + ' - '+tq_end.isoformat()[11:19]
        
        text_cc = 'Correlation analysis (ref. sat.,  coeff,  time lag [s]):    sw' + \
                    sats[iref] + '     ' + str(np.round(qicc_df['cc'][jj], decimals=3)) + \
                '      '+ str(qicc_df['opt_lag'][jj])         
 
        text_mva = 'MVA results (sat,  interval,  $\lambda_{max}/\lambda_{min}$,  ' +\
                    'angle VN [deg.])'
        swA_text = '      swA:   '+ qimva_df[indA]['TbegMVA'][jj].isoformat()[11:19] + \
                ' - '+qimva_df[indA]['TendMVA'][jj].isoformat()[11:19] + '      ' +\
                str(np.round(qimva_df[indA]['lrat'][jj], decimals=1)) + '     ' + \
                str(np.round(qimva_df[indA]['angVN'][jj], decimals=1))
        swC_text = '      swC:   '+ qimva_df[indC]['TbegMVA'][jj].isoformat()[11:19] + \
                ' - '+qimva_df[indC]['TendMVA'][jj].isoformat()[11:19] + '      ' +\
                str(np.round(qimva_df[indC]['lrat'][jj], decimals=1)) + '     ' + \
                str(np.round(qimva_df[indC]['angVN'][jj], decimals=1))                
        
        plt.figtext(0.25, 0.955, text_int, size='large')
        plt.figtext(0.1, 0.93, text_mva, size='large')
        plt.figtext(0.1, 0.91, swA_text, size='large')
        plt.figtext(0.1, 0.89, swC_text, size='large')
        plt.figtext(0.1, 0.865, text_cc, size='large')
        
        # PANEL PART
        # designes the panel frames 
        xle, xri, ybo, yto = 0.125, 0.92, 0.095, 0.15     # plot margins
        ratp = np.array([1, 1, 1, 1, 1, 1, 0.6 ]) # relative hight of each panel 
        hsep = 0.005                                    # vspace between panels 
        nrp = len(ratp) 
        
        hun = (1-yto - ybo - (nrp-1)*hsep)/ratp.sum() 
        yle = np.zeros(nrp)     # y left for each panel
        yri = np.zeros(nrp) 
        for ii in range(nrp): 
            yle[ii] = (1 - yto) -  ratp[: ii+1].sum()*hun - ii*hsep
            yri[ii] = yle[ii] + hun*ratp[ii]
        
        # creates axex for each panel
        ax = [0] * nrp
        for ii in range(nrp):
            ax[ii] =  fig.add_axes([xle, yle[ii], xri-xle, hun*ratp[ii]])
            ax[ii].set_xlim(tq_beg, tq_end)
            
        for ii in range(nrp -1):
            ax[ii].set_xticklabels([])
  

        ax[0].plot(qorbs_dB[indA][jj]['dBnec'])
        ax[0].legend(['dB_N', 'dB_E', 'dB_C' ], loc = (0.95, 0.1), handlelength=1)
        ax[0].set_ylabel('$dB_{NEC}$ sw'+sats[indA]+'\n[nT]', linespacing=1.7)
        ax[0].axvline(qimva_df[indA]['TbegMVA'][jj], ls=':', c='k', lw=0.7)
        ax[0].axvline(qimva_df[indC]['TendMVA'][jj], ls=':', c='k', lw=0.7)

        ax[1].plot(qorbs_dB[indC][jj]['dBnec'])
        ax[1].legend(['dB_N', 'dB_E', 'dB_C' ], loc = (0.95, 0.1), handlelength=1)
        ax[1].set_ylabel('$dB_{NEC}$ sw'+sats[indC]+'\n[nT]', linespacing=1.7)
        ax[1].get_shared_y_axes().join(ax[0], ax[1])
        ax[1].axvline(qimva_df[indC]['TbegMVA'][jj], ls=':', c='k', lw=0.7)
        ax[1].axvline(qimva_df[indC]['TendMVA'][jj], ls=':', c='k', lw=0.7)

        ax[2].plot(qorbs_dBmva[indA][jj][['dB_max','dB_min','dB_B']])
        ax[2].legend(['dB_max', 'dB_min', 'dB_B' ], loc = (0.95, 0.1), handlelength=1)
        ax[2].set_ylabel('$dB_{MVA}$ sw'+sats[indA]+'\n[nT]', linespacing=1.7)

        ax[3].plot(qorbs_dBmva[indC][jj][['dB_max','dB_min','dB_B']])
        ax[3].legend(['dB_max', 'dB_min', 'dB_B' ], loc = (0.95, 0.1), handlelength=1)
        ax[3].set_ylabel('$dB_{MVA}$ sw'+sats[indC]+'\n[nT]', linespacing=1.7)
        ax[3].get_shared_y_axes().join(ax[2], ax[3])

        ax[4].plot(tsec, dBsec)
#        qorbs_ref_cc[jj].plot(ax=ax[4])
        ax[4].plot(Bcc_df[jj])
        ax[4].legend(['dB_max_sw'+sats[isec], 'dB_max_sw'+sats[iref]], \
                                 loc = (0.9, 0.1), handlelength=1)
        ax[4].set_ylabel('$dB_{MVA}$ \n[nT]', linespacing=1.7)

        ax[5].plot(qorbs_fac[indA][jj]['FAC_flt'])
        ax[5].plot(qorbs_fac[indC][jj]['FAC_flt'])       
        ax[5].legend(['Jfac_sw'+sats[indA], 'Jfac_sw'+sats[indC]], \
                                 loc = (0.94, 0.1), handlelength=1)
        ax[5].set_ylabel('$J_{FAC}$ \n[nT]', linespacing=1.7)

        ax[6].plot(qorbs_dBmva[indA][jj]['ang_VN'])
        ax[6].plot(qorbs_dBmva[indC][jj]['ang_VN'])
        max_ang= max([qorbs_dBmva[indA][jj]['ang_VN'].max(),qorbs_dBmva[indC][jj]['ang_VN'].max()])
        min_ang= min([qorbs_dBmva[indA][jj]['ang_VN'].min(),qorbs_dBmva[indC][jj]['ang_VN'].min()])        
        off_ang = 0.2*(max_ang - min_ang)
        ax[6].set_ylim([min_ang - off_ang, max_ang + off_ang])
        ax[6].legend(['ang_sw'+sats[indA], 'ang_sw'+sats[indC]], loc = (0.95, 0.1), handlelength=1)
        ax[6].set_ylabel('ang_VN \n[deg]', linespacing=1.7)
        ax[6].axvline(qimva_df[indA]['TbegMVA'][jj], ls=':', c='b', lw=0.7)
        ax[6].axvline(qimva_df[indC]['TendMVA'][jj], ls=':', c='b', lw=0.7)        
        ax[6].axvline(qimva_df[indC]['TbegMVA'][jj], ls=':', c='tab:orange', lw=0.7)
        ax[6].axvline(qimva_df[indC]['TendMVA'][jj], ls=':', c='tab:orange', lw=0.7)        
        int_min = 2
        dif_min = tq_end.floor('min') - tq_beg.ceil('min')
        nr_ticks = dif_min/pd.to_timedelta(1, unit='m')//int_min +1
        pos_ticks = tq_beg.ceil('min') + np.arange(nr_ticks)*dtm.timedelta(minutes=int_min)
        ax[6].set_xticks(pos_ticks)
                  
         # Ephemerides
        latc = qorbs_Bnec[indC][jj]['Latitude'].values
        lonc = qorbs_Bnec[indC][jj]['Longitude'].values
        
        locx = ax[nrp-1].get_xticks()
        latc_ipl = np.round(np.interp(locx, mdt.date2num(qorbs_dBmva[indC][jj].index), \
                                    latc), decimals=2).astype('str')
        lonc_ipl = np.round(np.interp(locx, mdt.date2num(qorbs_dBmva[indC][jj].index), \
                                    lonc), decimals=2).astype('str')
        qdlat_ipl = np.round(np.interp(locx, mdt.date2num(qorbs_dBmva[indC][jj].index), \
                                    qorbs_Bnec[indC][jj]['QDLat']), decimals=2).astype('str')
        qdlon_ipl = np.round(np.interp(locx, mdt.date2num(qorbs_dBmva[indC][jj].index), \
                                    qorbs_Bnec[indC][jj]['QDLon']), decimals=2).astype('str')
        mlt_ipl = np.round(np.interp(locx, mdt.date2num(qorbs_dBmva[indC][jj].index), \
                                    qorbs_Bnec[indC][jj]['MLT']), decimals=1).astype('str')       

        lab_fin = ['']*len(locx)
        for ii in range(len(locx)):
            lab_ini = mdt.num2date(locx[ii]).strftime('%H:%M:%S')
            lab_fin[ii] = lab_ini + '\n' +latc_ipl[ii] + '\n' + lonc_ipl[ii] + \
            '\n'+ qdlat_ipl[ii] + '\n' +qdlon_ipl[ii] + '\n' + mlt_ipl[ii]
            
        ax[nrp-1].set_xticklabels(lab_fin)
        plt.figtext(0.01, 0.01, 'Time\nLat\nLon\nQLat\nQLon\nMLT')

        plt.show()
        fig.savefig(fname_fig)        
