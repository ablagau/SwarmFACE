#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.dates as mdt
import numpy as np
import pandas as pd
import datetime as dtm
import warnings
warnings.filterwarnings('ignore')


def save_mva1sat(jcorr_df, dBmva_df, mva_df, param):
    '''
    Save to ASCII file results from interactive MVA. Input from 
    perform_mva1sat.py
    '''            

    MVAline = mva_df.iloc[0] 
    tbeg = MVAline['TbegMVA'].strftime("%Y%m%d_%H%M%S")
    tend = MVAline['TendMVA'].strftime("%Y%m%d_%H%M%S")
    tmarg = dtm.timedelta(seconds=45)
    t1, t2 = MVAline['TbegMVA'] - tmarg, MVAline['TendMVA'] + tmarg
    
    str_tmva = MVAline['TbegMVA'].strftime("%Y-%m-%d   %H:%M:%S") +\
               ' - ' + MVAline['TendMVA'].strftime("%H:%M:%S")
    sat = param['sat']
    Bmodel = param['Bmodel']
    
    jcorr_df2 = jcorr_df.truncate(before=t1, after=t2)
    dBmva_df2 = dBmva_df.truncate(before=t1, after=t2)
    ang_df = jcorr_df2['incl']
    
    # Writes the MVA related quality indices
    fname_mva = 'sw'+sat[0]+'_MVA_'+ tbeg +'_'+ tend + '.dat'
    with open(fname_mva, 'w') as file:
        file.write('# Swarm sat: ' + str(sat[0]) + '\n')
        file.write('# Model: ' + Bmodel + '\n')
#        file.write('# time interval: ' + tbeg +'_'+ tend + '\n')   
        file.write('#########  MVA results  #########\n')
        file.write('# MVA interval: ' + str_tmva + '\n')    
        file.write('# B_unit:  '+ format(0,'9.1f')+'  ['+ 
                   format(MVAline['B_unit'][0], '7.3f') +', '+
                   format(MVAline['B_unit'][1], '7.3f') +', '+
                   format(MVAline['B_unit'][2], '7.3f') +']\n')
        file.write('# minvar:  '+ format(MVAline['lmin'],'9.1f')+'  ['+
                   format(MVAline['mindir'][0], '7.3f') + ', '+
                   format(MVAline['mindir'][1], '7.3f') + ', '+
                   format(MVAline['mindir'][2], '7.3f') +'] \n')  
        file.write('# maxvar:  '+ format(MVAline['lmax'],'9.1f')+'  ['+
                   format(MVAline['maxdir'][0], '7.3f') + ', '+
                   format(MVAline['maxdir'][1], '7.3f') + ', '+
                   format(MVAline['maxdir'][2], '7.3f') +'] \n') 
        file.write('# eigenvalues ratio: ' + format(MVAline['lmax']/MVAline['lmin'], '.1f') +'\n')    
        file.write('# FAC inclination wrt Vsat (tangential plane):  ' + \
            format(np.min(ang_df), '.1f') + ' -  ' + \
            format(np.max(ang_df), '.1f') + '  deg. \n') 
        file.write('################################\n') 
        file.write('# time [YYYY-mm-ddTHH:MM:SS],\tdBmva_{B, minvar, maxvar} [nT] \n')
    dBmva_df2.to_csv(fname_mva, mode='a', sep=",", date_format='%Y-%m-%d %H:%M:%S', 
                     float_format = '%11.4f', header=False) 

def plot_mva1sat(j_df, dat_df, jcorr_df, dBmva_df, mva_df, param):
    '''
    Plot results from interactive MVA. Input from 
    perform_mva1sat.py
    '''      
    
    MVAline = mva_df.iloc[0] 
    tbeg = MVAline['TbegMVA']
    tend = MVAline['TendMVA']
    tmva_int = [tbeg, tend]
    tmid = tbeg + 0.5*(tend - tbeg)
    
    tmarg = dtm.timedelta(seconds=60)
    t1, t2 = tbeg - tmarg, tend + tmarg

    j_df2 = j_df.truncate(before=t1, after=t2)
    dat_df2 = dat_df.truncate(before=t1, after=t2)
    jcorr_df2 = jcorr_df.truncate(before=t1, after=t2)
    dBmva_df2 = dBmva_df.truncate(before=t1, after=t2)
    dBmva_df3 = dBmva_df.truncate(before=tbeg, after=tend)
    
    Bnec = np.stack(dat_df2['B_NEC'].values, axis=0)
    Bmod = np.stack(dat_df2['B_NEC_CHAOS-all'].values, axis=0)
    dBnec_df = pd.DataFrame(Bnec-Bmod, columns=['Bn', 'Be', 'Bc'], index = dat_df2.index)
    ang_df = jcorr_df2['incl']
    
    mva_results = 'MVA interval:  '+ tbeg.strftime("%H:%M:%S") + ' - ' + \
                                     tend.strftime("%H:%M:%S") +'\n' + \
      r'$\mathbf{N_{GEO}}$' +':'+ '  ['+ '{:.3f},  '.format(MVAline['mindir'][0]) + \
                                         '{:.3f},  '.format(MVAline['mindir'][1]) + \
                                         '{:.3f}'.format(MVAline['mindir'][2]) + ']\n' +\
      'eigenvalues ratio: ' + '{:.2f}'.format(MVAline['lmax']/MVAline['lmin']) +'\n' + \
      'FAC inclination (positive from '+r'$\mathbf{V_{sat}}$'+ ' to '+ \
        r'$\mathbf{N_{tang}}$' + ' along '+ r'$\mathbf{R_{sat}}$' + '):   ' + \
        '{:.1f}'.format(np.min(ang_df)) + r'$^{\degree}$ ' +r'$\mathrm{\div}$   ' + \
        '{:.1f}'.format(np.max(ang_df)) + r'$^{\degree}$'  

    sat = param['sat']
    use_filter = param['use_filter']
        
    fig_size = (8.27,11.69)
    fig = plt.figure(figsize=fig_size, frameon=True)
    fname_fig = 'sw'+sat[0]+'_MVA_'+ tbeg.strftime("%Y%m%d_%H%M%S") +'_'+ \
                                     tend.strftime("%Y%m%d_%H%M%S") + '.eps'
    fig.suptitle('MVA results on Swarm '+sat[0] + '  ' +\
        tmid.strftime("%Y-%m-%d  %H:%M:%S"), \
        size='xx-large', weight = 'bold', y=0.99)
    
    plt.figtext(0.04, 0.95, mva_results, fontsize = 'x-large', \
                va='top', ha = 'left')
    
    xle, xri, ybo, yto = 0.125, 0.94, 0.45, 0.15     # plot margins
    ratp = np.array([1, 1, 1, 0.5]) #relative hight of each panel 
    hsep = 0.005                                    # vspace between panels 
    nrp = len(ratp) 
    
    hun = (1-yto - ybo - (nrp-1)*hsep)/ratp.sum() 
    yle = np.zeros(nrp)     # y left for each panel
    yri = np.zeros(nrp) 
    for ii in range(nrp): 
        yle[ii] = (1 - yto) -  ratp[: ii+1].sum()*hun - ii*hsep
        yri[ii] = yle[ii] + hun*ratp[ii]
    
    # creates axes for each panel
    ax = [0] * (nrp+1)
    for ii in range(nrp):
        ax[ii] =  fig.add_axes([xle, yle[ii], xri-xle, hun*ratp[ii]])
        ax[ii].set_xlim(pd.Timestamp(t1), pd.Timestamp(t2))
    
    for ii in range(nrp -1):
        ax[ii].set_xticklabels([])
            
    #Plots time-series quantities    
    ax[0].plot(dBnec_df)
    ax[0].set_ylabel('$dB_{GEO}$ sw'+sat[0]+'\n[nT]', linespacing=1.7)
    ax[0].axvline(tmva_int[0], ls='--', c='k')
    ax[0].axvline(tmva_int[1], ls='--', c='k')
    ax[0].legend(['dB_N', 'dB_E', 'dB_C' ], loc = (0.95, 0.1), handlelength=1)
    ax[0].axhline(0, ls='--', c='k')
    
    ax[1].plot(dBmva_df2)
    ax[1].set_ylabel('$dB_{MVA}$ sw'+sat[0]+'\n[nT]', linespacing=1.7)
    ax[1].axvline(tmva_int[0], ls='--', c='k')
    ax[1].axvline(tmva_int[1], ls='--', c='k')
    ax[1].legend(['dB_B', 'dB_min', 'dB_max' ], loc = (0.93, 0.1), handlelength=1)
    ax[1].axhline(0, ls='--', c='k')

    if use_filter:
        ax[2].plot(j_df2['FAC_flt'], label='FAC')
        ax[2].plot(jcorr_df2['FAC_flt'], label='FAC_inc')
    else:
        ax[2].plot(j_df2['FAC'], label='FAC')
        ax[2].plot(jcorr_df2['FAC'], label='FAC_inc')    
    ax[2].axhline(0, ls='--', c='k')
    ax[2].axvline(tmva_int[0], ls='--', c='k')
    ax[2].axvline(tmva_int[1], ls='--', c='k')
    ax[2].set_ylabel(r'$J_{FAC}$'+'\n'+r'$[\mu A/m^2]$', linespacing=1.7)
    ax[2].legend(loc = (0.93, 0.6), handlelength=1)     
    
    ax[3].plot(ang_df)
    ax[3].axvline(tmva_int[0], ls='--', c='k')
    ax[3].axvline(tmva_int[1], ls='--', c='k')
    ax[3].set_ylabel(r'$ang_NV$'+'\n'+r'$[deg]$', linespacing=1.7)    
    
    for ii in range(1,nrp -1):
        ax[ii].get_shared_x_axes().join(ax[0])
        
    ax[4] =  fig.add_axes([xle, 0.03, xri-xle, 0.30])
    ax[4].set_title('Hodogram of '+r'$dB_{minvar}$'+ ' vs. '+\
        r'$dB_{maxvar}$' , fontsize = 'xx-large', pad = 15)
    ax[4].plot(dBmva_df2.values[:,2], dBmva_df2.values[:,1], label='plot_range')
    ax[4].plot(dBmva_df3.values[:,2], dBmva_df3.values[:,1], label='MVA range')
    ax[4].plot(dBmva_df3.values[0,2], dBmva_df3.values[0,1], \
      label='start', color='green', marker='o', linewidth=2, markersize=8)
    ax[4].plot(dBmva_df3.values[-1,2], dBmva_df3.values[-1,1], \
      label='stop', color='red', marker='o', linewidth=2, markersize=8)
    ax[4].set_aspect('equal', adjustable='box')
    ax[4].set_xlabel(r'$dB_{maxvar}$'+'\n'+r'$[nT]$', linespacing=1.7)
    ax[4].set_ylabel(r'$dB_{minvar}$'+'\n'+r'$[nT]$', linespacing=1.7)
    ax[4].legend(loc = (0.9, 0.9), handlelength=1) 

    latc = dat_df2['Latitude'].values
    lonc = dat_df2['Longitude'].values
    
    locx = ax[nrp-1].get_xticks()
    latc_ipl = np.round(np.interp(locx, mdt.date2num(dat_df2.index), \
                                latc), decimals=2).astype('str')
    lonc_ipl = np.round(np.interp(locx, mdt.date2num(dat_df2.index), \
                                lonc), decimals=2).astype('str')
    qdlat_ipl = np.round(np.interp(locx, mdt.date2num(dat_df2.index), \
                                dat_df2['QDLat']), decimals=2).astype('str')
    qdlon_ipl = np.round(np.interp(locx, mdt.date2num(dat_df2.index), \
                                dat_df2['QDLon']), decimals=2).astype('str')
    mlt_ipl = np.round(np.interp(locx, mdt.date2num(dat_df2.index), \
                                dat_df2['MLT']), decimals=1).astype('str')
    
    lab_fin = ['']*len(locx)
    for ii in range(len(locx)):
        lab_ini = mdt.num2date(locx[ii]).strftime('%H:%M:%S')
        lab_fin[ii] = lab_ini + '\n' +latc_ipl[ii] + '\n' + lonc_ipl[ii]+ \
        '\n'+ qdlat_ipl[ii] + '\n' +qdlon_ipl[ii] + '\n' + mlt_ipl[ii]
        
    ax[nrp-1].set_xticklabels(lab_fin)
    plt.figtext(0.01, 0.366, 'Time\nLat\nLon\nQLat\nQLon\nMLT')

    plt.show() 
    fig.savefig(fname_fig)
    