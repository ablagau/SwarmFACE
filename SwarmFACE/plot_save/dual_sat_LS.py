#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.dates as mdt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import SwarmFACE.esaL2 as esaL2

def save_dual_sat_LS(j_df, param):
    '''
    Save to ASCII file results from dual-satellite LS algorithm. Input from 
    j2satLS.py
    '''  

    dtime_beg = param['dtime_beg']
    dtime_end = param['dtime_end']    
    sats = param['sats']
    tshift = param['tshift']    
    dt_along = param['dt_along']
    angTHR = param['angTHR']
    errTHR = param['errTHR']
    use_filter = param['use_filter']
    Bmodel = param['Bmodel']    
    timebads = param['timebads']

    angBN = j_df['angBN'].values    
    
    nbads = np.zeros(2, dtype=int)
    for ii in range(2):
        nbads[ii] = 0 if timebads['sc'+str(ii)] is None else len(timebads['sc'+str(ii)])
 
    str_trange = dtime_beg.replace('-','')[:8]+'_'+ \
         dtime_beg.replace(':','')[11:17] + '_'+dtime_end.replace(':','')[11:17]
    
    fname_out = 'FAC_LR_sw'+''.join(sats)+'_LS_'+str_trange +'.dat'
       
    # exports the results
    out_df = j_df[['Rmid_x','Rmid_y','Rmid_z','FAC','Jn','FAC_er', 'Jn_er',
                  'angBN','CN2']]  
    text_beg = '# time [YYYY-mm-ddTHH:MM:SS.f],\t  Rmid_x [km],\t  Rmid_y,\t\
      Rmid_z,\t  Jfac [microA/m^2],\t  Jn,\t  errJfac,\t  errJn,\t  '
        
    if use_filter:
        out_df = j_df
        text_beg = text_beg + 'Jfac_flt,\t  Jn_flt,\t  errJfac_flt,\t  errJn_flt,\t  '         
    text_header = text_beg + 'angBN [deg],\t  CN2 \n'
    
    with open(fname_out, 'w') as file:
        file.write('# Swarm sat: [' + ', '.join(sats) + ' ] \n')
        file.write('# Time-shift in sec.: [' + ', '.join(str(x) for x in tshift) + '] \n')  
        file.write('# Accepted B - quad plane angle threshold: ' + str(angTHR) + ' deg \n')  
        file.write('# Accepted Jn error threshold: ' + str(errTHR) + ' microA/m^2 \n')                     
        file.write('# Model: ' + Bmodel + '\n')
        file.write('# use filter: ' + str(use_filter) + '\n')                   
        file.write('# Number of bad data points in L1b files: ' + str(nbads.sum()) + '\n')
        for ii in range(2):
            if nbads[ii]:
                file.write('#   Sw'+sats[ii] + ' ' + str(nbads[ii]) + ' point(s) : '+ 
                    '; '.join(timebads['sc'+str(ii)].strftime('%H:%M:%S').values) + '\n')
        file.write(text_header)
    out_df.to_csv(fname_out, mode='a', sep=",", na_rep = 'NaN', date_format='%Y-%m-%dT%H:%M:%S.%f',\
                  float_format='%15.4f', header=False)
        
def plot_dual_sat_LS(j_df, dat_df, param):
    '''
    Plot results from dual-satellite LS algorithm. Input from 
    j2satLS.py
    '''

    dtime_beg = param['dtime_beg']
    dtime_end = param['dtime_end']    
    sats = param['sats']
    tshift = param['tshift']    
    dt_along = param['dt_along']
    angTHR = param['angTHR']
    errTHR = param['errTHR']
    use_filter = param['use_filter']
    Bmodel = param['Bmodel']    
    timebads = param['timebads']
    
    angBN = j_df['angBN'].values    
    
    nsc = len(sats)
    nbads = np.zeros(2, dtype=int)
    for ii in range(2):
        nbads[ii] = 0 if timebads['sc'+str(ii)] is None else len(timebads['sc'+str(ii)])    
    
    FAC_L2 = esaL2.dual(dtime_beg, dtime_end)
    
    str_trange = dtime_beg.replace('-','')[:8]+'_'+ \
         dtime_beg.replace(':','')[11:17] + '_'+dtime_end.replace(':','')[11:17]   
         
    fname_fig = 'FAC_LR_sw'+''.join(sats)+'_LS_'+str_trange +'.eps'    
    
    bad_ang = np.less(np.abs(np.cos(angBN*np.pi/180.)), np.cos(np.deg2rad(90 - angTHR)))
    
    line1_text = 'Time interval:  ' + dtime_beg[:10]+'  ' + \
                dtime_beg[11:19] + ' - '+dtime_end[11:19] 
    line2_text = 'sat: [' + ', '.join(sats) + ']    Time-shift: [' + \
            ', '.join(str(x) for x in tshift) + ']     bad points: '+str(nbads.sum()) \
            + '     small angle points: ' +str(len(np.where(bad_ang)[0]))
    
    # creates fig and axes objects
    fig_size = (8.27,11.69)
    fig = plt.figure(figsize=fig_size, frameon=True)
    fig.suptitle('FAC density estimate with dual satellite LS method\n', size='xx-large',\
                 weight = 'bold', y=0.99)
    plt.figtext(0.5, 0.962, line1_text, fontsize = 'x-large', \
                va='top', ha = 'center')
    plt.figtext(0.5, 0.94, line2_text, fontsize = 'x-large', \
                va='top', ha = 'center')

    # PANEL PART
    # designes the panel frames 
    xle, xri, ybo, yto = 0.125, 0.94, 0.3, 0.09     # plot margins
    ratp = np.array([1, 1, 0.7, 0.7, 1.5, 0.7 ])    #relative hight of each panel 
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
        ax[ii].set_xlim(pd.Timestamp(dtime_beg), pd.Timestamp(dtime_end))
      
    for ii in range(nrp -1):
        ax[ii].set_xticklabels([])
        
    #Plots time-series quantities    
    ax[0].plot(dat_df[('dBgeo', str(sats[0]))])    
    ax[0].set_ylabel('$dB_{GEO}$ sw' + str(sats[0]) +'\n[nT]', linespacing=1.7)
    ax[0].legend(['dB_x', 'dB_y', 'dB_z' ], loc=(0.95, 0.1), handlelength=1)    
    
    ax[1].plot(dat_df[('dBgeo',str(sats[1]))])
    ax[1].get_shared_y_axes().join(ax[0], ax[1])
    ax[1].set_ylabel('$dB_{GEO}$ sw' + str(sats[1]) +'\n[nT]', linespacing=1.7)
    ax[1].legend(['dB_x', 'dB_y', 'dB_z' ], loc=(0.95, 0.1), handlelength=1)   

    ax[2].plot(j_df['CN2'])
    ax[2].set_ylabel('log(CN2)\n')
    
    ax[3].plot(j_df['angBN'])
    ax[3].set_ylabel('angBN\n[deg]', linespacing=1.7)
    
    if use_filter:
        ax[4].plot(j_df['FAC_flt'], label='$\mathrm{J_{LS \_ '+''.join(sats)+' \_ flt}}$')
    else:
        ax[4].plot(j_df['FAC'], label='$\mathrm{J_{LS \_ '+''.join(sats)+'}}$')
    ax[4].plot(FAC_L2['FAC'], label='$\mathrm{J_{Level2}}$')
    ax[4].axhline(y=0, linestyle='--', color='k', linewidth=0.7)
    ax[4].set_ylabel('$J_{FAC}$\n[$\mu A/m^2$]', linespacing=1.7)
    ax[4].legend(loc = (0.95, 0.1), handlelength=1)

    if use_filter:
        ax[5].plot(j_df['FAC_flt_er'])
    else:
        ax[5].plot(j_df['FAC_er'])   
    ax[5].set_ylabel(r'$J_{FAC\_er}}$'+'\n'+'[$\mu A/m^2$]', linespacing=1.7)
    
    for ii in range(1,nrp -1):
        ax[ii].get_shared_x_axes().join(ax[0])
    
    # creates the ephemerides    
    latc = dat_df[('Rsph',str(sats[1]),'Lat')].values
    lonc = dat_df[('Rsph',str(sats[1]),'Lon')].values
    locx = ax[nrp-1].get_xticks()
    latc_ipl = np.round(np.interp(locx, mdt.date2num(dat_df.index), \
                                latc), decimals=2).astype('str')
    lonc_ipl = np.round(np.interp(locx, mdt.date2num(dat_df.index), \
                                lonc), decimals=2).astype('str')
    qdlat_ipl = np.round(np.interp(locx, mdt.date2num(dat_df.index), \
                                dat_df[('QDref',str(sats[1]),'QDLat')]), decimals=2).astype('str')
    qdlon_ipl = np.round(np.interp(locx, mdt.date2num(dat_df.index), \
                                dat_df[('QDref',str(sats[1]),'QDLon')]), decimals=2).astype('str')
    mlt_ipl = np.round(np.interp(locx, mdt.date2num(dat_df.index), \
                                dat_df[('QDref',str(sats[1]),'MLT')]), decimals=1).astype('str')
    
    lab_fin = ['']*len(locx)
    for ii in range(len(locx)):
        lab_ini = mdt.num2date(locx[ii]).strftime('%H:%M:%S')
        lab_fin[ii] = lab_ini + '\n' +latc_ipl[ii] + '\n' + lonc_ipl[ii] + \
        '\n'+ qdlat_ipl[ii] +'\n' +qdlon_ipl[ii] + '\n' + mlt_ipl[ii]
        
    ax[nrp-1].set_xticklabels(lab_fin)
    plt.figtext(0.01, 0.213, 'Time\nLat\nLon\nQLat\nQLon\nMLT')
 
    # INSET PART
    # designes the insets to plot s/c constellation 
    xmar, ymar, xsep = 0.13, 0.04, 0.05
    xwi = (1 - 2*xmar - 2*xsep)/3.      # inset width
    rat = fig_size[0]/fig_size[1]       # useful to fix the same scales on x and y 
    
    # creates axes for each panel
    ax_conf = [0, 0, 0.]
    for ii in range(3):
        ax_conf[ii] =  fig.add_axes([xmar + ii*(xwi+xsep), ymar, xwi, xwi*rat])
    
    Rc = j_df[['Rmid_x', 'Rmid_y', 'Rmid_z']].values
    R = dat_df['Rgeo'].values.reshape(-1,2,3)
    ndt4 = len(j_df)

    # recoves the apex positions
    R4s = np.full((ndt4,4,3),np.nan)
    R4s[:,0:2, :] = R[:ndt4, :, :]
    R4s[:,2:, :] = R[dt_along:, :, :]

    ic=np.array([1, len(j_df)//2, len(j_df)-2])   # indexes for ploting the s/c
    Ri_nec, Vi_nec = (np.full((len(ic),4,3),np.nan) for i in range(2))
    nlim_nec, elim_nec = (np.zeros((len(ic), 2)) for i in range(2))
    
    z_geo = np.array([0., 0., 1.])
    for ii in range(len(ic)):
        pos_i = ic[ii]
        # computes NEC associated with the mesocenter location Rc 
        Ci_unit = -Rc[pos_i,:]/np.linalg.norm(Rc[pos_i,:])[...,None]
        Ei_geo = np.cross(Ci_unit, z_geo)
        Ei_unit = Ei_geo/np.linalg.norm(Ei_geo)
        Ni_geo = np.cross(Ei_unit, Ci_unit)
        Ni_unit = Ni_geo/np.linalg.norm(Ni_geo)
        trmat_i = np.stack((Ni_unit, Ei_unit, Ci_unit))
        # transform the sats. position from GEO mesocentric to NEC mesocentric
        Ri_geo = R4s[pos_i, :, :]
        Ri_nec[ii, :, :] = np.matmul(trmat_i, Ri_geo[...,None]).reshape(Ri_geo.shape)
        nlim_nec[ii,:] = [max(Ri_nec[ii, :, 0]), min(Ri_nec[ii, :, 0])]
        elim_nec[ii,:] = [max(Ri_nec[ii, :, 1]), min(Ri_nec[ii, :, 1])]  
        Vi_geo = (R4s[pos_i+1, :, :] - R4s[pos_i-1, :, :])/2.
        Vi_geo_unit = Vi_geo/np.linalg.norm(Vi_geo,axis=-1)[...,None]
        Vi_nec[ii, :, :] = np.matmul(trmat_i, Vi_geo_unit[...,None]).reshape(Vi_geo.shape)
        
    # computes the (common) range along N and E
    dn_span = max(nlim_nec[:,0] - nlim_nec[:,1])
    de_span = max(elim_nec[:,0] - elim_nec[:,1])
    d_span = max(np.concatenate((nlim_nec[:,0] - nlim_nec[:,1], 
                                 elim_nec[:,0] - elim_nec[:,1])))*1.2
        
     # plots the s/c positions
    icolor = ['b', 'r']
    for ii in range(len(ic)):
        norig = np.mean(nlim_nec[ii, :])
        eorig = np.mean(elim_nec[ii, :])    
        ax_conf[ii].set_xlim( norig - d_span/2., norig + d_span/2. )
        ax_conf[ii].set_ylim( eorig - d_span/2., eorig + d_span/2. )   
        xquad, yquad = [], []
        for kk in [0, 1, 3, 2, 0]:
            xquad.append(Ri_nec[ii, kk, 0])
            yquad.append(Ri_nec[ii, kk, 1])       
        ax_conf[ii].plot(xquad, yquad, c='k', linestyle=':', linewidth=1)        
        ax_conf[ii].arrow(0, 0, d_span/10*Vi_nec[ii, kk, 0], d_span/10*Vi_nec[ii, kk, 1], \
                   color='k', head_width = 4)
        for jj in range(4):
            ax_conf[ii].scatter(Ri_nec[ii, jj, 0], Ri_nec[ii, jj, 1],  marker='o'  ,\
               c=icolor[jj % 2], label=sats[jj%2])      
             
    ax_conf[0].set_ylabel('East [km]')
    ax_conf[0].set_xlabel('North [km]')
    ax_conf[1].set_xlabel('North [km]')
    ax_conf[1].set_yticklabels([])
    ax_conf[2].set_xlabel('North [km]')
    ax_conf[2].set_yticklabels([])
    handles, labels = ax_conf[2].get_legend_handles_labels()
    ax_conf[2].legend(handles[0:2],['sw'+sats[0],'sw'+sats[1]], loc = (0.98, 0.7),\
                      labelcolor=icolor, handlelength=1, ncol = 1)
                                    
    plt.draw()
    fig.savefig(fname_fig)    