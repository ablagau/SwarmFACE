#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import signal
import sys
from .utils import *

muo = 4*np.pi*1e-7
    
def singleJfac(t, R, B, dB, alpha= None, N2d= None, N3d= None, tincl= None, 
               res = 'LR', er_db= 0.5, angTHR= 30., use_filter= True):
    '''
    Estimate the FAC density with single-satellite method

    Parameters
    ----------
    t: pandas.Index
        time stamps
    R : numpy.array
        satellite vector position in GEO frame
    B : numpy.array
        magnetic field vector in GEO frame
    dB : numpy.array
        magnetic perturbation vector in GEO frame
    alpha : float
        angle in the tangential plane between the (projection of) current
        sheet normal and the satellite velocity
    N3d : [float, float, float]
        current sheet normal vector in GEO frame
    N2d : [float, float, float]
        projection of the current sheet normal on the tangential plane
    tincl : [datetime, datetime]
         time interval when the information on current sheet inclination is valid
    res : str
        data resolution, 'LR' or 'HR'
    er_db : float
        error in magnetic field measurements
    angTHR : float
        minimum accepted angle between the magnetic field vector and the tangential plane
    use_filter : boolean
        'True' for data filtering

    Returns
    -------
    j_df : DataFrame
        results, including FAC and IRC densities
    '''

    # Constructs the differences & values at mid-intervals
    dt = t[1:].values - t[:-1].values
    tmid = t[:-1].values + dt*0.5
    Bmid = 0.5*(B[1:,:] + B[:-1,:])           
    Rmid = 0.5*(R[1:,:] + R[:-1,:])
    diff_dB = dB[1:,:] - dB[:-1,:]   
    V3d = R[1:,:] - R[:-1,:]
    Vorb = np.sqrt(np.sum(V3d*V3d, axis=-1))      
    # Defines important unit vectors
    eV3d, eBmid, eRmid = normvec(V3d), normvec(Bmid), normvec(Rmid)
    eV2d = normvec(np.cross(eRmid, np.cross(eV3d, eRmid)))    
    # Angle between B and R
    cosBR = np.sum(eBmid*eRmid, axis=-1)
    angBR = np.arccos(cosBR)*180./np.pi    
    bad_ang = np.abs(cosBR) < np.cos(np.deg2rad(90 - angTHR))

    # incl is the array of FAC incliation wrt Vsat (in tangential plane)    
    if N3d is not None:
        eN3d = normvec(N3d)
        eN2d = normvec(eN3d - np.sum(eN3d*eRmid,axis=-1).reshape(-1,1)*eRmid)
        incl = sign_ang(eV2d, eN2d, eRmid)
    elif alpha is not None:
        incl = alpha if isinstance(alpha, np.ndarray) else \
                                     np.full(len(tmid), alpha)        
    elif N2d is not None:
        eN2d = normvec(np.cross(eRmid, np.cross(N2d, eRmid)))
        incl = sign_ang(eV2d, eN2d, eRmid)
    else:
        incl = np.zeros(len(tmid))

    # considers the validity interval of FAC inclination 
    if tincl is not None:
        ind_incl = np.where((tmid >= tincl[0]) & (tmid <= tincl[1]))[0]
        if len(ind_incl) == 0:
            print('*********************************************************')
            print('*** tincl not consistent with the input time interval ***')
            print('*********************************************************')
            sys.exit()
        incl[0:ind_incl[0]] = incl[ind_incl[0]]
        incl[ind_incl[-1]:] = incl[ind_incl[-1]]

    # working in the tangential plane
    eNtang = normvec(rotvecax(eV2d, eRmid, incl))
    eEtang = normvec(np.cross(eRmid, eNtang))
    diff_dB_Etang = np.sum(diff_dB*eEtang, axis=-1)
    Dplane = np.sum(eNtang*eV2d, axis=-1)
    Jrad= diff_dB_Etang/Dplane/Vorb/muo*1.e-6
    Jrad_er= np.abs(er_db/Dplane/Vorb/muo*1.e-6)   
    
    # FAC density and error
    Jb = Jrad/cosBR
    Jb_er = np.abs(Jrad_er/cosBR)    
    Jb[bad_ang] = np.nan
    Jb_er[bad_ang] = np.nan    

    # filtering part    
    Jrad_flt, Jb_flt, Jrad_flt_er, Jb_flt_er = \
                                    (np.full(len(Jb),np.nan) for i in range(4))
    if use_filter:
        fc, butter_ord = 1/20, 5      # 20 s cutt-off freq., filter order
        bf, af = signal.butter(butter_ord, fc /(res_param(res)[1]/2), 'low')
        dB_flt = signal.filtfilt(bf, af, dB, axis=0)        
        diff_dB_flt = dB_flt[1:,:] - dB_flt[:-1,:]
        diff_dB_flt_Etang = np.sum(diff_dB_flt*eEtang, axis=-1)
        Jrad_flt = diff_dB_flt_Etang/Dplane/Vorb/muo*1.e-6                
        Jb_flt = Jrad_flt/cosBR   
        Jb_flt[bad_ang] = np.nan        
        Jb_flt_er = Jb_er/2.5  # this is an empirical factor
        Jrad_flt_er = Jrad_er/2.5

    # stores the output in a DataFrame        
    j_df = pd.DataFrame(np.stack((Rmid[:,0], Rmid[:,1], Rmid[:,2], Jb, Jrad, 
                                  Jb_er, Jrad_er, Jb_flt, Jrad_flt, Jb_flt_er, 
                                  Jrad_flt_er, angBR, incl)).transpose(), 
                    columns=['Rmid_x','Rmid_y','Rmid_z','FAC','IRC','FAC_er',
                             'IRC_er','FAC_flt','IRC_flt','FAC_flt_er',
                             'IRC_flt_er','angBR','incl'], index=tmid) 
    return j_df


def recivec3s(R):
    '''
    Compute the reciprocal vectors for a three-satellites configuration

    Parameters
    ----------
    R : numpy.array
        satellites vector position in format [index, sc, comp]

    Returns
    -------
    Rc : numpy.array
        position of the mesocenter
    Rmeso : numpy.array
        satellites position in the mesocentric frame
    nuvec : numpy.array
        normal to the spacecraft plane
    Q3S : numpy.array
        generalized reciprocal vectors
    Qtens : numpy.array
        reciprocal tensor
    Rtens : numpy.array
        position tensor
     '''

    Rc = np.mean(R, axis=-2)
    Rmeso = R - Rc[:, np.newaxis, :]
    r12 = Rmeso[:,1,:] - Rmeso[:,0,:]
    r13 = Rmeso[:,2,:] - Rmeso[:,0,:]
    r23 = Rmeso[:,2,:] - Rmeso[:,1,:]
    nuvec = np.cross(r12, r13)
    nuvec_norm = np.linalg.norm(nuvec, axis=-1, keepdims=True)
    nuvec = np.divide(nuvec, nuvec_norm)
    Q3S = np.stack((np.cross(nuvec,r23), np.cross(nuvec,-r13), np.cross(nuvec,r12)),axis = -2)
    Q3S = np.divide(Q3S, nuvec_norm[...,None])
    Qtens = np.sum(np.matmul(Q3S[:,:,:,None],Q3S[:,:,None, :]), axis = -3)    
    Rtens = np.sum(np.matmul(Rmeso[:,:,:,None],Rmeso[:,:,None, :]), axis = -3)
    return Rc, Rmeso, nuvec, Q3S, Qtens, Rtens

def threeJfac(dt, R, B, dB, er_db= 0.5, angTHR= 30., use_filter= True):
    '''
    Estimate the FAC density with three-satellite method

    Parameters
    ----------
    dt: pandas.Index
        time stamps
    R : numpy.array
        satellites vector position in GEO frame
    B : numpy.array
        magnetic field vector in GEO frame
    dB : numpy.array
        magnetic perturbation vector in GEO frame
    er_db : float
        error in magnetic field measurements
    angTHR : float
        minimum accepted angle between the magnetic field vector and the
        spacecraft plane
    use_filter : boolean
        'True' for data filtering

    Returns
    -------
    j_df : DataFrame
        results, including FAC and normal current densities
    '''

    # computes the mesocenter of the Swarm constellation (Rc), the s/c positions
    # in the mesocentric frame (Rmeso), the direction normal to spacecraft 
    # plane (nuvec), the reciprocal vectors (Q3S), the reciprocal tensor (Qtens), 
    # and the position tensor (Rtens)    
    Rc, Rmeso, nuvec, Q3S, Qtens, Rtens = recivec3s(R)
    eigval = np.sort(np.linalg.eigvals(Rtens), axis=-1)
    CN3 = np.log10(np.divide(eigval[:,2],eigval[:,1]))  # the condition number  

    # computes the direction of (un-subtracted) local magnetic field Bunit and #
    # the orientation of spacecraft plane with respect to Bunit (cosBN and angBN). 
    # Stores times when B is too close to the spacecraft plane, set by angTHR.    
    Bunit = B.mean(axis=-2)/np.linalg.norm(B.mean(axis=-2),axis=-1)[...,None]
    cosBN = np.matmul(Bunit[...,None,:],nuvec[...,:,None]).reshape(dt.shape)
    angBN = np.arccos(cosBN)*180./np.pi 
    bad_ang = np.where((np.abs(angBN) < 90+angTHR) & (np.abs(angBN) > 90-angTHR))    

    # Estimates the curl of B and Jn, i.e. current flowing along B    
    CurlB = np.sum( np.cross(Q3S,dB,axis=-1), axis=-2)
    CurlBn = np.matmul(CurlB[...,None,:],nuvec[...,:,None]).reshape(dt.shape)
    Jn= (1e-6/muo)*CurlBn
    Jb= Jn/cosBN    
   
    # Estimates the errors in Jn
    traceq = np.trace(Qtens, axis1=-2, axis2=-1) 
    Jn_er = 1e-6*er_db/muo*np.sqrt(traceq)
    Jb_er = Jn_er/np.absolute(cosBN)   
  
    Jb[bad_ang] = np.nan   

    # filtering part   
    ndt = len(dt)
    dB_flt = np.full((ndt,3,3),np.nan)
    Jn_flt, Jb_flt, Jn_flt_er, Jb_flt_er = \
                                    (np.full(len(Jb),np.nan) for i in range(4))
    if use_filter:
        fc, butter_ord = 1/20, 5      # 20 s cutt-off freq., filter order
        bf, af = signal.butter(butter_ord, fc /(1./2.), 'low')
        for sc in range(3):
            dB_flt[:,sc,:] = signal.filtfilt(bf, af, dB[:,sc,:], axis=0)
        CurlB_flt = np.sum( np.cross(Q3S,dB_flt,axis=-1), axis=-2)
        CurlBn_flt = np.matmul(CurlB_flt[...,None,:],nuvec[...,:,None]).reshape(dt.shape)
        Jn_flt= (1e-6/muo)*CurlBn_flt
        Jb_flt= Jn_flt/cosBN    
        Jb_flt_er = Jb_er/2.5  # this is an empirical factor
        Jn_flt_er = Jn_er/2.5     
        Jb_flt[bad_ang] = np.nan        
        
    # stores the output in a DataFrame 
    j_df = pd.DataFrame(np.stack((Rc[:,0], Rc[:,1], Rc[:,2], Jb, Jn, Jb_er, 
                                  Jn_er, Jb_flt, Jn_flt, Jb_flt_er, Jn_flt_er, 
                                  angBN, CN3)).transpose(),
                columns=['Rmid_x','Rmid_y','Rmid_z','FAC','Jn','FAC_er','Jn_er',
                         'FAC_flt','Jn_flt','FAC_flt_er','Jn_flt_er', 'angBN', 
                         'CN3'], index=dt)    
    return j_df
    
def recivec2s(R4s):
    '''
    Compute the reciprocal vectors of a four-point planar configuration

    Parameters
    ----------
    R4s : numpy.array
        position vectors of the apexes in format [index, apex, comp]

    Returns
    -------
    Q4s : numpy.array
        canonical base (reciprocal) vectors
    Rc : numpy_array
        position of the mesocenter
    nuvec : numpy.array
        normal to the spacecraft plane
    condnum : numpy.array
        condition number
    nonplanar : numpy.array
        nonplanarity
    tracer : numpy.array
        trace of the position tensor
    traceq : numpy.array
        trace of the reciprocal tensor
     '''
    # computes the reciprocal vectors of a 4 point planar configuration
    # work in the mesocenter frame
    Rc = np.mean(R4s, axis=-2)
    R4smc = R4s - Rc[:, np.newaxis, :]
    Rtens = np.sum(np.matmul(R4smc[:,:,:,None],R4smc[:,:,None, :]), axis = -3)
    eigval, eigvec = np.linalg.eigh(Rtens)
    # avoid eigenvectors flipping direction
    # for that, impose N to be closer to radial direction
    nprov = np.squeeze(eigvec[:,:,0])    # provisional normal
    cosRN = np.squeeze(np.matmul(Rc[...,None,:],nprov[...,:,None]))
    iposr = cosRN < 0
    eigvec[iposr,:,:] = -eigvec[iposr,:,:]
    # minimum variance direction is along normal N
    nuvec = np.squeeze(eigvec[:,:,0])
    intvec = np.squeeze(eigvec[:,:,1])
    maxvec = np.squeeze(eigvec[:,:,2])
    # nonplanarity and condition number
    nonplan = eigval[:,0]/eigval[:,2]
    condnum = eigval[:,2]/eigval[:,1]
    qtens = 1./eigval[:,2,np.newaxis,np.newaxis]*np.matmul(maxvec[:,:,None],maxvec[:,None, :]) +\
        1./eigval[:,1,np.newaxis,np.newaxis]*np.matmul(intvec[:,:,None],intvec[:,None, :])
    # Q4s are the planar canonical base vectors
    Q4s = np.squeeze(np.matmul(qtens[:,None, :,:],R4smc[:,:,:,None]))
    # trace of the position and reciprocal tensor 
    tracer = np.sum(np.square(np.linalg.norm(R4smc, axis = -1)), axis=-1)   
    traceq = np.sum(np.square(np.linalg.norm(Q4s, axis = -1)), axis=-1)
    return Q4s, Rc, nuvec, condnum, nonplan, tracer, traceq

def ls_dualJfac(dt, R, B, dB, dt_along= 5, er_db= 0.5, angTHR= 30., 
                errTHR=0.1, use_filter= True, saveconf=False):
    '''
    Estimate the FAC density with dual-satellite Least-Squares method

    Parameters
    ----------
    dt: pandas.Index
        time stamps
    R : numpy.array
        satellites vector position in GEO frame
    B : numpy.array
        magnetic field vector in GEO frame
    dB : numpy.array
        magnetic perturbation vector in GEO frame
    dt_along : integer
        along track separation in sec.
    er_db : float
        error in magnetic field measurements
    angTHR : float
        minimum accepted angle between the magnetic field vector and the
        quad plane
    errTHR : float
        accepted error for normal current density
    use_filter : boolean
        'True' for data filtering
    saveconf : boolean
        'True' to add the quad's parameters in the results

    Returns
    -------
    j_df : DataFrame
        results, including FAC and normal current densities
    '''

    ndt = len(dt)
    ndt4 = ndt - dt_along    
    dt4 = dt[:ndt4].shift(1000.*(dt_along/2),freq='ms')   # new data timeline

    # constructs of the quad
    R4s, B4s, dB4s = (np.full((ndt4,4,3),np.nan) for i in range(3))
    for arr4s, arr in zip([R4s, B4s, dB4s], [R,B,dB]):
        arr4s[:,0:2, :] = arr[:ndt4, :, :]
        arr4s[:,2:, :] = arr[dt_along:, :, :]
    
    # computes the planar canonical base vectors (Q4s), the position at mesocenter 
    # (Rc), the quad normal (nuvec), the condition number (cnum), nonplanarity 
    # (nonplan) and trace of the position (tracer) and reciprocal (traceq) tensors. 
    Q4s, Rc, nuvec, cnum, nonplan, tracer, traceq = recivec2s(R4s)
    CN2 = np.log10(cnum)  # log of condition number  
    
    # computes the direction of (un-subtracted) local magnetic field Bunit and #
    # the orientation of spacecraft plane with respect to Bunit (cosBN and angBN). 
    # Stores times when B is too close to the spacecraft plane, set by angTHR.    
    Bunit = B4s.mean(axis=-2)/np.linalg.norm(B4s.mean(axis=-2),axis=-1)[...,None]
    cosBN = np.matmul(Bunit[...,None,:],nuvec[...,:,None]).reshape(dt4.shape)
    angBN = np.arccos(cosBN)*180./np.pi 
    bad_ang = np.where((np.abs(angBN) < 90+angTHR) & (np.abs(angBN) > 90-angTHR))    

    # Estimates the curl of B, Jn, and Jb i.e. current along normal and along B    
    CurlB = np.sum( np.cross(Q4s,dB4s,axis=-1), axis=-2)
    CurlBn = np.matmul(CurlB[...,None,:],nuvec[...,:,None]).reshape(dt4.shape)
    Jn= (1e-6/muo)*CurlBn
    Jb= Jn/cosBN    
     
    # Estimates the errors in Jn
    Jn_er = 1e-6*er_db/muo*np.sqrt(traceq)
    Jb_er = Jn_er/np.absolute(cosBN)    
    bad_err = np.where(Jn_er > errTHR)    
    Jb[bad_ang] = np.nan   
    Jb[bad_err], Jn[bad_err]  = (np.nan for i in range(2))

    # filtering part   
    dB_flt = np.full((ndt,2,3),np.nan)
    dB4s_flt = np.full((ndt4,4,3),np.nan)
    Jn_flt, Jb_flt, Jn_flt_er, Jb_flt_er = \
                                    (np.full(len(Jb),np.nan) for i in range(4))
    if use_filter:
        fc, butter_ord = 1/20, 5      # 20 s cutt-off freq., filter order
        bf, af = signal.butter(butter_ord, fc /(1./2.), 'low')
        for sc in range(2):
            dB_flt[:,sc,:] = signal.filtfilt(bf, af, dB[:,sc,:], axis=0)
        dB4s_flt[:,0:2, :] = dB_flt[:ndt4, :, :]
        dB4s_flt[:,2:, :] = dB_flt[dt_along:, :, :]
        CurlB_flt = np.sum( np.cross(Q4s,dB4s_flt,axis=-1), axis=-2)
        CurlBn_flt = np.matmul(CurlB_flt[...,None,:],nuvec[...,:,None]).reshape(dt4.shape)
        Jn_flt= (1e-6/muo)*CurlBn_flt
        Jb_flt= Jn_flt/cosBN    
        Jb_flt_er = Jb_er/2.5  # this is an empirical factor
        Jn_flt_er = Jn_er/2.5    
        bad_flt_err = np.where(Jb_flt_er > errTHR) 
        Jb_flt[bad_ang] = np.nan     
        Jb_flt[bad_flt_err], Jn_flt[bad_flt_err]  = (np.nan for i in range(2))
        
    # stores the output in a DataFrame 
    j_df = pd.DataFrame(np.stack((Rc[:,0], Rc[:,1], Rc[:,2], Jb, Jn, Jb_er, 
                                  Jn_er, Jb_flt, Jn_flt, Jb_flt_er, Jn_flt_er, 
                                  angBN, CN2)).transpose(),
                columns=['Rmid_x','Rmid_y','Rmid_z','FAC','Jn','FAC_er','Jn_er',
                         'FAC_flt','Jn_flt','FAC_flt_er','Jn_flt_er', 'angBN', 
                         'CN2'], index=dt4)    
    if saveconf == True:
        # orders the quad's vertices and computes its parameters 
        EL, EM, el, em  = SortVertices(R4s, dB4s)[7:]        
        j_df = j_df.assign(EL=EL, EM=EM, el=el, em=em)    
    
    return j_df
    
def ffunct(tau3d,tauast=0.13,taunul=0.07,intpol='Linear'):
    '''
    Compute coefficients for a smooth transition between 1D and 2D
    gradient estimators in the SVD method.

    Parameters
    ----------
    tau3d : numpy.array
        S eigenvalues ratio
    tauast : float
        value between [0, 1]. tauast >= taunul. The interval [taunul, tauast]
        is used to match the 1D and 2D gradient estimators
    taunul : float
        value between [0, 1]. Below this threshold value the quad is
        considered degenerated
    intpol : string
        interpolation method used for matching the 1D and 2D gradient
        estimators. Should be ‘Linear’, ‘Cubic’, or None

    Returns
    -------
    f : numpy.array
        coefficient to smooth transition between 1D and 2D gradient estimators
    '''

    if intpol==None:
        f = np.ones(tau3d.shape)
        f[taunul>tau3d] = 0
    elif intpol=='Linear':
        f = (tau3d-taunul)/(tauast-taunul)
        f[tau3d>=tauast] = 1
        f[taunul>tau3d] = 0
    elif intpol=='Cubic':
        f = (3-2*(tau3d-taunul)/(tauast-taunul))*(tau3d-taunul)**2 /(tauast-taunul)**2
        f[tau3d>=tauast] = 1
        f[taunul>tau3d] = 0
    else:
        print('*** ffunct: Non-supported value of intpol ***')
        f = -1
    return f

def qmatrix_intpol(R,tauast=0.13, taunul=0.07, intpol='Linear'):
    '''
    Compute the SVD reciprocal matrix of a four-point planar configuration.
    A smooth transition between 1D and 2D gradient estimators is allowed.

    Parameters
    ----------
    R : numpy.array
        position vectors of the apexes in format [index, apex, comp]
    tauast : float
        value between [0, 1]. tauast >= taunul. The interval [taunul, tauast]
        is used to match the 1D and 2D gradient estimators
    taunul : float
        value between [0, 1]. Below this threshold value the quad is
        considered degenerated
    intpol : string
        interpolation method used for matching the 1D and 2D gradient
        estimators. Should be ‘Linear’, ‘Cubic’, or None

    Returns
    -------
    Q : numpy.array
        reciprocal matrix
    AD : numpy_array
        quad degeneracy
    Vt : numpy.array
        orthogonal matrix from SVD decomposition of R = USVt
    S : numpy.array
        diagonal matrix from SVD decomposition of R = USVt
     '''

    U,S,Vt = np.linalg.svd(R-R.mean(axis=-2)[...,None,:],full_matrices=False)
    tau3d = np.divide(S, S[:,0][...,None])
    ff = ffunct(tau3d,tauast=tauast,taunul=taunul,intpol=intpol) 
   
    Sinv = np.zeros(S.shape)
    Sinv = np.multiply(1/S, ff)
    degen = S < taunul*S[...,0][...,None]
    Sinv[degen] = 0      # redundant
    Q = np.matmul(U*Sinv[...,None,:],Vt)
    AD = ff.sum(axis=-1)
    return Q,AD,Vt,S     

def svd_dualJfac(dt, R, B, dB, dt_along=5, er_db=0.5, tauast=0.13, taunul=0.07,
                 intpol='Linear', angTHR=30., use_filter=True, saveconf=False):
    '''
    Estimate the FAC density with dual-satellite Singular
    Values Decomposition method

    Parameters
    ----------
    dt: pandas.Index
        time stamps
    R : numpy.array
        satellites vector position in GEO frame
    B : numpy.array
        magnetic field vector in GEO frame
    dB : numpy.array
        magnetic perturbation vector in GEO frame
    dt_along : integer
        along track separation in sec.
    er_db : float
        error in magnetic field measurements
    tauast : float
        value between [0, 1]. tauast >= taunul. The interval [taunul, tauast]
        is used to match the 1D and 2D gradient estimators
    taunul : float
        value between [0, 1]. Below this threshold value the quad is
        considered degenerated
    intpol : string
        interpolation method used for matching the 1D and 2D gradient
        estimators. Should be ‘Linear’, ‘Cubic’, or None
    angTHR : float
        minimum accepted angle between the magnetic field vector and the
        quad plane
    use_filter : boolean
        'True' for data filtering
    saveconf : boolean
        'True' to add the quad's parameters in the results

    Returns
    -------
    j_df : DataFrame
        results, including FAC and normal current densities
    '''

    ndt = len(dt)
    ndt4 = ndt - dt_along    
    dt4 = dt[:ndt4].shift(1000.*(dt_along/2),freq='ms')   # new data timeline

    # constructs of the quad
    R4s, B4s, dB4s = (np.full((ndt4,4,3),np.nan) for i in range(3))
    for arr4s, arr in zip([R4s, B4s, dB4s], [R,B,dB]):
        arr4s[:,0:2, :] = arr[:ndt4, :, :]
        arr4s[:,2:, :] = arr[dt_along:, :, :]

    Q,AD,Vt,S = qmatrix_intpol(R4s,tauast=tauast, taunul=taunul, intpol=intpol)
    tau = S[...,1]/S[...,0]
    Rc = R4s.mean(axis=-2)
    Runit = Rc/np.linalg.norm(Rc,axis=-1)[...,None]     
    Nunit = Vt[...,2,:]/np.linalg.norm(Vt[...,2,:],axis=-1)[...,None]
    # avoid eigenvectors flipping direction
    # for that, impose N to be closer to radial direction
    cosRN = np.sum(Runit*Nunit, axis=-1)
    iposr = cosRN < 0
    Nunit[iposr,:] = -Nunit[iposr,:]   
    # when geometry is degenerate, use Runit
    iposa = AD <= 1
    Nunit[iposa,:] = Runit[iposa,:]
    # computes the direction of (un-subtracted) local magnetic field Bunit and 
    # the orientation of spacecraft plane with respect to Bunit (cosBN and angBN). 
    # Stores times when B is too close to the spacecraft plane, set by angTHR.       
    Bunit = B4s.mean(axis=-2)/np.linalg.norm(B4s.mean(axis=-2),axis=-1)[...,None]
    cosBN = np.matmul(Bunit[...,None,:],Nunit[...,:,None]).reshape(dt4.shape)
    angBN = np.arccos(cosBN)*180./np.pi
    bad_ang = np.where((np.abs(angBN) < 90+angTHR) & (np.abs(angBN) > 90-angTHR)) 
    
    # Estimates the curl of B, Jn, and Jb i.e. current along normal and along B        
    CurlB = np.sum( np.cross(Q,dB4s,axis=-1), axis=-2 )    
    CurlBn = np.matmul(CurlB[...,None,:],Nunit[...,:,None]).reshape(dt4.shape) 
    Jn= (1e-6/muo)*CurlBn
    Jb= Jn/cosBN     

    Jn_er = (1e-6/muo)*er_db*np.linalg.norm(Q,axis=(-2,-1))   
    Jb_er = Jn_er/np.absolute(cosBN)
  
    Jb[bad_ang] = np.nan  
    Jb_er[bad_ang] = np.nan     
    
    # filtering part   
    dB_flt = np.full((ndt,2,3),np.nan)
    dB4s_flt = np.full((ndt4,4,3),np.nan)
    Jn_flt, Jb_flt, Jn_flt_er, Jb_flt_er = \
                                    (np.full(len(Jb),np.nan) for i in range(4))
    if use_filter:
        fc, butter_ord = 1/20, 5      # 20 s cutt-off freq., filter order
        bf, af = signal.butter(butter_ord, fc /(1./2.), 'low')
        for sc in range(2):
            dB_flt[:,sc,:] = signal.filtfilt(bf, af, dB[:,sc,:], axis=0)
        dB4s_flt[:,0:2, :] = dB_flt[:ndt4, :, :]
        dB4s_flt[:,2:, :] = dB_flt[dt_along:, :, :]
        CurlB_flt = np.sum( np.cross(Q,dB4s_flt,axis=-1), axis=-2 )    
        CurlBn_flt = np.matmul(CurlB_flt[...,None,:],Nunit[...,:,None]).reshape(dt4.shape)         
        Jn_flt= (1e-6/muo)*CurlBn_flt
        Jb_flt= Jn_flt/cosBN    
        Jb_flt[bad_ang] = np.nan
        Jb_flt_er = Jb_er/2.5  # this is an empirical factor
        Jn_flt_er = Jn_er/2.5    
            
    # stores the output in a DataFrame 
    j_df = pd.DataFrame(np.stack((Rc[:,0], Rc[:,1], Rc[:,2], Jb, Jn, Jb_er, 
                                  Jn_er, Jb_flt, Jn_flt, Jb_flt_er, Jn_flt_er, 
                                  angBN, AD, tau)).transpose(),
                columns=['Rmid_x','Rmid_y','Rmid_z','FAC','Jn','FAC_er','Jn_er',
                         'FAC_flt','Jn_flt','FAC_flt_er','Jn_flt_er', 'angBN', 
                         'AD','tau'], index=dt4)  
    if saveconf == True:
        # orders the quad's vertices and computes its parameters 
        EL, EM, el, em  = SortVertices(R4s, dB4s)[7:]        
        j_df = j_df.assign(EL=EL, EM=EM, el=el, em=em)    
        
    return j_df

def calcBI(R4, dB4):
    '''
    Compute curlB using the discrete BI integral along a four-point
    configuration.

    Parameters
    ----------
    R4 : numpy.array
        position vectors of the apexes in format [index, apex, comp]
    dB4 : numpy.array
        magnetic perturbation vectors at the apexes in format
        [index, apex, comp]

    Returns
    -------
     : numpy.array
        curl of magnetic field perturbation
     '''

    r10 = R4[:,1,:] - R4[:,0,:]
    r32 = R4[:,3,:] - R4[:,2,:]
    r21 = R4[:,2,:] - R4[:,1,:]
    r03 = R4[:,0,:] - R4[:,3,:]
    area = 0.5*np.linalg.norm(np.cross(-r10, r21), axis=-1) + \
            0.5*np.linalg.norm(np.cross(-r32, r03), axis=-1)
    int_along = np.sum((dB4[:,1,:] + dB4[:,0,:])*r10, axis = -1) +\
            np.sum((dB4[:,3,:] + dB4[:,2,:])*r32, axis = -1)
    int_cross = np.sum((dB4[:,2,:] + dB4[:,1,:])*r21, axis = -1) +\
            np.sum((dB4[:,0,:] + dB4[:,3,:])*r03, axis = -1) 
    return 0.5*(int_along + int_cross)/area

def bi_dualJfac(dt, R, B, dB, dt_along= 5, er_db= 0.5, angTHR= 30., 
                errTHR=0.1, use_filter= True, saveconf= False):
    '''
    Estimate the FAC density with dual-satellite Boundary
    Integral method

    Parameters
    ----------
    dt: pandas.Index
        time stamps
    R : numpy.array
        satellites vector position in GEO frame
    B : numpy.array
        magnetic field vector in GEO frame
    dB : numpy.array
        magnetic perturbation vector in GEO frame
    dt_along : integer
        along track separation in sec.
    er_db : float
        error in magnetic field measurements
    angTHR : float
        minimum accepted angle between the magnetic field vector and the
        quad plane
    errTHR : float
        accepted error for normal current density
    use_filter : boolean
        'True' for data filtering
    saveconf : boolean
        'True' to add the quad's parameters in the results

    Returns
    -------
    j_df : DataFrame
        results, including FAC and normal current densities
    '''

    ndt = len(dt)
    ndt4 = ndt - dt_along    
    dt4 = dt[:ndt4].shift(1000.*(dt_along/2),freq='ms')   # new data timeline
    # constructs the quad
    R4s, B4s, dB4s = (np.full((ndt4,4,3),np.nan) for i in range(3))
    for arr4s, arr in zip([R4s, B4s, dB4s], [R,B,dB]):
        arr4s[:,0:2, :] = arr[:ndt4, :, :]
        arr4s[:,2:, :] = arr[dt_along:, :, :]
    # orders the quad's vertices and computes the trace of BI reciprocal tensors 
    # The new 0,1,2,and 3 vertices are situated in quadrants Q3, Q4, Q1, and Q2
    # of the local proper reference    
    R4sa, dB4sa, trBI, Rc, nuvec, satsort, trLS, EL, EM, el, em  = \
                                                    SortVertices(R4s, dB4s)
    # computes the direction of (un-subtracted) local magnetic field Bunit and 
    # the orientation of spacecraft plane with respect to Bunit (cosBN and angBN). 
    # Stores times when B is too close to the spacecraft plane, set by angTHR.    
    Bunit = B4s.mean(axis=-2)/np.linalg.norm(B4s.mean(axis=-2),axis=-1)[...,None]
    cosBN = np.matmul(Bunit[...,None,:],nuvec[...,:,None]).reshape(dt4.shape)
    angBN = np.arccos(cosBN)*180./np.pi 
    bad_ang = np.where((np.abs(angBN) < 90+angTHR) & (np.abs(angBN) > 90-angTHR))     
    # Estimates the curl of B, Jn, and Jb i.e. current along normal and along B  
    CurlBn = calcBI(R4sa, dB4sa)  
    Jn= (1e-6/muo)*CurlBn
    Jb= Jn/cosBN    
    # Estimates the errors in Jn
    Jn_er = 1e-6*er_db/muo*np.sqrt(trBI)
    Jb_er = Jn_er/np.absolute(cosBN)  

    bad_err = np.where(Jn_er > errTHR)    
    Jb[bad_ang] = np.nan   
    Jb[bad_err], Jn[bad_err]  = (np.nan for i in range(2))
    # filtering part   
    dB_flt = np.full((ndt,2,3),np.nan)
    dB4s_flt, dB4sa_flt = (np.full((ndt4,4,3),np.nan) for i in range(2))
    Jn_flt, Jb_flt, Jn_flt_er, Jb_flt_er = \
                                    (np.full(len(Jb),np.nan) for i in range(4))
    if use_filter:
        fc, butter_ord = 1/20, 5      # 20 s cutt-off freq., filter order
        bf, af = signal.butter(butter_ord, fc /(1./2.), 'low')
        for sc in range(2):
            dB_flt[:,sc,:] = signal.filtfilt(bf, af, dB[:,sc,:], axis=0)
        dB4s_flt[:,0:2, :] = dB_flt[:ndt4, :, :]
        dB4s_flt[:,2:, :] = dB_flt[dt_along:, :, :]
        for jj in range(3):
            dB4sa_flt[:,:,jj] = np.take_along_axis(dB4s_flt[:,:,jj], satsort, -1)                
        CurlBn_flt = calcBI(R4sa, dB4sa_flt)
        Jn_flt= (1e-6/muo)*CurlBn_flt
        Jb_flt= Jn_flt/cosBN    
        Jb_flt_er = Jb_er/2.5  # this is an empirical factor
        Jn_flt_er = Jn_er/2.5    
        bad_flt_err = np.where(Jb_flt_er > errTHR) 
        Jb_flt[bad_ang] = np.nan     
        Jb_flt[bad_flt_err], Jn_flt[bad_flt_err]  = (np.nan for i in range(2))
    # stores the output in a DataFrame 
    j_df = pd.DataFrame(np.stack((Rc[:,0], Rc[:,1], Rc[:,2], Jb, Jn, Jb_er, 
                                  Jn_er, Jb_flt, Jn_flt, Jb_flt_er, Jn_flt_er, 
                                  angBN)).transpose(),
                columns=['Rmid_x','Rmid_y','Rmid_z','FAC','Jn','FAC_er','Jn_er',
                         'FAC_flt','Jn_flt','FAC_flt_er','Jn_flt_er', 'angBN'], index=dt4)    
    if saveconf == True:
        j_df = j_df.assign(EL=EL, EM=EM, el=el, em=em)
    
    return j_df

