"""
Contains User Inference/Analytic Models.

A model must fit the following requisites and structure :
--------------------------------------------------------
    1. must be a callable function that takes N numpy arrays as inputs
    2. /!\ returns N None for the N awaited outputs if at least one of the input is None /!\
    3. inputs may be freely formatted and transformed into what you want BUT...
    4. ...outputs must be formatted as numpy array for sending back
"""
import numpy as np
import gsw

# --------- utils ---------- #
def Is_None(*inputs):
    """ Test presence of at least one None in inputs """
    return any(item is None for item in inputs)

# ============================ #
#     Stanley et al. 2020      #
# ============================ #
def stanley_terms(T,S):
    """ Compute d2 rho(T,S) / dTdT """
    der = gsw.rho_second_derivatives(S,T,np.zeros_like(T))
    return der[2] / 2.0


# sigmaT = c * |lx_i · dT/dx_i|**2
def Std_Stanley(T, S, mask_u, mask_v, c=0.1):
    """ Computation of Stanley et al. (2020) model with analytical methods """
    if Is_None(T,S):
        return None
    else:
        dTdx  = np.diff(T, axis=0, append=T[-1:,:,:]) * mask_u
        dTdx = dTdx + np.diff(T,axis=0, prepend=T[0:1,:,:]) * np.roll(mask_u, 1, axis=0)
        wght = mask_u + np.roll(mask_u, 1, axis=0)
        wght[ wght == 0 ] = 1
        dTdx = dTdx / wght
        
        dTdy = np.diff(T, axis=1, append=T[:,-1:,:]) * mask_v
        dTdy = dTdy + np.diff(T,axis=1, prepend=T[:,0:1,:]) * np.roll(mask_v, 1, axis=1)
        wght = mask_v + np.roll(mask_v, 1, axis=1)
        wght[(wght == 0)] = 1
        dTdy = dTdy / wght
        
        return stanley_terms(T,S) * c * ( dTdx**2 + dTdy**2 )
