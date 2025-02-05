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

# --------- utils ---------- #
def Is_None(*inputs):
    """ Test presence of at least one None in inputs """
    return any(item is None for item in inputs)

# ============================ #
#             MLE              #
# ============================ #
def vert_buoyancy_flux(db,H,S,dl,C_Lf):
    """ Compute vertical buoyancy flux induced streamfunction with expression (13) from doi.org/10.1016/j.ocemod.2020.101678 """
    if Is_None(db,H):
        return None
    else:
        grad_b = db / dl
        return C_Lf * H**2 * S * grad_b
