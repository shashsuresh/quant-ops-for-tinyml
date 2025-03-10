"""
This file contains some conversion functions and defines for quantized operations
"""

QUANTIZED_GRADIENT = False
ROUNDING = 'round'
CONV_W_GRAD = True

import torch
import numpy as np

def convert_to_np(x):
    '''
    Casts input `x` into an NP array if it is a number
    '''
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return np.array([x])
    

def to_pytorch_tensor(x):
    '''
    Casts input `x` into a float tensor if it either an array or a number
    '''
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.float64:
            return x.float()
        else:
            return x.float() #For Back-propagation everything has to be in float
    elif isinstance(x, np.ndarray):
        org_dtype = x.dtype
        x = torch.from_numpy(x)
        if org_dtype in [np.int64, np.int32, np.int8]:
            return x.float() #For Back-propagation everything has to be in float
        elif org_dtype in [np.float64, np.float32]:
            return x.float()
        else:
            raise NotImplementedError("Type: ",org_dtype)
    else:
        return to_pytorch_tensor(np.array(x))


def round_tensor(x):
    '''
    A helper function to round tensors by either
        - Regular rounding
        - Flooring
        - Keeping them in float for debug
    ''' 
    if ROUNDING == 'round':
        return x.round()
    elif ROUNDING == 'floor':
        return x.int().float()
    elif ROUNDING == 'debug': #No rounding in debug mode
        return x
    else:
        raise NotImplementedError
