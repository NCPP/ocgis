import numpy as np
import itertools


def get_tile_indices(n,tdim,origin,nt):
    modulus = np.floor(nt)
    remainder = nt - modulus
    breaks = np.arange(origin,n,tdim,dtype=int)
    if remainder > 0:
        breaks = np.concatenate((breaks, [n]))
    return(breaks)

def get_tile_schema(nrow,ncol,tdim):
    row_origin = 0
    col_origin = 0
    tdim = float(tdim)
    nt_row = nrow/tdim
    nt_col = ncol/tdim
    ret = {}
    row_idx = get_tile_indices(nrow,tdim,row_origin,nt_row)
    col_idx = get_tile_indices(ncol,tdim,col_origin,nt_col)
    row_slices = get_slices(row_idx)
    col_slices = get_slices(col_idx)
    import ipdb;ipdb.set_trace()
    tile_id = 0
    for row,col in itertools.product(range(len(row_slices)),range(len(col_slices))):
        ret.update({tile_id:{'row':row_slices[row],'col':col_slices[col]}})
        tile_id += 1
    return(ret)

def get_slices(arr):
    ret = [None]*(arr.shape[0]-1)
    for idx in range(arr.shape[0]):
        try:
            start = arr[idx]
            stop = arr[idx+1]
            ret[idx] = [start,stop]
        except IndexError:
            break
    return(ret)