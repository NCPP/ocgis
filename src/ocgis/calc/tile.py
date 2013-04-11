import numpy as np
import itertools


def get_tile_schema(nrow,ncol,tdim,origin=0):
    ret = {}
    row_idx = np.arange(origin,nrow+tdim,step=tdim,dtype=int)
    col_idx = np.arange(origin,ncol+tdim,step=tdim,dtype=int)
    row_slices = get_slices(row_idx)
    col_slices = get_slices(col_idx)
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