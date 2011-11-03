import itertools
import numpy as np
from shapely.geometry.polygon import Polygon
import pdb


def itr_array(a):
    "a -- 2-d ndarray"
    ix = a.shape[0]
    jx = a.shape[1]
    for ii,jj in itertools.product(range(ix),range(jx)):
        yield ii,jj
        
def contains(grid,lower,upper,res=0.0):
    
    ## small ranges on coordinates requires snapping to closest coordinate
    ## to ensure values are selected through logical comparison.
    ugrid = np.unique(grid)
    lower = ugrid[np.argmin(np.abs(ugrid-(lower-0.5*res)))]
    upper = ugrid[np.argmin(np.abs(ugrid-(upper+0.5*res)))]
    
    s1 = grid >= lower
    s2 = grid <= upper
    ret = s1*s2

    return(ret)

def approx_resolution(vec):
    """
    >>> vec = [1,2,3,4,5]
    >>> approx_resolution(vec)
    1.0
    """
    diff = []
    for i in range(len(vec)):
        curr = vec[i]
        try:
            next = vec[i+1]
            diff.append(abs(curr-next))
        except IndexError:
            break
    return(np.mean(diff))

def make_poly(rtup,ctup):
    """
    rtup = (row min, row max)
    ctup = (col min, col max)
    """
    return Polygon(((ctup[0],rtup[0]),
                    (ctup[0],rtup[1]),
                    (ctup[1],rtup[1]),
                    (ctup[1],rtup[0])))
    
def sub_range(a):
    """
    >>> vec = np.array([2,5,9])
    >>> sub_range(vec)
    array([2, 3, 4, 5, 6, 7, 8, 9])
    """
    a = np.array(a)
    return(np.arange(a.min(),a.max()+1))

def keep(prep_igeom,igeom,target):
        if prep_igeom.intersects(target) and not target.touches(igeom):
            ret = True
        else:
            ret = False
        return(ret)

def union_sum(weight,value,normalize=True):
    ## renormalize the weights
    if normalize: weight = weight/weight.sum()
    ## this will hold the weighted data
    weighted = np.empty((value.shape[0],value.shape[1],1))
    ## next, weight and sum the data accordingly
    for dim_time in range(value.shape[0]):
        for dim_level in range(value.shape[1]):
            weighted[dim_time,dim_level,0] = np.sum(weight*value[dim_time,dim_level,:])
    return(weighted)
           
if __name__ == '__main__':
    import doctest
    doctest.testmod()