from util.ncconv.experimental.ocg_dataset import *
from warnings import warn
import numpy as np


def f_put(out,exc,arg):
    try:
        ret = f(arg)
    except RuntimeError:
        ## try again if a runtime error is received
        try:
            ret = f(arg)
        except RuntimeError:
            exc.append('RuntimeError')
    except:
        raise
        exc.append('unhandled fatal error')
    ## accounting for the empty intersection...
    if ret is not None:
        out.append(ret)

## this is the function to map
def f(arg):
#    from util.ncconv.experimental.ocg_dataset import OcgDataset
    
    uri = arg[0]
    ocg_opts = arg[1]
    var_name = arg[2]
    polygon = arg[3]
    time_range = arg[4]
    level_range = arg[5]
    clip = arg[6]
    union = arg[7]
    subpoly_proc = arg[8]
    allow_empty = arg[9]
    
    ## split out the polygon arguments
    gid = polygon['gid']
    poly = polygon['geom']
    ## there is potential to return empty data
    sub = None
    ocg_dataset = OcgDataset(uri,**ocg_opts)
    if subpoly_proc <= 1:
        try:
            sub = ocg_dataset.subset(var_name,poly,time_range,level_range)
            if clip: sub.clip(poly)
            if union: sub.union()
        except (MaskedDataError,ExtentError):
            if not allow_empty:
                raise
    else:
        subset_opts = dict(time_range=time_range,
                           polygon=poly)
        try:
            subs = ocg_dataset.split_subset(var_name,
                                             max_proc=subpoly_proc,
                                             subset_opts=subset_opts)
            subs = ocg_dataset.parallel_process_subsets(subs,
                                                        clip=clip,
                                                        union=union,
                                                        polygon=poly)
            sub = ocg_dataset.combine_subsets(subs,union=union)
        except (MaskedDataError,ExtentError):
            if not allow_empty:
                raise
    if gid is not None and union is True:
        sub.cell_id = np.array([gid])
    return(sub)


def multipolygon_operation(uri,
                           var_name,
                           ocg_opts = {},
                           polygons=None,
                           time_range=None,
                           level_range=None,
                           clip=False,
                           union=False,
                           in_parallel=False,
                           max_proc=4,
                           max_proc_per_poly=4,
                           allow_empty=True):

    ## we do not allow a union operation coupled with an intersects operation.
    ## it may return overlapping geometries.
    if not clip and union:
        raise(ValueError('Intersects + Union is not an allowed operation.'))
    
    ## make the sure the polygon object is iterable
    if not polygons:
        if clip:
            raise(ValueError('If clip is requested, polygon boundaries must be passed.'))
        polygons = [{'gid':None,'geom':None}]
    
    ## switch depending on parallel request
    if in_parallel:
        import multiprocessing as mp
        
        ## construct the pool. first, calculate number of remaining processes
        ## to allocate to sub-polygon processing.
        if len(polygons) <= max_proc:
            poly_proc = len(polygons)
            subpoly_proc = max_proc - len(polygons)
        else:
            poly_proc = max_proc
            subpoly_proc = 0
        ## distribute the remaining processes to the polygons
        subpoly_proc = subpoly_proc / len(polygons)
        ## ensure we don't exceed the maximum processes per polygon
        if subpoly_proc > max_proc_per_poly:
            subpoly_proc = max_proc_per_poly
        ## assemble the argument list
        out = mp.Manager().list()
        exc = mp.Manager().list()
        args = []
        for polygon in polygons:
            arg = [uri,ocg_opts,var_name,polygon,time_range,level_range,clip,union,subpoly_proc,allow_empty]
            args.append(arg)
        ## process list
        processes = [mp.Process(target=f_put,args=(out,exc,arg)) for arg in args]
        ## execute and manage processes
        for ii,process in enumerate(processes,start=1):
            process.start()
#            print('started')
            if ii >= poly_proc:
                while sum([p.is_alive() for p in processes]) >= poly_proc:
                    pass
        for p in processes:
            p.join()
        ## check for exceptions
        if len(exc) > 0:
            raise(RuntimeError('Child processes raised unhandled exceptions: {0}'.format(exc)))
        ## extract data
        subs = [sub for sub in out]
    else:
        subs = []
        ## loop through each polygon
        ocg_dataset = OcgDataset(uri,**ocg_opts)
        for polygon in polygons:
            try:
                sub = ocg_dataset.subset(var_name,polygon['geom'],time_range,level_range)
                if clip: sub.clip(polygon['geom'])
                if union:
                    sub.union()
                    ## apply the gid if passed. only when the geometries are
                    ## unioned is the user gid relevant.
                    gid = polygon.get('gid')
                    if gid is not None and union is True:
                        sub.cell_id = np.array([gid])
                ## hold for later merging
                subs.append(sub)
            except (MaskedDataError,ExtentError):
                if allow_empty:
                    warn('empty overlay encountered.')
                    continue
                else:
                    raise
    ## it is possible to to have no subdatasets if empty intersections are
    ## permitted.
    if allow_empty and len(subs) == 0:
        warn('all attempted geometric operations returned empty.')
        subs = [SubOcgDataset([],[],[])]
    elif not allow_empty and len(subs) == 0:
        raise(ValueError('empty overlays are not allowed but no SubOcgDataset objects captured.'))
    ## merge subsets
    for ii,sub in enumerate(subs):
        ## keep the first one separate to merge against
        if ii == 0:
            base = sub
        else:
            base = base.merge(sub)
    ## if the operation is intersects, there may be duplicate geometries. purge
    ## the dataset for those duplicate geometries and values.
    if clip == False:
        base.purge()
    return(base)