from ocg_dataset import OcgDataset
from util.ncconv.experimental.ocg_dataset import MaskedDataError, ExtentError,\
    SubOcgDataset
from warnings import warn


def f_put(out,arg):
    out.append(f(arg))

## this is the function to map
def f(arg):
    from util.ncconv.experimental.ocg_dataset import OcgDataset
    
    uri = arg[0]
    ocg_opts = arg[1]
    var_name = arg[2]
    polygon = arg[3]
    time_range = arg[4]
    level_range = arg[5]
    clip = arg[6]
    union = arg[7]
    subpoly_proc = arg[8]
    
    ocg_dataset = OcgDataset(uri,**ocg_opts)
    if subpoly_proc <= 1:
        sub = ocg_dataset.subset(var_name,polygon,time_range,level_range)
        if clip: sub.clip(polygon)
        if union: sub.union()
    else:
        subset_opts = dict(time_range=time_range,
                           polygon=polygon)
        subs = ocg_dataset.mapped_subset(var_name,
                                         max_proc=subpoly_proc,
                                         subset_opts=subset_opts)
        subs = ocg_dataset.parallel_process_subsets(subs,
                                                    clip=clip,
                                                    union=union,
                                                    polygon=polygon)
        sub = ocg_dataset.combine_subsets(subs,union=union)
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
    
    if in_parallel is True and allow_empty is True:
        raise(NotImplementedError('in parallel empty intersections not ready.'))
    
    ## make the sure the polygon object is iterable
    if polygons is None:
        if clip:
            raise(ValueError('If clip is requested, polygon boundaries must be passed.'))
        polygons = [None]
#    else:
#        ## ensure polygon list is iterable and the correct format
#        if type(polygons) not in [list,tuple]:
#            polygons = [polygons]
    
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
        args = []
        for polygon in polygons:
            arg = [uri,ocg_opts,var_name,polygon,time_range,level_range,clip,union,subpoly_proc]
            args.append(arg)
        ## process list
        processes = [mp.Process(target=f_put,args=(out,arg)) for arg in args]
        ## execute and manage processes
        for ii,process in enumerate(processes,start=1):
            process.start()
#            print('started')
            if ii >= poly_proc:
                while sum([p.is_alive() for p in processes]) >= poly_proc:
                    pass
        for p in processes:
            p.join() 
        ## extract data
        subs = [sub for sub in out]
    else:
        subs = []
        ## loop through each polygon
        ocg_dataset = OcgDataset(uri,**ocg_opts)
        for polygon in polygons:
            try:
                sub = ocg_dataset.subset(var_name,polygon,time_range,level_range)
                if clip: sub.clip(polygon)
                if union: sub.union()
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
        subs = [SubOcgDataset([],[],[],[])]
    elif not allow_empty and len(subs) == 0:
        raise(ValueError('empty overlays are not allowed but no SubOcgDataset objects captured.'))
    ## merge subsets
    for ii,sub in enumerate(subs):
        ## keep the first one separate to merge against
        if ii == 0:
            base = sub
        else:
            base = base.merge(sub)
    
    return(base)