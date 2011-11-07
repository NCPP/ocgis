from util.ncconv.experimental.ocg_dataset import OcgDataset, SubOcgDataset


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
    
    ocg_dataset = OcgDataset(uri,**ocg_opts)
    sub = ocg_dataset.subset(var_name,polygon,time_range,level_range)
    if clip: sub.clip(polygon)
    if union: sub.union()
    
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
                           max_proc_for_polys=4,
                           max_proc_per_poly=4):
    
    ## make the sure the polygon object is iterable
    if polygons is None:
        if clip:
            raise(ValueError('If clip is requested, polygon boundaries must be passed.'))
        polygons = [None]
    else:
        ## ensure polygon list is iterable
        if type(polygons) not in [list,tuple]:
            polygons = [polygons]
    
    ## switch depending on parallel request
    if in_parallel:
        import multiprocessing as mp
        
        ## assemble the argument list
        args = []
        for polygon in polygons:
            arg = [uri,ocg_opts,var_name,polygon,time_range,level_range,clip,union]
            args.append(arg)
        ## construct the pool
        pool = mp.Pool(max_proc_for_polys)
        subs = pool.map(f,args)
        
    else:
        subs = []
        ## loop through each polygon
        for polygon in polygons:
            ocg_dataset = OcgDataset(uri,**ocg_opts)
            sub = ocg_dataset.subset(var_name,polygon,time_range,level_range)
            if clip: sub.clip(polygon)
            if union: sub.union()
            subs.append(sub)
    ## merge subsets
    for ii,sub in enumerate(subs):
        ## keep the first one separate to merge against
        if ii == 0:
            base = sub
        else:
            base = base.merge(sub)
    
    return(base)