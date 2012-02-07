from util.ncconv.experimental.ocg_dataset import *
from warnings import warn
import numpy as np
import multiprocessing as mp
from util.ncconv.experimental.pmanager import ProcessManager


class SpatialOperationProcess(mp.Process):
    
    def __init__(self,*args):
        attrs = ['out','uri','ocg_opts','var_name','polygon','time_range',
                 'level_range','clip','union','subpoly_proc','allow_empty',
                 'max_retries','debug']
        for arg,attr in zip(args,attrs):
            setattr(self,attr,arg)
        
        ## we need a gid
        assert self.polygon['gid'] is not None, 'polygon GIDs may not be NoneType'  
        ## make sure at least one process is relegated to sub-polygon processing
        if self.subpoly_proc < 1:
            self.subpoly_proc = 1
        ## keep track of the attempts
        self._curr_try = 1
        
        super(SpatialOperationProcess,self).__init__()
            
    def run(self):
        try:
            ## split out the polygon arguments
            gid = self.polygon['gid']
            poly = self.polygon['geom']
            ## there is potential to return empty data
            sub = None
            ## initialize the dataset
            ocg_dataset = OcgDataset(self.uri,**self.ocg_opts)
            subset_opts = dict(time_range=self.time_range,
                               polygon=poly)
            try:
                subs = ocg_dataset.split_subset(self.var_name,
                                                max_proc=self.subpoly_proc,
                                                subset_opts=subset_opts)
                subs = ocg_dataset.parallel_process_subsets(subs,
                                                            clip=self.clip,
                                                            union=self.union,
                                                            polygon=poly,
                                                            debug=self.debug)
                sub = ocg_dataset.combine_subsets(subs,union=self.union)
            except (MaskedDataError,ExtentError):
                if not self.allow_empty:
                    raise
            if self.union is True:
                sub.gid = np.array([gid])
            self.out.append(sub)
        except RuntimeError:
            ## if the current try is less than the max_retries, try again. this
            ## is to attempt to overcome RuntimeErrors...
            if self._curr_try < self.max_retries:
                self._curr_try += 1
                self.run()
            else:
                raise
        except:
            raise


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
    ## if there are no assigned gids, assign some
    if polygons is not None:
        if polygons[0]['gid'] is None:
            for ii,poly in enumerate(polygons,start=1):
                poly['gid'] = ii
    ## we do not allow a union operation coupled with an intersects operation.
    ## it may return overlapping geometries.
    if not clip and union:
        raise(ValueError('Intersects + Union is not an allowed operation.'))
    
    ## make the sure the polygon object is iterable
    if not polygons:
        if clip:
            raise(ValueError('If clip is requested, polygon boundaries must be passed.'))
        polygons = [{'gid':1,'geom':None}]
    
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
    
    ## generate the processes
    
    debug = False
    max_retries = 5
    in_parallel = not debug
    
    processes = []
    for polygon in polygons:
        processes.append(
         SpatialOperationProcess(out,uri,ocg_opts,var_name,polygon,
                                 time_range,level_range,clip,union,
                                 subpoly_proc,allow_empty,max_retries,debug))
    if in_parallel:
        pmanager = ProcessManager(processes,poly_proc)
        pmanager.run()
    else:
        for process in processes:
            process.run()

    subs = list(out)

    ## it is possible to have no subdatasets if empty intersections are
    ## permitted.
    if allow_empty and len(subs) == 0:
        warn('all attempted geometric operations returned empty.')
        subs = [SubOcgDataset([],[],[])]
    elif not allow_empty and len(subs) == 0:
        raise(ValueError('empty overlays are not allowed but no SubOcgDataset objects captured.'))
    ## merge subsets. subset at this point represent polygons from the request.
    ## we need to carefully manage the value_sets if there is a union operation.
    value_set_coll = {}
    for ii,sub in enumerate(subs):
        ## if this is a union operation, collect the values.
        if union:
            value_set_coll.update({int(sub.gid):sub.value_set})
        ## keep the first one separate to merge against
        if ii == 0:
            base = sub
        else:
            base = base.merge(sub,union=union)
    ## store this collection for potential use by the statistics
    base.value_set = value_set_coll
    ## if the operation is intersects, there may be duplicate geometries. purge
    ## the dataset for those duplicate geometries and values.
    if clip == False:
        base.purge()
    return(base)