import itertools
from multiprocessing import Pool
from ocgis.api.interp.iocg.dataset.dataset import OcgDataset
from ocgis.calc.engine import OcgCalculationEngine
from ocgis.meta.interface.interface import SpatialInterfacePolygon
from ocgis.spatial.clip import clip
from ocgis.spatial.union import union
import copy


class SubsetOperation(object):
    '''Spatial and temporal subsetting plus calculation. Optional parallel
    extraction mode.
    
    desc :: dict :: Operational arguments for the interpreter to execute.
    serial=True :: bool :: Set to False to run in parallel.
    nprocs=1 :: int :: Number of processes to use when executing parallel
        operations.
    '''
    
    def __init__(self,ops,serial=True,nprocs=1):
        self.ops = ops
        self.serial = serial
        self.nprocs = nprocs
        
        ## update meta entries with OcgDataset objects checking for
        ## duplicate URIs. if the URI is the same, there is not reason to build
        ## the interface objects again.
        uri_map = {}
        for meta in self.ops.meta:
            key = '+++'.join(meta['uri'])
            if key in uri_map:
                ods = uri_map[key]
            else:
                ods = OcgDataset(meta['uri'],interface_overload=self.ops.interface)
                uri_map.update({key:ods})
            meta.update({'ocg_dataset':ods})

        ## create the calculation engine
        if self.ops.calc is None:
            self.cengine = None
        else:
            self.cengine = OcgCalculationEngine(self.ops.calc_grouping,
                                           ods.i.temporal.time.value,
                                           self.ops.calc,
                                           raw=self.ops.calc_raw,
                                           agg=self.ops.aggregate,
                                           time_range=self.ops.time_range)
            
        ## check for snippet request in the operations dictionary. if there is
        ## on, the time range should be set in the operations dictionary.
        if self.ops.snippet is True:
            ## only select the first level
            self.ops.level_range = [1]
            ## case of no calculation request
            if self.cengine is None:
                ref = self.ops.meta[0]['ocg_dataset'].i.temporal.time.value
                self.ops.time_range = [ref[0],ref[0]]
            ## case of a calculation. will need to select data based on temporal
            ## group.
            else:
                ## subset the calc engine time groups
                self.cengine.dgroups = [self.cengine.dgroups[0]]
                for key,value in self.cengine.dtime.iteritems():
                    self.cengine.dtime[key] = [value[0]]
                ## modify the time range to only pull a single group
                sub_time = self.cengine.timevec[self.cengine.dgroups[0]]
                self.ops.time_range = [sub_time.min(),sub_time.max()]
            
        ## set the spatial_interface
        self.spatial_interface = ods.i.spatial
        
    def __iter__(self):
        '''Return OcgCollection objects from the cache or directly from
        source data.
        
        yields
        
        OcgCollection'''
        
        ## simple iterator for serial operations
        if self.serial:
            it = itertools.imap(get_collection,self._iter_proc_args_())
        ## use a multiprocessing pool returning unordered geometries
        ## for the parallel case
        else:
            pool = Pool(processes=self.nprocs)
            it = pool.imap_unordered(get_collection,
                                     self._iter_proc_args_())
        ## the iterator return from the Pool requires calling its 'next'
        ## method and catching the StopIteration exception
        while True:
            try:
                yld = it.next()
                yield(yld)
            except StopIteration:
                break
        
    def _iter_proc_args_(self):
        '''Generate arguments for the extraction function.
        
        yields
        
        SubsetOperation
        geom_dict :: dict
        '''
        
#        ## these functions are for pickling problems of the OcgSpatialReference
#        ## which is a Swig object not consumable by pickle routines.
#        def _remove_projection_():
#            projection = copy(self.ops.meta'][0]['ocg_dataset.i.projection)
#            for meta in self.ops.meta:
#                meta['ocg_dataset'].i.projection = None
#            return(projection)
#        def _add_projection_(projection):
#            for meta in self.ops.meta:
#                meta['ocg_dataset'].i.projection = projection
        
#        try:
        ## copy for the iterator to avoid pickling the cache
        so_copy = copy.copy(self)
        for geom_dict in self.ops.geom:
#            projection = _remove_projection_()
#            try:
            yield(so_copy,geom_dict)
#            finally:
#                _add_projection_(projection)
#        except TypeError:
#            yield(self,{'id':1,'geom':None})
            
def get_collection((so,geom_dict)):
    '''Execute requested operations.
    
    so :: SubsetOperation
    geom_dict :: dict :: Geometry dictionary with the following structure:
        {'id':int,'geom':Shapely Polygon or MultiPolygon}
        
    returns
    
    OcgCollection
    geom_dict :: dict'''
    
    ## using the OcgDataset objects built in the SubsetOperation constructor
    ## do the spatial and temporal subsetting.
    return_collection=True
    for ii,meta in enumerate(so.ops.meta,start=1):
        ## collection are always returned but only the first one is needed.
        subset_return = \
          meta['ocg_dataset'].subset(meta['variable'],
                            polygon=geom_dict['geom'],
                            time_range=so.ops.time_range,
                            level_range=so.ops.level_range,
                            return_collection=return_collection)
        try:
            coll,ocg_variable = subset_return
            coll.geom_dict = geom_dict
        except TypeError:
            ocg_variable = subset_return
            
        ocg_variable.vid = ii
        
        if ii == 1:
            return_collection = False
            ## needed for time referencing during conversion.
            coll.cengine = so.cengine
        ## add the variable to the collection
        coll.add_variable(ocg_variable)
                    
    ## clipping operation.
    if so.ops.spatial_operation == 'clip':
        if isinstance(so.spatial_interface,SpatialInterfacePolygon):
            coll = clip(coll,geom_dict['geom'])
    ## data aggregation.
    if so.ops.aggregate:
        if isinstance(so.spatial_interface,SpatialInterfacePolygon):
            coll = union(geom_dict['id'],coll)
        else:
            raise(NotImplementedError)
#            coll['geom'] = np.array([[geom_dict['geom']]])
#            coll['geom_mask'] = np.array([[False]])
#            coll['gid'] = np.ma.array([[geom_dict['id']]],
#                                      mask=[[False]],dtype=int)
#            coll['value_agg'] = OrderedDict()
#            for key,value in coll['value'].iteritems():
#                coll['value_agg'].update({key:union_sum(coll['weights'],
#                                                        value)})

    ## do the requested calculations.
    if so.cengine is not None:
        coll = so.cengine.execute(coll)
    ## conversion of groups.
    if so.ops.output_grouping is not None:
        raise(NotImplementedError)
    else:
        return(coll,geom_dict)
    