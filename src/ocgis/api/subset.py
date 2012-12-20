import itertools
from multiprocessing import Pool
from ocgis.calc.engine import OcgCalculationEngine
from ocgis.interface.interface import SpatialInterfacePolygon
import copy
from ocgis.api.dataset.dataset import OcgDataset
from ocgis.util.spatial.wrap import unwrap_geoms, wrap_coll
from ocgis.api.dataset.collection.collection import OcgCollection


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
        ## duplicate URIs. if the URI is the same, there is no reason to build
        ## the interface objects again.
        uri_map = {}
        for dataset in self.ops.dataset:
            key = dataset['variable'] + '_' + dataset['uri']
            if key in uri_map:
                ods = uri_map[key]
            else:
                ods = OcgDataset(dataset,interface_overload=self.ops.interface)
                uri_map.update({key:ods})
            dataset.update({'ocg_dataset':ods})
            
        ## wrap the geometry dictionary if needed
        arch = self.ops.dataset[0]['ocg_dataset']
        if arch.i.spatial.is_360 and self.ops._get_object_('geom').is_empty is False:
            unwrap_geoms(self.ops.geom,arch.i.spatial.pm)

        ## create the calculation engine
        if self.ops.calc is None:
            self.cengine = None
        else:
            self.cengine = OcgCalculationEngine(self.ops.calc_grouping,
                                           self.ops.calc,
                                           raw=self.ops.calc_raw,
                                           agg=self.ops.aggregate
                                           )
            
        ## check for snippet request in the operations dictionary. if there is
        ## on, the time range should be set in the operations dictionary.
        if self.ops.snippet is True:
            raise(NotImplementedError)
            ## only select the first level
            self.ops.level_range = [1]
            ## case of no calculation request
            if self.cengine is None:
                ref = self.ops.dataset[0]['ocg_dataset'].i.temporal.value
                self.ops.time_range = [ref[0],ref[0]]
            else:
                import ipdb;ipdb.set_trace()
            
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
        
        ## copy for the iterator to avoid pickling the cache
        so_copy = copy.copy(self)
        for geom_dict in self.ops.geom:
            yield(so_copy,geom_dict)
            
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
    
    coll = OcgCollection(ugeom=geom_dict)
    for dataset in so.ops:
        ref = dataset['dataset']['ocg_dataset']
        ocg_variable = ref.subset(
                            polygon=geom_dict['geom'],
                            time_range=dataset['time_range'],
                            level_range=dataset['level_range'],
                            allow_empty=so.ops.allow_empty)
        ocg_variable._i = ref.i
        coll.projection = ref.i.spatial.projection
        coll.add_variable(ocg_variable)

    ## skip other operations if the dataset is empty
    if coll.is_empty:
        return(coll)
     
    ## clipping operation
    if so.ops.spatial_operation == 'clip':
        if isinstance(so.spatial_interface,SpatialInterfacePolygon):
            coll.clip(geom_dict['geom'])
            
    ## data aggregation.
    if so.ops.aggregate:
        coll.aggregate(new_id=coll.ugeom['ugid'])
        
    ## if it is a vector output, wrap the data (if requested).
    arch = so.ops.dataset[0]['ocg_dataset']
    if arch.i.spatial.is_360 and so.ops.output_format != 'nc' and so.ops.vector_wrap:
        wrap_coll(coll)

    ## do the requested calculations.
    if so.cengine is not None:
        so.cengine.execute(coll)
    ## conversion of groups.
    if so.ops.output_grouping is not None:
        raise(NotImplementedError)
    else:
        return(coll)
