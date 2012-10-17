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
    
    def __init__(self,desc,serial=True,nprocs=1):
        self.desc = desc
        self.serial = serial
        self.nprocs = nprocs
        
        ## update meta entries with OcgDataset objects checking for
        ## duplicate URIs. if the URI is the same, there is not reason to build
        ## the interface objects again.
        uri_map = {}
        for meta in self.desc['meta']:
            key = '+++'.join(meta['uri'])
            if key in uri_map:
                ods = uri_map[key]
            else:
                ods = OcgDataset(meta['uri'])
                uri_map.update({key:ods})
            meta.update({'ocg_dataset':ods})

        ## check for snippet request in the operations dictionary. if there is
        ## on, the time range should be set in the operations dictionary.
        request_snippet = self.desc.get('request_snippet')
        if request_snippet is True:
            ref = self.desc['meta'][0]['ocg_dataset'].i.temporal.time.value
            self.desc['time_range'] = [ref[0],ref[0]]
        ## create the calculation engine
        if self.desc['calc'] is None:
            self.cengine = None
        else:
            self.cengine = OcgCalculationEngine(self.desc['calc_grouping'],
                                           ods.i.temporal.time.value,
                                           self.desc['calc'],
                                           raw=self.desc['calc_raw'],
                                           agg=self.desc['aggregate'],
                                           time_range=self.desc['time_range'],
                                           mode=self.desc['mode'])
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
#            projection = copy(self.desc['meta'][0]['ocg_dataset'].i.projection)
#            for meta in self.desc['meta']:
#                meta['ocg_dataset'].i.projection = None
#            return(projection)
#        def _add_projection_(projection):
#            for meta in self.desc['meta']:
#                meta['ocg_dataset'].i.projection = projection
        
#        try:
        ## copy for the iterator to avoid pickling the cache
        so_copy = copy.copy(self)
        for geom_dict in self.desc['geom']:
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
    for ii,meta in enumerate(so.desc['meta'],start=1):
        ## collection are always returned but only the first one is needed.
        subset_return = \
          meta['ocg_dataset'].subset(meta['variable'],
                            polygon=geom_dict['geom'],
                            time_range=so.desc['time_range'],
                            level_range=so.desc['level_range'],
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
    if so.desc['spatial_operation'] == 'clip':
        if isinstance(so.spatial_interface,SpatialInterfacePolygon):
            coll = clip(coll,geom_dict['geom'])
    ## data aggregation.
    if so.desc['aggregate']:
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
    if so.desc['output_grouping'] is not None:
        raise(NotImplementedError)
    else:
        return(coll,geom_dict)
    