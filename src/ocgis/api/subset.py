import itertools
from multiprocessing import Pool
from ocgis.calc.engine import OcgCalculationEngine
from ocgis.util.spatial.wrap import unwrap_geoms, wrap_coll
from copy import deepcopy
from ocgis import env
from ocgis.interface.shp import ShpDataset
from ocgis.api.collection import RawCollection
from ocgis.interface.nc.dataset import NcDataset


class SubsetOperation(object):
    '''Spatial and temporal subsetting plus calculation. Optional parallel
    extraction mode.
    
    :param ops: An `~ocgis.OcgOperations` object.
    :type ops: `~ocgis.OcgOperations`
    :param serial: Set to `False` to run in parallel.
    :type serial: bool
    :param ncprocs: Number of processes to use when executing in parallel.
    :type ncprocs: int
    '''
    
    def __init__(self,ops,serial=True,nprocs=1):
        self.ops = ops
        self.serial = serial
        self.nprocs = nprocs
        
#        ## construct OcgDataset objects
#        for request_dataset in env.ops.dataset:
#            import ipdb;ipdb.set_trace()
#            iface = request_dataset.interface.copy()
#            iface.update({'s_abstraction':self.ops.abstraction})
#            ods = OcgDataset(request_dataset,
#                             interface_overload=iface)
#            
#            request_dataset.ocg_dataset = ods
        
        ## move to operations ##################################################
        
        
#        ## determine if dimensions are equivalent.
#        mappers = [EqualSpatialDimensionMapper,EqualTemporalDimensionMapper,EqualLevelDimensionMapper]
#        for mapper in mappers:
#            mapper(self.ops.dataset)
#            
#        ## ensure they are all the same type of spatial interfaces. raise an error
#        ## otherwise.
#        types = [type(ods['ocg_dataset'].i.spatial) for ods in self.ops.dataset]
#        if all([t == SpatialInterfacePolygon for t in types]):
#            self.itype = SpatialInterfacePolygon
#        elif all([t == SpatialInterfacePoint for t in types]):
#            self.itype = SpatialInterfacePoint
#        else:
#            raise(ValueError('Input datasets must have same geometry types. Perhaps overload "s_abstraction"?'))
#        
#        ## ensure all data has the same projection
#        projections = [ods.ocg_dataset.i.spatial.projection.sr.ExportToProj4() for ods in self.ops.dataset]
#        projection_test = [projections[0] == ii for ii in projections]
#        if not all(projection_test):
#            raise(ValueError('Input datasets must share a common projection.'))
#        
#        ## if the target dataset(s) has a different projection than WGS84, the
#        ## selection geometries will need to be projected.
#        if not self.ops._get_object_('geom').is_empty:
#            if self.ops.dataset[0].ocg_dataset.i.spatial.projection != self.ops.geom.ocgis.sr.ExportToProj4():
#                new_geom = self.ops.geom.ocgis.get_projected(self.ops.dataset[0].ocg_dataset.i.spatial.projection.sr)
#                self.ops.geom = new_geom

        ## create the calculation engine
        if self.ops.calc is None:
            self.cengine = None
        else:
            self.cengine = OcgCalculationEngine(self.ops.calc_grouping,
                                           self.ops.calc,
                                           raw=self.ops.calc_raw,
                                           agg=self.ops.aggregate)
            
#        ## check for snippet request in the operations dictionary. if there is
#        ## on, the time range should be set in the operations dictionary.
#        if self.ops.snippet is True:
#            for dataset in self.ops.dataset:
#                dataset.level_range = [1,1]
#                ref = dataset.ocg_dataset.i.temporal
#                if self.cengine is None or (self.cengine is not None and self.cengine.grouping is None):
#                    dataset.time_range = [ref.value[0],ref.value[0]]
#                else:
#                    tgdim = TemporalDimension(ref.tid,ref.value,
#                                              bounds=ref.bounds).\
#                                              group(self.cengine.grouping)
#                    times = ref.value[tgdim.dgroups[0]]
#                    dataset.time_range = [times.min(),times.max()]
        
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
            raise(NotImplementedError)
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
        
#        ## copy for the iterator to avoid pickling the cache
#        so_copy = copy.copy(self)
        if self.ops.geom is None:
            yield(self,None)
        elif isinstance(self.ops.geom,ShpDataset):
            for idx in range(self.ops.geom.spatial.geom):
                yield(env.geom[idx])
        else:
            yield(self,self.ops.geom)
            
def get_collection((so,geom)):
    '''Execute requested operations.
    
    so :: SubsetOperation
    geom_dict :: dict :: Geometry dictionary with the following structure:
        {'id':int,'geom':Shapely Polygon or MultiPolygon}
        
    returns
    
    OcgCollection
    geom_dict :: dict'''
    
#    tdk
#    import pickle
##    import ipdb;ipdb.set_trace()
##    ref = so.ops.dataset._s['rhsmax'].ocg_dataset.i.temporal
##    import ipdb;ipdb.set_trace()
##    for k,v in so.ops.dataset._s.iteritems():
##        v.ocg_dataset = None
##        import ipdb;ipdb.set_trace()
#    with open('/tmp/out.pkl','w') as f:
#        pickle.dump(so.ops,f)
##        pickle.dump(so.ops.dataset._s['rhsmax'].ocg_dataset,f)
##        pickle.dump(so.ops.dataset._s['rhsmax'],f)
##        pickle.dump(ref,f)
#    with open('/tmp/out.pkl','r') as f:
#        ops = pickle.load(f)
#    import ipdb;ipdb.set_trace()
#    /tdk
    
#    if env.VERBOSE:
#        print('processing geometry with UGID {0}...'.format(geom_dict['ugid']))
    
    ## using the OcgDataset objects built in the SubsetOperation constructor
    ## do the spatial and temporal subsetting.
    
    coll = RawCollection(ugeom=geom)
#    ## store geoms for later clipping. needed because some may be wrapped while
#    ## others unwrapped.
#    geom_copys = []
    for request_dataset in so.ops.dataset:
#        ## use a copy of the geometry dictionary, since it may be modified
#        geom_copy = deepcopy(geom_dict)
#        geom_copys.append(geom_copy)
#        
#        ref = dataset.ocg_dataset
        ## wrap the geometry dictionary if needed
        ods = NcDataset(request_dataset=request_dataset)
        if so.ops.geom is not None and ods.spatial.is_360:
#        if ref.i.spatial.is_360 and so.ops._get_object_('geom').is_empty is False:
            unwrap_geoms([geom_copy],ref.i.spatial.pm)
        if so.ops.slice_row is not None or so.ops.slice_column is not None:
            raise(NotImplementedError)
        ## perform the data subset
        ods = ods.get_subset(
                            spatial_operation=so.ops.spatial_operation,
                            polygon=geom,
                            temporal=request_dataset.time_range,
                            level=request_dataset.level_range,
                            allow_empty=so.ops.allow_empty,)
        if so.ops.spatial_operation == 'clip' and geom is not None:
            ods = ods.spatial.vector.clip()
        if so.ops.aggregate:
            try:
                new_geom_id = geom.uid
            except AttributeError:
                new_geom_id = 1
            ods.aggregate(new_geom_id=new_geom_id)
        if not env.OPTIMIZE_FOR_CALC:
            if ods.spatial.is_360 and so.ops.output_format != 'nc' and so.ops.vector_wrap:
                ods.wrap()
        if so.cengine is not None:
            so.cengine.execute(ods)
#                            slice_row=so.ops.slice_row,
#                            slice_column=so.ops.slice_column)
#        ## tell the keyed iterator if this should be used for identifiers.
#        ocg_variable._use_for_id = dataset._use_for_id
#        ## update the variable's alias
#        ocg_variable.alias = dataset.alias
#        ## maintain the interface for use by nc converter
#        ## TODO: i don't think this is required by other converters
#        ocg_variable._i = ref.i
#        ## TODO: projection is set multiple times. what happens with multiple
#        ## projections?
#        coll.projection = ref.i.spatial.projection
        coll.variables.update({request_dataset.alias:ods})
#        coll.add_variable(ocg_variable)

#    ## skip other operations if the dataset is empty
#    if len(coll.variables) == 0:
#        return(coll)
    
#    ## clipping operation
#    if so.ops.spatial_operation == 'clip':
#        if so.itype == SpatialInterfacePolygon:
#            for geom_copy,var in itertools.izip(geom_copys,
#                                                coll.variables.itervalues()):
#                var.clip(geom_copy['geom'])
            
#    ## data aggregation.
#    if so.ops.aggregate:
#        coll.aggregate(new_id=coll.ugeom['ugid'])
        
    ## if it is a vector output, wrap the data (if requested) but not if optimizations
    ## for calculations.
    ## TODO: every variable may not need to be wrapped
#    if not env.OPTIMIZE_FOR_CALC:
#        for dataset in so.ops.dataset:
#            ref = dataset['ocg_dataset']
#            if ref.i.spatial.is_360 and so.ops.output_format != 'nc' and so.ops.vector_wrap:
#                wrap_coll(coll)
#                break

    ## do the requested calculations.
    
    
    #tdk
#    import pickle
#    with open('/tmp/coll.pkl','w') as f:
#        pickle.dump(coll,f)
#    with open('/tmp/coll.pkl','r') as f:
#        coll = pickle.load(f)
#    import ipdb;ipdb.set_trace()
    #/tdk
    
    if env.VERBOSE:
        print(' complete.'.format(geom_dict['ugid']))
        
    ## conversion of groups.
    if so.ops.output_grouping is not None:
        raise(NotImplementedError)
    else:
        return(coll)
