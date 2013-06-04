import itertools
from multiprocessing import Pool
from ocgis.calc.engine import OcgCalculationEngine
from ocgis import env
from ocgis.interface.shp import ShpDataset
from ocgis.api.collection import RawCollection
from ocgis.exc import EmptyData, ExtentError, MaskedDataError
from ocgis.interface.projection import WGS84
from ocgis.util.spatial.wrap import Wrapper
from copy import deepcopy
from ocgis.util.logging_ocgis import ocgis_lh
import logging


class SubsetOperation(object):
    
    def __init__(self,ops,serial=True,nprocs=1,validate=True):
        self.ops = ops
        self.serial = serial
        self.nprocs = nprocs
        
        subset_log = ocgis_lh.get_logger('subset')
        
        if validate:
            ocgis_lh('validating request datasets',subset_log,level=logging.DEBUG)
            ops.dataset.validate()

        ## create the calculation engine
        if self.ops.calc is None:
            self.cengine = None
        else:
            ocgis_lh('initializing calculation engine',subset_log,level=logging.DEBUG)
            self.cengine = OcgCalculationEngine(self.ops.calc_grouping,
                                           self.ops.calc,
                                           raw=self.ops.calc_raw,
                                           agg=self.ops.aggregate)
            
        ## check for snippet request in the operations dictionary. if there is
        ## on, the time range should be set in the operations dictionary.
        if self.ops.snippet is True:
            ##TODO: move snippet to iteration
            ocgis_lh('getting snippet bounds',subset_log)
            for rd in self.ops.dataset:
                rd.level_range = [1,1]
                ods = rd.ds
                ## load the first time slice if there is calculation or the 
                ## calculation does not use a temporal group.
                if self.cengine is None or (self.cengine is not None and self.cengine.grouping is None):
                    ##TODO: improve slicing to not load all time values
                    ods._load_slice.update({'T':slice(0,1)})
                ## snippet for the computation. this currently requires loading
                ## all the data for the time dimension into memory.
                ##TODO: more efficiently pull dates for monthly grouping (for
                ##example).
                else:
                    ods.temporal.set_grouping(self.cengine.grouping)
                    tgdim = ods.temporal.group
                    times = ods.temporal.value[tgdim.dgroups[0]]
                    rd.time_range = [times.min(),times.max()]
        
    def __iter__(self):
        ''':rtype: AbstractCollection'''
        
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
        ''':rtype: tuple'''
        
        subset_log = ocgis_lh.get_logger('subset')
        
        ## if there is no geometry, yield None.
        if self.ops.geom is None:
            ocgis_lh('returning entire spatial domain - no selection geometry',subset_log)
            yield(self,None,subset_log)
            
        ## iterator through geometries in the ShpDataset
        elif isinstance(self.ops.geom,ShpDataset):
            ocgis_lh('{0} geometry(s) to process'.format(len(self.ops.geom)),subset_log)
            for geom in self.ops.geom:
                yield(self,geom,subset_log)
                
        ## otherwise, the data is likely a GeometryDataset with a single value.
        ## just return it.
        else:
            ocgis_lh('1 geometry to process'.format(len(self.ops.geom)),subset_log)
            yield(self,self.ops.geom,subset_log)
            
def get_collection((so,geom,logger)):
    '''
    :type so: SubsetOperation
    :type geom: None, GeometryDataset, ShpDataset
    :rtype: AbstractCollection
    '''
    
    ## initialize the collection object to store the subsetted data.
    coll = RawCollection(ugeom=geom)
    ## perform the operations on each request dataset
    ocgis_lh('{0} request dataset(s) to process'.format(len(so.ops.dataset)),logger)
    ## reference the geometry ugid
    ugid = None if geom is None else geom.spatial.uid[0]
    for request_dataset in so.ops.dataset:
        ## reference the request dataset alias
        alias = request_dataset.alias
        ocgis_lh('processing',logger,level=logging.DEBUG,alias=alias,ugid=ugid)
        ## copy the geometry
        copy_geom = deepcopy(geom)
        ## reference the dataset object
        ods = request_dataset.ds
        ## return a slice or do the other operations
        if so.ops.slice is not None:
            ods = ods.__getitem__(so.ops.slice)
        ## other subsetting operations
        else:
            ## if a geometry is passed and the target dataset is 360 longitude,
            ## unwrap the passed geometry to match the spatial domain of the target
            ## dataset.
            if copy_geom is None:
                igeom = None
            else:
                ## check projections adjusting projection the selection geometry
                ## if necessary
                if type(ods.spatial.projection) != type(copy_geom.spatial.projection):
                    msg = 'projecting selection geometry to match input projection: {0} to {1}'
                    msg = msg.format(copy_geom.spatial.projection.__class__.__name__,
                                     ods.spatial.projection.__class__.__name__)
                    ocgis_lh(msg,logger,alias=alias,ugid=ugid)
                    copy_geom.project(ods.spatial.projection)
                else:
                    ocgis_lh('projections match',logger,alias=alias,ugid=ugid)
                ## unwrap the data if it is geographic and 360
                if type(ods.spatial.projection) == WGS84 and ods.spatial.is_360:
                    ocgis_lh('unwrapping selection geometry with axis={0}'.format(ods.spatial.pm),
                             logger,alias=alias,ugid=ugid)
                    w = Wrapper(axis=ods.spatial.pm)
                    copy_geom.spatial.geom[0] = w.unwrap(deepcopy(copy_geom.spatial.geom[0]))
                igeom = copy_geom.spatial.geom[0]
            ## perform the data subset
            try:
                ## pull the temporal subset which may be a range or region
                temporal = request_dataset.time_range or request_dataset.time_region
                ods = ods.get_subset(spatial_operation=so.ops.spatial_operation,
                                     igeom=igeom,
                                     temporal=temporal,
                                     level=request_dataset.level_range)
                ## aggregate the geometries and data if requested
                if so.ops.aggregate:
                    ocgis_lh('aggregating target geometries and area-weighting values',
                             logger,alias=alias,ugid=ugid)
                    ## the new geometry will have the same id as the passed
                    ## geometry. if it does not have one, simple give it a value
                    ## of 1 as it is the only geometry requested for subsetting.
                    try:
                        new_geom_id = copy_geom.spatial.uid[0]
                    except AttributeError:
                        new_geom_id = 1
                    ## do the aggregation in place.
                    clip_geom = None if copy_geom is None else copy_geom.spatial.geom[0]
                    ods.aggregate(new_geom_id=new_geom_id,
                                  clip_geom=clip_geom)
                ## wrap the returned data depending on the conditions of the
                ## operations.
                if not env.OPTIMIZE_FOR_CALC:
                    if type(ods.spatial.projection) == WGS84 and \
                       ods.spatial.is_360 and \
                       so.ops.output_format != 'nc' and \
                       so.ops.vector_wrap:
                        ocgis_lh('wrapping output geometries',logger,alias=alias,
                                 ugid=ugid)
                        ods.spatial.vector.wrap()
                ## check for all masked values
                if not so.ops.file_only and ods.value.mask.all():
                    if so.ops.snippet or so.ops.allow_empty:
                        if so.ops.snippet:
                            ocgis_lh('all masked data encountered but allowed for snippet',
                                     logger,alias=alias,ugid=ugid,level=logging.WARN)
                        if so.ops.allow_empty:
                            ocgis_lh('all masked data encountered but empty returns allowed',
                                     logger,alias=alias,ugid=ugid,level=logging.WARN)
                        pass
                    else:
                        ocgis_lh(None,logger,exc=MaskedDataError(),alias=alias,ugid=ugid)
            ## there may be no data returned - this may be real or could be an
            ## error. by default, empty returns are not allowed
            except EmptyData:
                if so.ops.allow_empty:
                    ocgis_lh('the geometric operations returned empty but empty returns are allowed',
                             logger,alias=alias,ugid=ugid)
                    continue
                else:
                    ocgis_lh('empty geometric operation',logger,exc=ExtentError(),alias=alias,ugid=ugid)
        coll.variables.update({request_dataset.alias:ods})

    ## if there are calculations, do those now and return a new type of collection
    if so.cengine is not None:
        ocgis_lh('performing computations',logger,alias=alias,ugid=ugid)
        coll = so.cengine.execute(coll,file_only=so.ops.file_only)
    
    ## conversion of groups.
    if so.ops.output_grouping is not None:
        raise(NotImplementedError)
    else:
        return(coll)
