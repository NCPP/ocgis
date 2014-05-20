from ocgis.calc.engine import OcgCalculationEngine
from ocgis import env, constants
from ocgis.exc import EmptyData, ExtentError, MaskedDataError, EmptySubsetError,\
    ImproperPolygonBoundsError, VariableInCollectionError
from ocgis.util.spatial.wrap import Wrapper
from ocgis.util.logging_ocgis import ocgis_lh, ProgressOcgOperations
import logging
from ocgis.api.collection import SpatialCollection
from ocgis.interface.base.crs import CFWGS84, CFRotatedPole, WGS84
from shapely.geometry.point import Point
from ocgis.calc.base import AbstractMultivariateFunction,\
    AbstractKeyedOutputFunction
from ocgis.util.helpers import project_shapely_geometry,\
    get_rotated_pole_spatial_grid_dimension, get_default_or_apply
from shapely.geometry.multipoint import MultiPoint
from copy import deepcopy
import numpy as np


class SubsetOperation(object):
    '''
    :param :class:~`ocgis.OcgOperations` ops:
    :param bool request_base_size_only: If ``True``, return field objects following
     the spatial subset performing as few operations as possible.
    :param :class:`ocgis.util.logging_ocgis.ProgressOcgOperations` progress:
    '''
    
    def __init__(self,ops,request_base_size_only=False,progress=None):
        self.ops = ops
        self._request_base_size_only = request_base_size_only
        self._subset_log = ocgis_lh.get_logger('subset')
        self._progress = progress or ProgressOcgOperations()

        ## create the calculation engine
        if self.ops.calc == None or self._request_base_size_only == True:
            self.cengine = None
        else:
            ocgis_lh('initializing calculation engine',self._subset_log,level=logging.DEBUG)
            self.cengine = OcgCalculationEngine(self.ops.calc_grouping,
                                           self.ops.calc,
                                           raw=self.ops.calc_raw,
                                           agg=self.ops.aggregate,
                                           calc_sample_size=self.ops.calc_sample_size,
                                           progress=self._progress)
            
        ## in the case of netcdf output, geometries must be unioned. this is
        ## also true for the case of the selection geometry being requested as
        ## aggregated.
        if (self.ops.output_format == 'nc' or self.ops.agg_selection is True) \
         and self.ops.geom is not None:
            ocgis_lh('aggregating selection geometry',self._subset_log)
            build = True
            for element_geom in self.ops.geom:
                if build:
                    new_geom = element_geom['geom']
                    new_crs = element_geom['crs']
                    new_properties = {'UGID':1}
                    build = False
                else:
                    new_geom = new_geom.union(element_geom['geom'])
            itr = [{'geom':new_geom,'properties':new_properties,'crs':new_crs}]
            self.ops.geom = itr
        
    def __iter__(self):
        ''':rtype: AbstractCollection'''
        
        ocgis_lh('beginning iteration',logger='conv.__iter__',level=logging.DEBUG)
        self._ugid_unique_store = []
        self._geom_unique_store = []
        
        ## simple iterator for serial operations
        for coll in self._iter_collections_():
            yield(coll)
        
    def _iter_collections_(self):
        '''
        :yields: :class:~`ocgis.SpatialCollection`
        '''
        
        ## multivariate calculations require datasets come in as a list with all
        ## variable inputs part of the same sequence.
        if self.cengine is not None and self.cengine._check_calculation_members_(self.cengine.funcs,AbstractMultivariateFunction):
            itr_rd = [[r for r in self.ops.dataset]]
        ## otherwise, process geometries expects a single element sequence
        else:
            itr_rd = [[rd] for rd in self.ops.dataset]
        
        ## configure the progress object
        self._progress.n_subsettables = len(itr_rd)
        self._progress.n_geometries = get_default_or_apply(self.ops.geom,len,default=1)
        self._progress.n_calculations = get_default_or_apply(self.ops.calc,len,default=0)
        ## send some messages
        msg = '{0} dataset collection(s) to process.'.format(self._progress.n_subsettables)
        ocgis_lh(msg=msg,logger=self._subset_log)
        if self.ops.geom is None:
            msg = 'Entire spatial domain returned. No selection geometries requested.'
        else:
            msg = 'Each data collection will be subsetted by {0} selection geometries.'.format(self._progress.n_geometries)
        ocgis_lh(msg=msg,logger=self._subset_log)
        if self._progress.n_calculations == 0:
            msg = 'No calculations requested.'
        else:
            msg = 'The following calculations will be applied to each data collection: {0}.'.\
             format(', '.join([_['func'] for _ in self.ops.calc]))
        ocgis_lh(msg=msg,logger=self._subset_log)
        
        ## process the data collections
        for rds in itr_rd:
            msg = 'Processing URI(s): {0}'.format([rd.uri for rd in rds])
            ocgis_lh(msg=msg,logger=self._subset_log)
            
            for coll in self._process_subsettables_(rds):
                ## if there are calculations, do those now and return a new type of collection
                if self.cengine is not None:
                    ocgis_lh('Starting calculations.',
                             self._subset_log,
                             alias=coll.items()[0][1].keys()[0],
                             ugid=coll.keys()[0])
                    
                    ## look for any optimizations for temporal grouping.
                    if self.ops.optimizations is None:
                        tgds = None
                    else:
                        tgds = self.ops.optimizations.get('tgds')
                    ## execute the calculations
                    coll = self.cengine.execute(coll,file_only=self.ops.file_only,
                                                tgds=tgds)
                else:
                    ## if there are no calculations, mark progress to indicate
                    ## a geometry has been completed.
                    self._progress.mark()
                
                ## conversion of groups.
                if self.ops.output_grouping is not None:
                    raise(NotImplementedError)
                else:
                    ocgis_lh('subset yielding',self._subset_log,level=logging.DEBUG)
                    yield(coll)

    def _process_subsettables_(self,rds):
        '''
        :param rds: Sequence of :class:~`ocgis.RequestDataset` objects.
        :type rds: sequence
        :yields: :class:~`ocgis.SpatialCollection`
        '''
        ocgis_lh(msg='entering _process_geometries_',logger=self._subset_log,level=logging.DEBUG)
        
        ## select headers
        if self.ops.headers is not None:
            headers = self.ops.headers
        else:
            if self.cengine is not None:
                if self.cengine._check_calculation_members_(self.cengine.funcs,AbstractMultivariateFunction):
                    headers = constants.multi_headers
                else:
                    headers = constants.calc_headers
            else:
                headers = constants.raw_headers
                
        ## keyed output functions require appending headers regardless. there is
        ## only one keyed output function allowed in a request.
        if self.cengine is not None:
            if self.cengine._check_calculation_members_(self.cengine.funcs,AbstractKeyedOutputFunction):
                value_keys = self.cengine.funcs[0]['ref'].structure_dtype['names']
                headers = list(headers) + value_keys
                ## remove the 'value' attribute headers as this is replaced by the
                ## keyed output names.
                try:
                    headers.remove('value')
                ## it may not be in the list because of a user overload
                except ValueError:
                    pass
            else:
                value_keys = None
        else:
            value_keys = None
                    
        alias = '_'.join([r.alias for r in rds])
        ocgis_lh('processing...',self._subset_log,alias=alias,level=logging.DEBUG)
        ## return the field object
        try:
            ## look for field optimizations
            if self.ops.optimizations is not None and 'fields' in self.ops.optimizations:
                field = [self.ops.optimizations['fields'][rd.alias] for rd in rds]
            else:
                field = [rd.get(format_time=self.ops.format_time,
                                interpolate_spatial_bounds=self.ops.interpolate_spatial_bounds) 
                         for rd in rds]
            ## update the spatial abstraction to match the operations value. sfield
            ## will be none if the operation returns empty and it is allowed to have
            ## empty returns.
            for f in field:
                f.spatial.abstraction = self.ops.abstraction
                
            if len(field) > 1:
                try:
                    field[0].variables.add_variable(field[1].variables.first())
                ## this will fail for optimizations as the fields are already joined
                except VariableInCollectionError:
                    if self.ops.optimizations is not None and 'fields' in self.ops.optimizations:
                        pass
                    else:
                        raise
            field = field[0]
        ## this error is related to subsetting by time or level. spatial subsetting
        ## occurs below.
        except EmptySubsetError as e:
            if self.ops.allow_empty:
                ocgis_lh(msg='time or level subset empty but empty returns allowed',
                         logger=self._subset_log,level=logging.WARN)
                coll = SpatialCollection(headers=headers)
                coll.add_field(1,None,rd.alias,None)
                try:
                    yield(coll)
                finally:
                    return
            else:
                ocgis_lh(exc=ExtentError(message=str(e)),alias=rd.alias,logger=self._subset_log)
        
        ## set iterator based on presence of slice. slice always overrides geometry.
        if self.ops.slice is not None:
            itr = [{}]
        else:
            itr = [{}] if self.ops.geom is None else self.ops.geom
        
        for coll in self._process_geometries_(itr,field,headers,value_keys,alias):
            yield(coll)
    
    def _process_geometries_(self,itr,field,headers,value_keys,alias):
        '''
        :param sequence itr: Contains geometry dictionaries to process. If there
         are no geometries to process, this will be a sequence of one element with
         an empty dictionary.
        :param :class:`ocgis.interface.Field` field: The field object to use for
         operations.
        :param sequence headers: Sequence of strings to use as headers for the
         creation of the collection.
        :param sequence value_keys: Sequence of strings to use as headers for the
         keyed output functions.
        :param str alias: The request data alias currently being processed.
        :yields: :class:~`ocgis.SpatialCollection`
        '''
        ## loop over the iterator
        for gd in itr:
            ## always work with a new geometry dictionary
            gd = deepcopy(gd)
            ## CFRotatedPole takes special treatment. only do this if a subset
            ## geometry is available. this variable is needed to determine if 
            ## backtransforms are necessary.
            original_rotated_pole_crs = None
            if isinstance(field.spatial.crs,CFRotatedPole):
                ## only transform if there is a subset geometry
                if len(gd) > 0:
                    ## store row and column dimension metadata and names before
                    ## transforming as this information is lost w/out row and 
                    ## column dimensions on the transformations.
                    original_row_column_metadata = {'row':{'name':field.spatial.grid.row.name,
                                                           'meta':field.spatial.grid.row.meta},
                                                    'col':{'name':field.spatial.grid.col.name,
                                                           'meta':field.spatial.grid.col.meta}}
                    ## reset the geometries
                    field.spatial._geom = None
                    ## get the new grid dimension
                    field.spatial.grid = get_rotated_pole_spatial_grid_dimension(field.spatial.crs,field.spatial.grid)
                    ## update the CRS. copy the original CRS for possible later
                    ## transformation back to rotated pole.
                    original_rotated_pole_crs = deepcopy(field.spatial.crs)
                    field.spatial.crs = CFWGS84()
            
            ## initialize the collection object to store the subsetted data. if
            ## the output CRS differs from the field's CRS, adjust accordingly 
            ## when initializing.
            if self.ops.output_crs is not None and field.spatial.crs != self.ops.output_crs:
                collection_crs = self.ops.output_crs
            else:
                collection_crs = field.spatial.crs
                
            coll = SpatialCollection(crs=collection_crs,headers=headers,meta=gd.get('meta'),
                                     value_keys=value_keys)
            
            ## reference variables from the geometry dictionary
            geom = gd.get('geom')
            ## keep this around for the collection creation
            coll_geom = deepcopy(geom)
            crs = gd.get('crs')
            
            ## if there is a spatial abstraction, ensure it may be loaded.
            if self.ops.abstraction is not None:
                try:
                    getattr(field.spatial.geom,self.ops.abstraction)
                except ImproperPolygonBoundsError:
                    exc = ImproperPolygonBoundsError('A "polygon" spatial abstraction is not available without the presence of bounds.')
                    ocgis_lh(exc=exc,logger='subset')
                except Exception as e:
                    ocgis_lh(exc=e,logger='subset')
                    
            ## if there is a snippet, return the first realization, time, and level
            if self.ops.snippet:
                field = field[0,0,0,:,:]
            ## if there is a slice, use it to subset the field.
            elif self.ops.slice is not None:
                field = field.__getitem__(self.ops.slice)
                
            ## see if the selection crs matches the field's crs
            if crs is not None and crs != field.spatial.crs:
                geom = project_shapely_geometry(geom,crs.sr,field.spatial.crs.sr)
                crs = field.spatial.crs
            ## if the geometry is a point, we need to buffer it...
            if type(geom) in [Point,MultiPoint]:
                ocgis_lh(logger=self._subset_log,msg='buffering point geometry',level=logging.DEBUG)
                geom = geom.buffer(self.ops.search_radius_mult*field.spatial.grid.resolution)
                ## update the geometry to store in the collection
                coll_geom = deepcopy(geom)
            
            ## get the ugid following geometry manipulations
            if 'properties' in gd and 'UGID' in gd['properties']:
                ugid = gd['properties']['UGID']
            else:
                ugid = 1
                
            if geom is None:
                msg = 'No selection geometry. Returning all data. Assiging UGID as 1.'
            else:
                msg = 'Subsetting with selection geometry having UGID={0}'.format(ugid)
            ocgis_lh(msg=msg,logger=self._subset_log)
                
            ## check for unique ugids. this is an issue with point subsetting
            ## as the buffer radius changes by dataset.
            if ugid in self._ugid_unique_store and geom is not None:
                ## only update if the geometry is unique
                if not any([__.almost_equals(geom) for __ in self._geom_unique_store]):
                    prev_ugid = ugid
                    ugid = max(self._ugid_unique_store) + 1
                    self._ugid_unique_store.append(ugid)
                    msg = 'Updating UGID {0} to {1} to maintain uniqueness.'.format(prev_ugid,ugid)
                    ocgis_lh(msg,self._subset_log,level=logging.WARN,alias=alias,ugid=ugid)
                else:
                    self._geom_unique_store.append(geom)
            else:
                self._ugid_unique_store.append(ugid)
                self._geom_unique_store.append(geom)
                            
            ## try to update the properties
            try:
                gd['properties']['UGID'] = ugid
            except KeyError:
                if not isinstance(gd,dict):
                    raise
                
            ## unwrap the data if it is geographic and 360
            if geom is not None and crs == CFWGS84():
                if CFWGS84.get_is_360(field.spatial):
                    ocgis_lh('unwrapping selection geometry',self._subset_log,alias=alias,ugid=ugid,level=logging.DEBUG)
                    geom = Wrapper().unwrap(geom)
            ## perform the spatial operation
            if geom is not None:
                try:
                    if self.ops.spatial_operation == 'intersects':
                        sfield = field.get_intersects(geom, use_spatial_index=env.USE_SPATIAL_INDEX,
                                                      select_nearest=self.ops.select_nearest)
                    elif self.ops.spatial_operation == 'clip':
                        sfield = field.get_clip(geom, use_spatial_index=env.USE_SPATIAL_INDEX,
                                                select_nearest=self.ops.select_nearest)
                    else:
                        ocgis_lh(exc=NotImplementedError(self.ops.spatial_operation))
                except EmptySubsetError as e:
                    if self.ops.allow_empty:
                        ocgis_lh(alias=alias,ugid=ugid,msg='empty geometric operation but empty returns allowed',level=logging.WARN)
                        sfield = None
                    else:
                        msg = str(e) + ' This typically means the selection geometry falls outside the spatial domain of the target dataset.'
                        ocgis_lh(exc=ExtentError(message=msg),alias=alias,logger=self._subset_log)
            else:
                sfield = field
            
            ## if the base size is being requested, bypass the rest of the
            ## operations.
            if self._request_base_size_only == False:
                ## if empty returns are allowed, there be an empty field
                if sfield is not None:
                    ## aggregate if requested
                    if self.ops.aggregate:
                        ocgis_lh('executing spatial average',self._subset_log,alias=alias,ugid=ugid)
                        sfield = sfield.get_spatially_aggregated(new_spatial_uid=ugid)
                    
                    ## wrap the returned data.
                    if not env.OPTIMIZE_FOR_CALC:
                        if CFWGS84.get_is_360(sfield.spatial):
                            if self.ops.output_format != 'nc' and self.ops.vector_wrap:
                                ocgis_lh('wrapping output geometries',self._subset_log,alias=alias,ugid=ugid,
                                         level=logging.DEBUG)
                                ## modifying these values in place will change the values
                                ## in the base field. a copy is necessary.
                                sfield.spatial = deepcopy(sfield.spatial)
                                sfield.spatial.crs.wrap(sfield.spatial)
                                
                    ## check for all masked values
                    if env.OPTIMIZE_FOR_CALC is False and self.ops.file_only is False:
                        for variable in sfield.variables.itervalues():
                            ocgis_lh(msg='Fetching data for variable with alias "{0}".'.format(variable.alias),
                                     logger=self._subset_log)
                            if variable.value.mask.all():
                                ## masked data may be okay depending on other opeartional
                                ## conditions.
                                if self.ops.snippet or self.ops.allow_empty or (self.ops.output_format == 'numpy' and self.ops.allow_empty):
                                    if self.ops.snippet:
                                        ocgis_lh('all masked data encountered but allowed for snippet',
                                                 self._subset_log,alias=alias,ugid=ugid,level=logging.WARN)
                                    if self.ops.allow_empty:
                                        ocgis_lh('all masked data encountered but empty returns allowed',
                                                 self._subset_log,alias=alias,ugid=ugid,level=logging.WARN)
                                    if self.ops.output_format == 'numpy':
                                        ocgis_lh('all masked data encountered but numpy data being returned allowed',
                                                 logger=self._subset_log,alias=alias,ugid=ugid,level=logging.WARN)
                                else:
                                    ## if the geometry is also masked, it is an empty spatial
                                    ## operation.
                                    if sfield.spatial.abstraction_geometry.value.mask.all():
                                        ocgis_lh(exc=EmptyData,logger=self._subset_log)
                                    ## if none of the other conditions are met, raise the masked data error
                                    else:
                                        ocgis_lh(logger=self._subset_log,exc=MaskedDataError(),alias=alias,ugid=ugid)
                    
                    ## transform back to rotated pole if necessary
                    if original_rotated_pole_crs is not None:
                        if self.ops.output_crs is None and not isinstance(self.ops.output_crs,CFWGS84):
                            ## we need to load the values before proceeding. source
                            ## indices will disappear.
                            for variable in sfield.variables.itervalues():
                                variable.value
                            ## reset the geometries
                            sfield.spatial._geom = None
                            sfield.spatial.grid = get_rotated_pole_spatial_grid_dimension(
                             original_rotated_pole_crs,sfield.spatial.grid,inverse=True,
                             rc_original=original_row_column_metadata)
                            ## update the uid mask to match the spatial mask
                            sfield.spatial.uid = np.ma.array(sfield.spatial.uid,mask=sfield.spatial.get_mask())
                            sfield.spatial.crs = original_rotated_pole_crs
                    
                    ## update the coordinate system of the data output
                    if self.ops.output_crs is not None:
                        ## if the geometry is not None, it may need to be projected to match
                        ## the output crs.
                        if geom is not None and crs != self.ops.output_crs:
                            geom = project_shapely_geometry(geom,crs.sr,self.ops.output_crs.sr)
                            coll_geom = deepcopy(geom)
                        ## update the coordinate reference system of the spatial
                        ## dimension.
                        try:
                            sfield.spatial.update_crs(self.ops.output_crs)
                        ## this is likely a rotated pole origin
                        except RuntimeError as e:
                            if isinstance(sfield.spatial.crs,CFRotatedPole):
                                assert(isinstance(self.ops.output_crs,WGS84))
                                sfield.spatial._geom = None
                                sfield.spatial.grid = get_rotated_pole_spatial_grid_dimension(
                                 sfield.spatial.crs,sfield.spatial.grid)
                                sfield.spatial.crs = self.ops.output_crs
                            else:
                                ocgis_lh(exc=e,logger=self._subset_log)
                
            ## the geometry may need to be wrapped or unwrapped depending on
            ## the vector wrap situation
            coll.add_field(ugid,coll_geom,alias,sfield,properties=gd.get('properties'))

            yield(coll)
