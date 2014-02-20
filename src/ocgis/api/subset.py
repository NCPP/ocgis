from ocgis.calc.engine import OcgCalculationEngine
from ocgis import env, constants
from ocgis.exc import EmptyData, ExtentError, MaskedDataError, EmptySubsetError,\
    ImproperPolygonBoundsError
from ocgis.util.spatial.wrap import Wrapper
from ocgis.util.logging_ocgis import ocgis_lh
import logging
from ocgis.api.collection import SpatialCollection
from ocgis.interface.base.crs import CFWGS84
from shapely.geometry.point import Point
from ocgis.calc.base import AbstractMultivariateFunction,\
    AbstractKeyedOutputFunction
from ocgis.util.helpers import project_shapely_geometry
from shapely.geometry.multipoint import MultiPoint
from copy import deepcopy


class SubsetOperation(object):
    
    def __init__(self,ops,serial=True,nprocs=1):
        self.ops = ops
        self.serial = serial
        self.nprocs = nprocs
        
        self._subset_log = ocgis_lh.get_logger('subset')

        ## create the calculation engine
        if self.ops.calc is None:
            self.cengine = None
        else:
            ocgis_lh('initializing calculation engine',self._subset_log,level=logging.DEBUG)
            self.cengine = OcgCalculationEngine(self.ops.calc_grouping,
                                           self.ops.calc,
                                           raw=self.ops.calc_raw,
                                           agg=self.ops.aggregate,
                                           calc_sample_size=self.ops.calc_sample_size)
            
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
                
        ## simple iterator for serial operations
        if self.serial:
            for coll in self._iter_collections_():
                yield(coll)
        ## use a multiprocessing pool returning unordered geometries
        ## for the parallel case
        else:
            raise(ocgis_lh(exc=NotImplementedError('multiprocessing is not available')))

    def _process_geometries_(self,rds):
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
        ocgis_lh('processing...',self._subset_log,alias=alias)
        ## return the field object
        try:
            field = [rd.get(format_time=self.ops.format_time,
                            interpolate_spatial_bounds=self.ops.interpolate_spatial_bounds) 
                     for rd in rds]
            if len(field) > 1:
                field[0].variables.add_variable(field[1].variables.first())
            field = field[0]
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
                
        ## loop over the iterator
        for gd in itr:
            ## initialize the collection object to store the subsetted data. if
            ## the output CRS differs from the field's CRS, adjust accordingly 
            ## when initilizing.
            if self.ops.output_crs is not None and field.spatial.crs != self.ops.output_crs:
                collection_crs = self.ops.output_crs
            else:
                collection_crs = field.spatial.crs
                
            coll = SpatialCollection(crs=collection_crs,headers=headers,meta=gd.get('meta'),
                                     value_keys=value_keys)
            
            ## reference variables from the geometry dictionary
            geom = gd.get('geom')
            crs = gd.get('crs')
            
            if 'properties' in gd and 'UGID' in gd['properties']:
                ugid = gd['properties']['UGID']
            else:
                ## try to get lowercase ugid in case the shapefile is not perfectly
                ## formed. however, if there is no geometry accept the error and
                ## use the default geometry identifier.
                if len(gd) == 0:
                    ugid = 1
                else:
                    ugid = gd['properties']['ugid']
                    
            ocgis_lh('processing',self._subset_log,level=logging.DEBUG,alias=alias,ugid=ugid)
            
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
            ## unwrap the data if it is geographic and 360
            if geom is not None and crs == CFWGS84():
                if CFWGS84.get_is_360(field.spatial):
                    ocgis_lh('unwrapping selection geometry',self._subset_log,alias=alias,ugid=ugid)
                    geom = Wrapper().unwrap(geom)
            ## perform the spatial operation
            if geom is not None:
                try:
                    if self.ops.spatial_operation == 'intersects':
                        sfield = field.get_intersects(geom)
                    elif self.ops.spatial_operation == 'clip':
                        sfield = field.get_clip(geom)
                    else:
                        ocgis_lh(exc=NotImplementedError(self.ops.spatial_operation))
                except EmptySubsetError as e:
                    if self.ops.allow_empty:
                        ocgis_lh(alias=alias,ugid=ugid,msg='empty geometric operation but empty returns allowed',level=logging.WARN)
                        sfield = None
                    else:
                        ocgis_lh(exc=ExtentError(message=str(e)),alias=alias,logger=self._subset_log)
            else:
                sfield = field
                        
            ## if empty returns are allowed, there be an empty field
            if sfield is not None:
                ## aggregate if requested
                if self.ops.aggregate:
                    sfield = sfield.get_spatially_aggregated(new_spatial_uid=ugid)
                
                ## wrap the returned data.
                if not env.OPTIMIZE_FOR_CALC:
                    if CFWGS84.get_is_360(sfield.spatial):
                        if self.ops.output_format != 'nc' and self.ops.vector_wrap:
                            ocgis_lh('wrapping output geometries',self._subset_log,alias=alias,ugid=ugid)
                            ## modifying these values in place will change the values
                            ## in the base field. a copy is necessary.
                            sfield.spatial = deepcopy(sfield.spatial)
                            sfield.spatial.crs.wrap(sfield.spatial)
                            
                ## check for all masked values
                if env.OPTIMIZE_FOR_CALC is False and self.ops.file_only is False:
                    for variable in sfield.variables.itervalues():
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
            
            ## update the coordinate system of the data output
            if self.ops.output_crs is not None:
                ## if the geometry is not None, it may need to be projected to match
                ## the output crs.
                if geom is not None and crs != self.ops.output_crs:
                    geom = project_shapely_geometry(geom,crs.sr,self.ops.output_crs.sr)
                    
                sfield.spatial.update_crs(self.ops.output_crs)
            
            ## update the spatial abstraction to match the operations value. sfield
            ## will be none if the operation returns empty and it is allowed to have
            ## empty returns.
            if sfield is not None:
                sfield.spatial.abstraction = self.ops.abstraction
            
            coll.add_field(ugid,geom,alias,sfield,properties=gd.get('properties'))
            
            yield(coll)
    
    def _iter_collections_(self):
        
        ocgis_lh('{0} request dataset(s) to process'.format(len(self.ops.dataset)),'conv._iter_collections_')
        
        if self.cengine is None:
            itr_rd = ([rd] for rd in self.ops.dataset)
        else:
            if self.cengine._check_calculation_members_(self.cengine.funcs,AbstractMultivariateFunction):
                itr_rd = [[r for r in self.ops.dataset]]
            else:
                itr_rd = ([rd] for rd in self.ops.dataset)
        
        for rds in itr_rd:
            for coll in self._process_geometries_(rds):
                ## if there are calculations, do those now and return a new type of collection
                if self.cengine is not None:
                    ocgis_lh('performing computations',
                             self._subset_log,
                             alias=coll.items()[0][1].keys()[0],
                             ugid=coll.keys()[0])
                    coll = self.cengine.execute(coll,file_only=self.ops.file_only)
                
                ## conversion of groups.
                if self.ops.output_grouping is not None:
                    raise(NotImplementedError)
                else:
                    ocgis_lh('subset yielding',self._subset_log,level=logging.DEBUG)
                    yield(coll)
