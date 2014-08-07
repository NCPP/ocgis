from ocgis.calc.engine import OcgCalculationEngine
from ocgis import env, constants
from ocgis.exc import EmptyData, ExtentError, MaskedDataError, EmptySubsetError,\
    ImproperPolygonBoundsError, VariableInCollectionError
from ocgis.util.spatial.wrap import Wrapper
from ocgis.util.logging_ocgis import ocgis_lh, ProgressOcgOperations
import logging
from ocgis.api.collection import SpatialCollection
from ocgis.interface.base.crs import CFWGS84, CFRotatedPole, WGS84
from ocgis.calc.base import AbstractMultivariateFunction,\
    AbstractKeyedOutputFunction
from ocgis.util.helpers import get_default_or_apply, write_geom_dict
from copy import deepcopy, copy
import numpy as np
from ocgis.calc.eval_function import MultivariateEvalFunction
from shapely.geometry import Point, MultiPoint
from ocgis.interface.base.dimension.spatial import SpatialGeometryPolygonDimension


class SubsetOperation(object):
    """
    :param :class:~`ocgis.OcgOperations` ops:
    :param bool request_base_size_only: If ``True``, return field objects following
     the spatial subset performing as few operations as possible.
    :param :class:`ocgis.util.logging_ocgis.ProgressOcgOperations` progress:
    """
    
    def __init__(self,ops,request_base_size_only=False,progress=None):
        self.ops = ops
        self._request_base_size_only = request_base_size_only
        self._subset_log = ocgis_lh.get_logger('subset')
        self._progress = progress or ProgressOcgOperations()

        ## create the calculation engine
        if self.ops.calc == None or self._request_base_size_only == True:
            self.cengine = None
            self._has_multivariate_calculations = False
        else:
            ocgis_lh('initializing calculation engine',self._subset_log,level=logging.DEBUG)
            self.cengine = OcgCalculationEngine(self.ops.calc_grouping,
                                           self.ops.calc,
                                           raw=self.ops.calc_raw,
                                           agg=self.ops.aggregate,
                                           calc_sample_size=self.ops.calc_sample_size,
                                           progress=self._progress)
            self._has_multivariate_calculations = any([self.cengine._check_calculation_members_(self.cengine.funcs,k) \
             for k in [AbstractMultivariateFunction,MultivariateEvalFunction]])
            
        # in the case of netcdf output, geometries must be unioned. this is also true for the case of the selection
        # geometry being requested as aggregated.
        if (self.ops.output_format == 'nc' or self.ops.agg_selection is True) and self.ops.geom is not None:
            ocgis_lh('aggregating selection geometry', self._subset_log)
            build = True
            for sdim in self.ops.geom:
                _geom = sdim.geom.get_highest_order_abstraction().value[0, 0]
                if build:
                    new_geom = _geom
                    new_crs = sdim.crs
                    new_properties = {'UGID': 1}
                    build = False
                else:
                    new_geom = new_geom.union(_geom)
            self.ops.geom = [{'geom': new_geom, 'properties': new_properties, 'crs': new_crs}]

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
        :yields: :class:`~ocgis.SpatialCollection`
        '''
        
        ## multivariate calculations require datasets come in as a list with all
        ## variable inputs part of the same sequence.
        if self._has_multivariate_calculations:
            itr_rd = [[r for r in self.ops.dataset.itervalues()]]

        ## otherwise, process geometries expects a single element sequence
        else:
            itr_rd = [[rd] for rd in self.ops.dataset.itervalues()]
        
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

    def _process_subsettables_(self, rds):
        """
        :param rds: Sequence of :class:~`ocgis.RequestDataset` objects.
        :type rds: sequence
        :rtype: :class:~`ocgis.SpatialCollection`
        """

        ocgis_lh(msg='entering _process_geometries_',logger=self._subset_log,level=logging.DEBUG)
        
        ## select headers
        if self.ops.headers is not None:
            headers = self.ops.headers
        else:
            if self.cengine is not None:
                if self._has_multivariate_calculations:
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

        alias = '_'.join([r.name for r in rds])

        ocgis_lh('processing...',self._subset_log,alias=alias,level=logging.DEBUG)
        ## return the field object
        try:
            ## look for field optimizations
            if self.ops.optimizations is not None and 'fields' in self.ops.optimizations:
                field = [self.ops.optimizations['fields'][rd.alias] for rd in rds]
            else:
                field = [rd.get(format_time=self.ops.format_time,
                                interpolate_spatial_bounds=self.ops.interpolate_spatial_bounds) for rd in rds]
            # update the spatial abstraction to match the operations value. sfield will be none if the operation returns
            # empty and it is allowed to have empty returns.
            for f in field:
                f.spatial.abstraction = self.ops.abstraction

            if len(field) > 1:
                try:
                    ## reset the variable uid and let the collection handle its assignment
                    variable_to_add = field[1].variables.first()
                    variable_to_add.uid = None
                    field[0].variables.add_variable(variable_to_add)
                    ## reset the field names and let these be auto-generated
                    for f in field:
                        f._name = None
                ## this will fail for optimizations as the fields are already joined
                except VariableInCollectionError:
                    if self.ops.optimizations is not None and 'fields' in self.ops.optimizations:
                        pass
                    else:
                        raise
            field = field[0]
        # this error is related to subsetting by time or level. spatial subsetting occurs below.
        except EmptySubsetError as e:
            if self.ops.allow_empty:
                ocgis_lh(msg='time or level subset empty but empty returns allowed',
                         logger=self._subset_log,level=logging.WARN)
                coll = SpatialCollection(headers=headers)
                coll.add_field(1, None, None, name='_'.join([rd.name for rd in rds]))
                try:
                    yield(coll)
                finally:
                    return
            else:
                ocgis_lh(exc=ExtentError(message=str(e)),alias=rd.alias,logger=self._subset_log)
        
        ## set iterator based on presence of slice. slice always overrides geometry.
        if self.ops.slice is not None:
            itr = [None]
        else:
            itr = [None] if self.ops.geom is None else self.ops.geom
        for coll in self._process_geometries_(itr,field,headers,value_keys,alias):
            yield(coll)

    def _get_initialized_collection_(self, field, headers, value_keys):
        """
        Initialize the spatial collection object selecting the output CRS in the process.

        :param field:
        :type field: :class:`ocgis.interface.base.field.Field`
        :param headers:
        :type headers: list[str]
        :param value_keys:
        :type value_keys: list[str]
        :rtype: :class:`ocgis.api.collection.SpatialCollection`
        """

        # initialize the collection object to store the subsetted data. if the output CRS differs from the field's
        # CRS, adjust accordingly when initializing.
        if self.ops.output_crs is not None and field.spatial.crs != self.ops.output_crs:
            collection_crs = self.ops.output_crs
        else:
            collection_crs = field.spatial.crs
        coll = SpatialCollection(crs=collection_crs, headers=headers, value_keys=value_keys)
        return coll

    @staticmethod
    def _get_update_rotated_pole_state_(field, subset_sdim):
        """
        Rotated pole coordinate systems are handled internally by transforming the CRS to a geographic coordinate
        system.

        :param field:
        :type field: :class:`ocgis.interface.base.field.Field`
        :param subset_sdim:
        :type subset_sdim: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension` or None
        :rtype: None or :class:`ocgis.interface.base.crs.CFRotatedPole`
        :raises: AssertionError
        """

        # CFRotatedPole takes special treatment. only do this if a subset geometry is available. this variable is
        # needed to determine if backtransforms are necessary.
        original_rotated_pole_crs = None
        if isinstance(field.spatial.crs, CFRotatedPole):
            # only transform if there is a subset geometry
            if subset_sdim is not None:
                # update the CRS. copy the original CRS for possible later transformation back to rotated pole.
                original_rotated_pole_crs = copy(field.spatial.crs)
                field.spatial.update_crs(CFWGS84())
        return original_rotated_pole_crs

    def _assert_abstraction_available_(self, field):
        """
        Assert the spatial abstraction may be loaded on the field object if one is provided in the operations.

        :param field:
        :type field: :class:`ocgis.interface.base.field.Field`
        """

        if self.ops.abstraction is not None:
            try:
                getattr(field.spatial.geom, self.ops.abstraction)
            except ImproperPolygonBoundsError:
                msg = 'A "polygon" spatial abstraction is not available without the presence of bounds.'
                exc = ImproperPolygonBoundsError(msg)
                ocgis_lh(exc=exc, logger='subset')
            except Exception as e:
                ocgis_lh(exc=e, logger='subset')

    def _get_slice_or_snippet_(self, field):
        """
        Slice the incoming field if a slice or snippet argument is present.

        :param field:
        :type field: :class:`ocgis.interface.base.field.Field`
        :rtype: :class:`ocgis.interface.base.field.Field`
        """

        # if there is a snippet, return the first realization, time, and level
        if self.ops.snippet:
            field = field[0, 0, 0, :, :]
        # if there is a slice, use it to subset the field.
        elif self.ops.slice is not None:
            field = field.__getitem__(self.ops.slice)
        return field

    def _get_spatially_subsetted_field_(self, alias, field, subset_sdim, subset_ugid):
        """
        Spatially subset a field with a selection geometry.

        :param str alias: The request data alias currently being processed.
        :param field:
        :type field: :class:`ocgis.interface.base.field.Field`
        :param subset_sdim:
        :type subset_sdim: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        :rtype: None or :class:`ocgis.interface.base.field.Field`
        :raises: AssertionError, ExtentError
        """

        assert(subset_sdim is not None)

        subset_geom = subset_sdim.single.geom

        # check for unique ugids. this is an issue with point subsetting as the buffer radius changes by dataset.
        if subset_ugid in self._ugid_unique_store:
            # # only update if the geometry is unique
            if not any([__.almost_equals(subset_geom) for __ in self._geom_unique_store]):
                prev_ugid = subset_ugid
                ugid = max(self._ugid_unique_store) + 1

                # update the geometry property and uid
                subset_sdim.properties['UGID'][0] = ugid
                subset_sdim.uid[:] = ugid

                self._ugid_unique_store.append(ugid)
                self._geom_unique_store.append(subset_geom)
                msg = 'Updating UGID {0} to {1} to maintain uniqueness.'.format(prev_ugid, ugid)
                ocgis_lh(msg, self._subset_log, level=logging.WARN, alias=alias, ugid=ugid)
            else:
                pass
                # self._ugid_unique_store.append(subset_ugid)
                # self._geom_unique_store.append(subset_geom)
        else:
            self._ugid_unique_store.append(subset_ugid)
            self._geom_unique_store.append(subset_geom)

        # unwrap the data if it is geographic and 360
        if field.spatial.is_unwrapped and not subset_sdim.is_unwrapped:
            ocgis_lh('unwrapping selection geometry', self._subset_log, alias=alias, ugid=subset_ugid,
                     level=logging.DEBUG)
            subset_sdim.unwrap()
            # update the geometry reference as the spatial dimension was unwrapped and modified in place
            subset_geom = subset_sdim.single.geom

        # perform the spatial operation
        try:
            if self.ops.spatial_operation == 'intersects':
                sfield = field.get_intersects(subset_geom, use_spatial_index=env.USE_SPATIAL_INDEX,
                                              select_nearest=self.ops.select_nearest)
            elif self.ops.spatial_operation == 'clip':
                sfield = field.get_clip(subset_geom, use_spatial_index=env.USE_SPATIAL_INDEX,
                                        select_nearest=self.ops.select_nearest)
            else:
                ocgis_lh(exc=NotImplementedError(self.ops.spatial_operation))
        except EmptySubsetError as e:
            if self.ops.allow_empty:
                ocgis_lh(alias=alias, ugid=subset_ugid, msg='empty geometric operation but empty returns allowed',
                         level=logging.WARN)
                sfield = None
            else:
                msg = str(e) + ' This typically means the selection geometry falls outside the spatial domain of the target dataset.'
                ocgis_lh(exc=ExtentError(message=msg), alias=alias, logger=self._subset_log)

        # if the subset geometry is unwrapped and the vector wrap option is true, wrap the subset geometry.
        if subset_sdim.is_unwrapped and self.ops.vector_wrap:
            subset_sdim.wrap()

        return sfield

    def _update_subset_geometry_if_point_(self, field, subset_sdim, subset_ugid):
        """
        If the subset geometry is a point of multipoint, it will need to be buffered and the spatial dimension updated
        accordingly. If the subset geometry is a polygon, pass through.

        :param field:
        :type field: :class:`ocgis.interface.base.field.Field`
        :param subset_sdim:
        :type subset_sdim: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        :param int subset_ugid:
        :raises: AssertionError
        """

        if type(subset_sdim.single.geom) in [Point, MultiPoint]:
            assert(subset_sdim.abstraction == 'point')
            ocgis_lh(logger=self._subset_log, msg='buffering point geometry', level=logging.DEBUG)
            subset_geom = subset_sdim.single.geom.buffer(self.ops.search_radius_mult*field.spatial.grid.resolution)
            value = np.ma.array([[None]])
            value[0, 0] = subset_geom
            subset_sdim.geom._polygon = SpatialGeometryPolygonDimension(value=value, uid=subset_ugid)
            # the polygon should be used for subsetting, update the spatial dimension to use this abstraction
            subset_sdim.abstraction = 'polygon'
        assert(subset_sdim.abstraction == 'polygon')

    def _process_geometries_(self, itr, field, headers, value_keys, alias):
        """
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
        :rtype: :class:~`ocgis.SpatialCollection`
        """

        # process each geometry
        for subset_sdim in itr:
            # always work with a copy of the target geometry
            subset_sdim = deepcopy(subset_sdim)
            """:type subset_sdim: ocgis.interface.base.dimension.spatial.SpatialDimension"""

            # operate on the rotated pole coordinate system by first transforming it to CFWGS84
            original_rotated_pole_crs = self._get_update_rotated_pole_state_(field, subset_sdim)

            # initialize the collection storage
            coll = self._get_initialized_collection_(field, headers, value_keys)

            # check if the geometric abstraction is available on the field object
            self._assert_abstraction_available_(field)

            # return a slice or snippet if either of these are requested.
            field = self._get_slice_or_snippet_(field)

            # choose the subset ugid value
            if subset_sdim is None:
                msg = 'No selection geometry. Returning all data. Assigning UGID as 1.'
                subset_ugid = 1
            else:
                subset_ugid = subset_sdim.single.uid
                msg = 'Subsetting with selection geometry having UGID={0}'.format(subset_ugid)
            ocgis_lh(msg=msg, logger=self._subset_log)

            if subset_sdim is not None:
                # if the CRS's differ, update the spatial dimension to match the field
                if subset_sdim.crs is not None and subset_sdim.crs != field.spatial.crs:
                    subset_sdim.update_crs(field.spatial.crs)
                # if the geometry is a point, it needs to be buffered
                self._update_subset_geometry_if_point_(field, subset_sdim, subset_ugid)

            # if there is a selection geometry present, use it for the spatial subset. if not, all the field's data is
            # being returned.
            if subset_sdim is None:
                sfield = field
            else:
                sfield = self._get_spatially_subsetted_field_(alias, field, subset_sdim, subset_ugid)

            # if the base size is being requested, bypass the rest of the operations.
            if not self._request_base_size_only:
                ## if empty returns are allowed, there be an empty field
                if sfield is not None:
                    ## aggregate if requested
                    if self.ops.aggregate:
                        ocgis_lh('executing spatial average',self._subset_log,alias=alias,ugid=subset_ugid)
                        sfield = sfield.get_spatially_aggregated(new_spatial_uid=subset_ugid)
                    
                    # wrap the returned data.
                    if not env.OPTIMIZE_FOR_CALC:
                        if sfield is not None and sfield.spatial.is_unwrapped:
                            if self.ops.output_format != 'nc' and self.ops.vector_wrap:
                                ocgis_lh('wrapping output geometries', self._subset_log, alias=alias, ugid=subset_ugid,
                                         level=logging.DEBUG)
                                # deepcopy the spatial dimension before wrapping as wrapping will modify the spatial
                                # dimension on the parent field object. which may need to be reused for additional
                                # subsets.
                                sfield.spatial = deepcopy(sfield.spatial)
                                sfield.spatial.wrap()
                                
                    # check for all masked values
                    if env.OPTIMIZE_FOR_CALC is False and self.ops.file_only is False:
                        for variable in sfield.variables.itervalues():
                            ocgis_lh(msg='Fetching data for variable with alias "{0}".'.format(variable.alias),
                                     logger=self._subset_log)
                            if variable.value.mask.all():
                                # masked data may be okay...
                                if self.ops.snippet or self.ops.allow_empty or (self.ops.output_format == 'numpy' and self.ops.allow_empty):
                                    if self.ops.snippet:
                                        ocgis_lh('all masked data encountered but allowed for snippet',
                                                 self._subset_log,alias=alias,ugid=subset_ugid,level=logging.WARN)
                                    if self.ops.allow_empty:
                                        ocgis_lh('all masked data encountered but empty returns allowed',
                                                 self._subset_log,alias=alias,ugid=subset_ugid,level=logging.WARN)
                                    if self.ops.output_format == 'numpy':
                                        ocgis_lh('all masked data encountered but numpy data being returned allowed',
                                                 logger=self._subset_log,alias=alias,ugid=subset_ugid,level=logging.WARN)
                                else:
                                    # if the geometry is also masked, it is an empty spatial operation.
                                    if sfield.spatial.abstraction_geometry.value.mask.all():
                                        ocgis_lh(exc=EmptyData,logger=self._subset_log)
                                    # if none of the other conditions are met, raise the masked data error
                                    else:
                                        ocgis_lh(logger=self._subset_log, exc=MaskedDataError(), alias=alias,
                                                 ugid=subset_ugid)
                    
                    ## transform back to rotated pole if necessary
                    if original_rotated_pole_crs is not None:
                        if not isinstance(self.ops.output_crs, CFWGS84):
                            sfield.spatial.update_crs(original_rotated_pole_crs)

                    # update the coordinate system of the data output
                    if self.ops.output_crs is not None:
                        # if the geometry is not None, it may need to be projected to match the output crs.
                        if subset_sdim is not None and subset_sdim.crs != self.ops.output_crs:
                            subset_sdim.update_crs(self.ops.output_crs)
                        # update the subset field CRS
                        sfield.spatial.update_crs(self.ops.output_crs)
                
            # use the field's alias if it is provided. otherwise, let it be automatically assigned
            name = alias if sfield is None else None

            # pass selection geometry and properties for the field added to the collection. add the created field to the
            # output collection.
            if subset_sdim is None:
                subset_geom, subset_properties = [None, None]
            else:
                subset_geom = subset_sdim.single.geom
                subset_properties = subset_sdim.properties
                subset_ugid = subset_sdim.single.uid
            coll.add_field(subset_ugid, subset_geom, sfield, properties=subset_properties, name=name)

            yield coll
