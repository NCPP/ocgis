import logging
from copy import deepcopy

from ocgis import Variable
from ocgis import env, constants
from ocgis.calc.engine import OcgCalculationEngine
from ocgis.collection.field import OcgField
from ocgis.collection.spatial import SpatialCollection
from ocgis.constants import WrappedState, HeaderNames, WrapAction
from ocgis.exc import ExtentError, EmptySubsetError, BoundsAlreadyAvailableError
from ocgis.spatial.spatial_subset import SpatialSubsetOperation
from ocgis.util.helpers import get_default_or_apply
from ocgis.util.logging_ocgis import ocgis_lh, ProgressOcgOperations
from ocgis.variable.crs import CFWGS84, CFRotatedPole


class OperationsEngine(object):
    """
    :param :class:~`ocgis.OcgOperations` ops:
    :param bool request_base_size_only: If ``True``, return field objects following
     the spatial subset performing as few operations as possible.
    :param :class:`ocgis.util.logging_ocgis.ProgressOcgOperations` progress:
    """

    def __init__(self, ops, request_base_size_only=False, progress=None):
        self.ops = ops
        self._request_base_size_only = request_base_size_only
        self._subset_log = ocgis_lh.get_logger('subset')
        self._progress = progress or ProgressOcgOperations()

        # Create the calculation engine is calculations are present.
        if self.ops.calc is None or self._request_base_size_only:
            self.cengine = None
            self._has_multivariate_calculations = False
        else:
            ocgis_lh('initializing calculation engine', self._subset_log, level=logging.DEBUG)
            self.cengine = OcgCalculationEngine(self.ops.calc_grouping,
                                                self.ops.calc,
                                                calc_sample_size=self.ops.calc_sample_size,
                                                progress=self._progress,
                                                spatial_aggregation=self.ops.aggregate)
            self._has_multivariate_calculations = self.cengine.has_multivariate_functions

    def __iter__(self):
        """:rtype: :class:`ocgis.collection.base.AbstractCollection`"""

        ocgis_lh('beginning iteration', logger='conv.__iter__', level=logging.DEBUG)
        # self._ugid_unique_store = []
        # self._geom_unique_store = []

        # Yields collections with all operations applied.
        for coll in self._iter_collections_():
            yield coll

    def _iter_collections_(self):
        """:rtype: :class:`ocgis.collection.base.AbstractCollection`"""

        # The dataset object defines the iterator yielding request dataset or fields. The "value" could be any of the
        # inputs.
        # ops_dataset = self.ops._get_object_(Dataset.name)

        # Multivariate calculations require datasets come in as a list with all variable inputs part of the same
        # sequence.
        if self._has_multivariate_calculations:
            itr_rd = [[rd for rd in self.ops.dataset]]
        # Otherwise, process geometries expects a single element sequence.
        else:
            itr_rd = [[rd] for rd in self.ops.dataset]

        # Configure the progress object.
        self._progress.n_subsettables = len(itr_rd)
        self._progress.n_geometries = get_default_or_apply(self.ops.geom, len, default=1)
        self._progress.n_calculations = get_default_or_apply(self.ops.calc, len, default=0)

        # Some introductory logging.
        msg = '{0} dataset collection(s) to process.'.format(self._progress.n_subsettables)
        ocgis_lh(msg=msg, logger=self._subset_log)
        if self.ops.geom is None:
            msg = 'Entire spatial domain returned. No selection geometries requested.'
        else:
            msg = 'Each data collection will be subsetted by {0} selection geometries.'.format(
                self._progress.n_geometries)
        ocgis_lh(msg=msg, logger=self._subset_log)
        if self._progress.n_calculations == 0:
            msg = 'No calculations requested.'
        else:
            msg = 'The following calculations will be applied to each data collection: {0}.'. \
                format(', '.join([_['func'] for _ in self.ops.calc]))
        ocgis_lh(msg=msg, logger=self._subset_log)

        # Process the incoming datasets. Convert from request datasets to fields as needed.
        for rds in itr_rd:

            try:
                msg = 'Processing URI(s): {0}'.format([rd.uri for rd in rds])
            except AttributeError:
                # Field objects have no URIs. Multivariate calculations change how the request dataset iterator is
                # configured as well.
                msg = []
                for rd in rds:
                    try:
                        msg.append(rd.uri)
                    except AttributeError:
                        # Likely a field object which does have a name.
                        msg.append(rd.name)
                msg = 'Processing URI(s) / field names: {0}'.format(msg)
            ocgis_lh(msg=msg, logger=self._subset_log)

            for coll, original_rotated_pole_crs in self._process_subsettables_(rds):
                # If there are calculations, do those now and return a collection.
                if self.cengine is not None:
                    ocgis_lh('Starting calculations.', self._subset_log)

                    # Look for any temporal grouping optimizations.
                    if self.ops.optimizations is None:
                        tgds = None
                    else:
                        tgds = self.ops.optimizations.get('tgds')

                    # Execute the calculations.
                    if not coll.is_empty:
                        coll = self.cengine.execute(coll, file_only=self.ops.file_only, tgds=tgds)

                    # If we need to spatially aggregate and calculations used raw values, update the collection
                    # fields and subset geometries.
                    if self.ops.aggregate and self.ops.calc_raw:
                        coll_to_itr = coll.copy()
                        for sfield, container in coll_to_itr.iter_fields(yield_container=True):
                            sfield = _update_aggregation_wrapping_crs_(self, None, original_rotated_pole_crs,
                                                                       sfield, container, None)
                            coll.add_field(sfield, container, force=True)
                else:
                    # If there are no calculations, mark progress to indicate a geometry has been completed.
                    self._progress.mark()

                # # Spatially reorder the data.
                # ocgis_lh(msg='before spatial reorder', logger=self._subset_log, level=logging.DEBUG)
                # if self.ops.spatial_reorder:
                #     for field_to_reorder in coll.iter_fields():
                #         if field_to_reorder.grid is not None:
                #             wrapped_state = field_to_reorder.grid.wrapped_state
                #             if wrapped_state == WrappedState.WRAPPED:
                #                 field_to_reorder.grid.reorder()
                #             else:
                #                 msg = 'Reorder not supported for wrapped state "{}". Doing nothing.'.format(
                #                     str(wrapped_state))
                #                 ocgis_lh(msg=msg, logger=self._subset_log, level=logging.WARN)

                # Conversion of groups.
                if self.ops.output_grouping is not None:
                    raise NotImplementedError
                else:
                    ocgis_lh('subset yielding', self._subset_log, level=logging.DEBUG)
                    yield coll

    def _process_subsettables_(self, rds):
        """
        :param rds: Sequence of :class:~`ocgis.RequestDataset` objects.
        :type rds: sequence
        :rtype: :class:`ocgis.collection.base.AbstractCollection`
        """

        ocgis_lh(msg='entering _process_geometries_', logger=self._subset_log, level=logging.DEBUG)

        # This is used to define the group of request datasets for these like logging and exceptions.
        try:
            alias = '_'.join([r.field_name for r in rds])
        except AttributeError:
            # Allow field objects with do not expose the "field_name" attribute.
            try:
                alias = '_'.join([r.name for r in rds])
            except TypeError:
                # The alias is used for logging, etc. If it cannot be constructed easily, leave it as None.
                alias = None

        ocgis_lh('processing...', self._subset_log, alias=alias, level=logging.DEBUG)
        # Create the field object. Field objects may be passed directly to operations.
        try:
            # Look for field optimizations. Field optimizations typically include pre-loaded datetime objects.
            if self.ops.optimizations is not None and 'fields' in self.ops.optimizations:
                ocgis_lh('applying optimizations', self._subset_log, level=logging.DEBUG)
                field = [self.ops.optimizations['fields'][rd.field_name] for rd in rds]
            # No field optimizations.
            else:
                ocgis_lh('creating field objects', self._subset_log, level=logging.DEBUG)
                len_rds = len(rds)
                field = [None] * len_rds
                for ii in range(len_rds):
                    rds_element = rds[ii]
                    try:
                        field_object = rds_element.get(format_time=self.ops.format_time,
                                                       grid_abstraction=self.ops.abstraction)
                    except (AttributeError, TypeError):
                        # Likely a field object which does not need to be loaded from source.
                        if not self.ops.format_time:
                            raise NotImplementedError
                        # Check that is indeed a field before a proceeding.
                        if not isinstance(rds_element, OcgField):
                            raise
                        field_object = rds_element

                    field[ii] = field_object

                # Multivariate calculations require pulling variables across fields.
                if self._has_multivariate_calculations and len(field) > 1:
                    for midx in range(1, len(field)):
                        # Use the data variable tag if it is available. Otherwise, attempt to merge the fields raising
                        # warning if the variable exists in the squashed field.
                        if len(field[midx].data_variables) > 0:
                            vitr = field[midx].data_variables
                            is_data = True
                        else:
                            vitr = field[midx].values()
                            is_data = False
                        for mvar in vitr:
                            mvar = mvar.extract()
                            field[0].add_variable(mvar, is_data=is_data)

                # The first field in the list is always the target for other operations.
                field = field[0]

                # Apply any time or level subsetting provided through operations.
                if self.ops.time_range is not None:
                    field = field.time.get_between(*self.ops.time_range).parent
                if self.ops.time_region is not None:
                    field = field.time.get_time_region(self.ops.time_region).parent
                if self.ops.time_subset_func is not None:
                    field = field.time.get_subset_by_function(self.ops.time_subset_func).parent
                if self.ops.level_range is not None:
                    field = field.level.get_between(*self.ops.level_range).parent

                # Spatially reorder the data.
                ocgis_lh(msg='before spatial reorder', logger=self._subset_log, level=logging.DEBUG)
                if self.ops.spatial_reorder:
                    _update_wrapping_(self, field)
                    if field.grid is not None:
                        wrapped_state = field.grid.wrapped_state
                        if wrapped_state == WrappedState.WRAPPED:
                            field.grid.reorder()
                        else:
                            msg = 'Reorder not relevant for wrapped state "{}". Doing nothing.'.format(
                                str(wrapped_state))
                            ocgis_lh(msg=msg, logger=self._subset_log, level=logging.WARN)

                # Extrapolate the spatial bounds if requested.
                # TODO: Rename "interpolate" to "extrapolate".
                if self.ops.interpolate_spatial_bounds:
                    try:
                        name_x_variable = '{}_{}'.format(field.grid.x.name, constants.OCGIS_BOUNDS)
                        name_y_variable = '{}_{}'.format(field.grid.y.name, constants.OCGIS_BOUNDS)
                        field.grid.set_extrapolated_bounds(name_x_variable, name_y_variable, constants.OCGIS_BOUNDS)
                    except BoundsAlreadyAvailableError:
                        msg = 'Bounds/corners already on object. Ignoring "interpolate_spatial_bounds".'
                        ocgis_lh(msg=msg, logger=self._subset_log, level=logging.WARNING)

        # This error is related to subsetting by time or level. Spatial subsetting occurs below.
        except EmptySubsetError as e:
            if self.ops.allow_empty:
                ocgis_lh(msg='time or level subset empty but empty returns allowed', logger=self._subset_log,
                         level=logging.WARN)
                coll = self._get_initialized_collection_()
                name = '_'.join([rd.field_name for rd in rds])
                field = OcgField(name=name)
                coll.add_field(field, None)
                try:
                    yield coll, None
                finally:
                    return
            else:
                # Raise an exception as empty subsets are not allowed.
                ocgis_lh(exc=ExtentError(message=str(e)), alias=str([rd.field_name for rd in rds]),
                         logger=self._subset_log)

        # Set iterator based on presence of slice. Slice always overrides geometry.
        if self.ops.slice is not None:
            itr = [None]
        else:
            itr = [None] if self.ops.geom is None else self.ops.geom

        for coll, original_rotated_pole_crs in self._process_geometries_(itr, field, alias):
            # Conform units following the spatial subset.
            if self.ops.conform_units_to is not None:
                for to_conform in coll.iter_fields():
                    for dv in to_conform.data_variables:
                        dv.cfunits_conform(self.ops.conform_units_to)

            yield coll, original_rotated_pole_crs

    @staticmethod
    def _get_initialized_collection_():
        coll = SpatialCollection()
        return coll

    def _get_update_rotated_pole_state_(self, field, subset_field):
        """
        Rotated pole coordinate systems are handled internally by transforming the CRS to a geographic coordinate
        system.

        :param field:
        :type field: :class:`ocgis.new_interface.field.OcgField`
        :param subset_field:
        :type subset_field: :class:`ocgis.new_interface.field.OcgField` or None
        :rtype: None or :class:`ocgis.interface.base.crs.CFRotatedPole`
        :raises: AssertionError
        """

        # CFRotatedPole takes special treatment. only do this if a subset geometry is available. this variable is
        # needed to determine if backtransforms are necessary.
        original_rotated_pole_crs = None
        if isinstance(field.crs, CFRotatedPole):
            # Only transform if there is a subset geometry.
            if subset_field is not None or self.ops.aggregate or self.ops.spatial_operation == 'clip':
                # Update the CRS. Copy the original CRS for possible later transformation back to rotated pole.
                original_rotated_pole_crs = deepcopy(field.crs)
                ocgis_lh('initial rotated pole transformation...', self._subset_log, level=logging.DEBUG)
                field.update_crs(env.DEFAULT_COORDSYS)
                ocgis_lh('...finished initial rotated pole transformation', self._subset_log, level=logging.DEBUG)
        return original_rotated_pole_crs

    def _assert_abstraction_available_(self, field):
        """
        Assert the spatial abstraction may be loaded on the field object if one is provided in the operations.

        :param field:
        :type field: :class:`ocgis.new_interface.field.OcgField`
        """

        if self.ops.abstraction != 'auto':
            is_available = field.grid.is_abstraction_available(self.ops.abstraction)
            if not is_available:
                msg = 'A "{0}" spatial abstraction is not available.'.format(self.ops.abstraction)
                ocgis_lh(exc=ValueError(msg), logger='subset')

    def _get_slice_or_snippet_(self, field):
        """
        Slice the incoming field if a slice or snippet argument is present.

        :param field:
        :type field: :class:`ocgis.new_interface.field.OcgField`
        :rtype: :class:`ocgis.new_interface.field.OcgField`
        """

        # If there is a snippet, return the first realization, time, and level.
        if self.ops.snippet:
            the_slice = {'time': 0, 'realization': 0, 'level': 0}
        # If there is a slice, use it to subset the field. Only field slices are supported.
        elif self.ops.slice is not None:
            the_slice = self.ops.slice
        else:
            the_slice = None
        if the_slice is not None:
            field = field.get_field_slice(the_slice, strict=False, distributed=True)
        return field

    def _get_spatially_subsetted_field_(self, alias, field, subset_field, subset_ugid):
        """
        Spatially subset a field with a selection geometry.

        :param str alias: The request data alias currently being processed.
        :param field:
        :type field: :class:`ocgis.new_interface.field.OcgField`
        :param subset_field:
        :type subset_field: :class:`ocgis.new_interface.field.OcgField`
        :rtype: :class:`ocgis.new_interface.field.OcgField`
        :raises: AssertionError, ExtentError
        """
        assert subset_field is not None

        # subset_geom = subset_field.single.geom
        subset_geom = subset_field.geom.value.flatten()[0]

        # # Check for unique subset geometry identifiers. This is an issue with point subsetting as the buffer radius
        # # changes by dataset.
        # if subset_ugid in self._ugid_unique_store:
        #     # Only update if the geometry is unique.
        #     if not any([__.almost_equals(subset_geom) for __ in self._geom_unique_store]):
        #         prev_ugid = subset_ugid
        #         ugid = max(self._ugid_unique_store) + 1
        #
        #         # Update the geometry property and unique identifier.
        #         subset_field.properties['UGID'][0] = ugid
        #         subset_field.uid[:] = ugid
        #
        #         self._ugid_unique_store.append(ugid)
        #         self._geom_unique_store.append(subset_geom)
        #         msg = 'Updating UGID {0} to {1} to maintain uniqueness.'.format(prev_ugid, ugid)
        #         ocgis_lh(msg, self._subset_log, level=logging.WARN, alias=alias, ugid=ugid)
        #     else:
        #         pass
        # else:
        #     self._ugid_unique_store.append(subset_ugid)
        #     self._geom_unique_store.append(subset_geom)

        ocgis_lh('executing spatial subset operation', self._subset_log, level=logging.DEBUG, alias=alias,
                 ugid=subset_ugid)
        sso = SpatialSubsetOperation(field)
        try:
            # Execute the spatial subset and return the subsetted field.
            sfield = sso.get_spatial_subset(self.ops.spatial_operation, subset_field.geom,
                                            select_nearest=self.ops.select_nearest,
                                            optimized_bbox_subset=self.ops.optimized_bbox_subset)
        except EmptySubsetError as e:
            if self.ops.allow_empty:
                ocgis_lh(alias=alias, ugid=subset_ugid, msg='Empty geometric operation but empty returns allowed.',
                         level=logging.WARN)
                sfield = OcgField(name=field.name)
            else:
                msg = ' This typically means the selection geometry falls outside the spatial domain of the target ' \
                      'dataset.'
                msg = str(e) + msg
                ocgis_lh(exc=ExtentError(message=msg), alias=alias, logger=self._subset_log)

        # If the subset geometry is unwrapped and the vector wrap option is true, wrap the subset geometry.
        if self.ops.vector_wrap:
            if subset_field.wrapped_state == WrappedState.UNWRAPPED:
                subset_field.wrap()

        return sfield

    def _get_buffered_subset_geometry_if_point_(self, field, subset_field):
        """
        If the subset geometry is a point of multipoint, it will need to be buffered and the spatial dimension updated
        accordingly. If the subset geometry is a polygon, pass through.

        :param field:
        :type field: :class:`ocgis.new_interface.field.OcgField`
        :param subset_field:
        :type subset_field: :class:`ocgis.new_interface.field.OcgField`
        """

        if subset_field.geom.geom_type in ['Point', 'MultiPoint']:
            ocgis_lh(logger=self._subset_log, msg='buffering point geometry', level=logging.DEBUG)
            subset_field = subset_field.geom.get_buffer(self.ops.search_radius_mult * field.grid.resolution).parent
        assert subset_field.geom.geom_type in ['Polygon', 'MultiPolygon']

        return subset_field

    def _get_regridded_field_with_subset_(self, sfield, subset_field_for_regridding=None, with_buffer=True):
        """
        Regrid ``sfield`` subsetting the regrid destination in the process.

        :param sfield: The input field to regrid.
        :type sfield: :class:`ocgis.OcgField`
        :param subset_field_for_regridding: The original, unaltered spatial dimension to use for subsetting.
        :type subset_field_for_regridding: :class:`ocgis.OcgField`
        :param bool with_buffer: If ``True``, buffer the geometry used to subset the destination grid.
        :rtype: :class:`~ocgis.OcgField`
        """

        from ocgis.regrid.base import RegridOperation
        ocgis_lh(logger=self._subset_log, msg='Starting regrid operation...', level=logging.INFO)
        ro = RegridOperation(sfield, self.ops.regrid_destination, subset_field=subset_field_for_regridding,
                             with_buffer=with_buffer, regrid_options=self.ops.regrid_options)
        sfield = ro.execute()
        ocgis_lh(logger=self._subset_log, msg='Regrid operation complete.', level=logging.INFO)
        return sfield

    def _process_geometries_(self, itr, field, alias):
        """
        :param sequence itr: An iterator yielding :class:`~ocgis.SpatialDimension` objects.
        :param :class:`ocgis.interface.Field` field: The field object to use for
         operations.
        :param sequence headers: Sequence of strings to use as headers for the
         creation of the collection.
        :param sequence value_keys: Sequence of strings to use as headers for the
         keyed output functions.
        :param str alias: The request data alias currently being processed.
        :rtype: :class:~`ocgis.SpatialCollection`
        """

        ocgis_lh('processing geometries', self._subset_log, level=logging.DEBUG)
        # Process each geometry.
        for subset_field in itr:
            # Always work with a copy of the subset geometry. This gets twisted in interesting ways depending on the
            # subset target with wrapping, coordinate system conversion, etc.
            subset_field = deepcopy(subset_field)

            if self.ops.regrid_destination is not None:
                # If there is regridding, make another copy as this geometry may be manipulated during subsetting of
                # sources.
                subset_field_for_regridding = deepcopy(subset_field)

            # Operate on the rotated pole coordinate system by first transforming it to CFWGS84.
            original_rotated_pole_crs = self._get_update_rotated_pole_state_(field, subset_field)

            # Initialize the collection storage.
            coll = self._get_initialized_collection_()

            # Check if the geometric abstraction is available on the field object.
            self._assert_abstraction_available_(field)

            # Return a slice or snippet if either of these are requested.
            field = self._get_slice_or_snippet_(field)

            # Choose the subset UGID value.
            if subset_field is None:
                msg = 'No selection geometry. Returning all data. No unique geometry identifier.'
                subset_ugid = None
            else:
                subset_ugid = subset_field.geom.ugid.value[0]
                msg = 'Subsetting with selection geometry having UGID={0}'.format(subset_ugid)
            ocgis_lh(msg=msg, logger=self._subset_log)

            if subset_field is not None:
                # If the CRS's differ, update the spatial dimension to match the field.
                if subset_field.crs is not None and subset_field.crs != field.crs:
                    subset_field.update_crs(field.crs)
                # If the geometry is a point, it needs to be buffered.
                subset_field = self._get_buffered_subset_geometry_if_point_(field, subset_field)

            # If there is a selection geometry present, use it for the spatial subset. if not, all the field's data is
            # being returned.
            if subset_field is None:
                sfield = field
            else:
                sfield = self._get_spatially_subsetted_field_(alias, field, subset_field, subset_ugid)

            # If the base size is being requested, bypass the rest of the operations.
            if not self._request_base_size_only:
                # Perform regridding operations if requested.
                if self.ops.regrid_destination is not None and sfield._should_regrid:
                    sfield = self._get_regridded_field_with_subset_(
                        sfield,
                        subset_field_for_regridding=subset_field_for_regridding,
                        with_buffer=True)
                # If empty returns are allowed, there may be an empty field.
                if sfield is not None:
                    # Only update spatial stuff if there are no calculations and, if there are calculations, those
                    # calculations are not expecting raw values.
                    if self.ops.calc is None or (self.ops.calc is not None and not self.ops.calc_raw):
                        # Update spatial aggregation, wrapping, and coordinate systems.
                        sfield = _update_aggregation_wrapping_crs_(self, alias, original_rotated_pole_crs, sfield,
                                                                   subset_field, subset_ugid)

            # Add the created field to the output collection with the selection geometry.
            coll.add_field(sfield, subset_field)

            yield coll, original_rotated_pole_crs


def _update_aggregation_wrapping_crs_(obj, alias, original_rotated_pole_crs, sfield, subset_sdim, subset_ugid):
    # Aggregate if requested.
    if obj.ops.aggregate:
        ocgis_lh('executing spatial average', obj._subset_log, alias=alias, ugid=subset_ugid, level=logging.DEBUG)

        # There may be no geometries if we are working with a gridded dataset. Load the geometries if this is the case.
        sfield.set_abstraction_geom()

        # Union the geometries and spatially average the data variables.
        sfield = sfield.geom.get_unioned(spatial_average=sfield.data_variables).parent

        if subset_sdim is not None and subset_sdim.geom is not None:
            # Add the unique geometry identifier variable. This should match the selection geometry's identifier.
            new_gid_variable = Variable(name=HeaderNames.ID_GEOMETRY, value=subset_sdim.geom.ugid.value,
                                        dimensions=sfield.geom.dimensions)
            sfield.geom.set_ugid(new_gid_variable)

    # Wrap the returned data.
    if not env.OPTIMIZE_FOR_CALC:
        if sfield is not None and sfield.wrapped_state == WrappedState.UNWRAPPED:
            ocgis_lh('wrap target is empty: {}'.format(sfield.is_empty), obj._subset_log, level=logging.DEBUG)

            # There may be no geometries if we are working with a gridded dataset. Load the geometries if this
            # is the case.
            sfield.set_abstraction_geom()

            if not sfield.is_empty:
                if obj.ops.output_format in constants.VECTOR_OUTPUT_FORMATS and obj.ops.vector_wrap:
                    ocgis_lh('wrapping output geometries', obj._subset_log, alias=alias, ugid=subset_ugid,
                             level=logging.DEBUG)

                    # Deepcopy geometries before wrapping as wrapping will be performed inplace. The original field may
                    # need to be reused for additional subsets.
                    geom = sfield.geom
                    copied_geom = geom.get_value().copy()
                    geom.set_value(copied_geom)
                    geom.wrap()
                    ocgis_lh('finished wrapping output geometries', obj._subset_log, alias=alias, ugid=subset_ugid,
                             level=logging.DEBUG)

    # Transform back to rotated pole if necessary.
    if original_rotated_pole_crs is not None:
        if not isinstance(obj.ops.output_crs, CFWGS84):
            sfield.spatial.update_crs(original_rotated_pole_crs)

    # Update the coordinate system of the data output.
    if obj.ops.output_crs is not None:

        # If the geometry is not none, it may need to be projected to match the output coordinate system.
        if subset_sdim is not None and subset_sdim.crs != obj.ops.output_crs:
            subset_sdim.update_crs(obj.ops.output_crs)

        # Update the subsetted field's coordinate system.
        sfield = sfield.copy()
        sfield.update_crs(obj.ops.output_crs)

    # Wrap or unwrap the data if the coordinate system permits.
    _update_wrapping_(obj, sfield)

    return sfield


def _update_wrapping_(obj, field_object):
    """
    Update the wrapped state of the incoming field object. This only affects fields with wrappable coordinate systems.

    :param obj: :class:`ocgis.driver.subset.OperationsEngine`
    :param field_object: :class:`ocgis.Field`
    """
    if obj.ops.spatial_wrapping is not None:
        if field_object.crs is not None and field_object.crs.is_geographic:
            as_enum = obj.ops._get_object_('spatial_wrapping').as_enum
            if field_object.wrapped_state != as_enum:
                if as_enum == WrapAction.WRAP:
                    field_object.wrap()
                else:
                    field_object.unwrap()
