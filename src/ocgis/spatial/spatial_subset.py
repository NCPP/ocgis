from copy import deepcopy, copy

from ocgis import env, vm
from ocgis.base import raise_if_empty, AbstractOcgisObject
from ocgis.collection.field import Field
from ocgis.constants import WrappedState
from ocgis.variable.crs import CFRotatedPole, CFSpherical, Spherical
from ocgis.variable.geom import GeometryVariable


class SpatialSubsetOperation(AbstractOcgisObject):
    """
    Perform spatial subsets using :class:`~ocgis.Field` objects.

    :param field: The target field to subset.
    :type field: :class:`~ocgis.Field`
    :param output_crs: If provided, all output coordinates will be remapped to match. If ``'input'``, the default,
     use the coordinate system assigned to ``field``.
    :type output_crs: :class:`~ocgis.variable.crs.AbstractCRS`
    :param wrap: This is only relevant for spherical coordinate systems on ``field`` or when selected as the
     ``output_crs``. If ``None``, leave the wrapping the same as ``field``. If ``True``, wrap the coordinates. If
     ``False``, unwrap the coordinates. A "wrapped" spherical coordinate system has a longitudinal domain from -180 to
     180 degrees.
    :type wrap: bool
    """

    _rotated_pole_destination_crs = env.DEFAULT_COORDSYS

    def __init__(self, field, output_crs='input', wrap=None):
        if not isinstance(field, Field):
            raise ValueError('"field" must be an "Field" object.')
        raise_if_empty(field)

        self.field = field
        self.output_crs = output_crs
        self.wrap = wrap

        self._original_rotated_pole_state = None
        self._transformed_unwrapped_select = None

    @property
    def should_update_crs(self):
        """Return ``True`` if output from ``get_spatial_subset`` needs to have its CRS updated."""

        if self.output_crs == 'input':
            ret = False
        elif self.output_crs != self.field.crs:
            ret = True
        else:
            ret = False
        return ret

    def get_spatial_subset(self, operation, geom, use_spatial_index=env.USE_SPATIAL_INDEX, buffer_value=None,
                           buffer_crs=None, geom_crs=None, select_nearest=False, optimized_bbox_subset=False,
                           keep_touches='auto', return_slice=False, cascade=True):
        """
        Perform a spatial subset operation on ``target``.

        :param str operation: Either ``'intersects'`` or ``'clip'``.
        :param geom: The input geometry object to use for subsetting of ``target``.
        :type geom: :class:`shapely.geometry.base.BaseGeometry` | :class:`ocgis.GeometryVariable`
        :param bool use_spatial_index: If ``True``, use an ``rtree`` spatial index.
        :param bool select_nearest: If ``True``, select the geometry nearest ``polygon`` using
         :meth:`shapely.geometry.base.BaseGeometry.distance`.
        :rtype: Same as ``target``. If ``target`` is a :class:`ocgis.RequestDataset`,
         then a :class:`ocgis.interface.base.field.Field` will be returned.
        :param float buffer_value: The buffer radius to use in units of the coordinate system of ``subset_sdim``.
        :param buffer_crs: If provided, then ``buffer_value`` are not in units of the coordinate system of
         ``subset_sdim`` but in units of ``buffer_crs``.
        :param geom_crs: The coordinate reference system for the subset geometry.
        :type geom_crs: :class:`ocgis.crs.CRS`
        :param bool select_nearest: If ``True``, following the spatial subset operation, select the nearest geometry
         in the subset data to ``geom``. Centroid-based distance is used.
        :type buffer_crs: :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
        :param bool optimized_bbox_subset: If ``True``, only do a bounding box subset and do not perform more complext
         GIS subset operations such as constructing a spatial index.
        :param keep_touches: If ``'auto'`` (the default), keep geometries that touch only if the grid's spatial
         abstraction is point.
        :type keep_touches: :class:`bool` | :class:`str`
        :param bool return_slice: If ``True``, also return the slices used to limit the target's extent.
        :param cascade: If ``True`` (the default), set the mask across all variables in the target's parent collection.
        :return: If ``return_slice`` is ``False`` (the default), return the subsetted field. If ``return_slice`` is
         ``True``, this will be a tuple with the subsetted object as the first element and the slice used as the second.
        :rtype: :class:`~ocgis.Field` | :class:`tuple` of ``(<:class:`~ocgis.Field`>, <slice used>)``
        :raises: ValueError
        """

        # Parameter prep and validation --------------------------------------------------------------------------------

        if not isinstance(geom, GeometryVariable):
            geom = GeometryVariable(value=geom, name='geom', dimensions='one', crs=geom_crs)
        if geom.get_value().flatten().shape != (1,):
            msg = 'Only one subset geometry allowed. The shape of the geometry variable is {}.'.format(geom.shape)
            raise ValueError(msg)
        if optimized_bbox_subset:
            if self.field.grid is None:
                msg = 'Subset operation must be performed on a grid when "optimized_bbox_subset=True".'
                raise ValueError(msg)
            if operation != 'intersects':
                msg = 'Only "intersects" spatial operations when "optimized_bbox_subset=True".'
                raise ValueError(msg)

        # --------------------------------------------------------------------------------------------------------------

        # Buffer the subset if a buffer value is provided.
        if buffer_value is not None:
            geom = self._get_buffered_geometry_(geom, buffer_value, buffer_crs=buffer_crs)
        prepared = self._prepare_geometry_(geom)
        base_geometry = prepared.get_value().flatten()[0]

        # Prepare the target field.
        self._prepare_target_()

        # execute the spatial operation
        slc = None
        if operation == 'intersects':
            if self.field.grid is None:
                if keep_touches == 'auto':
                    keep_touches = False
                ret = self.field.geom.get_intersects(base_geometry, use_spatial_index=use_spatial_index,
                                                     cascade=cascade, return_slice=return_slice, keep_touches=keep_touches)
                if return_slice:
                    ret, slc = ret
                ret = ret.parent
            else:
                ret = self.field.grid.get_intersects(base_geometry, cascade=cascade,
                                                     optimized_bbox_subset=optimized_bbox_subset,
                                                     return_slice=return_slice, keep_touches=keep_touches)
                if return_slice:
                    ret, slc = ret
                ret = ret.parent
        elif operation in ('clip', 'intersection'):
            if self.field.grid is None:
                ret = self.field.geom.get_intersection(base_geometry, use_spatial_index=use_spatial_index,
                                                       cascade=cascade, return_slice=return_slice, keep_touches=keep_touches)
                if return_slice:
                    ret, slc = ret
                ret = ret.parent
            else:
                ret = self.field.grid.get_intersection(base_geometry, cascade=cascade, return_slice=return_slice, keep_touches=keep_touches)
                if return_slice:
                    ret, slc = ret
                # An intersection with a grid returns a geometry variable. Set this on the field.
                ret.parent.set_geom(ret)
                ret = ret.parent
        else:
            msg = 'The spatial operation "{0}" is not supported.'.format(operation)
            raise ValueError(msg)

        with vm.scoped_by_emptyable('return finalize', ret):
            if not vm.is_null:
                # Select the nearest geometry if requested.
                if select_nearest:
                    ret.set_abstraction_geom()
                    ret = ret.geom.get_nearest(base_geometry).parent

                # check for rotated pole and convert back to default CRS
                if self._original_rotated_pole_state is not None and self.output_crs == 'input':
                    ret.update_crs(self._original_rotated_pole_state)

                # wrap the data...
                if self._get_should_wrap_(ret):
                    ret.wrap()

                # convert the coordinate system if requested...
                if self.should_update_crs:
                    ret.update_crs(self.output_crs)

        # Return the target field to its original state.
        self._finalize_target()

        if return_slice:
            ret = (ret, slc)

        return ret

    @staticmethod
    def _get_buffered_geometry_(geom, buffer_value, buffer_crs=None):
        """
        Buffer a spatial dimension. If ``buffer_crs`` is provided, then ``buffer_value`` are in units of ``buffer_crs``
        and the coordinate system of ``geom`` may need to be updated.

        :param subset_sdim: The spatial dimension object to buffer.
        :type subset_sdim: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        :param float buffer_value: The buffer radius to use in units of the coordinate system of ``subset_sdim``.
        :param buffer_crs: If provided, then ``buffer_value`` are not in units of the coordinate system of
         ``subset_sdim`` but in units of ``buffer_crs``.
        :type buffer_crs: :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
        :rtype: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        """

        to_buffer = deepcopy(geom)
        if buffer_crs is not None:
            to_buffer.update_crs(buffer_crs)
        to_buffer.get_value()[0] = to_buffer.get_value()[0].buffer(buffer_value, cap_style=3)
        if buffer_crs is not None:
            to_buffer.update_crs(geom.crs)
        return to_buffer

    def _get_should_wrap_(self, field):
        """
        Return ``True`` if the output from ``get_spatial_subset`` should be wrapped.

        :param field: The target field to test for wrapped stated.
        :type fiedl: :class:`ocgis.new_interface.field.Field`
        """

        # The output needs to be wrapped and the input data is unwrapped. Output from get_spatial_subset is always
        # wrapped relative to the input target.
        if self.wrap and field.wrapped_state == WrappedState.UNWRAPPED:
            ret = True
        else:
            ret = False

        return ret

    def _prepare_geometry_(self, geom):
        """
        Compare ``geom`` geographic state with the target field and perform any necessary transformations to ensure a
        smooth subset operation.

        :param geom: The input geometry to use for subsetting.
        :type geom: :class:`~ocgis.GeometryVariable`
        :rtype: :class:`~ocgis.GeometryVariable`
        """
        assert isinstance(geom, GeometryVariable)

        # The subset geometry may be modified during this transaction. Use a deep copy to preserve the original
        # geometry's state to avoid error accumulations during transformations.
        prepared = geom.deepcopy()

        if geom.crs is not None:
            assert prepared.crs is not None

        # Update the subset geometry's coordinate system to match the field's.
        if isinstance(self.field.crs, CFRotatedPole):
            prepared.update_crs(CFSpherical())
        else:
            if prepared.crs is None and self.field.crs is not None:
                msg = "The subset geometry has no assigned CRS and the target field does. Set the subset geometry's " \
                      "CRS to continue."
                raise ValueError(msg)
            if prepared.crs is not None and self.field.crs is not None and prepared.crs != self.field.crs:
                prepared.update_crs(self.field.crs)

        # Update the subset geometry's spatial wrapping to match the target field.
        field_wrapped_state = self.field.wrapped_state
        prepared_wrapped_state = prepared.wrapped_state
        if field_wrapped_state == WrappedState.UNWRAPPED:
            if prepared_wrapped_state == WrappedState.WRAPPED:
                prepared.unwrap()
        elif field_wrapped_state == WrappedState.WRAPPED:
            if prepared_wrapped_state == WrappedState.UNWRAPPED:
                prepared.wrap()

        return prepared

    def _finalize_target(self):
        """Perform any transformations on the target field following spatial subsetting."""
        if self._transformed_unwrapped_select is not None:
            self.field.grid.x.v()[self._transformed_unwrapped_select] -= 360
            self._transformed_unwrapped_select = None

    def _prepare_target_(self):
        """Perform any transformations on the target field in preparation for spatial subsetting."""

        field = self.field
        crs = field.crs
        if isinstance(crs, CFRotatedPole):
            self._original_rotated_pole_state = copy(crs)
            field.update_crs(self._rotated_pole_destination_crs)
        elif field.grid is not None and isinstance(crs, Spherical) and field.wrapped_state == WrappedState.UNWRAPPED:
            xval = field.grid.x.v()
            select = xval < 0
            if select.any():
                xval[select] += 360
                self._transformed_unwrapped_select = select
