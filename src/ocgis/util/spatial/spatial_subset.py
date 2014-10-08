from copy import deepcopy, copy
from ocgis.interface.base.crs import CFRotatedPole, CFWGS84
from ocgis.interface.base.dimension.spatial import SpatialDimension
from ocgis import RequestDataset


class SpatialSubsetOperation(object):
    """
    Perform spatial subsets on various types of ``ocgis`` data objects.

    :param target: The target object to subset.
    :type target: :class:`ocgis.interface.base.field.Field` or
     :class:`ocgis.interface.base.dimension.spatial.SpatialDimension` or :class:`ocgis.RequestDataset`
    :param output_crs: If provided, all output coordinates will be remapped to match.
    :type output_crs: :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
    :param wrap: This is only relevant for spherical coordinate systems on ``target`` or when selected as the
     ``output_crs``. If ``None``, leave the wrapping the same as ``target``. If ``True``, wrap the coordinates. If
     ``False``, unwrap the coordinates. A "wrapped" spherical coordinate system has a longitudinal domain from -180 to
     180 degrees.
    :type wrap: bool
    :param bool use_cache: If ``True``, create a dictionary-based cache to retrieve previous subsets.
    """

    _rotated_pole_destination_crs = CFWGS84

    def __init__(self, target, output_crs='input', wrap=None):
        self.target = target
        self.output_crs = output_crs
        self.wrap = wrap

        self._original_rotated_pole_state = None
        self._field = None

    @property
    def field(self):
        """
        :rtype: :class:`ocgis.interface.base.field.Field`. If ``target`` is a
         :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`, then ``None`` is returned.
        """

        if self._field is None:
            if isinstance(self.target, RequestDataset):
                self._field = self.target.get()
            elif isinstance(self.target, SpatialDimension):
                msg = 'Target is a SpatialDimension.'
                raise AttributeError(msg)
            else:
                self._field = self.target
        return self._field

    @property
    def sdim(self):
        """
        :rtype: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        """

        try:
            ret = self.field.spatial
        # if "target" is a spatial dimension, then the field is None. return the object assuming it is a spatial
        # dimension.
        except AttributeError:
            ret = self.target
        return ret

    @property
    def should_update_crs(self):
        """Return ``True`` if output from ``get_spatial_subset`` needs to have its CRS updated."""

        if self.output_crs == 'input':
            ret = False
        elif self.output_crs != self.sdim.crs:
            ret = True
        else:
            ret = False
        return ret

    def get_spatial_subset(self, operation, subset_sdim, use_spatial_index=True, select_nearest=False,
                           buffer_value=None, buffer_crs=None):
        """
        Perform a spatial subset operation on ``target``.

        :param str operation: Either 'intersects' or 'clip'.
        :param subset_sdim: The input object to use for subsetting of ``target``.
        :type subset_sdim: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        :param bool use_spatial_index: If ``True``, use an ``rtree`` spatial index.
        :param bool select_nearest: If ``True``, select the geometry nearest ``polygon`` using
         :meth:`shapely.geometry.base.BaseGeometry.distance`.
        :rtype: Same as ``target``. If ``target`` is a :class:`ocgis.RequestDataset`,
         then a :class:`ocgis.interface.base.field.Field` will be returned.
        :param float buffer_value: The buffer radius to use in units of the coordinate system of ``subset_sdim``.
        :param buffer_crs: If provided, then ``buffer_value`` are not in units of the coordinate system of
         ``subset_sdim`` but in units of ``buffer_crs``.
        :type buffer_crs: :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
        :raises: ValueError
        """

        if subset_sdim.geom.polygon.shape != (1, 1):
            msg = 'Only one Polygon/MultiPolygon for a spatial operation.'
            raise ValueError(msg)

        # buffer the subset if a buffer value is provided
        if buffer_value is not None:
            subset_sdim = self._get_buffered_subset_sdim_(subset_sdim, buffer_value, buffer_crs=buffer_crs)

        if operation == 'clip':
            try:
                method = self.field.get_clip
            except AttributeError:
                # target is likely a spatial dimension
                method = self.sdim.get_clip
        elif operation == 'intersects':
            try:
                method = self.field.get_intersects
            except AttributeError:
                # target is likely a spatial dimension
                method = self.sdim.get_intersects
        else:
            msg = 'The spatial operation "{0}" is not supported.'.format(operation)
            raise ValueError(msg)

        self._prepare_target_()
        prepared = self._prepare_subset_sdim_(subset_sdim)
        polygon = prepared.geom.polygon.value[0, 0]

        # execute the spatial operation
        ret = method(polygon, use_spatial_index=use_spatial_index, select_nearest=select_nearest)

        # check for rotated pole and convert back to default CRS
        if self._original_rotated_pole_state is not None and self.output_crs == 'input':
            try:
                ret.spatial.update_crs(self._original_rotated_pole_state)
            except AttributeError:
                # like a spatial dimension
                ret.update_crs(self._original_rotated_pole_state)

        # wrap the data...
        if self._get_should_wrap_(ret):
            try:
                ret.spatial.wrap()
            except AttributeError:
                # likely a spatial dimension
                ret.wrap()

        # convert the coordinate system if requested...
        if self.should_update_crs:
            try:
                ret.spatial.update_crs(self.output_crs)
            except AttributeError:
                # likely a spatial dimension
                ret.update_crs(self.output_crs)

        return ret

    @staticmethod
    def _get_buffered_subset_sdim_(subset_sdim, buffer_value, buffer_crs=None):
        """
        Buffer a spatial dimension. If ``buffer_crs`` is provided, then ``buffer_value`` are in units of ``buffer_crs``
        and the coordinate system of ``subset_sdim`` may need to be updated.

        :param subset_sdim: The spatial dimension object to buffer.
        :type subset_sdim: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        :param float buffer_value: The buffer radius to use in units of the coordinate system of ``subset_sdim``.
        :param buffer_crs: If provided, then ``buffer_value`` are not in units of the coordinate system of
         ``subset_sdim`` but in units of ``buffer_crs``.
        :type buffer_crs: :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
        :rtype: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        """

        subset_sdim = deepcopy(subset_sdim)
        if buffer_crs is not None:
            original_crs = deepcopy(subset_sdim.crs)
            subset_sdim.update_crs(buffer_crs)
        ref = subset_sdim.geom.polygon.value[0, 0]
        subset_sdim.geom.polygon.value[0, 0] = ref.buffer(buffer_value, cap_style=3)
        if buffer_crs is not None:
            subset_sdim.update_crs(original_crs)
        return subset_sdim

    def _get_should_wrap_(self, target):
        """
        Return ``True`` if the output from ``get_spatial_subset`` should be wrapped.

        :param target: The target to test for wrapped stated.
        :type target: :class:`ocgis.interface.base.field.Field` or
         :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        """

        try:
            sdim = target.spatial
        except AttributeError:
            # likely a spatial dimension
            sdim = target

        # the output needs to be wrapped and the input data is unwrapped. output from get_spatial_subset is always
        # wrapped according to the input spatial dimension
        if self.wrap and sdim.is_unwrapped:
            ret = True
        else:
            ret = False

        return ret

    def _prepare_subset_sdim_(self, subset_sdim):
        """
        Compare ``subset_sdim`` geographic state with ``target`` and perform any necessary transformations to ensure a
        smooth subset operation.

        :param subset_sdim: The input object to use for subsetting of ``target``.
        :type subset_sdim: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        :rtype: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
        """

        prepared = deepcopy(subset_sdim)
        prepared.update_crs(self.sdim.crs)
        if self.sdim.is_unwrapped and not prepared.is_unwrapped:
            prepared.unwrap()
        return prepared

    def _prepare_target_(self):
        """
        Perform any transformations on ``target`` in preparation for spatial subsetting.
        """

        if isinstance(self.sdim.crs, CFRotatedPole):
            self._original_rotated_pole_state = copy(self.sdim.crs)
            self.sdim.update_crs(self._rotated_pole_destination_crs())