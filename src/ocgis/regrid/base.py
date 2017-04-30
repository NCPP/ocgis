import logging
from copy import deepcopy
from types import NoneType

import ESMF
import numpy as np
from ESMF.api.constants import RegridMethod

from ocgis import constants, DimensionMap
from ocgis import env
from ocgis.base import AbstractOcgisObject, get_dimension_names
from ocgis.collection.field import Field
from ocgis.constants import DimensionMapKey, KeywordArgument
from ocgis.exc import RegriddingError, CornersInconsistentError
from ocgis.spatial.grid import Grid, expand_grid
from ocgis.spatial.spatial_subset import SpatialSubsetOperation
from ocgis.util.helpers import iter_array, get_esmf_corners_from_ocgis_corners
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.variable.base import Variable
from ocgis.variable.crs import Spherical


class RegridOperation(AbstractOcgisObject):
    """
    Execute a regrid operation handling spatial subsetting and coordinate system transformations.

    :param field_src: The source field to regrid.
    :type field_src: :class:`ocgis.Field`
    :param field_dst: The destination regrid field.
    :type field_dst: :class:`ocgis.Field`
    :param subset_field: If provided, use this field to subset the regridding fields.
    :type subset_field: :class:`ocgis.Field`
    :param regrid_options: A dictionary of keyword options to pass to :func:`~ocgis.regrid.base.regrid_field`.
    :type regrid_options: dict
    :param bool revert_dst_crs: If ``True``, revert the destination grid coordinate system if it needed to be
     transformed. Typically, a number of source fields are regridded to a common destination and this transform
     should only occur once.
    """

    def __init__(self, field_src, field_dst, subset_field=None, regrid_options=None, revert_dst_crs=False):
        assert isinstance(field_src, Field)
        assert isinstance(field_dst, Field)
        assert isinstance(subset_field, (Field, NoneType))

        self.field_dst = field_dst
        self.field_src = field_src
        self.subset_field = subset_field
        self.revert_dst_crs = revert_dst_crs
        if regrid_options is None:
            self.regrid_options = {}
        else:
            self.regrid_options = regrid_options

    def execute(self):
        """
        Execute regridding operation.

        :rtype: :class:`ocgis.Field`
        """

        regrid_destination, backtransform_dst_crs = self._get_regrid_destination_()
        regrid_source, backtransform_src_crs = self._get_regrid_source_()

        # Regrid the input field.
        ocgis_lh(logger='regrid', msg='Creating regridded field...', level=logging.INFO)
        regridded_source = regrid_field(regrid_source, regrid_destination, **self.regrid_options)

        if backtransform_src_crs is not None:
            regridded_source.update_crs(backtransform_src_crs)
            # self.field_src.update_crs(backtransform_src_crs)
        if backtransform_dst_crs is not None and self.revert_dst_crs:
            self.field_dst.update_crs(backtransform_dst_crs)

        return regridded_source

    def _get_regrid_destination_(self):
        """
        Prepare destination field for regridding.

        :rtype: (:class:`~ocgis.Field`, :class:`~ocgis.CoordinateReferenceSystem` or ``None``)
        """

        # Transform the coordinate system of the regrid destination. ###################################################

        # Update the regrid destination coordinate system must be updated to match the source.
        if self.field_dst.crs != Spherical():
            ocgis_lh(logger='regrid',
                     msg='updating regrid destination to spherical. regrid destination crs is: {}'.format(
                         self.field_dst.crs), level=logging.DEBUG)
            backtransform_crs = deepcopy(self.field_dst.crs)
            self.field_dst.update_crs(Spherical())
        else:
            backtransform_crs = None

        # Spatially subset the regrid destination. #####################################################################
        if self.subset_field is None:
            ocgis_lh(logger='regrid', msg='no spatial subsetting', level=logging.DEBUG)
            regrid_destination = self.field_dst
        else:
            ss = SpatialSubsetOperation(self.field_dst)
            regrid_destination = ss.get_spatial_subset('intersects', self.subset_field.geom,
                                                       use_spatial_index=env.USE_SPATIAL_INDEX,
                                                       select_nearest=False)

        return regrid_destination, backtransform_crs

    def _get_regrid_source_(self):
        # Update the source coordinate system to spherical.
        if self.field_src.crs != Spherical():
            ocgis_lh(logger='regrid',
                     msg='updating regrid source to spherical. regrid source crs is: {}'.format(
                         self.field_src.crs), level=logging.DEBUG)
            backtransform_crs = deepcopy(self.field_src.crs)
            self.field_src.update_crs(Spherical())
        else:
            backtransform_crs = None

        return self.field_src, backtransform_crs


def get_ocgis_grid_from_esmf_grid(egrid, crs=None, dimension_map=None):
    """
    Create an OCGIS :class:`~ocgis.interface.base.dimension.spatial.SpatialDimension` object from an ESMF
    :class:`~ESMF.driver.grid.Grid`.

    :type egrid: :class:`ESMF.driver.grid.Grid`
    :param crs: The coordinate system to attach to the output spatial dimension.
    :type crs: :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
    :param dimension_map: Dimension map for the outgoing OCGIS field/grid.
    :type dimension_map: :class:`ocgis.DimensionMap`
    :rtype: :class:`~ocgis.Grid`
    """

    if dimension_map is None:
        dimension_map = {DimensionMapKey.X: {'variable': 'x', 'bounds': 'x_bounds', DimensionMapKey.DIMS: ['x']},
                         DimensionMapKey.Y: {'variable': 'y', 'bounds': 'y_bounds', DimensionMapKey.DIMS: ['y']}}
        dimension_map = DimensionMap.from_dict(dimension_map)
    else:
        assert isinstance(dimension_map, DimensionMap)

    # OCGIS grid values are built on centers.
    coords = egrid.coords[ESMF.StaggerLoc.CENTER]
    shape_coords_list = list(coords[0].shape)
    dtype_coords = coords[0].dtype
    # construct the ocgis grid array and fill
    grid_value = np.zeros([2] + shape_coords_list, dtype=dtype_coords)
    grid_value[0, ...] = coords[1]
    grid_value[1, ...] = coords[0]

    # Build OCGIS corners array if corners are present on the ESMF grid object.
    has_corners = get_esmf_grid_has_corners(egrid)
    if has_corners:
        corner = egrid.coords[ESMF.StaggerLoc.CORNER]
        grid_corners = np.zeros([2] + shape_coords_list + [4], dtype=dtype_coords)
        slices = [(0, 0), (0, 1), (1, 1), (1, 0)]
        for ii, jj in iter_array(coords[0], use_mask=False):
            row_slice = slice(ii, ii + 2)
            col_slice = slice(jj, jj + 2)
            row_corners = corner[1][row_slice, col_slice]
            col_corners = corner[0][row_slice, col_slice]
            for kk, slc in enumerate(slices):
                grid_corners[:, ii, jj, kk] = row_corners[slc], col_corners[slc]
    else:
        grid_corners = None

    # Does the grid have a mask?
    has_mask = False
    if egrid.mask is not None:
        if egrid.mask[ESMF.StaggerLoc.CENTER] is not None:
            has_mask = True
    if has_mask:
        # if there is a mask, update the grid values
        egrid_mask = egrid.mask[ESMF.StaggerLoc.CENTER]
        egrid_mask = np.invert(egrid_mask.astype(bool))

    # actually construct the masked arrays
    grid_value = np.ma.array(grid_value)
    if grid_corners is not None:
        grid_corners = np.ma.array(grid_corners)

    grid_dimensions = [dimension_map.get_dimensions(DimensionMapKey.Y)[0],
                       dimension_map.get_dimensions(DimensionMapKey.X)[0]]
    if grid_corners is not None:
        grid_bounds_dimensions = deepcopy(grid_dimensions)
        grid_bounds_dimensions.append(constants.DEFAULT_NAME_CORNERS_DIMENSION)

        name = dimension_map.get_bounds(DimensionMapKey.X)
        x_bounds = Variable(name=name, value=grid_corners[1, ...],
                            dimensions=['y', 'x', constants.DEFAULT_NAME_CORNERS_DIMENSION])

        name = dimension_map.get_bounds(DimensionMapKey.Y)
        y_bounds = Variable(name=name, value=grid_corners[0, ...],
                            dimensions=['y', 'x', constants.DEFAULT_NAME_CORNERS_DIMENSION])
    else:
        x_bounds, y_bounds = [None] * 2

    name = dimension_map.get_variable(DimensionMapKey.X)
    x = Variable(name=name, dimensions=grid_dimensions, value=grid_value[1, ...], bounds=x_bounds)

    name = dimension_map.get_variable(DimensionMapKey.Y)
    y = Variable(name=name, dimensions=grid_dimensions, value=grid_value[0, ...], bounds=y_bounds)

    ogrid = Grid(x, y, crs=crs)

    if has_mask:
        ogrid.set_mask(egrid_mask)

    return ogrid


def get_esmf_field_from_ocgis_field(ofield, esmf_field_name=None, **kwargs):
    """
    :param ofield: The OCGIS field to convert to an ESMF field.
    :type ofield: :class:`ocgis.Field`
    :param str esmf_field_name: An optional ESMF field name. If ``None``, use the name of the data variable on
     ``ofield``.
    :param dict kwargs: Any keyword arguments to :func:`ocgis.regrid.base.get_esmf_grid`.
    :return: An ESMF field object.
    :rtype: :class:`ESMF.api.field.Field`
    :raises: ValueError
    """

    if len(ofield.data_variables) > 1:
        msg = 'Only one data variable may be converted.'
        raise ValueError(msg)

    if len(ofield.data_variables) == 1:
        target_variable = ofield.data_variables[0]
        if esmf_field_name is None:
            esmf_field_name = target_variable.name
    else:
        target_variable = None

    egrid = get_esmf_grid(ofield.grid, **kwargs)

    # Find any dimension lengths that are not associated with the grid. These are considered "extra" dimensions in an
    # ESMF field.
    if target_variable is None:
        dimension_names = get_dimension_names(ofield.grid.dimensions)
    else:
        dimension_names = get_dimension_names(target_variable.dimensions)
    grid_dimension_names = get_dimension_names(ofield.grid.dimensions)

    if dimension_names != grid_dimension_names:
        if tuple(dimension_names[-2:]) != grid_dimension_names:
            raise ValueError('Grid dimensions must be last two dimensions of the data variable dimensions.')
        ndbounds = target_variable.shape[0:-2]
    else:
        # OCGIS field has no extra dimensions.
        ndbounds = None

    efield = ESMF.Field(egrid, name=esmf_field_name, ndbounds=ndbounds)

    if target_variable is not None:
        efield.data[:] = target_variable.get_value()

    return efield


def get_esmf_grid_has_corners(egrid):
    return egrid.has_corners


def get_ocgis_field_from_esmf_field(efield, dimensions=None, **kwargs):
    """
    :param efield: The ESMPy field object to convert to an OCGIS field.
    :type efield: :class:`ESMF.driver.field.Field`
    :param crs: The coordinate system of the ESMF field. If ``None``, this will default to a coordinate system
     mapped to the ``coord_sys`` attribute of the ESMF grid.
    :param tuple dimensions: Tuple of :class:`ocgis.Dimensions` objects corresponding to the dimensions of ``efield``.
     Required if there is an incoming data variable from ``efield``. If ``None``, assume there is no data variable to
     create from the ESMF field.
    :param dict kwargs: Any keyword arguments to the creation of the output OCGIS field. Values in ``kwargs`` are used
     in preference over any internal default values created in this function. The keyword argument
     :attrs:`~ocgis.constants.KeywordArgument.IS_DATA` may not be overloaded.
    :returns: An OCGIS field object.
    :rtype: :class:`~ocgis.Field`
    """

    kwargs = kwargs.copy()

    if dimensions is not None:
        kwargs[KeywordArgument.IS_DATA] = Variable(name=efield.name, value=efield.data, dimensions=dimensions)

    grid = kwargs.pop(KeywordArgument.GRID, None)
    dimension_map = kwargs.pop(KeywordArgument.DIMENSION_MAP, None)
    if grid is None:
        crs = kwargs.pop(KeywordArgument.CRS, None)
        if crs is None:
            crs = get_crs_from_esmf_field(efield)
        grid = get_ocgis_grid_from_esmf_grid(efield.grid, crs=crs, dimension_map=dimension_map)
    kwargs[KeywordArgument.GRID] = grid

    if KeywordArgument.CRS not in kwargs:
        kwargs[KeywordArgument.CRS] = Spherical()

    field = Field(**kwargs)

    grid_mask = grid.get_mask()
    if grid_mask is not None:
        field.grid.set_mask(grid_mask, cascade=True)

    return field


def get_crs_from_esmf_field(efield):
    """
    :type efield: :class:`ESMF.Field`
    """

    mp = {ESMF.CoordSys.SPH_DEG: Spherical()}

    coord_sys = efield.grid.coord_sys
    try:
        ret = mp[coord_sys]
    except KeyError:
        msg = 'ESMF coordinate system ("coord_sys") flag {} not supported.'.format(coord_sys)
        raise ValueError(msg)
    return ret


def get_esmf_grid(ogrid, regrid_method='auto', value_mask=None):
    """
    Create an ESMF :class:`~ESMF.driver.grid.Grid` object from an OCGIS :class:`ocgis.Grid` object.

    :param ogrid: The target OCGIS grid to convert to an ESMF grid.
    :type ogrid: :class:`~ocgis.Grid`
    :param regrid_method: If ``'auto'`` or :attr:`ESMF.api.constants.RegridMethod.CONSERVE`, use corners/bounds from the
     grid object. If :attr:`ESMF.api.constants.RegridMethod.CONSERVE`, the corners/bounds must be present on the grid.
    :param value_mask: If an ``bool`` :class:`numpy.array` with same shape as ``ogrid``, use a logical *or* operation
     with the OCGIS field mask when creating the input grid's mask. Values of ``True`` in ``value_mask`` are assumed to
     be masked. If ``None`` is provided (the default) then the mask will be set using the spatial mask on ``ogrid``.
    :type value_mask: :class:`numpy.array`
    :rtype: :class:`ESMF.driver.grid.Grid`
    """

    pkwds = get_periodicity_parameters(ogrid)

    egrid = ESMF.Grid(max_index=np.array(ogrid.shape), staggerloc=ESMF.StaggerLoc.CENTER,
                      coord_sys=ESMF.CoordSys.SPH_DEG, num_peri_dims=pkwds['num_peri_dims'], pole_dim=pkwds['pole_dim'],
                      periodic_dim=pkwds['periodic_dim'])

    ovalue_stacked = ogrid.get_value_stacked()
    row = egrid.get_coords(1, staggerloc=ESMF.StaggerLoc.CENTER)
    row[:] = ovalue_stacked[0, ...]
    col = egrid.get_coords(0, staggerloc=ESMF.StaggerLoc.CENTER)
    col[:] = ovalue_stacked[1, ...]

    # Use a logical or operation to merge with value_mask if present
    if value_mask is not None:
        # convert to boolean to make sure
        value_mask = value_mask.astype(bool)
        # do the logical or operation selecting values
        value_mask = np.logical_or(value_mask, ogrid.get_mask(create=True))
    else:
        value_mask = ogrid.get_mask()
    # Follows SCRIP convention where 1 is unmasked and 0 is masked.
    if value_mask is not None:
        esmf_mask = np.invert(value_mask).astype(np.int32)
        egrid.add_item(ESMF.GridItem.MASK, staggerloc=ESMF.StaggerLoc.CENTER, from_file=False)
        egrid.mask[0][:] = esmf_mask

    # Attempt to access corners if requested.
    if regrid_method == 'auto' and ogrid.has_bounds:
        regrid_method = RegridMethod.CONSERVE

    if regrid_method == RegridMethod.CONSERVE:
        # Conversion to ESMF objects requires an expanded grid (non-vectorized).
        # TODO: Create ESMF grids from vectorized OCGIS grids.
        expand_grid(ogrid)
        # Convert to ESMF corners from OCGIS corners.
        corners_esmf = np.zeros([2] + [element + 1 for element in ogrid.x.bounds.shape[0:2]],
                                dtype=ogrid.archetype.bounds.dtype)
        get_esmf_corners_from_ocgis_corners(ogrid.y.bounds.get_value(), fill=corners_esmf[0, :, :])
        get_esmf_corners_from_ocgis_corners(ogrid.x.bounds.get_value(), fill=corners_esmf[1, :, :])

        # adding corners. first tell the grid object to allocate corners
        egrid.add_coords(staggerloc=[ESMF.StaggerLoc.CORNER])
        # get the coordinate pointers and set the coordinates
        grid_corner = egrid.coords[ESMF.StaggerLoc.CORNER]
        # Note indexing is reversed for ESMF v. OCGIS. In ESMF, the x-coordinate (longitude) is the first
        # coordinate. If this is periodic, the last column corner wraps/connect to the first. This coordinate must
        # be removed.
        if pkwds['num_peri_dims'] is not None:
            grid_corner[0][:] = corners_esmf[1][:, 0:-1]
            grid_corner[1][:] = corners_esmf[0][:, 0:-1]
        else:
            grid_corner[0][:] = corners_esmf[1]
            grid_corner[1][:] = corners_esmf[0]

    return egrid


def get_periodicity_parameters(grid):
    """
    Get characteristics of a grid's periodicity. This is only applicable for grids with a spherical coordinate system.
    There are two classifications:
     1. A grid is periodic (i.e. it has global coverage). Periodicity is determined only with the x/longitude dimension.
     2. A grid is non-periodic (i.e. it has regional coverage).

    :param grid: :class:`~ocgis.Grid`
    :return: A dictionary containing periodicity parameters.
    :rtype: dict
    """
    # Check if grid may be flagged as "periodic" by determining if its extent is global. Use the centroids and the grid
    # resolution to determine this.
    is_periodic = False
    col = grid.x.get_value()
    resolution = grid.resolution
    min_col, max_col = col.min(), col.max()
    # Work only with unwrapped coordinates.
    if min_col < 0:
        min_col += 360.
    if max_col < 0:
        max_col += 360
    # Check the min and max column values are within a tolerance (the grid resolution) of global (0 to 360) edges.
    if (0. - resolution) <= min_col <= (0. + resolution):
        min_periodic = True
    else:
        min_periodic = False
    if (360. - resolution) <= max_col <= (360. + resolution):
        max_periodic = True
    else:
        max_periodic = False
    if min_periodic and max_periodic:
        is_periodic = True

    # If the grid is periodic, set the appropriate parameters.
    if is_periodic:
        num_peri_dims = 1
        pole_dim = 0
        periodic_dim = 1
    else:
        num_peri_dims, pole_dim, periodic_dim = [None] * 3

    ret = {'num_peri_dims': num_peri_dims, 'pole_dim': pole_dim, 'periodic_dim': periodic_dim}

    return ret


def iter_esmf_fields(ofield, regrid_method='auto', value_mask=None, split=True):
    """
    For all data or a single time coordinate, yield an ESMF :class:`~ESMF.driver.field.Field` from the input OCGIS field
    ``ofield``. Only one realization and level are allowed.

    :param ofield: The input OCGIS Field object.
    :type ofield: :class:`ocgis.Field`
    :param regrid_method: See :func:`~ocgis.regrid.base.get_esmf_grid`.
    :param value_mask: See :func:`~ocgis.regrid.base.get_esmf_grid`.
    :param bool split: If ``True``, yield a single time slice/coordinate from the source field. If ``False``, yield all
     time coordinates. When ``False``, OCGIS uses ESMF's ``ndbounds`` argument to field creation. Use ``True`` if
     there are memory limitations. Use ``False`` for faster performance.
    :rtype: tuple(str, :class:`ESMF.driver.field.Field`, int)

    The returned tuple elements are:

    ===== ================================= ===================================================================================================
    Index Type                              Description
    ===== ================================= ===================================================================================================
    0     str                               The variable name currently being converted.
    1     :class:`~ESMF.driver.field.Field` The ESMF Field object.
    2     int                               The current time index of the yielded ESMF field. If ``split=False``, This will be ``slice(None)``.
    ===== ================================= ===================================================================================================

    :raises: AssertionError
    """
    # Only one level and realization allowed.
    if ofield.level is not None and ofield.level.ndim > 0:
        assert ofield.level.shape[0] == 1
    if ofield.realization is not None:
        assert ofield.realization.shape[0] == 1

    # Retrieve the mask from the first variable.
    if value_mask is None:
        if ofield.time is not None:
            sfield = ofield.get_field_slice({'time': 0})
        else:
            sfield = ofield
        archetype = sfield.data_variables[0]
        value_mask = archetype.get_mask()

    # Create the ESMF grid.
    egrid = get_esmf_grid(ofield.grid, regrid_method=regrid_method, value_mask=value_mask)

    # Produce a new ESMF field for each variable.
    if ofield.time is not None:
        time_name = ofield.time.dimensions[0].name
    else:
        # If there is no time, there is no need to split anything by the time slice.
        split = False
        time_name = None

    # These dimensions will become singletons.
    dimension_names_to_squeeze = []
    if ofield.level is not None and ofield.level.ndim > 0:
        dimension_names_to_squeeze.append(ofield.level.dimensions[0].name)
    if ofield.realization is not None:
        dimension_names_to_squeeze.append(ofield.realization.dimensions[0].name)

    for variable in ofield.data_variables:
        variable = variable.extract()
        dimensions_to_squeeze = [idx for idx, d in enumerate(variable.dimensions) if
                                 d.name in dimension_names_to_squeeze]
        extra_dimensions = [d.name for d in variable.dimensions if d.name != time_name]
        slice_template = {name: None for name in extra_dimensions}
        variable_name = variable.name
        if split:
            for tidx in range(ofield.time.shape[0]):
                efield = ESMF.Field(egrid, name=variable_name)
                slice_template[time_name] = tidx
                efield.data[:] = np.squeeze(variable[slice_template].get_value(), axis=dimensions_to_squeeze)
                yield variable_name, efield, tidx
        else:
            if ofield.time is not None:
                ndbounds = [ofield.time.shape[0]]
                slice_template[time_name] = None
            else:
                ndbounds = None
            efield = ESMF.Field(egrid, name=variable_name, ndbounds=ndbounds)
            slice_template[ofield.x.dimensions[0].name] = None
            slice_template[ofield.y.dimensions[0].name] = None
            efield.data[:] = np.squeeze(variable[slice_template].get_value(), axis=dimensions_to_squeeze)
            tidx = slice(None)
            yield variable_name, efield, tidx


def check_fields_for_regridding(source, destination, regrid_method='auto'):
    """
    Perform a standard series of checks on regridding inputs.

    :param source: The source field.
    :type source: :class:`ocgis.Field`
    :param destination: The destination field.
    :type destination: :class:`ocgis.Field`
    :param regrid_method: If ``'auto'``, attempt to do conservative regridding if bounds/corners are available on both
      fields. Otherwise, do bilinear regridding. This may also be an ESMF regrid method flag.
    :type regrid_method: str or :attr:`ESMF.api.constants.RegridMethod`
    :raises: RegriddingError, CornersInconsistentError
    """

    # Check field objects are used.
    for element in [source, destination]:
        if not isinstance(element, Field):
            raise RegriddingError('OCGIS field objects only for regridding.')

    # Check their are grids on source and destination. Only structured grids are supported at this time.
    for element in [source, destination]:
        if element.grid is None:
            raise RegriddingError('Fields must have grids to do regridding.')

    # Check coordinate systems #########################################################################################

    for element in [source, destination]:
        if not isinstance(element.crs, Spherical):
            msg_a = 'Only spherical coordinate systems allowed for regridding.'
            raise RegriddingError(msg_a)

    if source.crs != destination.crs:
        msg = 'Source and destination coordinate systems must be equal.'
        raise RegriddingError(msg)

    # Check corners are available on all inputs ########################################################################

    if regrid_method == ESMF.RegridMethod.CONSERVE:
        has_corners_source = source.grid.has_bounds
        has_corners_destination = destination.grid.has_bounds
        if not has_corners_source or not has_corners_destination:
            msg = 'Corners are not available on all sources and destination. Consider changing "regrid_method".'
            raise CornersInconsistentError(msg)


def regrid_field(source, destination, regrid_method='auto', value_mask=None, split=True):
    """
    Regrid ``source`` data to match the grid of ``destination``.

    :param source: The source field.
    :type source: :class:`ocgis.Field`
    :param destination: The destination field.
    :type destination: :class:`ocgis.Field`
    :param regrid_method: See :func:`~ocgis.regrid.base.get_esmf_grid`.
    :param value_mask: See :func:`~ocgis.regrid.base.iter_esmf_fields`.
    :type value_mask: :class:`numpy.ndarray`
    :param bool split: See :func:`~ocgis.regrid.base.iter_esmf_fields`.
    :rtype: :class:`ocgis.Field`
    """

    # This function runs a series of asserts to make sure the sources and destination are compatible.
    check_fields_for_regridding(source, destination, regrid_method=regrid_method)

    # Regrid each source.
    ocgis_lh(logger='iter_regridded_fields', msg='starting source regrid loop', level=logging.DEBUG)
    # for source in sources:
    build = True
    fills = {}
    for variable_name, src_efield, tidx in iter_esmf_fields(source, regrid_method=regrid_method,
                                                            value_mask=value_mask, split=split):

        # We need to generate new variables given the change in shape
        if variable_name not in fills:
            if source.time is not None:
                new_dimensions = list(source.time.dimensions) + list(destination.grid.dimensions)
            else:
                new_dimensions = list(destination.grid.dimensions)
            source_variable = source[variable_name]
            new_variable = Variable(name=variable_name, dimensions=new_dimensions, dtype=source_variable.dtype,
                                    fill_value=source_variable.fill_value)
            fills[variable_name] = new_variable

        # Only build the regrid objects once.
        if build:
            # Build the destination grid once.
            esmf_destination_grid = get_esmf_grid(destination.grid, regrid_method=regrid_method, value_mask=value_mask)

            # Check for corners on the destination grid. If they exist, conservative regridding is possible.
            if regrid_method == 'auto':
                if get_esmf_grid_has_corners(esmf_destination_grid) and get_esmf_grid_has_corners(src_efield.grid):
                    regrid_method = ESMF.RegridMethod.CONSERVE
                else:
                    regrid_method = None

            if split:
                ndbounds = None
            else:
                ndbounds = [source.time.shape[0]]

            # Prepare the regridded sourced field. This amounts to exchanging the grids between the objects.
            regridded_source = source.copy()
            regridded_source.grid.extract(clean_break=True)
            regridded_source.set_grid(destination.grid.extract())

            build = False

        dst_efield = ESMF.Field(esmf_destination_grid, name='destination', ndbounds=ndbounds)
        fill_variable = fills[variable_name]
        fv = fill_variable.fill_value
        if fv is None:
            fv = np.ma.array([0], dtype=fill_variable.dtype).fill_value
        dst_efield.data.fill(fv)
        # Construct the regrid object. Weight generation actually occurs in this call.
        regrid = ESMF.Regrid(src_efield, dst_efield, unmapped_action=ESMF.UnmappedAction.IGNORE,
                             regrid_method=regrid_method, src_mask_values=[0], dst_mask_values=[0])
        # Perform the regrid operation. "zero_region" only fills values involved with regridding.
        regridded_esmf_field = regrid(src_efield, dst_efield, zero_region=ESMF.Region.SELECT)
        unmapped_mask = regridded_esmf_field.data[:] == fv
        # If all data is masked, raise an exception.
        if unmapped_mask.all():
            # Destroy ESMF objects.
            destroy_esmf_objects([regrid, dst_efield, esmf_destination_grid])
            msg = 'All regridded elements are masked. Do the input spatial extents overlap?'
            raise RegriddingError(msg)
        # Fill the new variables.
        fv_value = fill_variable.get_value()
        fv_mask = fill_variable.get_mask(create=True)
        try:
            fv_value[tidx, :, :] = regridded_esmf_field.data
        except IndexError:
            # Assume no time index.
            fv_value[:, :] = regridded_esmf_field.data
        if regridded_esmf_field.grid.mask[0] is not None:
            try:
                fv_mask[tidx, :, :] = np.invert(regridded_esmf_field.grid.mask[0].astype(bool))
            except IndexError:
                # Assume no time index.
                fv_mask[:, :] = np.invert(regridded_esmf_field.grid.mask[0].astype(bool))
        try:
            fill_variable_mask = np.logical_or(unmapped_mask, fv_mask[tidx, :, :])
        except IndexError:
            # Assume no time index.
            fill_variable_mask = np.logical_or(unmapped_mask, fv_mask[:, :])
        try:
            fv_mask[tidx, :, :] = fill_variable_mask
        except IndexError:
            # Assume no time index.
            fv_mask[:, :] = fill_variable_mask
        fill_variable.set_mask(fv_mask)

        # Destroy ESMF objects, but keep the grid until all splits have finished. If split=False, there is only one
        # split.
        destroy_esmf_objects([regrid, dst_efield, src_efield])

        # Create a new variable collection and add the variables to the output field.
        # source.variables = VariableCollection()
        for v in list(fills.values()):
            regridded_source.add_variable(v, is_data=True, force=True)

    destroy_esmf_objects([esmf_destination_grid])

    return regridded_source


def destroy_esmf_objects(objs):
    for obj in objs:
        obj.destroy()
