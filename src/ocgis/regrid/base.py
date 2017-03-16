import logging
from copy import deepcopy

import ESMF
import numpy as np

from ocgis import constants
from ocgis import env
from ocgis.base import AbstractOcgisObject
from ocgis.collection.field import OcgField
from ocgis.constants import DimensionMapKeys
from ocgis.exc import RegriddingError, CornersInconsistentError
from ocgis.spatial.grid import GridXY, expand_grid
from ocgis.spatial.spatial_subset import SpatialSubsetOperation
from ocgis.util.helpers import iter_array, get_esmf_corners_from_ocgis_corners
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.variable.base import Variable
from ocgis.variable.crs import Spherical, WGS84
from ocgis.variable.temporal import TemporalVariable


class RegridOperation(AbstractOcgisObject):
    """
    Execute a regrid operation handling spatial subsetting and coordinate system transformations.

    :param field_src: The source field to regrid.
    :type field_src: :class:`ocgis.Field`
    :param field_dst: The destination regrid field.
    :type field_dst: :class:`ocgis.Field`
    :param subset_sdim: If provided, use this spatial dimension to subset the regridding fields.
    :type subset_sdim: :class:`ocgis.SpatialDimension`
    :param with_buffer: If ``True``, use a buffer during spatial subsetting.
    :type with_buffer: bool
    :param regrid_options: A dictionary of keyword options to pass to :func:`~ocgis.regrid.base.iter_regridded_fields`.
    :type regrid_options: dict
    """

    def __init__(self, field_src, field_dst, subset_sdim=None, with_buffer=True, regrid_options=None):
        self.field_dst = field_dst
        self.field_src = field_src
        self.subset_sdim = subset_sdim
        self.with_buffer = with_buffer
        if regrid_options is None:
            self.regrid_options = {}
        else:
            self.regrid_options = regrid_options

        # If true, regridding required updating the coordinate system of the source field.
        self._regrid_required_source_crs_update = False
        # Holds original coordinate system of the source field for back transformations.
        self._original_sfield_crs = None

    def execute(self):
        """
        Execute regridding operation.

        :rtype: :class:`~ocgis.Field`
        """

        destination_sdim = self._get_regrid_destination_()
        self._update_regrid_source_coordinate_system_()

        # Regrid the input field.
        ocgis_lh(logger='regrid', msg='Creating regridded fields...', level=logging.INFO)
        regridded_source = list(iter_regridded_fields([self.field_src], destination_sdim, **self.regrid_options))[0]

        # Return the source field to its original coordinate system.
        if self._regrid_required_source_crs_update:
            ocgis_lh(logger='regrid', msg='Reverting source field to original coordinate system...', level=logging.INFO)
            regridded_source.spatial.update_crs(self._original_sfield_crs)
        else:
            regridded_source.spatial.crs = self._original_sfield_crs

        # Subset the output from the regrid operation as masked values may be introduced on the edges.
        if self.subset_sdim is not None:
            ss = SpatialSubsetOperation(regridded_source)
            regridded_source = ss.get_spatial_subset('intersects', self.subset_sdim,
                                                     use_spatial_index=env.USE_SPATIAL_INDEX,
                                                     select_nearest=False)

        return regridded_source

    def _get_regrid_destination_(self):
        """
        Prepare destination field for regridding.

        :rtype: :class:`~ocgis.SpatialDimension`
        """

        # Spatially subset the regrid destination. #####################################################################
        if self.subset_sdim is None:
            ocgis_lh(logger='regrid', msg='no spatial subsetting', level=logging.DEBUG)
            regrid_destination = self.field_dst
        else:
            if self.with_buffer:
                # Buffer the subset geometry by the resolution of the source field to improve chances of overlap between
                # source and destination extents.
                buffer_value = self.field_src.spatial.grid.resolution
                buffer_crs = self.field_src.spatial.crs
            else:
                buffer_value, buffer_crs = [None, None]
            ss = SpatialSubsetOperation(self.field_dst)
            regrid_destination = ss.get_spatial_subset('intersects', self.subset_sdim,
                                                       use_spatial_index=env.USE_SPATIAL_INDEX,
                                                       select_nearest=False, buffer_value=buffer_value,
                                                       buffer_crs=buffer_crs)

        # Transform the coordinate system of the regrid destination. ###################################################

        # Update the coordinate system of the regrid destination if required.
        try:
            destination_sdim = regrid_destination.spatial
        except AttributeError:
            # Likely a spatial dimension object already.
            destination_sdim = regrid_destination
        # If switched to true, the regrid destination coordinate system must be updated to match the source.
        update_regrid_destination_crs = False
        if not isinstance(regrid_destination.crs, Spherical):
            if isinstance(regrid_destination, Field):
                if isinstance(destination_sdim.crs, WGS84) and regrid_destination._has_assigned_coordinate_system:
                    update_regrid_destination_crs = True
                elif isinstance(destination_sdim.crs,
                                WGS84) and not regrid_destination._has_assigned_coordinate_system:
                    pass
                else:
                    update_regrid_destination_crs = True
            else:
                if not isinstance(destination_sdim.crs, Spherical):
                    update_regrid_destination_crs = True
        if update_regrid_destination_crs:
            ocgis_lh(logger='regrid',
                     msg='updating regrid destination to spherical. regrid destination crs is: {}'.format(
                         regrid_destination.crs), level=logging.DEBUG)
            destination_sdim.update_crs(Spherical())
        else:
            destination_sdim.crs = Spherical()

        # Remove the mask from the destination field. ##################################################################
        new_mask = np.zeros(destination_sdim.shape, dtype=bool)
        destination_sdim.set_mask(new_mask)

        return destination_sdim

    def _update_regrid_source_coordinate_system_(self):
        # Alias the original coordinate system for the back-transform.
        self._original_sfield_crs = self.field_src.spatial.crs

        # Check coordinate system on the source field.
        if not isinstance(self.field_src.spatial.crs, Spherical):
            # This has an _assigned_ WGS84 CRS. Hence, we cannot assume the default CRS.
            if isinstance(self.field_src.spatial.crs, WGS84) and self.field_src._has_assigned_coordinate_system:
                self._regrid_required_source_crs_update = True
            # The data has a coordinate system that is not WGS84.
            elif not isinstance(self.field_src.spatial.crs, WGS84):
                self._regrid_required_source_crs_update = True
        if self._regrid_required_source_crs_update:
            # Need to load values as source indices will disappear during CRS update.
            for variable in self.field_src.variables.itervalues():
                variable.value
            ocgis_lh(logger='regrid', msg='updating regrid source to spherical. regrid source crs is: {}'.format(
                self.field_src.spatial.crs), level=logging.DEBUG)
            self.field_src.spatial.update_crs(Spherical())
            ocgis_lh(logger='regrid', msg='completed crs update for regrid source'.format(self.field_src.spatial.crs),
                     level=logging.DEBUG)
        else:
            self.field_src.spatial.crs = Spherical()


def get_ocgis_grid_from_esmf_grid(egrid, crs=None):
    """
    Create an OCGIS :class:`~ocgis.interface.base.dimension.spatial.SpatialDimension` object from an ESMF
    :class:`~ESMF.driver.grid.Grid`.

    :type egrid: :class:`ESMF.driver.grid.Grid`
    :param crs: The coordinate system to attach to the output spatial dimension.
    :type crs: :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
    :rtype: :class:`~ocgis.interface.base.dimension.spatial.SpatialDimension`
    """

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
        try:
            egrid_mask = np.invert(egrid_mask.astype(bool))
        except:
            tkk
        mask_grid_value = np.zeros(grid_value.shape, dtype=bool)
        mask_grid_value[:, :, :] = egrid_mask
        if grid_corners is not None:
            mask_grid_corners = np.zeros(grid_corners.shape, dtype=bool)
            for ii, jj in iter_array(egrid_mask):
                mask_grid_corners[:, ii, jj, :] = egrid_mask[ii, jj]
    else:
        mask_grid_value = False
        mask_grid_corners = False

    # actually construct the masked arrays
    grid_value = np.ma.array(grid_value, mask=mask_grid_value)
    if grid_corners is not None:
        grid_corners = np.ma.array(grid_corners, mask=mask_grid_corners)

    if grid_corners is not None:
        x_bounds = Variable(name='x_bounds', value=grid_corners[1, ...],
                            dimensions=['y', 'x', constants.DEFAULT_NAME_CORNERS_DIMENSION])
        y_bounds = Variable(name='y_bounds', value=grid_corners[0, ...],
                            dimensions=['y', 'x', constants.DEFAULT_NAME_CORNERS_DIMENSION])
    else:
        x_bounds, y_bounds = [None] * 2

    x = Variable(name='x', dimensions=['y', 'x'], value=grid_value[1, ...], bounds=x_bounds)
    y = Variable(name='y', dimensions=['y', 'x'], value=grid_value[0, ...], bounds=y_bounds)

    ogrid = GridXY(x, y, crs=crs)

    return ogrid


def get_esmf_grid_has_corners(egrid):
    return egrid.has_corners


def get_ocgis_field_from_esmf_field(efield, crs=None, dimensions=None):
    """
    :param efield: The ESMPy field object to convert to an OCGIS field.
    :type efield: :class:`ESMF.driver.field.Field`
    :param crs: The coordinate system of the ESMF field. If ``None``, this will default to a coordinate system
     mapped to the ``coord_sys`` attribute of the ESMF grid.
    :param dimensions: A dictionary containing optional dimension definitions for realization, time, and level. The keys
     of this dictionary are the appropriate dimension types. If no overload dimensions are provided and the incoming
     field has shapes greater than one for non-spatial dimensions, values are filled using integer values starting with
     one.
    :type dimensions: dict

    +-------------+------------------------------------+
    |  Key        | Type                               |
    +=============+====================================+
    | temporal    | :class:`~ocgis.TemporalDimension`  |
    +-------------+------------------------------------+
    | level       | :class:`~ocgis.VectorDimension`    |
    +-------------+------------------------------------+
    | realization | :class:`~ocgis.VectorDimension`    |
    +-------------+------------------------------------+

    :returns: An OCGIS field object.
    :rtype: :class:`~ocgis.Field`
    """

    efield_shape = efield.data.shape
    assert len(efield_shape) == 5

    dimensions = dimensions or {}
    out_dimensions = [DimensionMapKeys.REALIZATION, DimensionMapKeys.TIME, DimensionMapKeys.LEVEL, DimensionMapKeys.Y,
                      DimensionMapKeys.X]

    try:
        realization = dimensions['realization']
    except KeyError:
        if efield_shape[0] > 1:
            realization_values = np.arange(1, efield_shape[0] + 1)
            realization = Variable(value=realization_values, name=DimensionMapKeys.REALIZATION,
                                   dimensions=DimensionMapKeys.REALIZATION)
        else:
            realization = None

    try:
        temporal = dimensions['temporal']
    except KeyError:
        if efield_shape[1] > 1:
            temporal_values = np.array([1] * efield_shape[1])
            temporal = TemporalVariable(value=temporal_values, format_time=False, name=DimensionMapKeys.TIME,
                                        dimensions=DimensionMapKeys.TIME)
        else:
            temporal = None

    try:
        level = dimensions['level']
    except KeyError:
        if efield_shape[2] > 1:
            level_values = np.arange(1, efield_shape[2] + 1)
            level = Variable(value=level_values, name=DimensionMapKeys.LEVEL, dimensions=DimensionMapKeys.LEVEL)
        else:
            level = None

    # Choose the default coordinate system.
    if crs is None:
        crs = get_crs_from_esmf_field(efield)

    variable = Variable(name=efield.name, value=efield.data, dimensions=out_dimensions)
    grid = get_ocgis_grid_from_esmf_grid(efield.grid, crs=crs)
    field = OcgField(variables=variable, realization=realization, time=temporal, level=level, grid=grid,
                     is_data=variable)

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


def get_esmf_grid(sdim, with_corners=True, value_mask=None):
    """
    Create an ESMF :class:`~ESMF.driver.grid.Grid` object from an OCGIS
    :class:`~ocgis.interface.base.dimension.spatial.SpatialDimension` object.

    :param sdim: The target spatial dimension to convert into an ESMF grid.
    :type sdim: :class:`~ocgis.interface.base.dimension.spatial.SpatialDimension`
    :param bool with_corners: If ``True``, attempt to access corners from ``sdim``.
    :param value_mask: If an ``bool`` :class:`numpy.array` with same shape as ``sdim``, use a logical *or* operation
     with the OCGIS field mask when creating the input grid's mask. Values of ``True`` in ``value_mask`` are assumed to
     be masked. If ``None`` is provided (the default) then the mask will be set using the first realization/time/level
     from the first variable contained in ``ofield``. This will include the spatial mask.
    :type value_mask: :class:`numpy.array`
    :rtype: :class:`ESMF.driver.grid.Grid`
    """

    ogrid = sdim.grid

    pkwds = get_periodicity_parameters(ogrid)

    egrid = ESMF.Grid(max_index=np.array(ogrid.shape), staggerloc=ESMF.StaggerLoc.CENTER,
                      coord_sys=ESMF.CoordSys.SPH_DEG, num_peri_dims=pkwds['num_peri_dims'], pole_dim=pkwds['pole_dim'],
                      periodic_dim=pkwds['periodic_dim'])
    row = egrid.get_coords(1, staggerloc=ESMF.StaggerLoc.CENTER)
    row[:] = ogrid.y.value
    col = egrid.get_coords(0, staggerloc=ESMF.StaggerLoc.CENTER)
    col[:] = ogrid.x.value

    # use a logical or operation to merge with value_mask if present
    if value_mask is not None:
        # convert to boolean to make sure
        value_mask = value_mask.astype(bool)
        # do the logical or operation selecting values
        value_mask = np.logical_or(value_mask, ogrid.get_mask())
    else:
        value_mask = ogrid.get_mask()
    # Follows SCRIP convention where 1 is unmasked and 0 is masked.
    if value_mask is not None:
        esmf_mask = np.invert(value_mask).astype(np.int32)
        egrid.add_item(ESMF.GridItem.MASK, staggerloc=ESMF.StaggerLoc.CENTER, from_file=False)
        egrid.mask[0][:] = esmf_mask

    # attempt to access corners if possible
    if with_corners:
        if ogrid.has_bounds:
            # Conversion to ESMF objects requires an expanded grid (non-vectorized).
            # TODO: Create ESMF grids from vectorized OCGIS grids.
            expand_grid(ogrid)
            # Convert to ESMF corners from OCGIS corners.
            corners_esmf = np.zeros([2] + [element + 1 for element in ogrid.x.bounds.shape[0:2]],
                                    dtype=ogrid.x.bounds.dtype)
            get_esmf_corners_from_ocgis_corners(ogrid.y.bounds.value, fill=corners_esmf[0, :, :])
            get_esmf_corners_from_ocgis_corners(ogrid.x.bounds.value, fill=corners_esmf[1, :, :])

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

    :param grid: :class:`~ocgis.interface.base.dimension.spatial.SpatialGridDimension`
    :return: A dictionary containing periodicity parameters.
    :rtype: dict
    """
    # Check if grid may be flagged as "periodic" by determining if its extent is global. Use the centroids and the grid
    # resolution to determine this.
    is_periodic = False
    col = grid.x.value
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


def iter_esmf_fields(ofield, with_corners=True, value_mask=None, split=True):
    """
    For all data or a single time coordinate, yield an ESMF :class:`~ESMF.driver.field.Field` from the input OCGIS field
    ``ofield``. Only one realization and level are allowed.

    :param ofield: The input OCGIS Field object.
    :type ofield: :class:`ocgis.Field`
    :param bool with_corners: See :func:`~ocgis.regrid.base.get_esmf_grid`.
    :param value_mask: See :func:`~ocgis.regrid.base.get_esmf_grid`.
    :param bool split: If ``True``, yield a single time slice/coordinate from the source field. If ``False``, yield all
     time coordinates. When ``False``, OCGIS uses ESMF's ``ndbounds`` argument to field creation. Use ``True`` if
     there are memory limitations. Use ``False`` for faster performance.
    :rtype: tuple(str, :class:`ESMF.driver.field.Field`, int)

    The returned tuple elements are:

    ===== ============================== ==================================================================================================
    Index Type                           Description
    ===== ============================== ==================================================================================================
    0     str                            The alias of the variable currently being converted.
    1     :class:`~ESMF.driver.field.Field` The ESMF Field object.
    2     int                            The current time index of the yielded ESMF field. If ``split=False``, This will be ``slice(None)``.
    ===== ============================== ==================================================================================================

    :raises: AssertionError
    """
    # Only one level and realization allowed.
    assert ofield.shape[0] == 1
    assert ofield.shape[2] == 1

    # Retrieve the mask from the first variable.
    if value_mask is None:
        sfield = ofield[0, 0, 0, :, :]
        variable = sfield.variables.first()
        value_mask = variable.value.mask.reshape(sfield.shape[-2:])
        assert np.all(value_mask.shape == sfield.shape[-2:])

    # Create the ESMF grid.
    egrid = get_esmf_grid(ofield.spatial, with_corners=with_corners, value_mask=value_mask)

    # Produce a new ESMF field for each variable.
    for variable_alias, variable in ofield.variables.iteritems():
        if split:
            for tidx in range(ofield.shape[1]):
                efield = ESMF.Field(egrid, name=variable_alias)
                efield.data[:] = variable.value[0, tidx, 0, :, :]
                yield variable_alias, efield, tidx
        else:
            efield = ESMF.Field(egrid, name=variable_alias, ndbounds=[ofield.shape[1]])
            efield.data[:] = variable.value[0, :, 0, :, :]
            tidx = slice(None)
            yield variable_alias, efield, tidx


def check_fields_for_regridding(sources, destination, with_corners=True):
    """
    Perform a standard series of checks on inputs to regridding.

    :param sources: Sequence of source fields to regrid.
    :type sources: sequence of :class:`ocgis.interface.base.field.Field`
    :param destination: The target grid contained in a field or spatial dimension.
    :type destination: :class:`ocgis.interface.base.field.Field` or
     :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
    :param bool with_corners: If ``True``, attempt to access corners from ``sdim``.
    :rtype: :class:`ocgis.interface.base.field.Field`
    :raises: RegriddingError, CornersInconsistentError
    """

    def _assert_spherical_crs_(crs):
        if type(crs) != Spherical:
            msg_a = 'Only spherical coordinate systems allowed for regridding.'
            raise RegriddingError(msg_a)

    try:
        sdim = destination.spatial
    # likely a SpatialDimension object
    except AttributeError:
        sdim = destination
    _assert_spherical_crs_(sdim.crs)

    ####################################################################################################################
    # check that corners are available on all inputs if with_corners is True
    ####################################################################################################################

    if with_corners:
        has_corners_sources = []
        for source in sources:
            if source.spatial.grid.corners is not None:
                has_corners_sources.append(True)
            else:
                has_corners_sources.append(False)
        if sdim.grid.corners is not None:
            has_corners_destination = True
        else:
            has_corners_destination = False
        if not all(has_corners_sources) or not has_corners_destination:
            msg = 'Corners are not available on all sources and destination. Consider setting "with_corners" to False.'
            raise CornersInconsistentError(msg)

    ####################################################################################################################
    # check coordinate systems. need to make sure only spherical coordinate systems are used and the coordinate system
    # definitions match.
    ####################################################################################################################

    for source in sources:
        _assert_spherical_crs_(source.spatial.crs)
        if source.spatial.crs != sdim.crs:
            msg = 'Source and destination coordinate systems must be equal.'
            raise RegriddingError(msg)


def iter_regridded_fields(sources, destination, with_corners='auto', value_mask=None, split=True):
    """
    Regrid ``sources`` to match the grid of ``destination``.

    :param sources: Sequence of source fields to regrid.
    :type sources: sequence of :class:`ocgis.Field`
    :param destination: The target grid contained in a field or spatial dimension.
    :type destination: :class:`ocgis.Field` or :class:`ocgis.SpatialDimension`
    :param with_corners: If ``'auto'``, automatically determine if corners should be used. They will be used if they
     are available on all ``sources`` and the ``destination``. ``True`` or ``False`` is also acceptable - see
     :func:`~ocgis.regrid.base.get_esmf_grid`.
    :type with_corners: str or bool
    :param value_mask: See :func:`~ocgis.regrid.base.iter_esmf_fields`.
    :type value_mask: :class:`numpy.ndarray`
    :param bool split: See :func:`~ocgis.regrid.base.iter_esmf_fields`.
    :rtype: :class:`ocgis.Field`
    """

    # The destination may be a field or spatial dimension.
    try:
        destination_sdim = destination.spatial
    # Likely a spatial dimension.
    except AttributeError:
        destination_sdim = destination

    # This function runs a series of asserts to make sure the sources and destination are compatible.
    try:
        check_fields_for_regridding(sources, destination, with_corners=with_corners)
    except CornersInconsistentError:
        if with_corners == 'auto':
            with_corners = False
        else:
            raise

    # Sources may be modified in the process, so make a copy of these grids.
    sources = deepcopy(sources)
    # This is the new shape of the output variables.
    new_shape_spatial = destination_sdim.shape
    # Regrid each source.
    ocgis_lh(logger='iter_regridded_fields', msg='starting source regrid loop', level=logging.DEBUG)
    for source in sources:
        build = True
        fills = {}
        for variable_alias, efield, tidx in iter_esmf_fields(source, with_corners=with_corners, value_mask=value_mask,
                                                             split=split):

            # We need to generate new variables given the change in shape
            if variable_alias not in fills:
                new_shape = list(source.variables[variable_alias].shape)
                new_shape[-2:] = new_shape_spatial
                fill_var = source.variables[variable_alias].get_empty_like(shape=new_shape)
                fills[variable_alias] = fill_var

            # Only build the regrid objects once.
            if build:
                # Build the destination grid once.
                esmf_destination_grid = get_esmf_grid(destination_sdim, with_corners=with_corners,
                                                      value_mask=value_mask)

                # Check for corners on the destination grid. If they exist, conservative regridding is possible.
                if get_esmf_grid_has_corners(esmf_destination_grid):
                    regrid_method = ESMF.RegridMethod.CONSERVE
                else:
                    regrid_method = None

                # Place a deepcopy of the destination spatial dimension as the output spatial dimension for the
                # regridded fields.
                out_sdim = deepcopy(destination_sdim)
                # If this is not conservative regridding then bounds and corners on the output should be stripped.
                if not with_corners or regrid_method is None:
                    if out_sdim.grid.row is not None:
                        out_sdim.grid.row.remove_bounds()
                        out_sdim.grid.col.remove_bounds()
                        out_sdim.grid._corners = None
                    # Remove any polygons if they exist.
                    out_sdim.geom._polygon = None
                    out_sdim.geom.grid = out_sdim.grid

                if split:
                    ndbounds = None
                else:
                    ndbounds = [source.shape[1]]

                build = False

            esmf_destination_field = ESMF.Field(esmf_destination_grid, name='destination', ndbounds=ndbounds)
            fill_variable = fills[variable_alias]
            esmf_destination_field.data.fill(fill_variable.fill_value)
            # Construct the regrid object. Weight generation actually occurs in this call.
            regrid = ESMF.Regrid(efield, esmf_destination_field, unmapped_action=ESMF.UnmappedAction.IGNORE,
                                 regrid_method=regrid_method, src_mask_values=[0], dst_mask_values=[0])
            # Perform the regrid operation. "zero_region" only fills values involved with regridding.
            regridded_esmf_field = regrid(efield, esmf_destination_field, zero_region=ESMF.Region.SELECT)
            unmapped_mask = regridded_esmf_field.data == fill_variable.fill_value
            # If all data is masked, raise an exception.
            if unmapped_mask.all():
                # Destroy ESMF objects.
                destroy_esmf_objects([regrid, esmf_destination_field, esmf_destination_grid])
                msg = 'All regridded elements are masked. Do the input spatial extents overlap?'
                raise RegriddingError(msg)
            # Fill the new variables.
            fill_variable.value.data[0, tidx, 0, :, :] = regridded_esmf_field.data
            fill_variable.value.mask[0, tidx, 0, :, :] = np.invert(regridded_esmf_field.grid.mask[0].astype(bool))
            fill_variable_mask = np.logical_or(unmapped_mask, fill_variable.value.mask[0, tidx, 0, :, :])
            fill_variable.value.mask[0, tidx, 0, :, :] = fill_variable_mask

            # Destroy ESMF objects, but keep the grid until all splits have finished. If split=False, there is only one
            # split.
            destroy_esmf_objects([regrid, esmf_destination_field, efield])

        # Set the output spatial dimension of the regridded field.
        source.spatial = out_sdim

        # Create a new variable collection and add the variables to the output field.
        source.variables = VariableCollection()
        for v in fills.itervalues():
            source.variables.add_variable(v)

        yield source

    destroy_esmf_objects([esmf_destination_grid])


def destroy_esmf_objects(objs):
    for obj in objs:
        obj.destroy()
