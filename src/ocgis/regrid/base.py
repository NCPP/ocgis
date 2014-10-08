from copy import deepcopy
import ESMF
import numpy as np
from ocgis.exc import CornersUnavailable, RegriddingError, CornersInconsistentError
from ocgis.interface.base.crs import Spherical
from ocgis.interface.base.dimension.spatial import SpatialGridDimension, SpatialDimension
from ocgis.interface.base.variable import VariableCollection
from ocgis.util.helpers import iter_array, make_poly


def get_sdim_from_esmf_grid(egrid):
    """
    Create an OCGIS :class:`~ocgis.interface.base.dimension.spatial.SpatialDimension` object from an ESMF
    :class:`~ESMF.api.grid.Grid`.

    :type egrid: :class:`ESMF.api.grid.Grid`
    :rtype: :class:`~ocgis.interface.base.dimension.spatial.SpatialDimension`
    """

    # get reference to esmf grid coordinates
    coords = egrid.coords[ESMF.StaggerLoc.CENTER]
    # extract the array shapes
    shape_coords_list = list(coords[0].shape)
    dtype_coords = coords[0].dtype
    # construct the ocgis grid array and fill
    grid_value = np.zeros([2] + shape_coords_list, dtype=dtype_coords)
    grid_value[0, ...] = coords[0]
    grid_value[1, ...] = coords[1]

    # check for corners on the esmf grid
    if all(egrid.coords_done[ESMF.StaggerLoc.CORNER]):
        # reference the corner array
        corner = egrid.coords[ESMF.StaggerLoc.CORNER]
        grid_corners = np.zeros([2] + shape_coords_list + [4], dtype=dtype_coords)
        slices = [(0, 0), (0, 1), (1, 1), (1, 0)]
        # collect the corners and insert into ocgis corners array
        for ii, jj in iter_array(coords[0], use_mask=False):
            row_slice = slice(ii, ii+2)
            col_slice = slice(jj, jj+2)
            row_corners = corner[0][row_slice, col_slice]
            col_corners = corner[1][row_slice, col_slice]
            for kk, slc in enumerate(slices):
                grid_corners[:, ii, jj, kk] = row_corners[slc], col_corners[slc]
    else:
        grid_corners = None

    # determine if a mask has been added to the grid
    has_mask = egrid.item_done[ESMF.StaggerLoc.CENTER][0]
    if has_mask:
        # if there is a mask, update the grid values
        egrid_mask = egrid.mask[ESMF.StaggerLoc.CENTER]
        egrid_mask = np.invert(egrid_mask.astype(bool))
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

    # make the spatial dimension object
    ogrid = SpatialGridDimension(value=grid_value, corners=grid_corners)
    sdim = SpatialDimension(grid=ogrid)

    return sdim


def get_esmf_grid_from_sdim(sdim, with_corners=True, value_mask=None):
    """
    Create an ESMF :class:`~ESMF.api.grid.Grid` object from an OCGIS
    :class:`~ocgis.interface.base.dimension.spatial.SpatialDimension` object.

    :param sdim: The target spatial dimension to convert into an ESMF grid.
    :type sdim: :class:`~ocgis.interface.base.dimension.spatial.SpatialDimension`
    :param bool with_corners: If ``True``, attempt to access corners from ``sdim``.
    :param value_mask: If present, use a logical *or* operation with ``sdim``'s mask when creating the output grid's
     mask. Values of ``True`` in ``value_mask`` are assumed to be masked.
    :type value_mask: boolean :class:`numpy.array` with same dimension as ``sdim``
    :rtype: :class:`ESMF.api.grid.Grid`
    """

    ogrid = sdim.grid
    egrid = ESMF.Grid(max_index=np.array(ogrid.value.shape[1:]), staggerloc=ESMF.StaggerLoc.CENTER,
                      coord_sys=ESMF.CoordSys.SPH_DEG)
    row = egrid.get_coords(0, staggerloc=ESMF.StaggerLoc.CENTER)
    row[:] = ogrid.value[0, ...]
    col = egrid.get_coords(1, staggerloc=ESMF.StaggerLoc.CENTER)
    col[:] = ogrid.value[1, ...]

    # use a logical or operation to merge with value_mask if present
    if value_mask is not None:
        # convert to boolean to make sure
        value_mask = value_mask.astype(bool)
        # do the logical or operation selecting values
        value_mask = np.logical_or(value_mask, ogrid.value.mask[0])
    else:
        value_mask = ogrid.value.mask[0]
    # follows SCRIP convention where 1 is unmasked and 0 is masked
    esmf_mask = np.invert(value_mask).astype(np.int8)
    egrid.add_item(ESMF.GridItem.MASK, staggerloc=ESMF.StaggerLoc.CENTER, from_file=False)
    egrid.mask[0][:] = esmf_mask

    # attempt to access corners if possible
    if with_corners:
        try:
            corners_esmf = sdim.grid.corners_esmf
            # adding corners. first tell the grid object to allocate corners
            egrid.add_coords(staggerloc=[ESMF.StaggerLoc.CORNER])
            # get the coordinate pointers and set the coordinates
            grid_corner = egrid.coords[ESMF.StaggerLoc.CORNER]
            grid_corner[0][:] = corners_esmf[0]
            grid_corner[1][:] = corners_esmf[1]
        except CornersUnavailable:
            pass

    return egrid


def iter_esmf_fields(ofield, with_corners=True, value_mask=None):
    """
    For each time coordinate, yield an ESMF :class:`~ESMF.api.field.Field` from the input OCGIS Field ``ofield``. Only
    one realization and level are allowed.

    :param ofield: The input OCGIS Field object.
    :type ofield: :class:`ocgis.interface.base.field.Field`
    :param bool with_corners: If ``True``, attempt to access corners from ``sdim``.
    :param value_mask: If an :class:`numpy.array`, use a logical *or* operation with ``sdim``'s mask when creating the
     input grid's mask. Values of ``True`` in ``value_mask`` are assumed to be masked. If ``None`` is provided (the
     default) then the mask will be set using the first realization/time/level from the first variable contained in
     ``ofield``.
    :type value_mask: boolean :class:`numpy.array` with same dimension as ``sdim``
    :rtype: tuple(int, str, :class:`ESMF.api.field.Field`)

    The returned tuple elements are:

    ===== ============================== =========================================================================
    Index Type                           Description
    ===== ============================== =========================================================================
    0     int                            The time coordinate index in ``ofield`` being converted to an ESMF Field.
    1     str                            The alias of the variable currently being converted.
    2     :class:`~ESMF.api.field.Field` The ESMF Field object.
    ===== ============================== =========================================================================

    :raises: AssertionError
    """
    #todo: provide other options for calculating value_mask
    # only one level and realization allowed
    assert ofield.shape[0] == 1
    assert ofield.shape[2] == 1

    # retrieve the mask from the first variable contained in ofield
    if value_mask is None:
        sfield = ofield[0, 0, 0, :, :]
        variable = sfield.variables.first()
        value_mask = variable.value.mask.reshape(sfield.shape[-2:])
        assert np.all(value_mask.shape == sfield.shape[-2:])

    # create the esmf grid
    egrid = get_esmf_grid_from_sdim(ofield.spatial, with_corners=with_corners, value_mask=value_mask)

    # produce a new esmf field for each variable and time step
    for variable_alias, variable in ofield.variables.iteritems():
        for tidx in range(ofield.shape[1]):
            efield = ESMF.Field(egrid, variable_alias)
            efield.data[:] = variable.value[0, tidx, 0, :, :]
            yield tidx, variable_alias, efield


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
            try:
                source.spatial.grid.corners
                has_corners_sources.append(True)
            except CornersUnavailable:
                has_corners_sources.append(False)
        try:
            sdim.grid.corners
            has_corners_destination = True
        except CornersUnavailable:
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

    ####################################################################################################################
    # check extents. the spatial extent of the destination grid must fully containing the extents of the source grids.
    ####################################################################################################################

    def _get_centroid_extent_polygon_(data):
        minx, miny, maxx, maxy = data[1].min(), data[0].min(), data[1].max(), data[0].max()
        return make_poly([miny, maxy], [minx, maxx])

    # the object-based extent calculation accounts for bounds which are not relevant without corners
    if with_corners:
        extent_destination = sdim.grid.extent_polygon
    else:
        extent_destination = _get_centroid_extent_polygon_(sdim.grid.value.data)
    for source in sources:
        if with_corners:
            extent_source = source.spatial.grid.extent_polygon
        else:
            extent_source = _get_centroid_extent_polygon_(source.spatial.grid.value.data)
        if not extent_source.intersection(extent_destination).almost_equals(extent_source):
            msg = 'The destination extent must contain (boundaries may touch) the source extent.'
            raise RegriddingError(msg)


def iter_regridded_fields(sources, destination, with_corners='choose', value_mask=None):
    """
    Regrid ``sources`` to match the grid of ``destination``.

    :param sources: Sequence of source fields to regrid.
    :type sources: sequence of :class:`ocgis.interface.base.field.Field`
    :param destination: The target grid contained in a field or spatial dimension.
    :type destination: :class:`ocgis.interface.base.field.Field` or
     :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
    :param with_corners: If ``'choose'``, automatically determine if corners should be used. They will be used if they
     are available on all ``sources`` and the ``destination``. If ``True``, attempt to access corners from ``sdim``
     raising an exception if corners are not available. If ``False``, do not use corners.
    :type with_corners: str or bool
    :param value_mask: If an :class:`numpy.array`, use a logical *or* operation with ``sdim``'s mask when creating the
     input grid's mask. Values of ``True`` in ``value_mask`` are assumed to be masked. If ``None`` is provided (the
     default) then the mask will be set using the first realization/time/level from the first variable contained in
     ``ofield``.
    :type value_mask: boolean :class:`numpy.array` with same dimension as ``sdim``
    :rtype: :class:`ocgis.interface.base.field.Field`
    """

    # reference the spatial dimension. this is needed as destination may be a Field or SpatialDimension
    try:
        sdim = destination.spatial
    # likely a SpatialDimension object
    except AttributeError:
        sdim = destination

    # this function runs a series of asserts to make sure the sources and destination are compatible
    try:
        check_fields_for_regridding(sources, destination, with_corners=with_corners)
    except CornersInconsistentError:
        if with_corners == 'choose':
            with_corners = False
        else:
            raise

    # get the destination esmf grid
    esmf_destination_grid = get_esmf_grid_from_sdim(sdim, with_corners=with_corners, value_mask=value_mask)
    # get the destination esmf field
    esmf_destination_field = ESMF.Field(esmf_destination_grid, 'destination')

    # check for corners on the destination grid. if they exist, conservative regridding is possible.
    regrid_method = ESMF.RegridMethod.CONSERVE
    corner = esmf_destination_grid.coords[ESMF.StaggerLoc.CORNER]
    for idx in [0, 1]:
        if corner[idx].shape == ():
            if np.all(corner[idx] == np.array(0.0)):
                regrid_method = None
                break

    # sources may be modified in the process, so make a copy of these grids
    sources = deepcopy(sources)
    # this is the new shape of the output variables
    new_shape_spatial = sdim.shape
    # regrid each source in turn
    for source in sources:
        build = True
        fills = {}
        # only regridding across the time dimension at this point
        for tidx, variable_alias, efield in iter_esmf_fields(source, with_corners=with_corners, value_mask=value_mask):

            # we need to generate new variables given the change in shape
            if variable_alias not in fills:
                new_shape = list(source.variables[variable_alias].shape)
                new_shape[-2:] = new_shape_spatial
                fill_var = source.variables[variable_alias].get_empty_like(shape=new_shape)
                fills[variable_alias] = fill_var

            # only build the grid once
            if build:
                regrid = ESMF.Regrid(efield, esmf_destination_field, unmapped_action=ESMF.UnmappedAction.IGNORE,
                                     regrid_method=regrid_method)
                # place a deepcopy of the destination spatial dimension as the output spatial dimension for the regridded
                # fields
                out_sdim = deepcopy(sdim)
                # if this is not conservative regridding then bounds and corners on the output should be stripped.
                if not with_corners or regrid_method is None:
                    if out_sdim.grid.row is not None:
                        out_sdim.grid.row.bounds
                        out_sdim.grid.row.bounds = None
                        out_sdim.grid.col.bounds
                        out_sdim.grid.col.bounds = None
                        out_sdim.grid._corners = None
                    # remove any polygons if they exist
                    if out_sdim._geom is not None:
                        out_sdim.geom._polygon = None
                build = False

            # perform the regrid operation and fill the new variabales
            regridded_esmf_field = regrid(efield, esmf_destination_field)
            fills[variable_alias].value.data[0, tidx, 0, :, :] = regridded_esmf_field.data
            fills[variable_alias].value.mask[0, tidx, 0, :, :] = np.invert(regridded_esmf_field.grid.mask[0].astype(bool))

        # set the output spatial dimension of the regridded field
        source.spatial = out_sdim

        # create a new variable collection and add the variables to the output field
        source.variables = VariableCollection()
        for v in fills.itervalues():
            source.variables.add_variable(v)

        yield source