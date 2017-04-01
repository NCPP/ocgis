from copy import deepcopy

import netCDF4 as nc
import numpy as np

import ocgis
from ocgis import constants
from ocgis.calc import tile
from ocgis.calc.base import AbstractMultivariateFunction
from ocgis.calc.engine import OcgCalculationEngine
from ocgis.constants import TagNames
from ocgis.ops.core import OcgOperations
from ocgis.util.helpers import ProgressBar


def compute(ops, tile_dimension, verbose=False, use_optimizations=True):
    """
    Used for computations on large arrays where memory limitations are a consideration. It is is also useful for
    extracting data from a server that has limitations on the size of requested data arrays. This function creates an
    empty destination NetCDF file that is then filled by executing the operations on chunks of the requested
    target dataset(s) and filling the destination NetCDF file.

    :param ops: The target operations to tile. There must be a calculation associated with
     the operations.
    :type ops: :class:`ocgis.OcgOperations`
    :param int tile_dimension: The target tile/chunk dimension. This integer value must be greater than zero.
    :param bool verbose: If ``True``, print more verbose information to terminal.
    :param bool use_optimizations: If ``True``, cache :class:`Field` and :class:`TemporalGroupDimension` objects for
     reuse during tile iteration.
    :raises: AssertionError, ValuError
    :returns: Path to the output NetCDF file.
    :rtype: str

    >>> from ocgis import RequestDataset, OcgOperations
    >>> from ocgis.util.large_array import compute
    >>> rd = RequestDataset(uri='/path/to/file', variable='tas')
    >>> ops = OcgOperations(dataset=rd,calc=[{'func':'mean','name':'mean'}],output_format='nc')
    >>> ret = compute(ops, 25)
    """

    assert isinstance(ops, OcgOperations)
    assert ops.calc is not None
    assert ops.output_format == constants.OUTPUT_FORMAT_NETCDF

    # Ensure that progress is not showing 100% at first.
    if ops.callback is not None:
        orgcallback = ops.callback

        def zeropercentagecallback(p, m):
            orgcallback(0., m)

        ops.callback = zeropercentagecallback

    tile_dimension = int(tile_dimension)
    if tile_dimension <= 0:
        raise ValueError('"tile_dimension" must be greater than 0')

    # Determine if we are working with a multivariate function.
    if OcgCalculationEngine._check_calculation_members_(ops.calc, AbstractMultivariateFunction):
        # Only one multivariate calculation allowed.
        assert len(ops.calc) == 1
        has_multivariate = True
    else:
        # Only one dataset allowed.
        assert len(list(ops.dataset)) == 1
        has_multivariate = False

    # work on a copy of the operations to create the template file
    ops_file_only = deepcopy(ops)
    # we need the output to be file only for the first request
    ops_file_only.file_only = True
    # save the environment flag for calculation optimizations.
    orig_oc = ocgis.env.OPTIMIZE_FOR_CALC

    try:
        # tell the software we are optimizing for calculations   
        ocgis.env.OPTIMIZE_FOR_CALC = True

        # first, write the template file
        if verbose:
            print('getting fill file...')
        fill_file = ops_file_only.execute()

        # if there is a geometry, we have to find the offset for the slice. we
        # also need to account for the subset mask.
        if ops.geom is not None:
            if verbose:
                print('geometry subset is present. calculating slice offsets...')
            ops_offset = deepcopy(ops)
            ops_offset.output_format = 'numpy'
            ops_offset.calc = None
            ops_offset.agg_selection = True
            ops_offset.snippet = False
            coll = ops_offset.execute()

            for row in coll.iter_melted(tag=TagNames.DATA_VARIABLES):
                assert row['variable']._value is None

            ref_field = coll.get_element()
            ref_grid = ref_field.grid
            row_offset = ref_grid.dimensions[0]._src_idx[0]
            col_offset = ref_grid.dimensions[1]._src_idx[0]
            mask_spatial = ref_grid.get_mask()
        # otherwise the offset is zero...
        else:
            row_offset = 0
            col_offset = 0
            mask_spatial = None

        # get the shape for the tile schema
        if verbose:
            print('getting tile schema shape inputs...')
        shp_variable = ops.calc[0]['name']
        template_rd = ocgis.RequestDataset(uri=fill_file, variable=shp_variable)
        template_field = template_rd.get()
        shp = template_field.grid.shape

        if use_optimizations:
            # if there is a calculation grouping, optimize for it. otherwise, pass
            # this value as None.
            try:
                # tgd_field = ops.dataset.first().get()
                archetype_dataset = list(ops.dataset)[0]
                tgd_field = archetype_dataset.get()
                template_tgd = tgd_field.temporal.get_grouping(deepcopy(ops.calc_grouping))
                if not has_multivariate:
                    key = archetype_dataset.field_name
                else:
                    key = '_'.join([__.field_name for __ in ops.dataset])
                optimizations = {'tgds': {key: template_tgd}}
            except TypeError:
                optimizations = None

            # load the fields and pass those for optimization
            field_optimizations = {}
            for rd in ops.dataset:
                gotten_field = rd.get(format_time=ops.format_time)
                field_optimizations.update({rd.field_name: gotten_field})
            optimizations = optimizations or {}
            optimizations['fields'] = field_optimizations
        else:
            optimizations = None

        if verbose:
            print('getting tile schema...')
        schema = tile.get_tile_schema(shp[0], shp[1], tile_dimension)
        lschema = len(schema)

        # Create new callbackfunction where the 0-100% range is converted to a subset corresponding to the no. of
        # blocks to be calculated
        if ops.callback is not None:
            percentageDone = 0
            callback = ops.callback

            def newcallback(p, m):
                p = (p / lschema) + percentageDone
                orgcallback(p, m)

            ops.callback = newcallback

        if verbose:
            print(('output file is: {0}'.format(fill_file)))
            print(('tile count: {0}'.format(lschema)))

        fds = nc.Dataset(fill_file, 'a')
        try:
            if verbose:
                progress = ProgressBar('tiles progress')
            if ops.callback is not None and callback:
                callback(0, "Initializing calculation")
            for ctr, indices in enumerate(iter(schema.values()), start=1):
                # appropriate adjust the slices to account for the spatial subset
                row = [ii + row_offset for ii in indices['row']]
                col = [ii + col_offset for ii in indices['col']]

                # copy the operations and modify arguments
                ops_slice = deepcopy(ops)
                ops_slice.geom = None
                ops_slice.slice = [None, None, None, row, col]
                ops_slice.output_format = 'numpy'
                ops_slice.optimizations = optimizations
                # return the object slice
                ret = ops_slice.execute()
                for field in ret.iter_fields():
                    for variable in field.data_variables:
                        vref = fds.variables[variable.name]
                        # we need to remove the offsets to adjust for the zero-based fill file.
                        slice_row = slice(row[0] - row_offset, row[1] - row_offset)
                        slice_col = slice(col[0] - col_offset, col[1] - col_offset)
                        # if there is a spatial mask, update accordingly
                        if mask_spatial is not None:
                            set_variable_spatial_mask(variable, mask_spatial, slice_row, slice_col)
                            fill_mask = field.grid.get_mask(create=True)
                            fill_mask[:, :] = mask_spatial[slice_row, slice_col]
                            fill_mask = np.ma.array(np.zeros(fill_mask.shape), mask=fill_mask)
                            fds.variables[field.grid.mask_variable.name][slice_row, slice_col] = fill_mask
                        fill_value = variable.get_masked_value()
                        # fill the netCDF container variable adjusting for shape
                        if len(vref.shape) == 3:
                            vref[:, slice_row, slice_col] = fill_value
                        elif len(vref.shape) == 4:
                            vref[:, :, slice_row, slice_col] = fill_value
                        else:
                            raise NotImplementedError(vref.shape)

                        fds.sync()
                if verbose:
                    progress.progress(int((float(ctr) / lschema) * 100))
                if ops.callback is not None and callback:
                    percentageDone = ((float(ctr) / lschema) * 100)
        finally:
            fds.close()
    finally:
        ocgis.env.OPTIMIZE_FOR_CALC = orig_oc
    if verbose:
        progress.endProgress()
        print('complete.')

    return fill_file


def set_variable_spatial_mask(variable, mask_spatial, slice_row, slice_col):
    """
    Update the mask on ``variable`` in-place to match ``mask_spatial``. The array slice updated is constrained by
    ``slice_row`` and ``slice_col``.

    :param variable: The target variable to update.
    :type variable: :class:`ocgis.Variable`
    :param mask_spatial: The boolean mask array resulting from a spatial operation on the ``variable``'s field. Must
     have same spatial dimensions as ``variable``.
    :type mask_spatial: boolean ndarray
    :param slice_row: The row slice to update.
    :type slice_row: slice
    :param slice_col: The column slice to update.
    :type slice_col: slice
    """

    fill_mask = np.zeros(variable.shape, dtype=bool)
    fill_mask[..., :, :] = mask_spatial[slice_row, slice_col]
    vmask = variable.get_mask(create=True)
    vmask = np.logical_or(fill_mask, vmask[:, :])
    variable.set_mask(vmask)
