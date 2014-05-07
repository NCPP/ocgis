import ocgis
from ocgis.calc import tile
import netCDF4 as nc
from ocgis.util.helpers import ProgressBar
import numpy as np
from ocgis.api.operations import OcgOperations
from copy import deepcopy
from ocgis.calc.engine import OcgCalculationEngine
from ocgis.calc.base import AbstractMultivariateFunction


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
    :raises: AssertionError
    :returns: Path to the output NetCDF file.
    :rtype: str

    >>> from ocgis import RequestDataset, OcgOperations
    >>> from ocgis.util.large_array import compute
    >>> rd = RequestDataset(uri='/path/to/file',variable='tas')
    >>> ops = OcgOperations(dataset=rd,calc=[{'func':'mean','name':'mean'}],output_format='nc')
    >>> ret = compute(ops, 25)
    """

    ## validate arguments
    assert(isinstance(ops, OcgOperations))
    assert(ops.calc is not None)
    assert(ops.output_format == 'nc')
    tile_dimension = int(tile_dimension)
    if tile_dimension <= 0:
        raise (ValueError('"tile_dimension" must be greater than 0'))

    ## determine if we are working with a multivariate function
    if OcgCalculationEngine._check_calculation_members_(ops.calc, AbstractMultivariateFunction):
        ## only one multivariate calculation allowed
        assert (len(ops.calc) == 1)
        has_multivariate = True
    else:
        ## only one calculation allowed
        assert (len(ops.dataset) == 1)
        has_multivariate = False

    ## work on a copy of the operations to create the template file
    ops_file_only = deepcopy(ops)
    ## we need the output to be file only for the first request
    ops_file_only.file_only = True
    ## save the environment flag for calculation optimizations.
    orig_oc = ocgis.env.OPTIMIZE_FOR_CALC

    try:
        ## tell the software we are optimizing for calculations   
        ocgis.env.OPTIMIZE_FOR_CALC = True

        ## first, write the template file
        if verbose: print('getting fill file...')
        fill_file = ops_file_only.execute()
        ## if there is a geometry, we have to find the offset for the slice. we
        ## also need to account for the subset mask.
        if ops.geom is not None:
            if verbose:
                print('geometry subset is present. calculating slice offsets...')
            ops_offset = deepcopy(ops)
            ops_offset.output_format = 'numpy'
            ops_offset.calc = None
            ops_offset.agg_selection = True
            ops_offset.snippet = False
            coll = ops_offset.execute()

            for row in coll.get_iter_melted():
                ## assert the values are not loaded...
                assert(row['variable']._value is None)
                ## assert only 3 or 4 dimensional data is being used
                assert(row['field'].shape_as_dict['R'] == 1)

            ref_spatial = coll[1][ops_offset.dataset[0].alias].spatial
            row_offset = ref_spatial.grid.row._src_idx[0]
            col_offset = ref_spatial.grid.col._src_idx[0]
            mask_spatial = ref_spatial.get_mask()
        ## otherwise the offset is zero...
        else:
            row_offset = 0
            col_offset = 0
            mask_spatial = None

        ## get the shape for the tile schema
        if verbose: print('getting tile schema shape inputs...')
        #        if has_multivariate == False:
        #            shp_variable = '{0}_{1}'.format(ops.calc[0]['name'],ops.dataset[0].alias)
        #        else:
        #            shp_variable = ops.calc[0]['name']
        shp_variable = ops.calc[0]['name']
        template_rd = ocgis.RequestDataset(uri=fill_file, variable=shp_variable)
        template_field = template_rd.get()
        shp = template_field.shape[-2:]

        if use_optimizations:
            ## if there is a calculation grouping, optimize for it. otherwise, pass
            ## this value as None.
            try:
                tgd_field = ops.dataset[0].get()
                template_tgd = tgd_field.temporal.get_grouping(deepcopy(ops.calc_grouping))
                if not has_multivariate:
                    key = ops.dataset[0].alias
                else:
                    key = '_'.join([__.alias for __ in ops.dataset])
                optimizations = {'tgds': {key: template_tgd}}
            except TypeError:
                optimizations = None

            ## load the fields and pass those for optimization
            field_optimizations = {}
            for rd in ops.dataset:
                gotten_field = rd.get(format_time=ops.format_time)
                field_optimizations.update({rd.alias: gotten_field})
            optimizations = optimizations or {}
            optimizations['fields'] = field_optimizations
        else:
            optimizations = None

        if verbose: print('getting tile schema...')
        schema = tile.get_tile_schema(shp[0], shp[1], tile_dimension)

        if verbose:
            print('output file is: {0}'.format(fill_file))
            lschema = len(schema)
            print('tile count: {0}'.format(lschema))

        fds = nc.Dataset(fill_file, 'a')
        try:
            if verbose:
                progress = ProgressBar('tiles progress')
            for ctr, indices in enumerate(schema.itervalues(), start=1):
                ## appropriate adjust the slices to account for the spatial subset
                row = [ii + row_offset for ii in indices['row']]
                col = [ii + col_offset for ii in indices['col']]
                ## copy the operations and modify arguments
                ops_slice = deepcopy(ops)
                ops_slice.geom = None
                ops_slice.slice = [None, None, None, row, col]
                ops_slice.output_format = 'numpy'
                ops_slice.optimizations = optimizations
                ## return the object slice
                ret = ops_slice.execute()
                for field_map in ret.itervalues():
                    for field in field_map.itervalues():
                        field_shape = field.shape_as_dict
                        for alias, variable in field.variables.iteritems():
                            vref = fds.variables[alias]
                            assert (isinstance(variable.value, np.ma.MaskedArray))
                            ## we need to remove the offsets to adjust for the zero-based
                            ## fill file.
                            slice_row = slice(row[0] - row_offset, row[1] - row_offset)
                            slice_col = slice(col[0] - col_offset, col[1] - col_offset)
                            ## if there is a spatial mask, update accordingly
                            if mask_spatial is not None:
                                fill_mask = np.zeros(variable.value.shape, dtype=bool)
                                fill_mask[..., :, :] = mask_spatial[slice_row, slice_col]
                                variable.value.mask = fill_mask
                            ## squeeze out extra dimensions from ocgis
                            fill_value = np.squeeze(variable.value)
                            ## fill the netCDF container variable adjusting for shape
                            if len(vref.shape) == 3:
                                reshape = (field_shape['T'], field_shape['Y'], field_shape['X'])
                                vref[:, slice_row, slice_col] = fill_value.reshape(*reshape)
                            elif len(vref.shape) == 4:
                                reshape = (field_shape['T'], field_shape['Z'], field_shape['Y'], field_shape['X'])
                                vref[:, :, slice_row, slice_col] = fill_value.reshape(*reshape)
                            else:
                                raise (NotImplementedError(vref.shape))

                            ## write the data to disk
                            fds.sync()
                if verbose:
                    progress.progress(int((float(ctr) / lschema) * 100))
        finally:
            fds.close()
    finally:
        ocgis.env.OPTIMIZE_FOR_CALC = orig_oc
    if verbose:
        progress.endProgress()
        print('complete.')

    return(fill_file)
