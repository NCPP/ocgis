import itertools
import logging
from abc import ABCMeta
from collections import OrderedDict
from copy import deepcopy

import netCDF4 as nc
import numpy as np
import six
from netCDF4._netCDF4 import VLType, MFDataset, MFTime
from ocgis import constants, vm
from ocgis import env
from ocgis.base import orphaned, raise_if_empty
from ocgis.collection.field import Field
from ocgis.constants import MPIWriteMode, DimensionMapKey, KeywordArgument, DriverKey, CFName, SourceIndexType
from ocgis.driver.base import AbstractDriver, driver_scope
from ocgis.exc import ProjectionDoesNotMatch, PayloadProtectedError, NoDataVariablesFound, \
    GridDeficientError
from ocgis.util.helpers import itersubclasses, get_iter, get_formatted_slice, get_by_key_list, is_auto_dtype, get_group
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.variable.base import SourcedVariable, ObjectType, get_slice_sequence_using_local_bounds
from ocgis.variable.crs import CFCoordinateReferenceSystem, CoordinateReferenceSystem, CFRotatedPole, CFSpherical, \
    AbstractProj4CRS
from ocgis.variable.dimension import Dimension
from ocgis.variable.temporal import TemporalVariable
from ocgis.vmachine.mpi import OcgDist


class DriverNetcdf(AbstractDriver):
    """
    Driver for netCDF files that avoids any hint of metadata.

    Driver keyword arguments (``driver_kwargs``) to the request dataset:

    * Any keyword arguments to dataset or multi-file dataset creation.
    """
    extensions = ('.*\.nc', 'http.*')
    key = DriverKey.NETCDF
    output_formats = 'all'
    common_extension = 'nc'

    @property
    def data_model(self):
        return self.metadata_source['file_format']

    @staticmethod
    def get_data_variable_names(group_metadata, group_dimension_map):
        return tuple()

    @classmethod
    def get_variable_for_writing_temporal(cls, temporal_variable):
        return temporal_variable.value_numtime

    def get_variable_value(self, variable):
        return get_value_from_request_dataset(variable)

    @classmethod
    def write_variable(cls, var, dataset, write_mode=MPIWriteMode.NORMAL, **kwargs):
        """
        Write a variable to an open netCDF dataset object.

        :param var: Variable object.
        :param dataset: Open netCDF dataset object.
        :param kwargs: Arguments to netCDF variable creation with additional keyword arguments below.
        :keyword bool file_only: (``=False``) If ``True``, do not write the value to the output file. Create an empty
         netCDF file.
        :keyword bool unlimited_to_fixed_size: (``=False``) If ``True``, convert the unlimited dimension to a fixed size.
        """
        # There should never be any write operations associated with an empty variable.
        raise_if_empty(var)

        # Write the parent collection if available on the variable.
        if not var.is_orphaned:
            parent_kwargs = {}
            parent_kwargs[KeywordArgument.VARIABLE_KWARGS] = kwargs
            return var.parent.write(dataset, **parent_kwargs)

        assert isinstance(dataset, nc.Dataset)

        file_only = kwargs.pop(KeywordArgument.FILE_ONLY, False)
        unlimited_to_fixed_size = kwargs.pop(KeywordArgument.UNLIMITED_TO_FIXED_SIZE, False)

        # No data should be written during a global write. Data will be filled in during the append process.
        if write_mode == MPIWriteMode.TEMPLATE:
            file_only = True

        if var.name is None:
            msg = 'A variable "name" is required.'
            raise ValueError(msg)

        # Dimension creation should not occur during a fill operation. The dimensions and variables have already been
        # created.
        if write_mode != MPIWriteMode.FILL:
            dimensions = var.dimensions

            dtype = cls.get_variable_write_dtype(var)
            if isinstance(dtype, ObjectType):
                dtype = dtype.create_vltype(dataset, dimensions[0].name + '_VLType')
            # Assume we are writing string data if the data type is object.
            elif dtype == str or var.is_string_object:
                dtype = 'S1'

            if len(dimensions) > 0:
                # Special handling for string variables.
                if dtype == 'S1':
                    max_length = var.string_max_length_global
                    assert max_length is not None
                    dimensions = [var.dimensions[0], Dimension('{}_ocgis_slen'.format(var.name), max_length)]
                dimensions = list(dimensions)
                # Convert the unlimited dimension to fixed size if requested.
                for idx, d in enumerate(dimensions):
                    if d.is_unlimited and unlimited_to_fixed_size:
                        dimensions[idx] = Dimension(d.name, size=var.shape[idx])
                        break
                # Create the dimensions.
                for dim in dimensions:
                    create_dimension_or_pass(dim, dataset, write_mode=write_mode)
                dimensions = [d.name for d in dimensions]

            # Only use the fill value if something is masked.
            is_nc3 = dataset.data_model.startswith('NETCDF3')
            if ((len(dimensions) > 0 and var.has_masked_values) and (
                    write_mode == MPIWriteMode.TEMPLATE or not file_only)) or (
                    is_nc3 and not var.has_allocated_value and len(
                dimensions) > 0):
                fill_value = cls.get_variable_write_fill_value(var)
            else:
                # Copy from original attributes.
                if '_FillValue' not in var.attrs:
                    fill_value = None
                else:
                    fill_value = cls.get_variable_write_fill_value(var)

        if write_mode == MPIWriteMode.FILL:
            ncvar = dataset.variables[var.name]
        else:
            ncvar = dataset.createVariable(var.name, dtype, dimensions=dimensions, fill_value=fill_value, **kwargs)

        # Do not fill values on file_only calls. Also, only fill values for variables with dimension greater than zero.
        if not file_only and not var.is_empty and not isinstance(var, CoordinateReferenceSystem):
            if isinstance(var.dtype, ObjectType) and not isinstance(var, TemporalVariable):
                bounds_local = var.dimensions[0].bounds_local
                for idx in range(bounds_local[0], bounds_local[1]):
                    ncvar[idx] = np.array(var.get_value()[idx - bounds_local[0]])
            else:
                fill_slice = get_slice_sequence_using_local_bounds(var)
                data_value = cls.get_variable_write_value(var)
                # Only write allocated values.
                if data_value is not None:
                    if var.dtype == str or var.is_string_object:
                        for idx in range(fill_slice[0].start, fill_slice[0].stop):
                            try:
                                curr_value = data_value[idx - fill_slice[0].start]
                            except Exception as e:
                                msg = "Variable name is '{}'. Original message: ".format(var.name) + e.message
                                raise e.__class__(msg)
                            for sidx, sval in enumerate(curr_value):
                                ncvar[idx, sidx] = sval
                    elif var.ndim == 0:
                        ncvar[:] = data_value
                    else:
                        try:
                            ncvar.__setitem__(fill_slice, data_value)
                        except Exception as e:
                            msg = "Variable name is '{}'. Original message: ".format(var.name) + e.message
                            raise e.__class__(msg)

        # Only set variable attributes if this is not a fill operation.
        if write_mode != MPIWriteMode.FILL:
            var.write_attributes_to_netcdf_object(ncvar)
            if var.units is not None:
                ncvar.setncattr('units', str(var.units))

        dataset.sync()

    @classmethod
    def _write_variable_collection_main_(cls, vc, opened_or_path, write_mode, **kwargs):
        assert write_mode is not None

        dataset_kwargs = kwargs.get('dataset_kwargs', {})
        variable_kwargs = kwargs.get('variable_kwargs', {})

        # When filling a dataset, we use append mode.
        if write_mode == MPIWriteMode.FILL:
            mode = 'a'
        else:
            mode = 'w'

        # For an asynchronous write, treat everything like a single rank.
        if write_mode == MPIWriteMode.ASYNCHRONOUS:
            possible_ranks = [0]
        else:
            possible_ranks = vm.ranks

        # Write the data on each rank.
        for idx, rank_to_write in enumerate(possible_ranks):
            # The template write only occurs on the first rank.
            if write_mode == MPIWriteMode.TEMPLATE and rank_to_write != 0:
                pass
            # If this is not a template write, fill the data.
            elif write_mode == MPIWriteMode.ASYNCHRONOUS or vm.rank == rank_to_write:
                with driver_scope(cls, opened_or_path=opened_or_path, mode=mode, **dataset_kwargs) as dataset:
                    # Write global attributes if we are not filling data.
                    if write_mode != MPIWriteMode.FILL:
                        vc.write_attributes_to_netcdf_object(dataset)
                    # This is the main variable write loop.
                    variables_to_write = get_variables_to_write(vc)
                    for variable in variables_to_write:
                        # Load the variable's data before orphaning. The variable needs its parent to know which
                        # group it is in.
                        variable.load()
                        # Call the individual variable write method in fill mode. Orphaning is required as a
                        # variable will attempt to write its parent first.
                        with orphaned(variable, keep_dimensions=True):
                            variable.write(dataset, write_mode=write_mode, **variable_kwargs)
                    # Recurse the children.
                    for child in list(vc.children.values()):
                        if write_mode != MPIWriteMode.FILL:
                            group = nc.Group(dataset, child.name)
                        else:
                            group = dataset.groups[child.name]
                        child.write(group, write_mode=write_mode, **kwargs)
                    dataset.sync()
            vm.barrier()

    def _get_metadata_main_(self):
        with driver_scope(self) as ds:
            ret = parse_metadata(ds)
        return ret

    @classmethod
    def _get_write_modes_(cls, the_vm, **kwargs):
        dataset_kwargs = kwargs.get(KeywordArgument.DATASET_KWARGS, {})

        ret = AbstractDriver._get_write_modes_(the_vm, **kwargs)
        if env.USE_NETCDF4_MPI and the_vm.size > 1:
            if dataset_kwargs.get('format', 'NETCDF4') == 'NETCDF4' and dataset_kwargs.get('parallel', True):
                ret = [MPIWriteMode.ASYNCHRONOUS]
        return ret

    def _init_variable_from_source_main_(self, variable, variable_object):
        init_variable_using_metadata_for_netcdf(variable, self.rd.metadata)

    @staticmethod
    def _open_(uri, mode='r', **kwargs):
        """
        :rtype: object
        """
        kwargs = kwargs.copy()
        group_indexing = kwargs.pop('group_indexing', None)
        lvm = kwargs.pop('vm', vm)

        if isinstance(uri, six.string_types):
            # Open the dataset in parallel if we want to use the netCDF MPI capability. It may not be available even in
            # parallel.
            if mode == 'w' and lvm.size > 1:
                if kwargs.get('format', 'NETCDF4') == 'NETCDF4':
                    if kwargs.get('parallel') is None and env.USE_NETCDF4_MPI:
                        kwargs['parallel'] = True
                    if kwargs.get('parallel') and kwargs.get('comm') is None:
                        kwargs['comm'] = lvm.comm
            ret = nc.Dataset(uri, mode=mode, **kwargs)
        else:
            ret = nc.MFDataset(uri, **kwargs)

        if group_indexing is not None:
            for group_name in get_iter(group_indexing):
                ret = ret.groups[group_name]

        return ret


@six.add_metaclass(ABCMeta)
class AbstractDriverNetcdfCF(DriverNetcdf):
    def create_dimension_map(self, group_metadata, strict=False):
        dmap = super(AbstractDriverNetcdfCF, self).create_dimension_map(group_metadata, strict=strict)

        # Check for time, level, and realization variables. This involves checking for the presence of any bounds
        # variables.
        axes = {'realization': 'R', 'time': 'T', 'level': 'Z'}
        entries = self._create_dimension_map_entries_dict_(axes, group_metadata, strict)
        self._update_dimension_map_with_entries_(entries, dmap)

        # Check for coordinate system variables. This will check every variable.
        crs_name = get_coordinate_system_variable_name(self, group_metadata)
        if crs_name is not None:
            dmap.set_crs(crs_name)

        # Check for a spatial mask.
        variables = group_metadata['variables']
        for varname, var in variables.items():
            if 'ocgis_role' in var.get('attrs', {}):
                if var['attrs']['ocgis_role'] == 'spatial_mask':
                    dmap.set_spatial_mask(varname, attrs=var['attrs'])
                    break

        return dmap

    @staticmethod
    def _create_dimension_map_entries_dict_(axes, group_metadata, strict, attr_name='axis'):
        variables = group_metadata['variables']
        check_bounds = list(axes.keys())
        if 'realization' in check_bounds:
            check_bounds.pop(check_bounds.index('realization'))

        # Get the main entry for each axis.
        for k, v in list(axes.items()):
            axes[k] = create_dimension_map_entry(v, variables, strict=strict, attr_name=attr_name)

        # Attempt to find bounds for each entry (ignoring realizations).
        for k in check_bounds:
            if axes[k] is not None:
                keys = ['bounds']
                if k == 'time':
                    keys += ['climatology']
                bounds_var = get_by_key_list(variables[axes[k]['variable']]['attrs'], keys)
                if bounds_var is not None:
                    if bounds_var not in variables:
                        msg = 'Bounds listed for variable "{0}" but the destination bounds variable "{1}" does not exist.'. \
                            format(axes[k]['variable'], bounds_var)
                        ocgis_lh(msg, logger='nc.driver', level=logging.WARNING)
                        bounds_var = None
                axes[k]['bounds'] = bounds_var
        entries = {k: v for k, v in list(axes.items()) if v is not None}
        return entries

    def _get_crs_main_(self, group_metadata):
        return get_crs_variable(group_metadata, self.rd.rotated_pole_priority)

    @staticmethod
    def _update_dimension_map_with_entries_(entries, dimension_map):
        for k, v in entries.items():
            variable_name = v.pop('variable')
            dimension_map.set_variable(k, variable_name, **v)


class DriverNetcdfCF(AbstractDriverNetcdfCF):
    """
    Metadata-aware netCDF driver interpreting CF-Grid by default.
    """
    key = DriverKey.NETCDF_CF
    _esmf_fileformat = 'GRIDSPEC'
    _default_crs = env.DEFAULT_COORDSYS
    _priority = True

    @staticmethod
    def array_resolution(value, axis):
        """See :meth:`ocgis.driver.base.AbstractDriver.array_resolution`"""
        if value.size == 1:
            return 0.0
        else:
            resolution_limit = constants.RESOLUTION_LIMIT
            is_vectorized = value.ndim == 1
            if is_vectorized:
                target = np.abs(np.diff(np.abs(value[0:resolution_limit])))
            else:
                if axis == 0:
                    target = np.abs(np.diff(np.abs(value[:, 0:resolution_limit]), axis=axis))
                elif axis == 1:
                    target = np.abs(np.diff(np.abs(value[0:resolution_limit, :]), axis=axis))
                else:
                    raise NotImplementedError(axis)
            ret = np.mean(target)
            return ret

    def create_dimension_map(self, group_metadata, strict=False):
        dmap = super(DriverNetcdfCF, self).create_dimension_map(group_metadata, strict=strict)

        # Get dimension variable metadata. This involves checking for the presence of any bounds variables.
        axes = {'x': 'X', 'y': 'Y'}
        entries = self._create_dimension_map_entries_dict_(axes, group_metadata, strict, attr_name='axis')
        self._update_dimension_map_with_entries_(entries, dmap)

        # Hack to get the original coordinate system. Tell the driver to get rotated pole if it can from the metadata.
        original_rpp = self.rd.rotated_pole_priority
        try:
            self.rd.rotated_pole_priority = True
            raw_crs = self.get_crs(group_metadata)
        finally:
            self.rd.rotated_pole_priority = original_rpp
        # Swap out correct axis variables for spherical as opposed to rotated latitude longitude
        if not self.rd.rotated_pole_priority and isinstance(raw_crs, CFRotatedPole):
            crs = self.get_crs(group_metadata)
            if isinstance(crs, CFSpherical):
                axes = {DimensionMapKey.X: {'value': CFName.StandardName.LONGITUDE, 'axis': 'X'},
                        DimensionMapKey.Y: {'value': CFName.StandardName.LATITUDE, 'axis': 'Y'}}
                crs_entries = self._create_dimension_map_entries_dict_(axes, group_metadata, strict,
                                                                       attr_name=CFName.STANDARD_NAME)
                self._update_dimension_map_with_entries_(crs_entries, dmap)
                env.SET_GRID_AXIS_ATTRS = False
                env.COORDSYS_ACTUAL = raw_crs
        return dmap

    @staticmethod
    def get_data_variable_names(group_metadata, group_dimension_map):
        axes_needed = [DimensionMapKey.TIME, DimensionMapKey.X, DimensionMapKey.Y]
        dvars = []

        dimension_names_needed = []
        for axis in axes_needed:
            axis_variable = group_dimension_map.get_variable(axis)
            if axis_variable is None:
                # A required axis is missing in the dimension map. Hence, there are no dimensioned variables in the
                # group.
                return tuple()
            else:
                dimension_names_needed += group_dimension_map.get_dimension(axis)
        dimension_names_needed = set(dimension_names_needed)

        for vk, vv in list(group_metadata['variables'].items()):
            variable_dimension_names = set(vv['dimensions'])
            intersection = dimension_names_needed.intersection(variable_dimension_names)
            if len(intersection) == len(axes_needed):
                dvars.append(vk)

        return tuple(dvars)

    def get_distributed_dimension_name(self, dimension_map, dimensions_metadata):
        x_variable = dimension_map.get_variable(DimensionMapKey.X)
        y_variable = dimension_map.get_variable(DimensionMapKey.Y)
        if x_variable and y_variable:
            sizes = np.zeros(2, dtype={'names': ['dim', 'size'], 'formats': [object, int]})

            dimension_name_x = dimension_map.get_dimension(DimensionMapKey.X)[0]
            dimension_name_y = dimension_map.get_dimension(DimensionMapKey.Y)[0]

            sizes[0] = (dimension_name_x, dimensions_metadata[dimension_name_x]['size'])
            sizes[1] = (dimension_name_y, dimensions_metadata[dimension_name_y]['size'])
            max_index = np.argmax(sizes['size'])
            ret = sizes['dim'][max_index]
        else:
            ret = None
        return ret

    @staticmethod
    def get_grid(field):
        from ocgis import Grid
        try:
            ret = Grid(parent=field)
        except GridDeficientError:
            ret = None
        return ret

    @staticmethod
    def _gc_iter_dst_grid_slices_(grid_chunker):
        slice_store = []
        ydim_name = grid_chunker.dst_grid.dimensions[0].name
        xdim_name = grid_chunker.dst_grid.dimensions[1].name
        dst_grid_shape_global = grid_chunker.dst_grid.shape_global
        for idx in range(grid_chunker.dst_grid.ndim):
            splits = grid_chunker.nchunks_dst[idx]
            size = dst_grid_shape_global[idx]
            slices = create_slices_for_dimension(size, splits)
            slice_store.append(slices)
        for slice_y, slice_x in itertools.product(*slice_store):
            yield {ydim_name: create_slice_from_tuple(slice_y),
                   xdim_name: create_slice_from_tuple(slice_x)}

    @classmethod
    def _get_field_write_target_(cls, field):
        """Collective!"""

        if field.crs is not None:
            field.crs.format_spatial_object(field)

        grid = field.grid
        if grid is not None:
            # If any grid pieces are masked, ensure the mask is created across all grids.
            has_mask = vm.gather(grid.has_mask)
            if vm.rank == 0:
                if any(has_mask):
                    create_mask = True
                else:
                    create_mask = False
            else:
                create_mask = None
            create_mask = vm.bcast(create_mask)
            if create_mask and not grid.has_mask:
                grid.get_mask(create=True)

            # Putting units on bounds for netCDF-CF can confuse some parsers.
            if grid.has_bounds:
                field = field.copy()
                field.x.bounds.attrs.pop('units', None)
                field.y.bounds.attrs.pop('units', None)

        # Remove the current coordinate system if this is a dummy coordinate system.
        if env.COORDSYS_ACTUAL is not None:
            field = field.copy()
            field.set_crs(env.COORDSYS_ACTUAL, should_add=True)

        return field


def create_slice_from_tuple(tup):
    return slice(tup[0], tup[1])


def create_slices_for_dimension(size, splits):
    ompi = OcgDist(size=splits)
    dimname = 'foo'
    ompi.create_dimension(dimname, size, dist=True)
    ompi.update_dimension_bounds()
    slices = []
    for rank in range(splits):
        dimension = ompi.get_dimension(dimname, rank=rank)
        slices.append(dimension.bounds_local)
    return slices


def parse_metadata(rootgrp, fill=None):
    if fill is None:
        fill = OrderedDict()
    if 'groups' not in fill:
        fill['groups'] = OrderedDict()
    update_group_metadata(rootgrp, fill)
    for group in list(rootgrp.groups.values()):
        new_fill = fill['groups'][group.name] = OrderedDict()
        parse_metadata(group, fill=new_fill)
    return fill


def read_from_collection(target, request_dataset, parent=None, name=None, source_name=constants.UNINITIALIZED,
                         uid=None):
    # Allow an empty variable renaming map. This occurs when there are no visible data variables to the metadata
    # parser.
    try:
        rename_variable_map = request_dataset.rename_variable_map
    except NoDataVariablesFound:
        rename_variable_map = {}

    ret = Field(attrs=get_netcdf_attributes(target), parent=parent, name=name, source_name=source_name, uid=uid)
    pred = request_dataset.predicate
    for varname, ncvar in target.variables.items():
        if pred is not None and not pred(varname):
            continue
        source_name = varname
        name = rename_variable_map.get(varname, varname)
        sv = SourcedVariable(name=name, request_dataset=request_dataset, parent=ret, source_name=source_name)
        ret[name] = sv

    for group_name, ncgroup in list(target.groups.items()):
        child = read_from_collection(ncgroup, request_dataset, parent=ret, name=group_name, uid=uid)
        ret.add_child(child)
    return ret


def init_variable_using_metadata_for_netcdf(variable, metadata):
    source = get_group(metadata, variable.group, has_root=False)
    desired_name = variable.source_name
    var = source['variables'][desired_name]

    if vm.is_null:
        variable.convert_to_empty()
    else:
        # Update data type and fill value.
        if is_auto_dtype(variable._dtype) or var.get('dtype_packed') is not None:
            var_dtype = var['dtype']
            desired_dtype = deepcopy(var_dtype)
            if isinstance(var_dtype, VLType):
                desired_dtype = ObjectType(var_dtype)
            elif var['dtype_packed'] is not None:
                desired_dtype = deepcopy(var['dtype_packed'])
            variable._dtype = desired_dtype

        if variable._fill_value == 'auto':
            if var.get('fill_value_packed') is not None:
                desired_fill_value = var['fill_value_packed']
            else:
                desired_fill_value = var.get('fill_value')
            variable._fill_value = deepcopy(desired_fill_value)

        variable_attrs = variable._attrs
        # Offset and scale factors are not supported by OCGIS. The data is unpacked when written to a new output file.
        # TODO: Consider supporting offset and scale factors for write operations.
        exclude = ['add_offset', 'scale_factor']
        for k, v in list(var.get('attrs', {}).items()):
            if k in exclude:
                continue
            if k not in variable_attrs:
                variable_attrs[k] = deepcopy(v)
        # The conform units to value should be the default units value. Units will be converted on variable load.
        conform_units_to = var.get('conform_units_to')
        if conform_units_to is not None:
            variable_attrs['units'] = conform_units_to


def get_coordinate_system_variable_name(driver_object, group_metadata):
    rd = driver_object.rd
    crs_name = None
    if rd._has_assigned_coordinate_system:
        if rd._crs is not None:
            crs_name = rd._crs.name
    else:
        crs = driver_object.get_crs(group_metadata)
        if crs is not None:
            crs_name = crs.name
    return crs_name


def get_value_from_request_dataset(variable):
    if variable.protected:
        raise PayloadProtectedError(variable.name)

    rd = variable._request_dataset
    with driver_scope(rd.driver) as source:
        if variable.group is not None:
            for vg in variable.group:
                if vg is None:
                    continue
                else:
                    source = source.groups[vg]
        desired_name = variable.source_name or rd.variable

        # Reference the variable in the source dataset.
        ncvar = source.variables[desired_name]

        # Allow multi-unit time values for temporal variables.
        if isinstance(variable, TemporalVariable) and isinstance(source, MFDataset) and rd.format_time:
            # MFTime may fail if time_bnds do not have a calendar attribute.
            # Use rd.dimension_map.set_bounds('time', None) to disable indexing on time_bnds.
            try:
                ncvar = MFTime(ncvar, units=variable.units, calendar=variable.calendar)
            except TypeError:
                # Older versions of netcdf4-python do not support the calendar argument.
                ncvar = MFTime(ncvar, units=variable.units)
        ret = get_variable_value(ncvar, variable.dimensions)
    return ret


def get_variables_to_write(vc):
    from ocgis.variable.geom import GeometryVariable
    ret = []
    for variable in vc.values():
        if isinstance(variable, GeometryVariable):
            continue
        else:
            ret.append(variable)
    return ret


def get_variable_value(variable, dimensions):
    if dimensions is not None and len(dimensions) > 0:
        to_format = [None] * len(dimensions)
        for idx in range(len(dimensions)):
            current_dimension = dimensions[idx]
            si_type = current_dimension._src_idx_type
            if si_type is None:
                if current_dimension.bounds_local is None:
                    to_insert = slice(0, len(current_dimension))
                else:
                    to_insert = slice(*current_dimension.bounds_local)
            elif si_type == SourceIndexType.FANCY:
                to_insert = current_dimension._src_idx
            elif si_type == SourceIndexType.BOUNDS:
                to_insert = slice(*current_dimension._src_idx)
            else:
                raise NotImplementedError(si_type)
            to_format[idx] = to_insert
        slc = get_formatted_slice(to_format, len(dimensions))
    else:
        slc = slice(None)
    try:
        ret = variable.__getitem__(slc)
    except IndexError:
        # TODO: Hack! Slicing the top-level MFTime variable does not work with multifiles.
        ret = super(MFTime, variable).__getitem__(slc)
    return ret


def create_dimension_or_pass(dim, dataset, write_mode=MPIWriteMode.NORMAL):
    if dim.name not in dataset.dimensions:
        if dim.is_unlimited:
            size = None
        elif write_mode in (MPIWriteMode.TEMPLATE, MPIWriteMode.ASYNCHRONOUS):
            lower, upper = dim.bounds_global
            size = upper - lower
        else:
            size = dim.size
        dataset.createDimension(dim.name, size)


def get_crs_variable(metadata, rotated_pole_priority, to_search=None):
    found = []
    variables = metadata['variables']

    for vname, var in list(variables.items()):
        if to_search is not None:
            if vname not in to_search:
                continue
        for potential in itersubclasses(CFCoordinateReferenceSystem):
            try:
                crs = potential.load_from_metadata(vname, metadata, strict=False)
                found.append(crs)
                break
            except ProjectionDoesNotMatch:
                continue

    fset = set([f.name for f in found])
    if len(fset) > 1:
        msg = 'Multiple coordinate systems found. There should be only one.'
        raise ValueError(msg)
    elif len(fset) == 0:
        crs = None
    else:
        crs = found[0]

    # Allow spherical coordinate systems to overload rotated pole if they are present in the metadata.
    if not rotated_pole_priority and isinstance(crs, CFRotatedPole):
        try:
            crs = CFSpherical.load_from_metadata(None, metadata, depth=2)
        except ProjectionDoesNotMatch:
            # No spherical coordinate system data available. Fall back on previous crs.
            pass

    # If there is no CRS, check for OCGIS coordinate systems.
    if crs is None:
        try:
            crs = AbstractProj4CRS.load_from_metadata(metadata)
        except ProjectionDoesNotMatch:
            pass

    return crs


def create_dimension_map_entry(src, variables, strict=False, attr_name='axis'):
    """
    Create a dimension map entry dictionary by searching variable metadata using attribute constraints.

    :param src: The source information to use for constructing the entry. If ``src`` is a dictionary, it must have two
     entries. The key ``'value'`` corresponds to the string attribute value. The key ``'axis'`` is the representative
     axis to assign the source value (for example ``'X'`` or ``'Y'``).
    :type src: str | dict
    :param dict variables: The metadata entries for the group's variables.
    :param bool strict: If ``False``, do not use a strict interpretation of metadata. Allow some standard approaches for
     handling metadata exceptions.
    :param str attr_name: Name of the attribute to use for checking the attribute values form ``src``.
    :return: dict
    """
    if isinstance(src, dict):
        axis = src['axis']
        attr_value = src['value']
    else:
        axis = src
        attr_value = src

    axis_vars = []
    for variable in list(variables.values()):
        vattrs = variable.get('attrs', {})
        if vattrs.get(attr_name) == attr_value:
            if len(variable['dimensions']) == 0:
                pass
            else:
                axis_vars.append(variable['name'])

    # Try to find by default names.
    if not strict and len(axis_vars) == 0:
        possible_names = CFName.get_axis_mapping().get(axis, [])
        for pn in possible_names:
            if pn in list(variables.keys()):
                axis_vars.append(variables[pn]['name'])

    if len(axis_vars) == 1:
        var_name = axis_vars[0]
        dims = list(variables[var_name]['dimensions'])

        if not strict:
            # Use default index positions for X/Y dimensions.
            if axis in ('X', 'Y') and len(dims) > 1:
                if axis == 'Y':
                    dims = [dims[0]]
                elif axis == 'X':
                    dims = [dims[1]]

        ret = {'variable': var_name, DimensionMapKey.DIMENSION: dims}
    elif len(axis_vars) > 1:
        msg = 'Multiple axis (axis="{}") possibilities found using variable(s) "{}". Use a dimension map to specify ' \
              'the appropriate coordinate dimensions.'
        ocgis_lh(msg.format(axis, axis_vars), level=logging.WARN, logger='ocgis.driver.nc', force=True)
        ret = None
    else:
        ret = None
    return ret


def get_netcdf_attributes(target):
    attributes = OrderedDict()
    for attname in target.ncattrs():
        try:
            attributes[attname] = target.getncattr(attname)
        except AttributeError:
            # TODO: Report bug to netCDF4-python.
            # May fail for multi-file datasets. Try to access the underlying dictionary.
            attributes[attname] = target.__dict__[attname]
    return attributes


def remove_netcdf_attribute(filename, variable_name, attr_name):
    with driver_scope(DriverNetcdf, opened_or_path=filename, mode='a') as ds:
        var = ds[variable_name]
        var.delncattr(attr_name)


def update_group_metadata(rootgrp, fill):
    global_attributes = get_netcdf_attributes(rootgrp)
    fill.update({'global_attributes': global_attributes})

    # get file format
    fill.update({'file_format': rootgrp.file_format})

    # get variables
    variables = OrderedDict()
    for key, value in rootgrp.variables.items():
        subvar = OrderedDict()
        for attr in value.ncattrs():
            subvar.update({attr: getattr(value, attr)})

        # Remove scale factors and offsets from the metadata.
        if 'scale_factor' in subvar:
            dtype_packed = value[0].dtype
            fill_value_packed = np.ma.array([], dtype=dtype_packed).fill_value
        else:
            dtype_packed = None
            fill_value_packed = None

        # Attempt to find the fill value.
        try:
            fill_value = value.__dict__['_FillValue']
        except KeyError:
            try:
                fill_value = value.fill_value
            except AttributeError:
                try:
                    fill_value = value.missing_value
                except AttributeError:
                    fill_value = 'auto'

        variables.update({key: {'dimensions': value.dimensions,
                                'attrs': subvar,
                                'dtype': value.dtype,
                                'name': value._name,
                                'fill_value': fill_value,
                                'dtype_packed': dtype_packed,
                                'fill_value_packed': fill_value_packed}})
    fill.update({'variables': variables})

    # get dimensions
    dimensions = OrderedDict()
    for key, value in rootgrp.dimensions.items():
        subdim = {key: {'name': key, 'size': len(value), 'isunlimited': value.isunlimited()}}
        dimensions.update(subdim)
    fill.update({'dimensions': dimensions})
