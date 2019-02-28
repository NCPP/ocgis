import abc
import json
from abc import ABCMeta
from contextlib import contextmanager
from copy import deepcopy
from warnings import warn

import numpy as np
import six
from ocgis import constants, GridUnstruct
from ocgis import vm
from ocgis.base import AbstractOcgisObject, raise_if_empty
from ocgis.base import get_variable_names
from ocgis.collection.field import Field
from ocgis.constants import MPIWriteMode, TagName, KeywordArgument, OcgisConvention, VariableName, DecompositionType
from ocgis.driver.dimension_map import DimensionMap
from ocgis.exc import DefinitionValidationError, NoDataVariablesFound, DimensionMapError, VariableMissingMetadataError, \
    GridDeficientError
from ocgis.util.helpers import get_group
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.variable.base import SourcedVariable
from ocgis.variable.dimension import Dimension
from ocgis.vmachine.mpi import OcgDist


# TODO: Make the driver accept no arguments at initialization. The request dataset should be passed around as a parameter.

@six.add_metaclass(abc.ABCMeta)
class AbstractDriver(AbstractOcgisObject):
    """
    Base class for all drivers.
    
    :param rd: The input request dataset object.
    :type rd: :class:`~ocgis.RequestDataset`
    """

    common_extension = None  # The standard file extension commonly associated with the canonical file format.
    default_axes_positions = (0, 1)  # Standard axes index for Y and X respectively.
    _default_crs = None
    _esmf_fileformat = None  # The associated ESMF file type. This may be None.
    _esmf_grid_class = constants.ESMFGridClass.GRID  # The ESMF grid class type.
    _priority = False

    def __init__(self, rd):
        self.rd = rd
        self._metadata_raw = None
        self._dimension_map_raw = None
        self._dist = None

    def __eq__(self, other):
        return self.key == other.key

    def __str__(self):
        return '"{0}"'.format(self.key)

    @property
    def crs(self):
        """Blocks setting the CRS attribute on a driver."""

        raise NotImplementedError

    @property
    def data_model(self):
        """
        Return the data model definition for the driver if it has one. NetCDF has NETCDF3_CLASSIC, etc.

        :rtype: str | ``None``
        """
        return None

    @property
    def dimension_map_raw(self):
        if self._dimension_map_raw is None:
            self._dimension_map_raw = create_dimension_map_raw(self, self.metadata_raw)
        return self._dimension_map_raw

    @property
    def dist(self):
        if self._dist is None:
            self._dist = self.create_dist()
        return self._dist

    @abc.abstractproperty
    def extensions(self):
        """
        :returns: A sequence of regular expressions used to match appropriate URIs.
        :rtype: (str,)

        >>> ('.*\.shp',)
        """

    @abc.abstractproperty
    def key(self):
        """:rtype: str"""

    @property
    def metadata_raw(self):
        if self._metadata_raw is None:
            # If there is no URI, we must be working with metadata passed to the request dataset.
            if self.rd._uri is None and self.rd.opened is None:
                res = self.rd.metadata
            else:
                res = self.get_metadata()
            self._metadata_raw = res
        return self._metadata_raw

    @property
    def metadata_source(self):
        return self.rd.metadata

    @abc.abstractproperty
    def output_formats(self):
        """
        :returns: A list of acceptable output formats for the driver. If this is `'all'`, then the driver's data may be
         converted to all output formats.
        :rtype: list[str, ...]
        """

    @staticmethod
    def array_resolution(value, axis):
        """
        Optionally overloaded by subclasses to calculate the "resolution" of an array. This makes the most sense in the
        context of coordinate variables. This method should return a float resolution.

        :param value: The value array target.
        :type value: :class:`numpy.ndarray`
        :param int axis: The target axis for the resolution calculation.
        :rtype: float
        """
        raise NotImplementedError

    @classmethod
    def close(cls, obj, rd=None):
        # If the request dataset has an opened file object, do not close the file as we expect the client to handle
        # closing/finalization options.
        if rd is not None and rd.opened is not None:
            pass
        else:
            cls._close_(obj)

    def get_crs(self, group_metadata):
        """:rtype: ~ocgis.interface.base.crs.CoordinateReferenceSystem"""

        crs = self._get_crs_main_(group_metadata)
        if crs is None:
            ret = self._default_crs
        else:
            ret = crs
        return ret

    def create_dimension_map(self, group_metadata, **kwargs):
        """
        Create a dimension map for a group from its metadata.

        :param dict group_metadata: Group metadata to use when creating the dimension map.
        :rtype: :class:`~ocgis.DimensionMap`
        """
        ret = DimensionMap()
        ret.set_driver(self.key)
        return ret

    @staticmethod
    def create_dimensions(group_metadata):
        """
        Create dimension objects. The key may differ from the dimension name. In which case, we can assume the dimension
        is being renamed.

        :param dict group_metadata: Metadata dictionary for the target group.
        :rtype: list
        """
        gmd = group_metadata.get('dimensions', {})
        dims = {}
        for k, v in gmd.items():
            size = v['size']
            if v.get('isunlimited', False):
                size_current = size
                size = None
            else:
                size_current = None
            dims[k] = Dimension(v.get('name', k), size=size, size_current=size_current, src_idx='auto', source_name=k)
        return dims

    def create_dist(self, metadata=None):
        """
        Create a distribution from global metadata. In general, this should not be overloaded by subclasses.

        :param dict metadata: Global metadata to use for creating a distribution.

        :rtype: :class:`ocgis.OcgDist`
        """
        ompi = OcgDist(size=vm.size, ranks=vm.ranks)

        # Convert metadata into a grouping consistent with the MPI dimensions.
        if metadata is None:
            metadata = self.metadata_source
        metadata = {None: metadata}
        for group_index in iter_all_group_keys(metadata):
            group_meta = get_group(metadata, group_index)

            # Add the dimensions to the distribution object.
            dimensions = self.create_dimensions(group_meta)

            # Only build a distribution if the group has more than one dimension.
            if len(dimensions) == 0:
                _ = ompi.get_group(group=group_index)
            else:
                for dimension_name, dimension_meta in list(group_meta['dimensions'].items()):
                    target_dimension = dimensions[dimension_name]
                    target_dimension.dist = group_meta['dimensions'][dimension_name].get('dist', False)
                    ompi.add_dimension(target_dimension, group=group_index)
                try:
                    dimension_map = self.rd.dimension_map.get_group(group_index)
                except DimensionMapError:
                    # Likely a user-provided dimension map.
                    continue

                # dimension_map = get_group(self.rd.dimension_map, group_index, has_root=False)
                distributed_dimension_name = self.get_distributed_dimension_name(dimension_map,
                                                                                 group_meta['dimensions'],
                                                                                 decomp_type=self.rd.decomp_type)
                # Allow no distributed dimensions to be returned.
                if distributed_dimension_name is not None:
                    for target_rank in range(ompi.size):
                        distributed_dimension = ompi.get_dimension(distributed_dimension_name, group=group_index,
                                                                   rank=target_rank)
                        distributed_dimension.dist = True

        ompi.update_dimension_bounds()
        return ompi

    def create_field(self, *args, **kwargs):
        """
        Create a field object. In general, this should not be overloaded by subclasses.

        :keyword bool format_time: ``(=True)`` If ``False``, do not convert numeric times to Python date objects.
        :keyword str grid_abstraction: ``(='auto')`` If provided, use this grid abstraction.
        :keyword raw_field: ``(=None)`` If provided, modify this field instead.
        :type raw_field: None | :class:`~ocgis.Field`
        :param kwargs: Additional keyword arguments to :meth:`~ocgis.driver.base.AbstractDriver.create_raw_field`.
        :return: :class:`ocgis.Field`
        """
        kwargs = kwargs.copy()
        raw_field = kwargs.pop('raw_field', None)
        format_time = kwargs.pop(KeywordArgument.FORMAT_TIME, True)
        grid_abstraction = kwargs.pop(KeywordArgument.GRID_ABSTRACTION, self.rd.grid_abstraction)
        grid_is_isomorphic = kwargs.pop('grid_is_isomorphic', self.rd.grid_is_isomorphic)

        if raw_field is None:
            # Get the raw variable collection from source.
            new_kwargs = kwargs.copy()
            new_kwargs['source_name'] = None
            raw_field = self.create_raw_field(*args, **new_kwargs)

        # Get the appropriate metadata for the collection.
        group_metadata = self.get_group_metadata(raw_field.group, self.metadata_source)
        # Always pull the dimension map from the request dataset. This allows it to be overloaded.
        dimension_map = self.get_group_metadata(raw_field.group, self.rd.dimension_map)

        # Modify the coordinate system variable. If it is overloaded on the request dataset, then the variable
        # collection needs to be updated to hold the variable and any alternative coordinate systems needs to be
        # removed.
        to_remove = None
        to_add = None
        crs = self.get_crs(group_metadata)
        if self.rd._has_assigned_coordinate_system:
            to_add = self.rd._crs
            if crs is not None:
                to_remove = crs.name
        else:
            if self.rd._crs is not None and self.rd._crs != 'auto':
                to_add = self.rd._crs
                if crs is not None:
                    to_remove = crs.name
            elif crs is not None:
                to_add = crs
        if to_remove is not None:
            raw_field.pop(to_remove, None)
        if to_add is not None:
            raw_field.add_variable(to_add, force=True)
        # Overload the dimension map with the CRS.
        if to_add is not None:
            # dimension_map[DimensionMapKey.CRS][DimensionMapKey.VARIABLE] = to_add.name
            dimension_map.set_crs(to_add.name)

        # Remove the mask variable if present in the raw dimension map and the source dimension map is set to None.
        if self.rd.dimension_map.get_spatial_mask() is None and self.dimension_map_raw.get_spatial_mask() is not None:
            raw_field.pop(self.dimension_map_raw.get_spatial_mask())

        # Convert the raw variable collection to a field.
        # TODO: Identify a way to remove this code block; field should be appropriately initialized; format_time and grid_abstraction are part of a dimension map.
        kwargs[KeywordArgument.DIMENSION_MAP] = dimension_map
        kwargs[KeywordArgument.FORMAT_TIME] = format_time
        if grid_abstraction != 'auto':
            kwargs[KeywordArgument.GRID_ABSTRACTION] = grid_abstraction
        if grid_is_isomorphic != 'auto':
            kwargs['grid_is_isomorphic'] = grid_is_isomorphic
        field = Field.from_variable_collection(raw_field, *args, **kwargs)

        # If this is a source grid for regridding, ensure the flag is updated.
        field.regrid_source = self.rd.regrid_source
        # Update the assigned coordinate system flag.
        field._has_assigned_coordinate_system = self.rd._has_assigned_coordinate_system

        # Apply any requested subsets.
        if self.rd.time_range is not None:
            field = field.time.get_between(*self.rd.time_range).parent
        if self.rd.time_region is not None:
            field = field.time.get_time_region(self.rd.time_region).parent
        if self.rd.time_subset_func is not None:
            field = field.time.get_subset_by_function(self.rd.time_subset_func).parent
        if self.rd.level_range is not None:
            field = field.level.get_between(*self.rd.level_range).parent

        # These variables have all the dimensions needed for a data classification. Use overloaded values from the
        # request dataset if they are provided.
        try:
            data_variable_names = list(get_variable_names(self.rd.rename_variable))
        except NoDataVariablesFound:
            # It is okay to have no data variables in a field.
            data_variable_names = []
            pass
        for dvn in data_variable_names:
            field.append_to_tags(TagName.DATA_VARIABLES, dvn, create=True)

        # Load child fields.
        for child in list(field.children.values()):
            kwargs['raw_field'] = child
            field.children[child.name] = self.create_field(*args, **kwargs)

        return field

    def create_raw_field(self, group_metadata=None, group_name=None, name=None, source_name=constants.UNINITIALIZED,
                         parent=None, uid=None):
        """
        Create a raw field object. This field object should interpret metadata explicitly (i.e. no dimension map). In
        general this method should not be overloaded by subclasses.

        :param dict group_metadata: Metadata dictionary for the current group.
        :param str group_name: The name of the current group being processed.
        :param str name: See :class:`~ocgis.base.AbstractNamedObject`
        :param str source_name: See :class:`~ocgis.base.AbstractNamedObject`
        :param parent: :class:`~ocgis.variable.base.AbstractContainer`
        :param int uid: See :class:`~ocgis.variable.base.AbstractContainer`
        :return: :class:`~ocgis.Field`
        """
        if group_metadata is None:
            group_metadata = self.rd.metadata

        field = Field(name=name, source_name=source_name, uid=uid, attrs=group_metadata.get('global_attributes'))
        for var in self.create_variables(group_metadata, parent=field).values():
            field.add_variable(var, force=True)
        if parent is not None:
            parent.add_child(field)
        for k, v in group_metadata.get('groups', {}).items():
            _ = self.create_raw_field(v, name=k, parent=field, group_name=k)
        return field

    def create_variables(self, group_metadata, parent=None):
        """
        Create a dictionary of variable objects. The keys are the variable names.

        :param dict group_metadata: Metadata for the group to create variables from.
        :param parent: See :class:`~ocgis.variable.base.AbstractContainer`
        :return: dict
        """
        vmeta = group_metadata['variables']
        vars = {}
        for k, v in vmeta.items():
            # Dimensions are set in this sourced variable initialization call. This is to allow the group hierarchy to
            # be determined by the parents' relationships.
            nvar = SourcedVariable(name=v.get('name', k), dtype=v.get('dtype'), request_dataset=self.rd, source_name=k,
                                   parent=parent)
            vars[k] = nvar
        return vars

    @staticmethod
    def get_data_variable_names(group_metadata, group_dimension_map):
        """
        Return a tuple of data variable names using the metadata and dimension map.
        """
        return tuple(group_metadata['variables'].keys())

    def get_distributed_dimension_name(self, dimension_map, dimensions_metadata, decomp_type=DecompositionType.OCGIS):
        """Return the preferred distributed dimension name."""
        return None

    def get_dump_report(self, indent=0, group_metadata=None, first=True, global_attributes_name='global'):
        lines = []
        if first:
            lines.append('OCGIS Driver Key: ' + self.key + ' {')
            group_metadata = group_metadata or self.metadata_source
        else:
            indent += 2
        lines += get_dump_report_for_group(group_metadata, global_attributes_name=global_attributes_name, indent=indent)
        for group_name, group_metadata in list(group_metadata.get('groups', {}).items()):
            lines.append('')
            lines.append(' ' * indent + 'group: ' + group_name + ' {')
            dump_lines = self.get_dump_report(group_metadata=group_metadata, first=False, indent=indent,
                                              global_attributes_name=group_name)
            lines += dump_lines
            lines.append(' ' * indent + '  }' + ' // group: {}'.format(group_name))
        if first:
            lines.append('}')
        return lines

    @classmethod
    def get_esmf_fileformat(cls):
        """
        Get the ESMF file format associated with the driver. The string should be an accessible attribute on :attr:`ESMF.constants.FileFormat`.

        :rtype: str
        """
        from ocgis.regrid.base import ESMF
        return getattr(ESMF.constants.FileFormat, cls._esmf_fileformat)

    @classmethod
    def get_esmf_grid_class(cls):
        """
        Get the ESMF grid class.

        :rtype: :class:`ESMF.Grid` | :class:`ESMF.Mesh`
        """
        return constants.ESMFGridClass.get_esmf_class(cls._esmf_grid_class)

    @staticmethod
    def get_grid(field):
        """
        Construct a grid object and return it. If a grid object may not be constructed, return ``None``.

        :param field: The field object to use for constructing the grid.
        :type field: :class:`~ocgis.Field`
        :rtype: :class:`ocgis.spatial.grid.AbstractGrid` | ``None``
        """
        raise NotImplementedError

    @staticmethod
    def get_group_metadata(group_index, metadata, has_root=False):
        return get_group(metadata, group_index, has_root=has_root)

    def get_metadata(self):
        """
        :rtype: dict
        """

        metadata_subclass = self._get_metadata_main_()

        # Use the predicate (filter) if present on the request dataset.
        # TODO: Should handle groups?
        pred = self.rd.predicate
        if pred is not None:
            to_pop = []
            for var_name in metadata_subclass['variables'].keys():
                if not pred(var_name):
                    to_pop.append(var_name)
            for var_name in to_pop:
                metadata_subclass['variables'].pop(var_name)

        return metadata_subclass

    def get_source_metadata_as_json(self):

        def _jsonformat_(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    _jsonformat_(v)
                else:
                    try:
                        # Numpy arrays need to be convered to lists.
                        v = v.tolist()
                    except AttributeError:
                        v = str(v)
                d[k] = v

        meta = deepcopy(self.metadata_source)
        _jsonformat_(meta)
        return json.dumps(meta)

    @staticmethod
    def get_or_create_spatial_mask(*args, **kwargs):
        """
        Get or create the spatial mask variable and return its mask value as a boolean array.

        :param tuple args: See table

        ===== ======================================================= ===================================
        Index Type                                                    Description
        ===== ======================================================= ===================================
        0     :class:`ocgis.spatial.base.AbstractXYZSpatialContainer` Target XYZ spatial container
        1:    <varying>                                               See :meth:`ocgis.Variable.get_mask`
        ===== ======================================================= ===================================

        :param dict kwargs: See keyword arguments to :meth:`~ocgis.Variable.get_mask`. If ``create`` and ``check_value``
         are ``True`` and no mask variable exists, then coordinate variable masks are combined using a logical OR
         operation.
        :rtype: :class:`numpy.ndarray`
        """
        from ocgis.spatial.base import create_spatial_mask_variable

        args = list(args)
        sobj = args[0]  # Spatial object containing coordinate variables
        args = args[1:]

        create = kwargs.get(KeywordArgument.CREATE, False)
        check_value = kwargs.get(KeywordArgument.CHECK_VALUE, False)

        # Check for existing mask variable
        mask_variable = sobj.mask_variable

        ret = None
        if mask_variable is None:
            if create:
                if check_value:
                    # Combine coordinate variable masks using a logical OR operation
                    for ii, cvar in enumerate(sobj.coordinate_variables):
                        currmask = cvar.get_mask(create=True, check_value=True)
                        if ii == 0:
                            mask_value = currmask
                        else:
                            mask_value = np.logical_or(currmask, mask_value)
                else:
                    # Let the mask creation function handling the creation of mask values
                    mask_value = None

                # Create the spatial mask variable and set it
                mask_variable = create_spatial_mask_variable(VariableName.SPATIAL_MASK, mask_value, sobj.dimensions)
                sobj.set_mask(mask_variable)

        if mask_variable is not None:
            # Validate the mask variable checking attributes, data types, etc.
            sobj.driver.validate_spatial_mask(mask_variable)
            # Get the actual boolean array from the mask variable
            ret = mask_variable.get_mask(*args, **kwargs)

        return ret

    def get_variable_collection(self, **kwargs):
        """Here for backwards compatibility."""
        return self.create_raw_field(**kwargs)

    def get_variable_metadata(self, variable_object):
        variable_metadata = get_variable_metadata_from_request_dataset(self, variable_object)
        return variable_metadata

    @classmethod
    def get_variable_for_writing(cls, variable):
        """
        Allows variables to overload which member to use for writing. For example, temporal variables always want
        numeric times.
        """
        from ocgis.variable.temporal import TemporalVariable
        if isinstance(variable, TemporalVariable):
            ret = cls.get_variable_for_writing_temporal(variable)
        else:
            ret = variable
        return ret

    @classmethod
    def get_variable_for_writing_temporal(cls, temporal_variable):
        # Only return datetime objects directly if time is being formatted. Otherwise return the standard value, these
        # could be datetime objects if passed during initialization.
        if temporal_variable.format_time:
            ret = temporal_variable.value_datetime
        else:
            ret = temporal_variable.value
        return ret

    @abc.abstractmethod
    def get_variable_value(self, variable):
        """Get value for the variable."""

    @classmethod
    def get_variable_write_dtype(cls, variable):
        return cls.get_variable_for_writing(variable).dtype

    @classmethod
    def get_variable_write_fill_value(cls, variable):
        ret = cls.get_variable_for_writing(variable).fill_value
        return ret

    @classmethod
    def get_variable_write_value(cls, variable):
        from ocgis.variable.temporal import TemporalVariable
        if variable.has_allocated_value:
            if isinstance(variable, TemporalVariable):
                ret = cls.get_variable_for_writing(variable)
            else:
                if variable.has_mask:
                    ret = cls.get_variable_for_writing(variable).get_masked_value()
                else:
                    ret = cls.get_variable_for_writing(variable).get_value()
        else:
            ret = None
        return ret

    def init_variable_from_source(self, variable):
        variable_metadata = self.get_variable_metadata(variable)

        # Create the dimensions if they are not present.
        if variable._dimensions is None:
            dist = self.dist
            desired_dimensions = variable_metadata.get('dimensions')
            # Variable may not have any associated dimensions (attribute-only variable).
            if desired_dimensions is not None:
                new_dimensions = []
                for d in desired_dimensions:
                    try:
                        to_append = dist.get_dimension(d, group=variable.group, rank=vm.rank)
                    except KeyError:
                        # Rank may not be mapped due to scoped VM.
                        if vm.is_live:
                            raise
                        else:
                            variable.convert_to_empty()
                            break
                    else:
                        new_dimensions.append(to_append)
                super(SourcedVariable, variable).set_dimensions(new_dimensions)

        # Call the subclass variable initialization routine.
        self._init_variable_from_source_main_(variable, variable_metadata)
        # The variable is now allocated.
        variable._allocated = True

    def init_variable_value(self, variable):
        """Set the variable value from source data conforming units in the process."""
        from ocgis.variable.temporal import TemporalVariable

        value = self.get_variable_value(variable)
        variable.set_value(value, update_mask=True)
        # Conform the units if requested. Need to check if this variable is inside a group to find the appropriate
        # metadata.
        meta = get_variable_metadata_from_request_dataset(self, variable)
        conform_units_to = meta.get('conform_units_to')
        if conform_units_to is not None:
            # The initialized units for the variable are overloaded by the destination / conform to units.
            if isinstance(variable, TemporalVariable):
                from_units = TemporalVariable(units=meta['attrs']['units'],
                                              calendar=meta['attrs'].get('calendar'))
                from_units = from_units.cfunits
            else:
                from_units = meta['attrs']['units']
            variable.cfunits_conform(conform_units_to, from_units=from_units)

    @staticmethod
    def inquire_opened_state(opened_or_path):
        """
        Return ``True`` if the input is an opened file object.

        :param opened_or_path: Output file path or an open file object.
        :rtype: bool
        """
        poss = tuple(list(six.string_types) + [tuple, list])
        if isinstance(opened_or_path, poss):
            ret = False
        else:
            ret = True
        return ret

    def inspect(self):
        """
        Inspect the request dataset printing information to stdout.
        """

        for line in self.get_dump_report():
            print(line)

    @staticmethod
    def iterator_formatter(name, value, mask):
        return [(name, value)]

    @staticmethod
    def iterator_formatter_time_bounds(name, value, mask):
        return AbstractDriver.iterator_formatter(name, value, mask)

    @staticmethod
    def iterator_formatter_time_value(name, value, mask):
        return AbstractDriver.iterator_formatter(name, value, mask)

    @classmethod
    def open(cls, uri=None, mode='r', rd=None, **kwargs):
        if uri is None and rd is None:
            raise ValueError('A URI or request dataset is required.')

        if rd is not None and rd.opened is not None:
            ret = rd.opened
        else:
            if rd is not None and uri is None:
                uri = rd.uri
            ret = cls._open_(uri, mode=mode, **kwargs)
        return ret

    @staticmethod
    def set_spatial_mask(sobj, value, cascade=False):
        """
        Set the spatial mask on an XYZ spatial container.

        :param sobj: Target XYZ spatial container
        :type sobj: :class:`ocgis.spatial.base.AbstractXYZSpatialContainer`
        :param value: The spatial mask value. This may be a variable or a boolean array. If it is a boolean array, a
         spatial mask variable will be created and this array set as its mask.
        :type value: :class:`ocgis.Variable` | :class:`numpy.ndarray`
        :param bool cascade: If ``True``, cascade the mask across shared dimensions on the grid.
        """
        from ocgis.spatial.grid import grid_set_mask_cascade
        from ocgis.spatial.base import create_spatial_mask_variable
        from ocgis.variable.base import Variable

        if value is None:
            # Remove the mask variable from the parent object and update the dimension map.
            if sobj.has_mask:
                sobj.parent.remove_variable(sobj.mask_variable)
                sobj.dimension_map.set_spatial_mask(None)
        else:
            if isinstance(value, Variable):
                # Set the mask variable from the incoming value.
                sobj.parent.add_variable(value, force=True)
                mask_variable = value
            else:
                # Convert the incoming boolean array into a mask variable.
                mask_variable = sobj.mask_variable
                if mask_variable is None:
                    dimensions = sobj.dimensions
                    mask_variable = create_spatial_mask_variable(VariableName.SPATIAL_MASK, value, dimensions)
                    sobj.parent.add_variable(mask_variable)
                else:
                    mask_variable.set_mask(value)
            sobj.dimension_map.set_spatial_mask(mask_variable)

            if cascade:
                grid_set_mask_cascade(sobj)

    def validate_field(self, field):
        pass

    @classmethod
    def validate_ops(cls, ops):
        """
        :param ops: An operation object to validate.
        :type ops: :class:`~ocgis.OcgOperations`
        :raises: DefinitionValidationError
        """

        if cls.output_formats != 'all':
            if ops.output_format not in cls.output_formats:
                msg = 'Output format not supported for driver "{0}". Supported output formats are: {1}'.format(cls.key,
                                                                                                               cls.output_formats)
                ocgis_lh(logger='driver', exc=DefinitionValidationError('output_format', msg))

    @staticmethod
    def validate_spatial_mask(mask_variable):
        """
        Validate the spatial mask variable.

        :param mask_variable: Target spatial mask variable
        :type mask_variable: :class:`ocgis.Variable`
        :raises: ValueError
        """
        if mask_variable.attrs.get('ocgis_role') != 'spatial_mask':
            msg = 'Mask variable "{}" must have an "ocgis_role" attribute with a value of "spatial_mask".'.format(
                mask_variable.name)
            raise ValueError(msg)

    def write_variable(self, *args, **kwargs):
        """Write a variable. Not applicable for tabular formats."""
        raise NotImplementedError

    @classmethod
    def write_field(cls, field, opened_or_path, **kwargs):
        raise_if_empty(field)
        vc_to_write = cls._get_field_write_target_(field)
        cls.write_variable_collection(vc_to_write, opened_or_path, **kwargs)

    @classmethod
    def write_variable_collection(cls, vc, opened_or_path, **kwargs):
        raise_if_empty(vc)

        if 'ranks_to_write' in kwargs:
            raise TypeError("write_variable_collection() got an unexepcted keyword argument 'ranks_to_write'")

        write_mode = kwargs.pop(KeywordArgument.WRITE_MODE, None)

        if vm.size > 1:
            if cls.inquire_opened_state(opened_or_path):
                raise ValueError('Only paths allowed for parallel writes.')

        if write_mode is None:
            write_modes = cls._get_write_modes_(vm, **kwargs)
        else:
            write_modes = [write_mode]

        # vm.rank_print('tkd: write_modes', write_modes)

        # Global string lengths are needed by the write. Set those while we still have global access.
        for var in vc.values():
            if var._string_max_length_global is None:
                var.set_string_max_length_global()

        for write_mode in write_modes:
            cls._write_variable_collection_main_(vc, opened_or_path, write_mode, **kwargs)

    @staticmethod
    def _close_(obj):
        """
        Close and finalize the open file object.
        """
        obj.close()

    @staticmethod
    def _gc_nchunks_dst_(grid_chunker):
        """
        Calculate the default chunking decomposition for a destination grid in the grid chunker. The decomposition should
        be a tuple of integers.

        >>> (10, 5)
        >>> (12,)

        :param grid_chunker: The grid chunker object.
        :type grid_chunker: :class:`~ocgis.spatial.grid_chunker.GridChunker`
        :rtype: tuple
        """
        raise NotImplementedError

    @classmethod
    def _get_write_modes_(cls, the_vm, **kwargs):
        if the_vm.size > 1:
            write_modes = [MPIWriteMode.TEMPLATE, MPIWriteMode.FILL]
        else:
            write_modes = [MPIWriteMode.NORMAL]
        return write_modes

    def _get_crs_main_(self, group_metadata):
        """Return the coordinate system variable or None if not found."""
        return None

    @abc.abstractmethod
    def _get_metadata_main_(self):
        """
        Return the base metadata object. The superclass will finalize the metadata doing things like adding dimension
        maps for each group.

        :rtype: dict
        """

    @classmethod
    def _get_field_write_target_(cls, field):
        return field

    @abc.abstractmethod
    def _init_variable_from_source_main_(self, variable_object, variable_metadata):
        """Initialize everything but dimensions on the target variable."""

    @staticmethod
    def _open_(uri, mode='r', **kwargs):
        """
        :rtype: object
        """

        return open(uri, mode=mode, **kwargs)

    @classmethod
    @abc.abstractmethod
    def _write_variable_collection_main_(cls, vc, opened_or_path, write_mode, **kwargs):
        """
        :param vc: :class:`~ocgis.new_interface.variable.VariableCollection`
        :param opened_or_path: Opened file object or path to the file object to open.
        :param comm: The MPI communicator.
        :param rank: The MPI rank.
        :param size: The MPI size.
        """


@six.add_metaclass(abc.ABCMeta)
class AbstractTabularDriver(AbstractDriver):
    """
    Base class for tabular drivers (no optimal single variable access).
    """

    def get_distributed_dimension_name(self, dimension_map, dimensions_metadata, decomp_type=DecompositionType.OCGIS):
        """Return the preferred distributed dimension name."""
        return list(dimensions_metadata.values())[0]['name']

    @staticmethod
    def iterator_formatter_time_bounds(name, value, mask):
        if mask:
            formatted_value = None
        else:
            formatted_value = str(value)
        ret = [(name, formatted_value)]
        return ret

    @staticmethod
    def iterator_formatter_time_value(name, value, mask):
        if mask:
            formatted_value = None
        else:
            formatted_value = str(value)
        ret = [(name, formatted_value)]
        try:
            ret.append(['YEAR', value.year])
            ret.append(['MONTH', value.month])
            ret.append(['DAY', value.day])
        except AttributeError:
            # Assume this is not a datetime object.
            ret.append(['YEAR', None])
            ret.append(['MONTH', None])
            ret.append(['DAY', None])
        return ret


@six.add_metaclass(ABCMeta)
class AbstractUnstructuredDriver(AbstractOcgisObject):
    default_axes_positions = (0, 0)  # Standard axes index for Y and X respectively.
    _esmf_grid_class = constants.ESMFGridClass.MESH

    @staticmethod
    def get_element_dimension(gc):
        """See :meth:`ocgis.spatial.geomc.AbstractGeometryCoordinates.element_dim`"""
        cindex = gc.cindex
        if cindex is None:
            ret = gc.archetype.dimensions[0]
        else:
            ret = cindex.dimensions[0]
        return ret

    @staticmethod
    def get_grid(field):
        try:
            ret = GridUnstruct(parent=field)
        except GridDeficientError:
            ret = None
        return ret

    @staticmethod
    def get_multi_break_value(cindex):
        """See :meth:`ocgis.spatial.geomc.AbstractGeometryCoordinates.multi_break_value`"""
        mbv_name = OcgisConvention.Name.MULTI_BREAK_VALUE
        return cindex.attrs.get(mbv_name)


@contextmanager
def driver_scope(ocgis_driver, opened_or_path=None, mode='r', **kwargs):
    kwargs = kwargs.copy()

    if opened_or_path is None:
        try:
            # Attempt to get the request dataset from the driver. If not there, assume we are working with the driver
            # class and not an instance created with a request dataset.
            rd = ocgis_driver.rd
        except AttributeError:
            rd = None
        if rd is None:
            raise ValueError('Without a driver instance and no open object or file path, nothing can be scoped.')
        else:
            if rd.opened is not None:
                opened_or_path = rd.opened
            else:
                opened_or_path = rd.uri
    else:
        rd = None

    if ocgis_driver.inquire_opened_state(opened_or_path):
        should_close = False
    else:
        should_close = True
        if rd is not None and rd.driver_kwargs is not None:
            kwargs.update(rd.driver_kwargs)
        opened_or_path = ocgis_driver.open(uri=opened_or_path, mode=mode, rd=rd, **kwargs)

    try:
        yield opened_or_path
    finally:
        if should_close:
            ocgis_driver.close(opened_or_path)


def find_variable_by_attribute(variables_metadata, attribute_name, attribute_value):
    ret = []
    for variable_name, variable_metadata in list(variables_metadata.items()):
        for k, v in list(variable_metadata['attrs'].items()):
            if k == attribute_name and v == attribute_value:
                ret.append(variable_name)
    return ret


def format_attribute_for_dump_report(attr_value):
    if isinstance(attr_value, six.string_types):
        ret = '"{}"'.format(attr_value)
    else:
        ret = attr_value
    return ret


def create_dimension_map_raw(driver, group_metadata):
    ret = DimensionMap.from_metadata(driver, group_metadata)
    return ret


def get_dump_report_for_group(group, global_attributes_name='global', indent=0):
    lines = []

    if len(group['dimensions']) > 0:
        lines.append('dimensions:')
        template = '    {0} = {1} ;{2}'
        for key, value in list(group['dimensions'].items()):
            if value.get('isunlimited', False):
                one = 'ISUNLIMITED'
                two = ' // {0} currently'.format(value['size'])
            else:
                one = value['size']
                two = ''
            lines.append(template.format(key, one, two))

    if len(group['variables']) > 0:
        lines.append('variables:')
        var_template = '    {0} {1}({2}) ;'
        attr_template = '      {0}:{1} = {2} ;'
        for key, value in list(group['variables'].items()):
            dims = [str(d) for d in value['dimensions']]
            dims = ', '.join(dims)
            lines.append(var_template.format(value['dtype'], key, dims))
            for key2, value2 in value.get('attrs', {}).items():
                lines.append(attr_template.format(key, key2, format_attribute_for_dump_report(value2)))

    global_attributes = group.get('global_attributes', {})
    if len(global_attributes) > 0:
        lines.append('')
        lines.append('// {} attributes:'.format(global_attributes_name))
        template = '    :{0} = {1} ;'
        for key, value in list(global_attributes.items()):
            try:
                lines.append(template.format(key, format_attribute_for_dump_report(value)))
            except UnicodeEncodeError:
                # for a unicode string, if "\u" is in the string and an inappropriate unicode character is used, then
                # template formatting will break.
                msg = 'Unable to encode attribute "{0}". Skipping printing of attribute value.'.format(key)
                warn(msg)

    if indent > 0:
        indent_string = ' ' * indent
        for idx, current in enumerate(lines):
            if len(current) > 0:
                lines[idx] = indent_string + current

    return lines


def get_variable_metadata_from_request_dataset(driver, variable):
    variables_metadata = get_group(driver.metadata_source, variable.group, has_root=False)['variables']
    try:
        ret = variables_metadata[variable._source_name]
    except KeyError:
        raise VariableMissingMetadataError(variable._source_name)
    return ret


def iter_all_group_keys(ddict, entry=None, has_root=True):
    if not has_root:
        ddict = {None: ddict}
    if entry is None:
        entry = [None]
    yield entry
    for keyseq in iter_group_keys(ddict, entry):
        for keyseq2 in iter_all_group_keys(ddict, keyseq):
            yield keyseq2


def iter_group_keys(ddict, keyseq):
    for key in get_group(ddict, keyseq).get('groups', {}):
        yld = deepcopy(keyseq)
        yld.append(key)
        yield yld
