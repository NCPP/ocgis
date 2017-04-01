import logging
import os
import re
from copy import deepcopy

import six

from ocgis import constants
from ocgis import env
from ocgis.constants import DriverKeys
from ocgis.driver.registry import get_driver_class, driver_registry
from ocgis.driver.request.base import AbstractRequestObject
from ocgis.exc import RequestValidationError, NoDataVariablesFound, VariableNotFoundError
from ocgis.util.helpers import get_iter, locate, validate_time_subset, get_tuple, get_by_sequence
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.util.units import get_units_object, get_are_units_equivalent
from ocgis.vm.mpi import MPI_COMM


# tdk: clean-up
class RequestDataset(AbstractRequestObject):
    # todo: document vector format
    """
    A :class:`ocgis.RequestDataset` contains all the information necessary to find and subset a variable (by time
    and/or level) contained in a local or OpenDAP-hosted CF dataset.

    >>> from ocgis import RequestDataset
    >>> uri = 'http://some.opendap.dataset'
    >>> # It is also okay to enter the path to a local file.
    >>> uri = '/path/to/local/file.nc'
    >>> variable = 'tasmax'
    >>> rd = RequestDataset(uri, variable)

    :param uri: The absolute path (URLs included) to the data's location.
    :type uri: str or sequence

    >>> uri = 'http://some.opendap.dataset'
    >>> uri = '/path/to/local/file.nc'
    # Multifile datasets are supported for local and remote targets.
    >>> uri = ['/path/to/local/file1.nc', '/path/to/local/file2.nc']

    .. warning:: There is no internal checking on the ordering of the files. If the datasets should be concatenated
     along the time dimension, it may be a good idea to run the sequence of URIs through a time sorting function
     :func:`~ocgis.util.helpers.get_sorted_uris_by_time_dimension`.

    :param variable: The target variable. If the argument value is ``None``, then a search on the target data object
     will be performed to find variables having a minimum set of dimensions (i.e. time and space). The value of this
     property will then be updated.
    :type variable: str or sequence or None

    >>> variable = 'tas'
    >>> variable = ['tas', 'tasmax']

    :param alias: An alternative name to identify the returned variable's data. If ``None``, this defaults to
     ``variable``. If variables having the same name occur in a request, this argument is required.
    :type alias: str or sequence

    >>> alias = 'tas_alias'
    >>> alias = ['tas_alias', 'tasmax_alias']

    :param time_range: Upper and lower bounds for time dimension subsetting. If ``None``, return all time points.
    :type time_range: [:class:`datetime.datetime`, :class:`datetime.datetime`]
    :param time_region: A dictionary with keys of ``'month'`` and/or ``'year'`` and values as sequences corresponding to
     target month and/or year values. Empty region selection for a key may be set to ``None``.
    :type time_region: dict

    >>> time_region = {'month':[6,7],'year':[2010,2011]}
    >>> time_region = {'year':[2010]}

    :param time_subset_func: See :meth:`ocgis.interface.base.dimension.temporal.TemporalDimension.get_subset_by_function`
     for usage instructions.
    :type time_subset_func: :class:`FunctionType`
    :param level_range: Upper and lower bounds for level dimension subsetting. If ``None``, return all levels.
    :type level_range: [int, int] or [float, float]
    :param crs: Overload the autodiscovered coordinate system.
    :type crs: :class:`ocgis.crs.CoordinateReferenceSystem`

    >>> from ocgis.variable.crs import CFWGS84
    >>> crs = CFWGS84()

    :param t_units: Overload the autodiscover `time units`_.
    :type t_units: str
    :param t_calendar: Overload the autodiscover `time calendar`_.
    :type t_calendar: str
    :param str t_conform_units_to: Conform the time dimension to the provided units. The calendar may not be changed.
     The option dependency ``cfunits-python`` is required.

    >>> t_conform_units_to = 'days since 1949-1-1'

    :param s_abstraction: Abstract the geometry data to either ``'point'`` or ``'polygon'``. If ``'polygon'`` is not
     possible due to missing bounds, ``'point'`` will be used instead.
    :type s_abstraction: str

    .. note:: The ``abstraction`` argument in :class:`ocgis.OcgOperations` will overload this.

    :param dimension_map: Maps dimensions to axes in the case of a projection/realization axis or an uncommon axis
     ordering. All axes must be in the dictionary.
    :type dimension_map: dict

    >>> dimension_map = {'T': 'time', 'X': 'longitude', 'Y': 'latitude', 'R': 'projection'}

    :param units: The units of the source variable. This will be read from metadata if this value is ``None``.
    :type units: str or :class:`cfunits.Units` or sequence
    :param conform_units_to: Destination units for conversion. If this parameter is set, then the :mod:`cfunits` module
     must be installed.
    :type conform_units_to: str or :class:`cfunits.Units` or sequence
    :param str driver: If ``None``, autodiscover the appropriate driver. Other accepted values are listed below.

    ============ ================= =============================================
    Value        File Extension(s) Description
    ============ ================= =============================================
    ``'netCDF'`` ``'nc'``          A netCDF file using a CF metadata convention.
    ``'vector'`` ``'shp'``         An ESRI Shapefile.
    ============ ================= =============================================

    :param str field_name: Name of the requested field in the output collection. If ``None``, defaults to the variable
     name or names joined by ``_``.
    :param bool regrid_source: If ``False``, do not regrid this dataset. This is relevant only if a
     ``regrid_destination`` dataset is present. Please see :ref:`esmpy-regridding` for an overview.
    :param bool regrid_destination: If ``True``, use this dataset as the destination grid for a regridding operation.
     Only one :class:`~ocgis.RequestDataset` may be set as the destination grid. Please see :ref:`esmpy-regridding` for
     an overview.

    .. _time units: http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html#num2date
    .. _time calendar: http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html#num2date

    :param dist: Overloaded dimension distribution
    :type dist: :class:`~ocgis.new_interface.mpi.OcgMpi`
    :param comm: The MPI communicator.
    :type comm: :class:`mpi4py.MPI_COMM`
    :param bool use_default_dist: If ``True`` (the default), use a default MPI distribution determined by the driver.
     If ``False``, do not apply a default distribution. In the absence of an overloaded distribution, defined by the
     keyword argument ``dist``, no variables/dimensions will be distributed.
    """

    # tdk: RESUME: driver-specific option for netcdf: grid_abstraction - perhaps driver_options?
    def __init__(self, uri=None, variable=None, units=None, time_range=None, time_region=None,
                 time_subset_func=None, level_range=None, conform_units_to=None, crs='auto', t_units=None,
                 t_calendar=None, t_conform_units_to=None, grid_abstraction='auto', dimension_map=None,
                 field_name=None, driver=None, regrid_source=True, regrid_destination=False, metadata=None,
                 format_time=True, opened=None, dist=None, comm=None, use_default_dist=True, uid=None,
                 rename_variable=None):

        self._is_init = True

        self._field_name = field_name
        self._level_range = None
        self._time_range = None
        self._time_region = None
        self._time_subset_func = None
        self._dimension_map = deepcopy(dimension_map)
        self._metadata = deepcopy(metadata)
        self._uri = None
        self._rename_variable = rename_variable
        self.use_default_dist = use_default_dist
        self.uid = uid

        # Set the default MPI communicator.
        self.comm = comm or MPI_COMM
        # Set the default dimension distribution.
        self._dist = dist

        # This is an "open" file-like object that may be passed in-place of file location parameters.
        self.opened = opened
        if self.opened is not None and driver is None:
            msg = 'If "opened" is not None, then a "driver" must be provided.'
            ocgis_lh(logger='request', exc=RequestValidationError('driver', msg))

        # Field creation options.
        self.format_time = format_time
        self.grid_abstraction = grid_abstraction
        # Flag used for regridding to determine if the coordinate system was assigned during initialization.
        self._has_assigned_coordinate_system = False if crs == 'auto' else True

        if uri is None:
            if opened is None:
                ocgis_lh(logger='request', exc=RequestValidationError('uri', 'Cannot be None'))
        else:
            self._uri = get_uri(uri)

        if driver is None:
            klass = get_autodiscovered_driver(uri)
        else:
            klass = get_driver(driver)
        self.driver = klass(self)

        if variable is not None:
            variable = get_tuple(variable)
        self._variable = variable

        self.time_range = time_range
        self.time_region = time_region
        self.time_subset_func = time_subset_func
        self.level_range = level_range

        self._crs = deepcopy(crs)

        self.regrid_source = regrid_source
        self.regrid_destination = regrid_destination

        self.units = units
        self.conform_units_to = conform_units_to

        self._is_init = False

        self._validate_time_subset_()

        # Update metadata for time variable.
        tvar = get_by_sequence(self.dimension_map, ['time', 'variable'])
        if tvar is not None:
            m = self.metadata['variables'][tvar]
            if t_units is not None:
                m['attributes']['units'] = t_units
            if t_calendar is not None:
                m['attributes']['calendar'] = t_calendar
            if t_conform_units_to is not None:
                from ocgis.util.units import get_units_object
                t_calendar = m['attributes'].get('calendar', constants.DEFAULT_TEMPORAL_CALENDAR)
                t_conform_units_to = get_units_object(t_conform_units_to, calendar=t_calendar)
                m['conform_units_to'] = t_conform_units_to

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    # tdk: remove
    @property
    def _name(self):
        raise NotImplementedError

    def __iter__(self):
        attrs = ['variable', 'units', 'conform_units_to']
        for ii in range(len(self)):
            yield {a: get_tuple(getattr(self, a))[ii] for a in attrs}

    def __len__(self):
        try:
            ret = len(get_tuple(self.variable))
        except NoDataVariablesFound:
            ret = 0
        return ret

    @property
    def conform_units_to(self):
        ret = []
        m = self.metadata['variables']
        for v in get_iter(self.variable):
            ret.append(m[v].get('conform_units_to'))
        ret = get_first_or_tuple(ret)
        return ret

    @conform_units_to.setter
    def conform_units_to(self, value):
        if value is not None:
            value = get_tuple(value)
            if len(value) != len(get_tuple(self.variable)):
                msg = 'Must match "variable" element-wise. The sequence lengths differ.'
                raise RequestValidationError('units', msg)
            if env.USE_CFUNITS:
                validate_units('conform_units_to', value)
            # If we are conforming units, assert that units are equivalent.
            validate_unit_equivalence(get_tuple(self.units), value)

            m = self.metadata['variables']
            for v, u in zip(get_tuple(self.variable), value):
                m[v]['conform_units_to'] = u

    @property
    def _conform_units_to(self):
        raise NotImplementedError

    @property
    def crs(self):
        if self._crs == 'auto':
            ret = self.driver.get_crs(self.metadata)
        else:
            ret = self._crs
        return ret

    @property
    def level_range(self):
        return self._level_range.value

    @level_range.setter
    def level_range(self, value):
        from ocgis.ops.parms.definition import LevelRange

        self._level_range = LevelRange(value)

    @property
    def dimension_map(self):
        if self._dimension_map is None:
            dimension_map_raw = self.driver.dimension_map_raw
            self._dimension_map = deepcopy(dimension_map_raw)
        self.driver.format_dimension_map(self._dimension_map, self.metadata)
        return self._dimension_map

    @property
    def dist(self):
        if self._dist is None:
            ret = self.driver.dist
        else:
            ret = self._dist
        return ret

    @property
    def field_name(self):
        if self._field_name is None:
            # Use renamed variables for field names. Often there is a single variable in the request. This ensures
            # unique field names if renamed variables are unique.
            ret = list(get_iter(self.rename_variable))
            if len(ret) > 1:
                msg = 'No default "field_name" based on variables name possible with multiple data variables: {}. ' \
                      'Using default field name: {}.'.format(self.variable, constants.MiscNames.DEFAULT_FIELD_NAME)
                ocgis_lh(msg=msg, level=logging.WARN)
                ret = constants.MiscNames.DEFAULT_FIELD_NAME
            else:
                ret = ret[0]
        else:
            ret = self._field_name
        return ret

    @field_name.setter
    def field_name(self, value):
        self._field_name = value

    @property
    def has_data_variables(self):
        """Return ``True`` if data variables are found in the target dataset."""
        try:
            assert self.variable
            ret = True
        except NoDataVariablesFound:
            ret = False
        return ret

    @property
    def metadata(self):
        if self._metadata is None:
            metadata = self.driver.metadata_raw
            self._metadata = deepcopy(metadata)
        return self._metadata

    @property
    def time_range(self):
        return self._time_range.value

    @property
    def rename_variable(self):
        if self._rename_variable is None:
            ret = self.variable
        else:
            ret = get_first_or_tuple(list(get_iter(self._rename_variable)))
        return ret

    @rename_variable.setter
    def rename_variable(self, value):
        value = get_tuple(value)
        self._rename_variable = value

    @property
    def rename_variable_map(self):
        ret = {}
        for name, rename in zip(get_iter(self.variable), get_iter(self.rename_variable)):
            ret[name] = rename
        return ret

    @time_range.setter
    def time_range(self, value):
        from ocgis.ops.parms.definition import TimeRange

        self._time_range = TimeRange(value)
        # ensure the time range and region overlaps
        if not self._is_init:
            self._validate_time_subset_()

    @property
    def time_region(self):
        return self._time_region.value

    @time_region.setter
    def time_region(self, value):
        from ocgis.ops.parms.definition import TimeRegion

        self._time_region = TimeRegion(value)
        # ensure the time range and region overlaps
        if not self._is_init:
            self._validate_time_subset_()

    @property
    def time_subset_func(self):
        # tdk: implement
        return self._time_subset_func.value

    @time_subset_func.setter
    def time_subset_func(self, value):
        from ocgis.ops.parms.definition import TimeSubsetFunction

        self._time_subset_func = TimeSubsetFunction(value)

    @property
    def units(self):
        ret = []
        for v in get_iter(self.variable):
            ret.append(self.metadata['variables'][v]['attributes'].get('units'))
        ret = get_first_or_tuple(ret)
        return ret

    @units.setter
    def units(self, value):
        if value is not None:
            value = get_tuple(value)
            if len(value) != len(get_tuple(self.variable)):
                msg = 'Must match "variable" element-wise. The sequence lengths differ.'
                raise RequestValidationError('units', msg)
            if env.USE_CFUNITS:
                validate_units('units', value)

            m = self.metadata['variables']
            for v, u in zip(get_tuple(self.variable), value):
                m[v]['attributes']['units'] = u

    # tdk: remove
    @property
    def _units(self):
        raise NotImplementedError

    @property
    def uri(self):
        return get_first_or_tuple(self._uri)

    @property
    def variable(self):
        if self._variable is None:
            ret = self.driver.get_data_variable_names(self.metadata, self.dimension_map)
        else:
            for vname in self._variable:
                if vname not in list(self.metadata['variables'].keys()):
                    raise VariableNotFoundError(self.uri, vname)
            ret = self._variable

        try:
            ret = get_first_or_tuple(ret)
        except IndexError:
            raise NoDataVariablesFound

        return ret

    def get(self, *args, **kwargs):
        return self.get_field(*args, **kwargs)

    def get_field(self, *args, **kwargs):
        """
        :rtype: :class:`~ocgis.interface.base.Field`
        """
        # Allow for get overloads in the method call.
        overloads = ['format_time', 'grid_abstraction', 'uid']
        for overload in overloads:
            if overload not in kwargs:
                kwargs[overload] = getattr(self, overload)
        if 'name' not in kwargs:
            try:
                name = self.field_name
            except NoDataVariablesFound:
                name = constants.MiscNames.DEFAULT_FIELD_NAME
            kwargs['name'] = name
        return self.driver.get_field(*args, **kwargs)

    def get_variable_collection(self, **kwargs):
        """
        :rtype: `VariableCollection`
        """
        return self.driver.get_variable_collection(**kwargs)

    def inspect(self):
        """
        Print a string containing important information about the source driver.
        """

        return self.driver.inspect()

    def _get_meta_rows_(self):
        if self.time_range is None:
            tr = None
        else:
            tr = '{0} to {1} (inclusive)'.format(self.time_range[0], self.time_range[1])
        if self.level_range is None:
            lr = None
        else:
            lr = '{0} to {1} (inclusive)'.format(self.level_range[0], self.level_range[1])

        rows = ['    URI: {0}'.format(self.uri),
                '    Variable: {0}'.format(self.variable),
                '    Time Range: {0}'.format(tr),
                '    Time Region/Selection: {0}'.format(self.time_region),
                '    Level Range: {0}'.format(lr)]
        return rows

    def _validate_time_subset_(self):
        if not validate_time_subset(self.time_range, self.time_region):
            raise RequestValidationError("time_range/time_region", '"time_range" and "time_region" must overlap.')


def get_first_or_tuple(value):
    if len(value) > 1:
        ret = tuple(value)
    else:
        ret = value[0]
    return ret


def get_is_none(value):
    return all([v is None for v in get_iter(value)])


def validate_units(keyword, sequence):
    # Check all units are convertible into the appropriate backend.
    try:
        list(map(get_units_object, sequence))
    except ValueError as e:
        raise RequestValidationError(keyword, e.message)


def validate_unit_equivalence(src_units, dst_units):
    from ocgis.ops.parms.definition import ConformUnitsTo

    for s, d in zip(src_units, dst_units):
        s, d = list(map(get_units_object, (s, d)))
        if not get_are_units_equivalent((s, d)):
            msg = 'The units specified in "{2}" ("{0}") are not equivalent to the source units "{1}".'
            raise RequestValidationError(ConformUnitsTo.name, msg.format(s, d, ConformUnitsTo.name))


def get_autodiscovered_driver(uri):
    """
    :param str uri: The target URI containing data for which to choose a driver.
    :returns: The correct driver for opening the ``uri``.
    :rtype: :class:`ocgis.api.request.driver.base.AbstractDriver`
    :raises: RequestValidationError
    """

    possible = []
    for element in get_iter(uri):
        for driver in driver_registry.drivers:
            for pattern in driver.extensions:
                if re.match(pattern, element) is not None:
                    possible.append(driver)

    exc_msg = None
    ret = None
    if len(possible) == 0:
        exc_msg = 'Driver not found for URI: {0}'.format(uri)
    elif len(possible) == 1:
        ret = possible[0]
    else:
        sub_possible = []
        for p in possible:
            if p._priority is True:
                sub_possible.append(p)
        sub_possible_keys = [sp.key for sp in sub_possible]
        if len(set(sub_possible_keys)) == 1:
            ret = sub_possible[0]
        else:
            exc_msg = 'More than one possible driver matched URI: {}'.format(uri)

    if exc_msg is None:
        return ret
    else:
        ocgis_lh(logger='request', exc=RequestValidationError('driver/uri', exc_msg))


def get_driver(driver):
    return get_driver_class(key_or_class=driver, default=DriverKeys.NETCDF_CF)


def get_uri(uri, ignore_errors=False, followlinks=True):
    out_uris = []
    if isinstance(uri, six.string_types):
        uris = [uri]
    else:
        uris = uri
    assert (len(uri) >= 1)
    for uri in uris:
        ret = None
        # check if the path exists locally
        if os.path.exists(uri) or '://' in uri:
            ret = uri
        # if it does not exist, check the directory locations
        else:
            if env.DIR_DATA is not None:
                if isinstance(env.DIR_DATA, six.string_types):
                    dirs = [env.DIR_DATA]
                else:
                    dirs = env.DIR_DATA
                for directory in dirs:
                    for filepath in locate(uri, directory, followlinks=followlinks):
                        ret = filepath
                        break
            if ret is None:
                if not ignore_errors:
                    msg = 'File not found: "{0}". Check env.DIR_DATA or ensure a fully qualified URI is used.'.format(
                        uri)
                    ocgis_lh(logger='request', exc=ValueError(msg))
            else:
                if not os.path.exists(ret) and not ignore_errors:
                    msg = 'Path does not exist and is likely not a remote URI: "{0}". Set "ignore_errors" to True if ' \
                          'this is not the case.'
                    msg = msg.format(ret)
                    ocgis_lh(msg, exc=ValueError(msg))
        out_uris.append(ret)
    return out_uris
