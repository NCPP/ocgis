from collections import OrderedDict
from copy import deepcopy
import inspect
import logging
import os
import itertools
import re

from ocgis.api.request.driver.vector import DriverVector
from ocgis.interface.base.field import Field
from ocgis.api.collection import AbstractCollection
from ocgis.api.request.driver.nc import DriverNetcdf
from ocgis.exc import RequestValidationError, NoUnitsError, NoDimensionedVariablesFound
from ocgis.interface.base.crs import CFWGS84
from ocgis.util.helpers import get_iter, locate, validate_time_subset, get_tuple
from ocgis import env
from ocgis.util.logging_ocgis import ocgis_lh


class RequestDataset(object):
    #todo: document vector format
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

    :param level_range: Upper and lower bounds for level dimension subsetting. If ``None``, return all levels.
    :type level_range: [int, int] or [float, float]
    :param crs: Overload the autodiscovered coordinate system.
    :type crs: :class:`ocgis.crs.CoordinateReferenceSystem`

    >>> from ocgis.crs import CFWGS84
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

    :param str name: Name of the requested data in the output collection. If ``None``, defaults to ``alias``. If this is
     a multivariate request (i.e. more than one variable) and this is ``None``, then the aliases will be joined by
     ``'_'`` to create the name.
    :param bool regrid_source: If ``False``, do not regrid this dataset. This is relevant only if a
     ``regrid_destination`` dataset is present. Please see :ref:`esmpy-regridding` for an overview.
    :param bool regrid_destination: If ``True``, use this dataset as the destination grid for a regridding operation.
     Only one :class:`~ocgis.RequestDataset` may be set as the destination grid. Please see :ref:`esmpy-regridding` for
     an overview.

    .. _time units: http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html#num2date
    .. _time calendar: http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html#num2date
    """

    # contains key-value links to drivers. as new drivers are added, this dictionary must be updated.
    _Drivers = OrderedDict()
    _Drivers[DriverNetcdf.key] = DriverNetcdf
    _Drivers[DriverVector.key] = DriverVector

    def __init__(self, uri=None, variable=None, alias=None, units=None, time_range=None, time_region=None,
                 time_subset_func=None, level_range=None, conform_units_to=None, crs=None, t_units=None,
                 t_calendar=None, t_conform_units_to=None, did=None, meta=None, s_abstraction=None, dimension_map=None,
                 name=None, driver=None, regrid_source=True, regrid_destination=False):

        self._alias = None
        self._conform_units_to = None
        self._name = None
        self._level_range = None
        self._time_range = None
        self._time_range = None
        self._time_subset_func = None
        self._units = None

        self._is_init = True

        # flag used for regridding to determine if the coordinate system was assigned during initialization
        self._has_assigned_coordinate_system = False if crs is None else True
        self._source_metadata = None

        if uri is None:
            raise RequestValidationError('uri', 'Cannot be None')
        else:
            self._uri = self._get_uri_(uri)

        if driver is None:
            klass = self._get_autodiscovered_driver_(self.uri)
        else:
            try:
                klass = self._Drivers[driver]
            except KeyError:
                raise RequestValidationError('driver', 'Driver not found: {0}'.format(driver))
        self.driver = klass(self)

        if variable is not None:
            variable = get_tuple(variable)
        self._variable = variable

        self.alias = alias
        self.name = name
        self.time_range = time_range
        self.time_region = time_region
        self.time_subset_func = time_subset_func
        self.level_range = level_range

        self._crs = deepcopy(crs)

        self.t_units = t_units
        self.t_calendar = t_calendar
        self.t_conform_units_to = t_conform_units_to

        self._dimension_map = deepcopy(dimension_map)
        self.did = did
        self.meta = meta or {}

        self.units = units
        self.conform_units_to = conform_units_to

        self.s_abstraction = s_abstraction
        try:
            self.s_abstraction = self.s_abstraction.lower()
            assert self.s_abstraction in ('point', 'polygon')
        except AttributeError:
            if s_abstraction is None:
                pass
            else:
                raise
        self.regrid_source = regrid_source
        self.regrid_destination = regrid_destination

        self._is_init = False

        self._validate_time_subset_()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __iter__(self):
        attrs = ['alias', 'variable', 'units', 'conform_units_to']
        for ii in range(len(self)):
            yield {a: get_tuple(getattr(self, a))[ii] for a in attrs}

    def __len__(self):
        try:
            ret = len(get_tuple(self.variable))
        except NoDimensionedVariablesFound:
            ret = 0
        return ret

    def __str__(self):
        msg = '{0}({1})'
        argspec = inspect.getargspec(self.__class__.__init__)
        parms = []
        for name in argspec.args:
            if name == 'self':
                continue
            else:
                as_str = '{0}={1}'
                value = getattr(self, name)
                if isinstance(value, basestring):
                    fill = '"{0}"'.format(value)
                else:
                    fill = value
                as_str = as_str.format(name, fill)
            parms.append(as_str)
        msg = msg.format(self.__class__.__name__, ', '.join(parms))
        return msg

    @property
    def alias(self):
        if self._alias is None:
            ret = get_tuple(self.variable)
        else:
            ret = self._alias
        return get_first_or_sequence(ret)

    @alias.setter
    def alias(self, value):
        if value is not None:
            self._alias = get_tuple(value)
            if len(self._alias) != len(get_tuple(self.variable)):
                raise RequestValidationError('alias', 'Each variable must have an alias. The sequence lengths differ.')

    @property
    def conform_units_to(self):
        if self._conform_units_to is None:
            ret = get_tuple([None] * len(get_tuple(self.variable)))
        else:
            ret = self._conform_units_to
        ret = get_first_or_sequence(ret)
        return ret

    @conform_units_to.setter
    def conform_units_to(self, value):
        if value is not None:
            value = get_tuple(value)
            if len(value) != len(get_tuple(self.variable)):
                msg = 'Must match "variable" element-wise. The sequence lengths differ.'
                raise RequestValidationError('conform_units_to', msg)
            validate_units('conform_units_to', value)
        self._conform_units_to = value

    @property
    def crs(self):
        if self._crs is None:
            ret = self.driver.get_crs()
            if ret is None:
                ocgis_lh('No "grid_mapping" attribute available assuming WGS84: {0}'.format(self.uri),
                         'request', logging.WARN)
                ret = CFWGS84()
        else:
            ret = self._crs
        return ret

    @property
    def level_range(self):
        return self._level_range.value

    @level_range.setter
    def level_range(self, value):
        from ocgis.api.parms.definition import LevelRange

        self._level_range = LevelRange(value)

    @property
    def name(self):
        if self._name is None:
            ret = '_'.join(get_tuple(self.alias))
        else:
            ret = self._name
        return ret

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def dimension_map(self):
        return self._dimension_map

    @property
    def source_metadata(self):
        if self._source_metadata is None:
            self._source_metadata = self.driver.get_source_metadata()
        return self._source_metadata

    @property
    def time_range(self):
        return self._time_range.value

    @time_range.setter
    def time_range(self, value):
        from ocgis.api.parms.definition import TimeRange

        self._time_range = TimeRange(value)
        # ensure the time range and region overlaps
        if not self._is_init:
            self._validate_time_subset_()

    @property
    def time_region(self):
        return self._time_region.value

    @time_region.setter
    def time_region(self, value):
        from ocgis.api.parms.definition import TimeRegion

        self._time_region = TimeRegion(value)
        # ensure the time range and region overlaps
        if not self._is_init:
            self._validate_time_subset_()

    @property
    def time_subset_func(self):
        # tdk: test
        # tdk: document param to request dataset
        return self._time_subset_func.value

    @time_subset_func.setter
    def time_subset_func(self, value):
        from ocgis.api.parms.definition import TimeSubsetFunction

        self._time_subset_func = TimeSubsetFunction(value)

    @property
    def units(self):
        if self._units is None:
            ret = get_tuple([None] * len(get_tuple(self.variable)))
        else:
            ret = self._units
        ret = get_first_or_sequence(ret)
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
        self._units = value

    @property
    def uri(self):
        return get_first_or_sequence(self._uri)

    @property
    def variable(self):
        if self._variable is None:
            self._variable = get_tuple(self.driver.get_dimensioned_variables())
        try:
            ret = get_first_or_sequence(self._variable)
        except IndexError:
            raise NoDimensionedVariablesFound
        return ret

    def get(self, **kwargs):
        """
        :rtype: :class:`~ocgis.interface.base.Field`
        """
        if not get_is_none(self._conform_units_to):
            src_units = []
            dst_units = []
            for rdict in self:
                if rdict['conform_units_to'] is not None:
                    variable_units = rdict.get('units') or self._get_units_from_metadata_(rdict['variable'])
                    if variable_units is None:
                        raise NoUnitsError(rdict['variable'])
                    src_units.append(variable_units)
                    dst_units.append(rdict['conform_units_to'])
            validate_unit_equivalence(src_units, dst_units)
        return self.driver.get_field(**kwargs)

    def inspect(self):
        """
        Print a string containing important information about the source driver.
        """

        return self.driver.inspect()

    def inspect_as_dct(self):
        '''
        Return a dictionary representation of the target's metadata. If the variable
        is `None`. An attempt will be made to find the target dataset's time bounds
        raising a warning if none is found or the time variable is lacking units
        and/or calendar attributes.

        >>> rd = ocgis.RequestDataset('rhs_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc','rhs')
        >>> ret = rd.inspect_as_dct()
        >>> ret.keys()
        ['dataset', 'variables', 'dimensions', 'derived']
        >>> ret['derived']
        OrderedDict([('Start Date', '2011-01-01 12:00:00'), ('End Date', '2020-12-31 12:00:00'), ('Calendar', '365_day'), ('Units', 'days since 1850-1-1'), ('Resolution (Days)', '1'), ('Count', '8192'), ('Has Bounds', 'True'), ('Spatial Reference', 'WGS84'), ('Proj4 String', '+proj=longlat +datum=WGS84 +no_defs '), ('Extent', '(-1.40625, -90.0, 358.59375, 90.0)'), ('Interface Type', 'NcPolygonDimension'), ('Resolution', '2.80091351339')])

        :rtype: :class:`collections.OrderedDict`
        '''
        from ocgis import Inspect
        ip = Inspect(request_dataset=self)
        ret = ip._as_dct_()
        return ret

    @classmethod
    def _get_autodiscovered_driver_(cls, uri):
        """
        :param str uri: The target URI containing data for which to choose a driver.
        :returns: The correct driver for opening the ``uri``.
        :rtype: :class:`ocgis.api.request.driver.base.AbstractDriver`
        :raises: RequestValidationError
        """

        for element in get_iter(uri):
            for driver in cls._Drivers.itervalues():
                for pattern in driver.extensions:
                    if re.match(pattern, element) is not None:
                        return driver

        msg = 'Driver not found for URI: {0}'.format(uri)
        raise RequestValidationError('driver/uri', msg)

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
                '    Alias: {0}'.format(self.alias),
                '    Time Range: {0}'.format(tr),
                '    Time Region/Selection: {0}'.format(self.time_region),
                '    Level Range: {0}'.format(lr),
                '    Overloaded Parameters:',
                '      PROJ4 String: {0}'.format(self.crs),
                '      Time Units: {0}'.format(self.t_units),
                '      Time Calendar: {0}'.format(self.t_calendar)]
        return rows

    def _get_units_from_metadata_(self, variable):
        return self.source_metadata['variables'][variable]['attrs'].get('units')

    @staticmethod
    def _get_uri_(uri, ignore_errors=False, followlinks=True):
        out_uris = []
        if isinstance(uri, basestring):
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
                    if isinstance(env.DIR_DATA, basestring):
                        dirs = [env.DIR_DATA]
                    else:
                        dirs = env.DIR_DATA
                    for directory in dirs:
                        for filepath in locate(uri, directory, followlinks=followlinks):
                            ret = filepath
                            break
                if ret is None:
                    if not ignore_errors:
                        raise (ValueError(
                            'File not found: "{0}". Check env.DIR_DATA or ensure a fully qualified URI is used.'.format(
                                uri)))
                else:
                    if not os.path.exists(ret) and not ignore_errors:
                        raise (ValueError(
                            'Path does not exist and is likely not a remote URI: "{0}". Set "ignore_errors" to True if this is not the case.'.format(
                                ret)))
            out_uris.append(ret)
        return out_uris

    def _validate_time_subset_(self):
        if not validate_time_subset(self.time_range, self.time_region):
            raise RequestValidationError("time_range/time_region", '"time_range" and "time_region" must overlap.')


class RequestDatasetCollection(AbstractCollection):
    """
    A set of :class:`ocgis.RequestDataset` and/or :class:`~ocgis.Field` objects.

    >>> from ocgis import RequestDatasetCollection, RequestDataset
    >>> uris = ['http://some.opendap.dataset1', 'http://some.opendap.dataset2']
    >>> variables = ['tasmax', 'tasmin']
    >>> request_datasets = [RequestDatset(uri,variable) for uri,variable in zip(uris,variables)]
    >>> rdc = RequestDatasetCollection(request_datasets)
    ...
    >>> # Update object in place.
    >>> rdc = RequestDatasetCollection()
    >>> for rd in request_datasets:
    ...     rdc.update(rd)

    :param target: A sequence of request dataset or field objects.
    :type target: sequence[:class:`~ocgis.RequestDataset` and/or :class:`~ocgis.Field` objects, ...]
    """

    def __init__(self, target=None):
        super(RequestDatasetCollection, self).__init__()

        self._unique_id_store = []

        if target is not None:
            for element in get_iter(target, dtype=(dict, RequestDataset, Field)):
                self.update(element)

    def __str__(self):
        ret = '{klass}(request_datasets=[{request_datasets}])'
        request_datasets = ', '.join([str(rd) for rd in self.itervalues()])
        return ret.format(klass=self.__class__.__name__, request_datasets=request_datasets)

    def iter_request_datasets(self):
        """
        :returns: An iterator over only the request dataset objects contained in the collection. Field objects are
         excluded.
        :rtype: `~ocgis.RequestDataset`
        """

        for value in self.itervalues():
            if isinstance(value, Field):
                continue
            else:
                yield value

    def update(self, target):
        """
        Add an object to the collection.
        
        :param target: The object to add.
        :type target: :class:`~ocgis.RequestDataset` or :class:`~ocgis.Field`
        """

        try:
            new_key = target.name
        except AttributeError:
            target = RequestDataset(**target)
            new_key = target.name

        unique_id = self._get_unique_id_(target)

        if unique_id is None:
            if len(self._unique_id_store) == 0:
                unique_id = 1
            else:
                unique_id = max(self._unique_id_store) + 1
            self._unique_id_store.append(unique_id)
            self._set_unique_id_(target, unique_id)
        else:
            self._unique_id_store.append(unique_id)

        if new_key in self._storage:
            raise KeyError('Name "{0}" already in collection. Names must be unique'.format(target.name))
        else:
            self._storage.update({target.name: target})

    def _get_meta_rows_(self):
        """
        :returns: A list of strings containing metadata on the collection objects.
        :rtype: list[str, ...]
        """

        rows = ['* dataset=']
        for value in self.itervalues():
            try:
                rows += value._get_meta_rows_()
            except AttributeError:
                # likely a field object
                msg = '{klass}(name={name}, ...)'.format(klass=value.__class__.__name__, name=value.name)
                rows.append(msg)
            rows.append('')

        return rows

    @staticmethod
    def _get_unique_id_(target):
        """
        :param target: The object to retrieve the unique identifier from.
        :type target: :class:`~ocgis.RequestDataset` or :class:`~ocgis.Field`
        :returns: The unique identifier of the object if available. ``None`` will be returned if no unique can be found.
        :rtype: int or ``None``
        """

        try:
            ret = target.did
        except AttributeError:
            ret = target.uid

        return ret

    @staticmethod
    def _set_unique_id_(target, uid):
        """
        :param target: The target object for setting the unique identifier.
        :type target: :class:`~ocgis.RequestDataset` or :class:`~ocgis.Field`
        :param int target: The unique identifier.
        """

        if isinstance(target, RequestDataset):
            target.did = uid
        elif isinstance(target, Field):
            target.uid = uid


def get_first_or_sequence(value):
    if len(value) > 1:
        ret = value
    else:
        ret = value[0]
    return ret


def get_is_none(value):
    return all([v is None for v in get_iter(value)])


def validate_units(keyword, sequence):
    from cfunits import Units
    try:
        map(Units, sequence)
    except ValueError as e:
        raise RequestValidationError(keyword, e.message)


def validate_unit_equivalence(src_units, dst_units):
    # import the cfunits package and attempt to construct a units object.
    # if this is okay, save the units string
    from cfunits import Units
    for s, d in itertools.izip(src_units, dst_units):
        if not Units(s).equivalent(Units(d)):
            raise RequestValidationError('conform_units_to',
             'The units specified in "conform_units_to" ("{0}") are not equivalent to the source units "{1}".'.\
             format(d.format(names=True), s.format(names=True)))
