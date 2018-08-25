import logging
from collections import OrderedDict
from collections import deque
from copy import deepcopy

import six
from ocgis import env, DimensionMap, VariableCollection
from ocgis.base import get_dimension_names, get_variable_names, get_variables, renamed_dimensions_on_variables, \
    revert_renamed_dimensions_on_variables, raise_if_empty
from ocgis.constants import DimensionMapKey, WrapAction, TagName, HeaderName, DimensionName, UNINITIALIZED, \
    KeywordArgument, DMK
from ocgis.util.helpers import get_iter, is_xarray
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.variable.base import Variable, get_bounds_names_1d, create_typed_variable_from_data_model
from ocgis.variable.crs import CoordinateReferenceSystem
from ocgis.variable.dimension import Dimension
from ocgis.variable.geom import GeometryVariable
from ocgis.variable.iterator import Iterator
from ocgis.variable.temporal import TemporalGroupVariable, TemporalVariable
from shapely.geometry import shape


class Field(VariableCollection):
    """
    A field behaves like a variable collection but with additional metadata on its component variables.
    
    .. note:: Accepts all parameters to :class:`~ocgis.VariableCollection`.

    Additional keyword arguments are:

    :param dimension_map: (``=None``) Maps variables to axes, dimensions, bounds, and default attributes. It is possible
     to fully-specify a default field by providing a list of ``variables`` and the dimension map. 
     Instrumented/coordinate variables may be provided with keyword arguments. The dimension map is updated internally 
     in those cases.
    :type dimension_map: :class:`~ocgis.DimensionMap` | :class:`dict`
    :param is_data: (``=None``) Set these variables or variable names (if names are provided, the variables must be
     provided through ``variables``) as data variables. Data variables often contain the field information of interest
     such as temperature, relative humidity, etc.
    :type is_data: `sequence` of :class:`~ocgis.Variable` | `sequence` of :class:`str`
    :param realization: (``=None``) A realization or ensemble variable. Its value is typically an integer representing
     its record count across global realizations.
    :type realization: :class:`~ocgis.Variable`
    :param time: (``=None``) A time variable.
    :type time: :class:`~ocgis.TemporalVariable`
    :param level: (``=None``) A level variable. This may also be considered the field's z-coordinate.
    :type level: :class:`~ocgis.Variable`
    :param grid: (``=None``) A grid object. x/y-coordinates will be pulled from the grid automatically. Any level or
     z-coordinate must be provided using ``level``.
    :type grid: :class:`~ocgis.Grid`
    :param geom: (``=None``) The geometry variable.
    :type geom: :class:`~ocgis.GeometryVariable`
    :param crs: (``='auto'``) A coordinate reference system variable. If ``'auto'``, use the coordinate system from
     the ``grid`` or ``geom``. ``geom`` is given preference if both are present.
    :type crs: :class:`str` | ``None`` | :class:`~ocgis.variable.crs.AbstractCRS`
    :param str format_time: See keyword argument ``format_time`` for :class:`~ocgis.TemporalVariable`.
    :param str grid_abstraction: See keyword argument ``abstraction`` for :class:`~ocgis.Grid`.
    :param grid_is_isomporphic: (``='auto'``) If ``True``, the grid is isomorphic with repeated, topologically adjancent
     cells (i.e. a logically rectangular grid). If ``False``, the grid elements change shapes (i.e. boundaries like a
     watershed). If ``'auto'``, let the driver determine the grid default.
    :type grid_is_isomorphic: bool | str
    """

    def __init__(self, **kwargs):
        kwargs = kwargs.copy()
        dimension_map = kwargs.pop('dimension_map', None)

        # Flag updated by driver to indicate if the coordinate system is assigned or implied.
        self._has_assigned_coordinate_system = False
        # Flag to indicate if this is a regrid destination.
        self.regrid_destination = kwargs.pop('regrid_destination', False)
        # Flag to indicate if this is a regrid source.
        self.regrid_source = kwargs.pop('regrid_source', True)

        # Other incoming data objects may have a coordinate system which should be used.
        crs = kwargs.pop(KeywordArgument.CRS, 'auto')

        # Add grid variable metadata to dimension map.
        grid = kwargs.pop(KeywordArgument.GRID, 'auto')

        # Configure the driver.
        driver = kwargs.pop(KeywordArgument.DRIVER, 'auto')

        # Extract standard coordinate variables from the field keyword arguments.
        k = (DimensionMapKey.GEOM, DimensionMapKey.REALIZATION, DimensionMapKey.TIME, DimensionMapKey.LEVEL)
        s = OrderedDict()
        for ii in k:
            s[ii] = kwargs.pop(ii, None)

        grid_abstraction = kwargs.pop(KeywordArgument.GRID_ABSTRACTION, 'auto')
        if grid_abstraction is None:
            raise ValueError("'{}' may not be None.".format(KeywordArgument.GRID_ABSTRACTION))
        grid_is_isomorphic = kwargs.pop('grid_is_isomorphic', 'auto')
        if grid_is_isomorphic is None:
            raise ValueError("'{}' may not be None.".format('grid_is_isomorphic'))

        # TODO: This should maybe be part of the dimension map? Time variables are not dependent on fields.
        self.format_time = kwargs.pop(KeywordArgument.FORMAT_TIME, True)

        # Use tags to set data variables.
        is_data = kwargs.pop(KeywordArgument.IS_DATA, [])

        VariableCollection.__init__(self, **kwargs)

        dimension_map = deepcopy(dimension_map)
        if dimension_map is None:
            dimension_map = DimensionMap()
        elif isinstance(dimension_map, dict):
            dimension_map = DimensionMap.from_dict(dimension_map)
        self.dimension_map = dimension_map

        self.set_grid(grid, crs=crs)
        if driver != 'auto':
            self.dimension_map.set_driver(driver)
        if grid_abstraction != 'auto':
            self.dimension_map.set_grid_abstraction(grid_abstraction)
        if grid_is_isomorphic != 'auto':
            self.dimension_map.set_property(DMK.IS_ISOMORPHIC, grid_is_isomorphic)

        # Append the data variable tagged variable names.
        is_data = list(get_iter(is_data, dtype=Variable))
        is_data_variable_names = get_variable_names(is_data)
        for idvn in is_data_variable_names:
            self.append_to_tags(TagName.DATA_VARIABLES, idvn, create=True)
        for idx, dvn in enumerate(is_data_variable_names):
            if dvn not in self:
                if isinstance(is_data[idx], Variable):
                    self.add_variable(is_data[idx])

        # Configure the field updating the dimension map in the process.
        cvar = s[DimensionMapKey.REALIZATION]
        if cvar is not None:
            self.set_realization(cvar)
        cvar = s[DimensionMapKey.TIME]
        if cvar is not None:
            self.set_time(cvar)
        cvar = s[DimensionMapKey.LEVEL]
        if cvar is not None:
            self.set_level(cvar)
        cvar = s[DimensionMapKey.GEOM]
        if cvar is not None:
            self.set_geom(cvar, crs=crs)
        if crs != 'auto':
            self.set_crs(crs)

    @property
    def _should_regrid(self):
        raise NotImplementedError

    @property
    def axes_shapes(self):
        """
        :return: Axis variables shapes.
        :rtype: dict
        """
        ret = {}
        if self.realization is None:
            r = 1
        else:
            r = self.realization.shape[0]
        ret['R'] = r
        if self.time is None:
            t = 0
        else:
            t = self.time.shape[0]
        ret['T'] = t
        if self.level is None or self.level.ndim == 0:
            l = 0
        else:
            l = self.level.shape[0]
        ret['Z'] = l
        if self.y is None:
            y = 0
        else:
            y = self.y.shape[0]
        ret['Y'] = y
        if self.x is None:
            x = 0
        else:
            x = self.x.shape[0]
        ret['X'] = x
        return ret

    @property
    def bounds_variables(self):
        """
        Create a tuple of bounds variables associated with :meth:`~ocgis.collection.field.Field.coordinate_variables`.

        :rtype: tuple(:class:`ocgis.Variable`, ...)
        """
        ret = [c.bounds for c in self.coordinate_variables if c.bounds is not None]
        ret = tuple(ret)
        return ret

    @property
    def coordinate_variables(self):
        """
        Return a tuple of coordinate variables. This will attempt to access spatial coordinate variables on the field's
        grid. If no grid is available, spatial coordinates will be pulled from the dimension map. Time will always be
        pulled from the field. The tuple may have a length of zero if no coordinate variables are available on the
        field.

        :rtype: tuple
        """
        grid = self.grid
        if grid is not None:
            ret = list(grid.coordinate_variables)
            if self.time is not None:
                ret.insert(0, self.time)
        else:
            poss = [self.x, self.y, self.level, self.time]
            poss = [p for p in poss if p is not None]
            ret = tuple(poss)
        return ret

    @property
    def crs(self):
        """
        :return: Get the field's coordinate reference system. Return ``None`` if no coordinate system is assigned.
        :rtype: :class:`~ocgis.variable.crs.AbstractCRS`
        """
        ret = self.dimension_map.get_crs(parent=self, nullable=True)
        return ret

    @property
    def data_variables(self):
        """
        Data variables are the "value" variables for the field. They are often variables like temperature or relative
        humidity. The default tag :attr:`ocgis.constants.TagName.DATA_VARIABLES` is used for data variables.
        
        :returns: A sequence of variables tagged with the default data variable tag.
        :rtype: `sequence` of :class:`~ocgis.Variable`
        """
        try:
            ret = tuple(self.get_by_tag(TagName.DATA_VARIABLES))
        except KeyError:
            ret = tuple()
        return ret

    @property
    def driver(self):
        """
        Return the driver class associated with the dimension map.

        :rtype: :class:`ocgis.driver.base.AbstractDriver`
        """
        return self.dimension_map.get_driver(as_class=True)

    @property
    def realization(self):
        """
        :return: Get the field's realization variable. Return ``None`` if no realization is assigned.
        :rtype: :class:`~ocgis.Variable` | ``None``
        """
        return get_field_property(self, 'realization')

    @property
    def temporal(self):
        """Alias for :attr:`~ocgis.Field.time`"""
        return self.time

    @property
    def time(self):
        """
        :return: Get the field's time variable. Return ``None`` if no time is assigned.
        :rtype: :class:`~ocgis.TemporalVariable` | ``None``
        """
        ret = get_field_property(self, 'time')
        # tdk: FEATURE: time selection should be handled by the driver
        if ret is not None:
            import xarray as xr
            if not isinstance(ret, xr.DataArray):
                if not isinstance(ret, TemporalGroupVariable):
                    ret = TemporalVariable.from_variable(ret, format_time=self.format_time)
        return ret

    @property
    def wrapped_state(self):
        """
        :return: The wrapped state for the field.
        :rtype: :attr:`ocgis.constants.WrappedState`
        :raises: :class:`~ocgis.exc.EmptyObjectError`
        """

        raise_if_empty(self)

        if self.crs is None:
            ret = None
        else:
            ret = self.crs.get_wrapped_state(self)
        return ret

    @property
    def level(self):
        """
        :return: Get the field's level variable. Return ``None`` if no level is assigned.
        :rtype: :class:`~ocgis.Variable` | ``None``
        """

        return get_field_property(self, DMK.LEVEL)

    @property
    def x(self):
        """
        :return: Get the field's x-coordinate variable. Return ``None`` if no x-coordinate is assigned.
        :rtype: :class:`~ocgis.Variable` | ``None``
        """

        return get_field_property(self, DMK.X)

    @property
    def y(self):
        """
        :return: Get the field's y-coordinate variable. Return ``None`` if no y-coordinate is assigned.
        :rtype: :class:`~ocgis.Variable` | ``None``
        """

        return get_field_property(self, DMK.Y)

    @property
    def z(self):
        """Alias for :attr:`~ocgis.Field.level`."""

        return self.level

    @property
    def grid(self):
        """
        :return: Get the field's grid object. Return ``None`` if no grid is present.
        :rtype: :class:`~ocgis.spatial.grid.AbstractGrid` | ``None``
        """
        return self.driver.get_grid(self)

    @property
    def geom(self):
        """
        :return: Get the field's geometry variable. Return ``None`` if no geometry is available.
        :rtype: :class:`~ocgis.GeometryVariable` | ``None``
        """

        ret = get_field_property(self, 'geom')
        if ret is not None:
            crs = self.crs
            # Overload the geometry coordinate system if set on the field. Otherwise, this will use the coordinate
            # system on the geometry variable.
            if crs is not None:
                try:
                    ret.crs = crs
                except:
                    # tdk: FEATURE: no knowledge in xarray of a crs on spatial objects
                    import xarray as xr
                    if isinstance(ret, xr.DataArray):
                        pass
                    else:
                        raise

        return ret

    @property
    def grid_abstraction(self):
        return self.dimension_map.get_grid_abstraction()

    @property
    def has_data_variables(self):
        """
        :return: ``True`` if the field has data variables.
        :rtype: bool
        """
        if len(self.data_variables) > 0:
            ret = True
        else:
            ret = False
        return ret

    def add_variable(self, variable, force=False, is_data=False):
        """
        ..note:: Accepts all parameters to :meth:`~ocgis.VariableCollection.add_variable`.
        
        Additional keyword arguments are:
        
        :param bool is_data: If ``True``, the variable is considered a data variable.
        """

        super(Field, self).add_variable(variable, force=force)
        if is_data:
            tagged = get_variable_names(self.get_by_tag(TagName.DATA_VARIABLES, create=True))
            if variable.name not in tagged:
                self.append_to_tags(TagName.DATA_VARIABLES, variable.name)

    def copy(self):
        """
        :return: A shallow copy of the field. The field's dimension map is deep copied.
        :rtype: :class:`~ocgis.Field`
        """

        ret = super(Field, self).copy()
        # Changes to a field's shallow copy should be able to adjust attributes in the dimension map as needed.
        ret.dimension_map = deepcopy(ret.dimension_map)
        return ret

    @classmethod
    def from_records(cls, records, schema=None, crs=UNINITIALIZED, uid=None, union=False, data_model=None):
        """
        Create a :class:`~ocgis.Field` from Fiona-like records.

        :param records: A sequence of records returned from an Fiona file object.
        :type records: `sequence` of :class:`dict`
        :param schema: A Fiona-like schema dictionary. If ``None`` and any records properties are ``None``, then this
         must be provided.
        :type schema: dict

        >>> schema = {'geometry': 'Point', 'properties': {'UGID': 'int', 'NAME', 'str:4'}}

        :param crs: If :attr:`ocgis.constants.UNINITIALIZED`, default to :attr:`ocgis.env.DEFAULT_COORDSYS`.
        :type crs: :class:`dict` | :class:`~ocgis.variable.crs.AbstractCoordinateReferenceSystem`
        :param str uid: If provided, use this attribute name as the unique identifier. Otherwise search for
         :attr:`env.DEFAULT_GEOM_UID` and, if not present, construct a 1-based identifier with this name.
        :param bool union: If ``True``, union the geometries from records yielding a single geometry with a unique
         identifier value of ``1``.
        :param str data_model: See :meth:`~ocgis.driver.nc.create_typed_variable_from_data_model`.
        :returns: Field object constructed from records.
        :rtype: :class:`~ocgis.Field`
        """

        if uid is None:
            uid = env.DEFAULT_GEOM_UID

        if isinstance(crs, dict):
            crs = CoordinateReferenceSystem(value=crs)
        elif crs == UNINITIALIZED:
            crs = env.DEFAULT_COORDSYS

        if union:
            deque_geoms = None
            deque_uid = [1]
        else:
            # Holds geometry objects.
            deque_geoms = deque()
            # Holds unique identifiers.
            deque_uid = deque()

        build = True
        for ctr, record in enumerate(records, start=1):

            # Get the geometry from a keyword present on the input dictionary or construct from the coordinates
            # sequence.
            try:
                current_geom = record['geom']
            except KeyError:
                current_geom = shape(record['geometry'])

            if union:
                if build:
                    deque_geoms = current_geom
                else:
                    deque_geoms = deque_geoms.union(current_geom)
            else:
                deque_geoms.append(current_geom)

            # Set up the properties array
            if build:
                build = False

                if uid in record['properties']:
                    has_uid = True
                else:
                    has_uid = False

            # The geometry unique identifier may be present as a property. Otherwise the enumeration counter is used for
            # the identifier.
            if not union:
                if has_uid:
                    to_append = int(record['properties'][uid])
                else:
                    to_append = ctr
                deque_uid.append(to_append)

        # If we are unioning, the target geometry is not yet a sequence.
        if union:
            deque_geoms = [deque_geoms]

        # Dimension for the outgoing field.
        if union:
            size = 1
        else:
            size = ctr
        dim = Dimension(name=DimensionName.GEOMETRY_DIMENSION, size=size)

        # Set default geometry type if no schema is provided.
        if schema is None:
            geom_type = 'auto'
        else:
            geom_type = schema['geometry']

        geom = GeometryVariable(value=deque_geoms, geom_type=geom_type, dimensions=dim)
        uid = create_typed_variable_from_data_model('int', data_model=data_model, name=uid, value=deque_uid,
                                                    dimensions=dim)
        geom.set_ugid(uid)

        field = Field(geom=geom, crs=crs)

        # All records from a unioned geometry are not relevant.
        if not union:
            from ocgis.driver.vector import get_dtype_from_fiona_type, get_fiona_type_from_pydata

            if schema is None:
                has_schema = False
            else:
                has_schema = True

            for idx, record in enumerate(records):
                if idx == 0 and not has_schema:
                    schema = {'properties': OrderedDict()}
                    for k, v in list(record['properties'].items()):
                        schema['properties'][k] = get_fiona_type_from_pydata(v)
                if idx == 0:
                    for k, v in list(schema['properties'].items()):
                        if k == uid.name:
                            continue
                        dtype = get_dtype_from_fiona_type(v, data_model=data_model)
                        var = Variable(name=k, dtype=dtype, dimensions=dim)
                        if v.startswith('str:'):
                            var.set_string_max_length_global(value=int(v.split(':')[1]))
                        field.add_variable(var)
                for k, v in list(record['properties'].items()):
                    if k == uid.name:
                        continue

                    field[k].get_value()[idx] = v

        data_variables = [uid.name]
        if not union:
            data_variables += [k for k in list(schema['properties'].keys()) if k != uid.name]
        field.append_to_tags(TagName.DATA_VARIABLES, data_variables, create=True)

        return field

    @classmethod
    def from_variable_collection(cls, vc, *args, **kwargs):
        """Create a field from a variable collection.
        
        :param vc: The template variable collection.
        :type vc: :class:`~ocgis.VariableCollection`
        :rtype: :class:`~ocgis.Field`
        """

        if 'name' not in kwargs:
            kwargs['name'] = vc.name
        if 'source_name' not in kwargs:
            kwargs['source_name'] = vc.source_name
        kwargs['attrs'] = vc.attrs
        kwargs['parent'] = vc.parent
        kwargs['children'] = vc.children
        kwargs[KeywordArgument.UID] = vc.uid
        kwargs['variables'] = vc.values()
        if 'force' not in kwargs:
            kwargs['force'] = True
        ret = cls(*args, **kwargs)
        return ret

    def get_field_slice(self, dslice, strict=True, distributed=False):
        """
        Slice the field using a dictionary. Keys are dimension map standard names defined by 
        :class:`ocgis.constants.DimensionMapKey`. Dimensions are temporarily renamed for the duration of the slice.
        
        :param dict dslice: The dictionary slice.
        :param strict: If ``True`` (the default), any dimension names in ``dslice`` are required to be in the target
         field.
        :param bool distributed: If ``True``, this is should be considered a parallel/global slice.
        :return: A shallow copy of the sliced field.
        :rtype: :class:`~ocgis.Field`
        """

        name_mapping = get_name_mapping(self.dimension_map)
        with renamed_dimensions_on_variables(self, name_mapping) as mapping_meta:
            # When strict is False, we don't care about extra dimension names in the slice. This is useful for a general
            # slicing operation such as slicing for time with or without the dimension.
            if not strict:
                to_pop = [dname for dname in list(dslice.keys()) if dname not in self.dimensions]
                for dname in to_pop:
                    dslice.pop(dname)
            if distributed:
                data_variable = self.data_variables[0]
                data_variable_dimensions = data_variable.dimensions
                data_variable_dimension_names = get_dimension_names(data_variable_dimensions)
                the_slice = []
                for key in data_variable_dimension_names:
                    try:
                        the_slice.append(dslice[key])
                    except KeyError:
                        if strict:
                            raise
                        else:
                            the_slice.append(None)
            else:
                the_slice = dslice

            if distributed:
                ret = self.data_variables[0].get_distributed_slice(the_slice).parent
            else:
                ret = super(Field, self).__getitem__(the_slice)

        revert_renamed_dimensions_on_variables(mapping_meta, ret)
        return ret

    def get_report(self, should_print=False):
        """
        :param bool should_print: If ``True``, print the report lines in addition to returning them. 
        :return: A sequence of strings with descriptive field information.
        :rtype: :class:`list` of :class:`str`
        """

        field = self
        m = OrderedDict([['=== Realization ================', 'realization'],
                         ['=== Time =======================', 'time'],
                         ['=== Level ======================', 'level'],
                         ['=== Geometry ===================', 'geom'],
                         ['=== Grid =======================', 'grid']])
        lines = []
        for k, v in m.items():
            sub = [k, '']
            dim = getattr(field, v)
            if dim is None:
                sub.append('No {0} dimension/container/variable.'.format(v))
            else:
                sub += dim.get_report()
            sub.append('')
            lines += sub

        if should_print:
            for line in lines:
                print(line)

        return lines

    def iter(self, **kwargs):
        """
        :return: Yield record dictionaries for variables in the field applying standard names to dimensions by default.
        :rtype: dict
        """

        if self.is_empty:
            raise StopIteration

        from ocgis.driver.registry import get_driver_class

        standardize = kwargs.pop(KeywordArgument.STANDARDIZE, KeywordArgument.Defaults.STANDARDIZE)
        tag = kwargs.pop(KeywordArgument.TAG, TagName.DATA_VARIABLES)
        driver = kwargs.get(KeywordArgument.DRIVER)
        primary_mask = kwargs.pop(KeywordArgument.PRIMARY_MASK, None)
        header_map = kwargs.pop(KeywordArgument.HEADER_MAP, None)
        melted = kwargs.pop(KeywordArgument.MELTED, False)
        variable = kwargs.pop(KeywordArgument.VARIABLE, None)
        followers = kwargs.pop(KeywordArgument.FOLLOWERS, None)
        allow_masked = kwargs.get(KeywordArgument.ALLOW_MASKED, False)

        if melted and not standardize:
            raise ValueError('"standardize" must be True when "melted" is True.')

        if KeywordArgument.ALLOW_MASKED not in kwargs:
            kwargs[KeywordArgument.ALLOW_MASKED] = False

        if driver is not None:
            driver = get_driver_class(driver)

        # Holds follower variables to pass to the generic iterator.
        if followers is None:
            followers = []
        else:
            for ii, f in enumerate(followers):
                if not isinstance(f, Iterator):
                    followers[ii] = get_variables(f, self)[0]

        if variable is None:
            # The primary variable(s) to iterate.
            tagged_variables = self.get_by_tag(tag, create=True)
            if len(tagged_variables) == 0:
                msg = 'Tag "{}" has no associated variables. Nothing to iterate.'.format(tag)
                raise ValueError(msg)
            variable = tagged_variables[0]
            if len(tagged_variables) > 1:
                followers += tagged_variables[1:]
        else:
            variable = get_variables(variable, self)[0]

        if self.geom is not None:
            if primary_mask is None:
                primary_mask = self.geom
            if standardize:
                add_geom_uid = True
            else:
                add_geom_uid = False
            followers.append(self.geom.get_iter(**{KeywordArgument.ADD_GEOM_UID: add_geom_uid,
                                                   KeywordArgument.ALLOW_MASKED: allow_masked,
                                                   KeywordArgument.PRIMARY_MASK: primary_mask}))
            geom = self.geom
        else:
            geom = None

        if self.realization is not None:
            followers.append(self.realization.get_iter(driver=driver, allow_masked=allow_masked,
                                                       primary_mask=primary_mask))
        if self.time is not None:
            followers.append(self.time.get_iter(add_bounds=True, driver=driver, allow_masked=allow_masked,
                                                primary_mask=primary_mask))
        if self.level is not None:
            followers.append(self.level.get_iter(add_bounds=True, driver=driver, allow_masked=allow_masked,
                                                 primary_mask=primary_mask))

        # Collect repeaters from the target variable and followers. This initializes the iterator twice, but the
        # operation is not expensive.
        itr_for_repeaters = Iterator(variable, followers=followers)
        found = kwargs.get(KeywordArgument.REPEATERS)
        if found is not None:
            found = [ii[0] for ii in found]
        repeater_headers = itr_for_repeaters.get_repeaters(headers_only=True, found=found)

        if standardize:
            if header_map is None:
                header_map = OrderedDict()
                if len(repeater_headers) > 0:
                    for k in repeater_headers:
                        header_map[k] = k
                if self.geom is not None and self.geom.ugid is not None:
                    header_map[self.geom.ugid.name] = self.geom.ugid.name
                if self.realization is not None:
                    header_map[self.realization.name] = HeaderName.REALIZATION
                if self.time is not None:
                    header_map[self.time.name] = HeaderName.TEMPORAL
                    update_header_rename_bounds_names(HeaderName.TEMPORAL_BOUNDS, header_map, self.time)
                    header_map['YEAR'] = 'YEAR'
                    header_map['MONTH'] = 'MONTH'
                    header_map['DAY'] = 'DAY'
                if self.level is not None:
                    header_map[self.level.name] = HeaderName.LEVEL
                    update_header_rename_bounds_names(HeaderName.LEVEL_BOUNDS, header_map, self.level)

        if melted:
            melted = tagged_variables
        else:
            melted = None

        kwargs[KeywordArgument.HEADER_MAP] = header_map
        kwargs[KeywordArgument.MELTED] = melted
        kwargs[KeywordArgument.VARIABLE] = variable
        kwargs[KeywordArgument.FOLLOWERS] = followers
        kwargs[KeywordArgument.GEOM] = geom

        for yld in super(Field, self).iter(**kwargs):
            yield yld

    def iter_data_variables(self, tag_name=TagName.DATA_VARIABLES):
        """
        :param str tag_name: The tag to iterate. 
        :return: Yields variables associated with ``tag``.
        :rtype: :class:`~ocgis.Variable`
        """

        for var in self.get_by_tag(tag_name):
            yield var

    def iter_mapped(self, include_crs=False):
        for k, v in list(self.dimension_map.items()):
            if k == DimensionMapKey.CRS and not include_crs:
                continue
            else:
                yield k, getattr(self, k)

    def set_abstraction_geom(self, force=True, create_ugid=False, ugid_name=HeaderName.ID_GEOMETRY, ugid_start=1,
                             set_ugid_as_data=False):
        """
        Set the abstraction geometry for the field using the field's geometry variable or the field's grid abstraction
        geometry.
        
        :param bool force: If ``True`` (the default), clobber any existing geometry variables.
        :param bool create_ugid: If ``True``, create a unique identifier integer :class:`~ocgis.Variable` for the 
         abstraction geometry. Only creates the variable if the geometry does not already have a ``ugid``.
        :param str ugid_name: Name for the ``ugid`` variable.
        :param int ugid_start: Starting value to use for the unique identifier.
        :param bool set_ugid_as_data: If ``True``, set the ``ugid`` variable as data on the field. Useful for writing
         shapefiles which require at least one data variable.
        :raises: ValueError
        """

        if self.geom is None:
            if self.grid is None:
                raise ValueError('No grid available to set abstraction geometry.')
            else:
                self.set_geom_from_grid(force=force)
        if not is_xarray(self.geom):
            if self.geom.ugid is None and create_ugid:
                ocgis_lh(msg='before self.geom.create_ugid_global in {}'.format(self.__class__), level=logging.DEBUG)
                self.geom.create_ugid_global(ugid_name, start=ugid_start)
                ocgis_lh(msg='after self.geom.create_ugid_global in {}'.format(self.__class__), level=logging.DEBUG)
            if set_ugid_as_data:
                self.add_variable(self.geom.ugid, force=True, is_data=True)

    def set_crs(self, value, force=True, should_add=True):
        """
        Set the field's coordinate reference system. If coordinate system is already present on the field. Remove this
        variable.
        
        :param value: The coordinate reference system variable or ``None``.
        :type value: :class:`~ocgis.variable.crs.AbstractCRS` | ``None``
        :param force: See :meth:`~ocgis.VariableCollection.add_variable`
        :param bool should_add: If ``True``, add the variable to the field object. If ``False``, do not add the
         variable to the field variable storage. This is useful for updating metadata on the dimension map only.
        """

        if self.crs is not None:
            self.pop(self.crs.name)
        if value is not None:
            if should_add:
                self.add_variable(value, force=force)
            # tdk: FEATURE: format_spatial_object should be implemented on driver
            try:
                value.format_spatial_object(self)
            except:
                import xarray as xr
                if isinstance(value, xr.DataArray):
                    pass
                else:
                    raise
        self.dimension_map.set_crs(value)

    def set_driver(self, driver):
        self.dimension_map.set_driver(driver)

    def set_element_node_connectivity(self, value, force=True, should_add=True):
        """
        Set the element node connectivity variable. This variable maps coordinate values to element nodes using an
        index.

        :param value: The element node connectivity variable.
        :type value: :class:`~ocgis.Variable`
        :param bool force: See :meth:`~ocgis.VariableCollection.add_variable`.
        :param bool should_add: If ``True`` (the default), add the variable to collection.
        """
        set_field_property(self, DimensionMapKey.ELEMENT_NODE_CONNECTIVITY, value, force, dimension=value.dimensions[0],
                           should_add=should_add)

    def set_geom(self, variable, crs='auto', force=True, dimensionless='auto', should_add=True):
        """
        Set the field's geometry variable. 

        :param variable: The geometry variable or ``None``.
        :type variable: :class:`~ocgis.GeometryVariable` | ``None``
        :param crs: If ``'auto'`` (the default), use the coordinate system of the incoming geometry variable.
        :param bool force: If ``True`` (the default), clobber any existing geometry variable.
        :param bool dimensionless: If ``'auto'``, automatically determine dimensionless state for the variable. See
         :meth:`~ocgis.Dimension.set_variable`.
        :param bool should_add: If ``True``, add the variable to the field object. If ``False``, do not add the
         variable to the field variable storage. This is useful for updating metadata on the dimension map only.
        :raises: ValueError
        """
        if dimensionless == 'auto':
            if variable.ndim > 1:
                dimensionless = True
            else:
                dimensionless = False

        if crs != 'auto':
            self.set_crs(crs, should_add=should_add)
        else:
            if not is_xarray(variable):
                variable_crs = variable.crs
                if crs != 'auto':
                    if variable_crs is not None and variable_crs != crs:
                        raise ValueError('Geometry and field do not have matching coordinate reference systems.')
                else:
                    self.set_crs(variable_crs, should_add=should_add)
        set_field_property(self, DimensionMapKey.GEOM, variable, force, dimensionless=dimensionless,
                           should_add=should_add)

    def set_grid(self, grid, crs='auto', force=True, should_add=True):
        """
        Set the field's grid.

        :param grid: The grid object. If ``'auto'``, pass-through.
        :type grid: :class:`~ocgis.Grid` | None | str
        :param crs: If ``'auto'`` (the default), use the coordinate system of the incoming grid object.
        :param bool force: If ``True`` (the default), clobber any existing grid member variables.
        :param bool should_add: If ``True``, add the variable to the field object. If ``False``, do not add the
         variable to the field variable storage. This is useful for updating metadata on the dimension map only.
        """
        if grid is None:
            raise ValueError("'grid' may not be None.")
        if grid != 'auto':
            if crs != 'auto' and grid.crs is not None and crs != grid.crs:
                raise ValueError('Grid and field coordinate systems do not match.')

            if should_add:
                for v in grid.parent.values():
                    self.add_variable(v, force=force)
            self.dimension_map.update(grid.dimension_map)

    def set_geom_from_grid(self, force=True):
        """
        Set the field's geometry from its grid's abstraction geometry.
        
        :param bool force: If ``True`` (the default), clobber any existing geometry variables. 
        """

        new_geom = self.grid.get_abstraction_geometry()
        self.set_geom(new_geom, force=force)

    def set_level(self, variable, force=True, should_add=True):
        """
        Set the field's level variable.

        :param variable: The variable to use.
        :type variable: :class:`~ocgis.TemporalVariable` | ``None``
        :param force: See :meth:`~ocgis.VariableCollection.add_variable`
        :param bool should_add: If ``True``, add the variable to the field object. If ``False``, do not add the
         variable to the field variable storage. This is useful for updating metadata on the dimension map only.
        """

        set_field_property(self, DimensionMapKey.LEVEL, variable, force, dimension=None, should_add=should_add)

    def set_level_repr(self, variable, force=True, should_add=True):
        """
        Set the field's representative level variable.

        :param variable: The variable to use.
        :type variable: :class:`~ocgis.TemporalVariable` | ``None``
        :param force: See :meth:`~ocgis.VariableCollection.add_variable`
        :param bool should_add: If ``True``, add the variable to the field object. If ``False``, do not add the
         variable to the field variable storage. This is useful for updating metadata on the dimension map only.
        """

        set_field_property(self, DimensionMapKey.LEVEL_REPR, variable, force, dimension=None, should_add=should_add)

    def set_realization(self, variable, force=True, should_add=True):
        """
        Set the field's realization variable.

        :param variable: The variable to use.
        :type variable: :class:`~ocgis.TemporalVariable` | ``None``
        :param force: See :meth:`~ocgis.VariableCollection.add_variable`
        :param bool should_add: If ``True``, add the variable to the field object. If ``False``, do not add the
         variable to the field variable storage. This is useful for updating metadata on the dimension map only.
        """

        set_field_property(self, DimensionMapKey.REALIZATION, variable, force, dimension=None, should_add=should_add)

    def set_time(self, variable, force=True, should_add=True):
        """
        Set the field's time variable.

        :param variable: The variable to use.
        :type variable: :class:`~ocgis.TemporalVariable` | ``None``
        :param force: See :meth:`~ocgis.VariableCollection.add_variable`
        :param bool should_add: If ``True``, add the variable to the field object. If ``False``, do not add the
         variable to the field variable storage. This is useful for updating metadata on the dimension map only.
        """

        set_field_property(self, DimensionMapKey.TIME, variable, force, dimension=None, should_add=should_add)

    def set_x(self, variable, dimension, force=True, should_add=True):
        """
        Set the field's x-coordinate variable.

        :param variable: The source variable.
        :type variable: :class:`~ocgis.Variable`
        :param dimension: The representative field dimension for the variable. Required as the representative dimension
         cannot be determined with greater than one dimension on the coordinate variable.
        :type dimension: :class:`~ocgis.Dimension`
        :param bool force: If ``True`` (the default), clobber any existing geometry variables.
        :param bool should_add: If ``True``, add the variable to the field object. If ``False``, do not add the
         variable to the field variable storage. This is useful for updating metadata on the dimension map only.
        """

        set_field_property(self, DimensionMapKey.X, variable, force, dimension=dimension, should_add=should_add)

    def set_y(self, variable, dimension, force=True, should_add=True):
        """
        Set the field's y-coordinate variable.

        :param variable: The source variable.
        :type variable: :class:`~ocgis.Variable`
        :param dimension: The representative field dimension for the variable. Required as the representative dimension
         cannot be determined with greater than one dimension on the coordinate variable.
        :type dimension: :class:`~ocgis.Dimension`
        :param force: See :meth:`~ocgis.VariableCollection.add_variable`
        :param bool should_add: If ``True``, add the variable to the field object. If ``False``, do not add the
         variable to the field variable storage. This is useful for updating metadata on the dimension map only.
        """

        set_field_property(self, DimensionMapKey.Y, variable, force, dimension=dimension, should_add=should_add)

    def to_xarray(self, **kwargs):
        """
        Convert the field to a :class:`xarray.Dataset` with CF metadata interpretation.

        Limitations:
        * Bounds are treated as data arrays inside the ``xarray`` dataset.
        * Integer masked arrays are upcast to float data types in ``xarray``.
        * Group hierarchies are not supported in ``xarray``.

        :keyword bool decode_cf: (``=True``) If ``True``, run the ``xarray`` function ``decode_cf`` on the returned
         dataset.
        :param dict kwargs: Optional keyword arguments to dataset creation. See :meth:`ocgis.VariableCollection.to_xarray`
         for additional information.
        :rtype: :class:`xarray.Dataset`
        """
        from xarray import decode_cf

        kwargs = kwargs.copy()
        should_decode_cf = kwargs.pop('decode_cf', True)

        ret = super(Field, self).to_xarray(**kwargs)
        if should_decode_cf:
            ret = decode_cf(ret)
        return ret

    def unwrap(self):
        """
        Unwrap the field's coordinates contained in its grid and/or geometry.
        
        :raises: :class:`~ocgis.exc.EmptyObjectError` 
        """

        raise_if_empty(self)

        wrap_or_unwrap(self, WrapAction.UNWRAP)

    def update_crs(self, to_crs, from_crs=None):
        """
        See :meth:`ocgis.spatial.base.AbstractOperationsSpatialObject.update_crs`
        """

        raise_if_empty(self)

        if from_crs is None:
            from_crs = self.crs

        if self.grid is not None:
            self.grid.update_crs(to_crs, from_crs=from_crs)
        if self.geom is not None:
            self.geom.update_crs(to_crs, from_crs=from_crs)

        self.dimension_map.set_crs(to_crs)

    def wrap(self, inplace=True):
        """
        Wrap the field's coordinates contained in its grid and/or geometry.

        :raises: :class:`~ocgis.exc.EmptyObjectError` 
        """

        wrap_or_unwrap(self, WrapAction.WRAP, inplace=inplace)

    @classmethod
    def read(cls, *args, **kwargs):
        raise NotImplementedError("Use request dataset 'get' method.")

    def write(self, *args, **kwargs):
        """
        See :meth:`ocgis.VariableCollection.write`.

        .. note:: If no ``driver`` is provided, then the field's dimension map driver will be used.
        """
        from ocgis.driver.registry import get_driver_class

        to_load = (DimensionMapKey.REALIZATION, DimensionMapKey.TIME, DimensionMapKey.LEVEL, DimensionMapKey.Y,
                   DimensionMapKey.X)

        # Attempt to load all instrumented dimensions once. Do not do this for the geometry variable. This is done to
        # ensure proper attributes are applied to dimension variables before writing.
        for k in to_load:
            getattr(self, k)

        driver = kwargs.pop('driver', None)
        if driver is None:
            driver = self.dimension_map.get_driver(as_class=True)
        else:
            driver = get_driver_class(driver)
        args = list(args)
        args.insert(0, self)
        return driver.write_field(*args, **kwargs)


def get_field_property(field, name, strict=False):
    variable = field.dimension_map.get_variable(name)
    bounds = field.dimension_map.get_bounds(name)
    if variable is None:
        ret = None
    else:
        try:
            ret = field[variable]
        except KeyError:
            if strict:
                raise
            else:
                ret = None
        if ret is not None:
            ret.attrs.update(field.dimension_map.get_attrs(name))
            if bounds is not None:
                try:
                    ret.set_bounds(field.get(bounds), force=True)
                except AttributeError:
                    # tdk: FEATURE: bounds management should be handled by the driver i guess
                    import xarray as xr
                    if isinstance(ret, xr.DataArray):
                        ret.attrs['bounds'] = bounds
    return ret


def get_name_mapping(dimension_map):
    name_mapping = {}
    to_slice = [DimensionMapKey.REALIZATION, DimensionMapKey.TIME, DimensionMapKey.LEVEL, DimensionMapKey.Y,
                DimensionMapKey.X]

    # Slice the geometry variable as well if the dimension length is one. Not sure entirely why this is in there.
    if len(dimension_map.get_dimension(DimensionMapKey.GEOM)) == 1:
        to_slice.append(DimensionMapKey.GEOM)

    for k in to_slice:
        variable_name = dimension_map.get_variable(k)
        if variable_name is not None:
            dimension_names = dimension_map.get_dimension(k)
            # Use the variable name if there are no dimension names available.
            if len(dimension_names) == 0:
                dimension_name = variable_name
            else:
                dimension_name = dimension_names[0]

            if dimension_name not in dimension_names:
                dimension_names.append(dimension_name)
            name_mapping[k] = dimension_names
    return name_mapping


def set_field_property(field, dmap_key, variable, force, dimension=None, dimensionless=False, nullable=True,
                       should_add=True):
    dimension_map = field.dimension_map
    curr = dimension_map.get_variable(dmap_key)
    # Remove the variable if it exists in the field. The "in field" check is needed when the field initialization has an
    # incoming dimension map so this function thinks the variable should exist in the field.
    if should_add and curr is not None and curr in field:
        if id(field[curr]) != id(variable):
            field.remove_variable(curr)
    if variable is None:
        if not nullable:
            raise ValueError("'variable' is None and the property has 'nullable=False'")
    else:
        # Allow string variable names to be passed.
        if should_add and not isinstance(variable, six.string_types):
            field.add_variable(variable, force=force)
    update_dimension_map_with_variable(dimension_map, dmap_key, variable, dimension, dimensionless=dimensionless)


def update_dimension_map_with_variable(dimension_map, key, variable, dimension, dimensionless=False):
    dimension_map.set_variable(key, variable, dimension=dimension, dimensionless=dimensionless)


def update_header_rename_bounds_names(bounds_name_desired, header_rename, variable):
    if variable.has_bounds:
        for ii, original_bounds_name in enumerate(get_bounds_names_1d(variable.name)):
            header_rename[original_bounds_name] = bounds_name_desired[ii]


def wrap_or_unwrap(field, action, inplace=True):
    if action not in (WrapAction.WRAP, WrapAction.UNWRAP):
        raise ValueError('"action" not recognized: {}'.format(action))

    if field.grid is not None:
        if not inplace:
            x = field.grid.x
            x.set_value(x.value.copy())
        if action == WrapAction.WRAP:
            field.grid.wrap()
        else:
            field.grid.unwrap()
    elif field.geom is not None:
        if not inplace:
            geom = field.geom
            geom.set_value(geom.value.copy())
        if action == WrapAction.WRAP:
            field.geom.wrap()
        else:
            field.geom.unwrap()
    else:
        raise ValueError('No grid or geometry to wrap/unwrap.')

    # Bounds are not handled by wrap/unwrap operations. They should be removed from the dimension map if present.
    if field.grid is not None:
        for key in [DimensionMapKey.X, DimensionMapKey.Y]:
            field.dimension_map.set_bounds(key, None)
