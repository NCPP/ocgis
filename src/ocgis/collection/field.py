from collections import OrderedDict
from collections import deque
from copy import deepcopy

from shapely.geometry import shape

from ocgis import env
from ocgis.base import get_dimension_names, get_variable_names, get_variables, renamed_dimensions_on_variables, \
    revert_renamed_dimensions_on_variables
from ocgis.constants import DimensionMapKeys, WrapAction, TagNames, HeaderNames, DimensionNames, UNINITIALIZED, \
    KeywordArguments
from ocgis.spatial.grid import GridXY
from ocgis.util.helpers import get_iter
from ocgis.variable.base import VariableCollection, Variable, get_bounds_names_1d
from ocgis.variable.crs import CoordinateReferenceSystem
from ocgis.variable.dimension import Dimension
from ocgis.variable.geom import GeometryVariable
from ocgis.variable.iterator import Iterator
from ocgis.variable.temporal import TemporalGroupVariable, TemporalVariable

_DIMENSION_MAP = OrderedDict()
_DIMENSION_MAP['realization'] = {'attrs': {'axis': 'R'}, 'variable': None, 'names': []}
_DIMENSION_MAP['time'] = {'attrs': {'axis': 'T'}, 'variable': None, 'bounds': None, 'names': []}
_DIMENSION_MAP['level'] = {'attrs': {'axis': 'Z'}, 'variable': None, 'bounds': None, 'names': []}
_DIMENSION_MAP['y'] = {'attrs': {'axis': 'Y'}, 'variable': None, 'bounds': None, 'names': []}
_DIMENSION_MAP['x'] = {'attrs': {'axis': 'X'}, 'variable': None, 'bounds': None, 'names': []}
_DIMENSION_MAP['geom'] = {'attrs': {'axis': 'ocgis_geom'}, 'variable': None, 'names': []}
_DIMENSION_MAP['crs'] = {'attrs': None, 'variable': None}


class OcgField(VariableCollection):
    def __init__(self, *args, **kwargs):
        dimension_map = deepcopy(kwargs.pop('dimension_map', None))
        self.dimension_map = get_merged_dimension_map(dimension_map)

        # Flag updated by driver to indicate if the coordinate system is assigned or implied.
        self._has_assigned_coordinate_system = False
        # Flag to indicate if this is a regrid destination.
        self.regrid_destination = kwargs.pop('regrid_destination', False)
        # Flag to indicate if this is a regrid source.
        self.regrid_source = kwargs.pop('regrid_source', True)

        # Add grid variable metadata to dimension map.
        grid = kwargs.pop('grid', None)
        if grid is not None:
            update_dimension_map_with_variable(self.dimension_map, DimensionMapKeys.X, grid.x, grid.dimensions[1])
            update_dimension_map_with_variable(self.dimension_map, DimensionMapKeys.Y, grid.y, grid.dimensions[0])
        # Add realization variable metadata to dimension map.
        rvar = kwargs.pop('realization', None)
        if rvar is not None:
            update_dimension_map_with_variable(self.dimension_map, DimensionMapKeys.REALIZATION, rvar,
                                               rvar.dimensions[0])
        # Add time variable metadata to dimension map.
        tvar = kwargs.pop('time', None)
        if tvar is not None:
            update_dimension_map_with_variable(self.dimension_map, DimensionMapKeys.TIME, tvar, tvar.dimensions[0])
        # Add level variable metadata to dimension map.
        lvar = kwargs.pop('level', None)
        if lvar is not None:
            update_dimension_map_with_variable(self.dimension_map, DimensionMapKeys.LEVEL, lvar, lvar.dimensions[0])
        # Add the coordinate system.
        crs = kwargs.pop('crs', None)
        if crs is not None:
            update_dimension_map_with_variable(self.dimension_map, DimensionMapKeys.CRS, crs, None)
        # Add the geometry variable.
        geom = kwargs.pop('geom', None)
        if geom is not None:
            update_dimension_map_with_variable(self.dimension_map, DimensionMapKeys.GEOM, geom, None)

        self.grid_abstraction = kwargs.pop('grid_abstraction', 'auto')
        if self.grid_abstraction is None:
            raise ValueError('"grid_abstraction" may not be None.')

        self.format_time = kwargs.pop('format_time', True)

        # Use tags to set data variables.
        is_data = kwargs.pop(KeywordArguments.IS_DATA, [])

        VariableCollection.__init__(self, *args, **kwargs)

        # Append the data variable tagged variable names.
        is_data = list(get_iter(is_data, dtype=Variable))
        is_data_variable_names = get_variable_names(is_data)
        for idvn in is_data_variable_names:
            self.append_to_tags(TagNames.DATA_VARIABLES, idvn, create=True)
        for idx, dvn in enumerate(is_data_variable_names):
            if dvn not in self:
                if isinstance(is_data[idx], Variable):
                    self.add_variable(is_data[idx])

        # Add grid variables to the variable collection.
        if grid is not None:
            for var in list(grid.parent.values()):
                self.add_variable(var, force=True)
        if tvar is not None:
            self.add_variable(tvar, force=True)
        if rvar is not None:
            self.add_variable(rvar, force=True)
        if lvar is not None:
            self.add_variable(lvar, force=True)
        if crs is not None:
            self.add_variable(crs, force=True)
        if geom is not None:
            self.add_variable(geom, force=True)

    @property
    def _should_regrid(self):
        raise NotImplementedError

    @property
    def axes_shapes(self):
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
        ret['L'] = l
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
    def crs(self):
        return get_field_property(self, 'crs')

    @property
    def data_variables(self):
        try:
            ret = tuple(self.get_by_tag(TagNames.DATA_VARIABLES))
        except KeyError:
            ret = tuple()
        return ret

    @property
    def realization(self):
        return get_field_property(self, 'realization')

    @property
    def temporal(self):
        return self.time

    @property
    def time(self):
        ret = get_field_property(self, 'time')
        if ret is not None:
            if not isinstance(ret, TemporalGroupVariable):
                ret = TemporalVariable.from_variable(ret, format_time=self.format_time)
        return ret

    @property
    def wrapped_state(self):
        if self.crs is None:
            ret = None
        else:
            ret = self.crs.get_wrapped_state(self)
        return ret

    @property
    def level(self):
        return get_field_property(self, 'level')

    @property
    def y(self):
        return get_field_property(self, 'y')

    @property
    def x(self):
        return get_field_property(self, 'x')

    @property
    def grid(self):
        x = self.x
        y = self.y
        if x is None or y is None:
            ret = None
        else:
            ret = GridXY(self.x, self.y, parent=self, crs=self.crs, abstraction=self.grid_abstraction)
        return ret

    @property
    def geom(self):
        ret = get_field_property(self, 'geom')
        if ret is not None:
            crs = self.crs
            # Overload the geometry coordinate system if set on the field. Otherwise, this will use the coordinate
            # system on the geometry variable.
            if crs is not None:
                ret.crs = crs
        return ret

    @property
    def has_data_variables(self):
        if len(self.data_variables) > 0:
            ret = True
        else:
            ret = False
        return ret

    def add_variable(self, variable, force=False, is_data=False):
        super(OcgField, self).add_variable(variable, force=force)
        if is_data:
            tagged = get_variable_names(self.get_by_tag(TagNames.DATA_VARIABLES, create=True))
            if variable.name not in tagged:
                self.append_to_tags(TagNames.DATA_VARIABLES, variable.name)

    def copy(self):
        ret = super(OcgField, self).copy()
        # Changes to a field's shallow copy should be able to adjust attributes in the dimension map as needed.
        ret.dimension_map = deepcopy(ret.dimension_map)
        return ret

    @classmethod
    def from_records(cls, records, schema=None, crs=UNINITIALIZED, uid=None, union=False):
        """
        Create a :class:`ocgis.interface.base.dimension.SpatialDimension` from Fiona-like records.

        :param records: A sequence of records returned from an Fiona file object.
        :type records: sequence
        :param schema: A Fiona-like schema dictionary. If ``None`` and any records properties are ``None``, then this
         must be provided.
        :type schema: dict

        >>> schema = {'geometry': 'Point', 'properties': {'UGID': 'int', 'NAME', 'str:4'}}

        :param crs: If :attr:`ocgis.constants.UNINITIALIZED`, default to :attr:`ocgis.env.DEFAULT_COORDSYS`.
        :type crs: dict or :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
        :param str uid: If provided, use this attribute name as the unique identifier. Otherwise search for
         :attr:`env.DEFAULT_GEOM_UID` and, if not present, construct a 1-based identifier with this name.
        :param bool union: If ``True``, union the geometries from records yielding a single geometry with a unique
         identifier value of ``1``.
        :returns: Field object constructed from records.
        :rtype: :class:`ocgis.new_interface.field.OcgField`
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
        dim = Dimension(name=DimensionNames.GEOMETRY_DIMENSION, size=size)

        # Set default geometry type if no schema is provided.
        if schema is None:
            geom_type = 'auto'
        else:
            geom_type = schema['geometry']

        geom = GeometryVariable(value=deque_geoms, geom_type=geom_type, dimensions=dim)
        uid = Variable(name=uid, value=deque_uid, dimensions=dim)
        geom.set_ugid(uid)

        field = OcgField(geom=geom, crs=crs)

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
                        dtype = get_dtype_from_fiona_type(v)
                        var = Variable(name=k, dtype=dtype, dimensions=dim)
                        field.add_variable(var)
                for k, v in list(record['properties'].items()):
                    if k == uid.name:
                        continue

                    field[k].get_value()[idx] = v

        data_variables = [uid.name]
        if not union:
            data_variables += [k for k in list(schema['properties'].keys()) if k != uid.name]
        field.append_to_tags(TagNames.DATA_VARIABLES, data_variables, create=True)

        return field

    @classmethod
    def from_variable_collection(cls, vc, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = vc.name
        if 'source_name' not in kwargs:
            kwargs['source_name'] = vc.source_name
        kwargs['attrs'] = vc.attrs
        kwargs['parent'] = vc.parent
        kwargs['children'] = vc.children
        kwargs['uid'] = vc.uid
        ret = cls(*args, **kwargs)
        for v in list(vc.values()):
            ret.add_variable(v, force=True)
        return ret

    def get_field_slice(self, dslice, strict=True, distributed=False):
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
                ret = super(OcgField, self).__getitem__(the_slice)

        revert_renamed_dimensions_on_variables(mapping_meta, ret)
        return ret

    def get_report(self, should_print=False):
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
        if self.is_empty:
            raise StopIteration

        from ocgis.driver.registry import get_driver_class

        standardize = kwargs.pop(KeywordArguments.STANDARDIZE, KeywordArguments.Defaults.STANDARDIZE)
        tag = kwargs.pop(KeywordArguments.TAG, TagNames.DATA_VARIABLES)
        driver = kwargs.get(KeywordArguments.DRIVER)
        primary_mask = kwargs.pop(KeywordArguments.PRIMARY_MASK, None)
        header_map = kwargs.pop(KeywordArguments.HEADER_MAP, None)
        melted = kwargs.pop(KeywordArguments.MELTED, False)
        variable = kwargs.pop(KeywordArguments.VARIABLE, None)
        followers = kwargs.pop(KeywordArguments.FOLLOWERS, None)
        allow_masked = kwargs.get(KeywordArguments.ALLOW_MASKED, False)

        if melted and not standardize:
            raise ValueError('"standardize" must be True when "melted" is True.')

        if KeywordArguments.ALLOW_MASKED not in kwargs:
            kwargs[KeywordArguments.ALLOW_MASKED] = False

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
            followers.append(self.geom.get_iter(**{KeywordArguments.ADD_GEOM_UID: add_geom_uid,
                                                   KeywordArguments.ALLOW_MASKED: allow_masked,
                                                   KeywordArguments.PRIMARY_MASK: primary_mask}))
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
        found = kwargs.get(KeywordArguments.REPEATERS)
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
                    header_map[self.realization.name] = HeaderNames.REALIZATION
                if self.time is not None:
                    header_map[self.time.name] = HeaderNames.TEMPORAL
                    update_header_rename_bounds_names(HeaderNames.TEMPORAL_BOUNDS, header_map, self.time)
                    header_map['YEAR'] = 'YEAR'
                    header_map['MONTH'] = 'MONTH'
                    header_map['DAY'] = 'DAY'
                if self.level is not None:
                    header_map[self.level.name] = HeaderNames.LEVEL
                    update_header_rename_bounds_names(HeaderNames.LEVEL_BOUNDS, header_map, self.level)

        if melted:
            melted = tagged_variables
        else:
            melted = None

        kwargs[KeywordArguments.HEADER_MAP] = header_map
        kwargs[KeywordArguments.MELTED] = melted
        kwargs[KeywordArguments.VARIABLE] = variable
        kwargs[KeywordArguments.FOLLOWERS] = followers
        kwargs[KeywordArguments.GEOM] = geom

        for yld in super(OcgField, self).iter(**kwargs):
            yield yld

    def iter_data_variables(self, tag_name=TagNames.DATA_VARIABLES):
        for var in self.get_by_tag(tag_name):
            yield var

    def iter_mapped(self, include_crs=False):
        for k, v in list(self.dimension_map.items()):
            if k == DimensionMapKeys.CRS and not include_crs:
                continue
            else:
                yield k, getattr(self, k)

    def set_abstraction_geom(self, force=True, create_ugid=False, ugid_name=HeaderNames.ID_GEOMETRY, comm=None,
                             ugid_start=1, set_ugid_as_data=False):
        """Collective!"""
        if self.geom is None:
            if self.grid is None:
                raise ValueError('No grid available to set abstraction geometry.')
            else:
                self.set_geom_from_grid(force=force)
        if self.geom.ugid is None and create_ugid:
            self.geom.create_ugid_global(ugid_name, comm=comm, start=ugid_start)
        if set_ugid_as_data:
            self.add_variable(self.geom.ugid, force=True, is_data=True)

    def set_crs(self, value):
        if self.crs is not None:
            self.pop(self.crs.name)
        if value is not None:
            variable_name = value.name
            self.add_variable(value)
        else:
            variable_name = None
        self.dimension_map[DimensionMapKeys.CRS][DimensionMapKeys.VARIABLE] = variable_name

    def set_geom(self, variable, force=True):
        if variable is None:
            self.dimension_map[DimensionMapKeys.GEOM] = None
        else:
            dmap_entry = self.dimension_map[DimensionMapKeys.GEOM]
            dmap_entry[DimensionMapKeys.VARIABLE] = variable.name
            if variable.crs != self.crs and not self.is_empty:
                raise ValueError('Geometry and field do not have matching coordinate reference systems.')
            self.add_variable(variable, force=force)

    def set_grid(self, grid, force=True):
        for member in grid.get_member_variables():
            self.add_variable(member, force=force)
        self.grid_abstraction = grid.abstraction
        self.set_x(grid.x, grid.dimensions[1])
        self.set_y(grid.y, grid.dimensions[0])

    def set_geom_from_grid(self, force=True):
        new_geom = self.grid.get_abstraction_geometry()
        self.set_geom(new_geom, force=force)

    def set_time(self, variable):
        if self.time is not None:
            time_bounds = self.time.bounds
            if time_bounds is not None:
                bounds_dimension_name = time_bounds.dimensions[1].name
            time_name = self.time.name
            self.dimensions.pop(self.time.dimensions[0].name)
            self.pop(time_name)
            if time_bounds is not None:
                self.dimensions.pop(bounds_dimension_name)
                self.pop(time_bounds.name)
        self.add_variable(variable)
        self.dimension_map[DimensionMapKeys.TIME][DimensionMapKeys.VARIABLE] = variable.name
        if variable.has_bounds:
            self.dimension_map[DimensionMapKeys.TIME][DimensionMapKeys.BOUNDS] = variable.bounds.name

    def set_x(self, variable, dimension, force=True):
        self.add_variable(variable, force=force)
        update_dimension_map_with_variable(self.dimension_map, DimensionMapKeys.X, variable, dimension)

    def set_y(self, variable, dimension, force=True):
        self.add_variable(variable, force=force)
        update_dimension_map_with_variable(self.dimension_map, DimensionMapKeys.Y, variable, dimension)

    def unwrap(self):
        wrap_or_unwrap(self, WrapAction.UNWRAP)

    def update_crs(self, to_crs):
        if self.grid is not None:
            self.grid.update_crs(to_crs)
        else:
            self.geom.update_crs(to_crs)
        self.dimension_map[DimensionMapKeys.CRS]['variable'] = to_crs.name

    def wrap(self, inplace=True):
        wrap_or_unwrap(self, WrapAction.WRAP, inplace=inplace)

    def write(self, *args, **kwargs):
        from ocgis.driver.nc import DriverNetcdfCF
        from ocgis.driver.registry import get_driver_class

        # Attempt to load all instrumented dimensions once. Do not do this for the geometry variable. This is done to
        # ensure proper attributes are applied to dimension variables before writing.
        for k in list(self.dimension_map.keys()):
            if k != 'geom':
                getattr(self, k)

        driver = kwargs.pop('driver', DriverNetcdfCF)
        driver = get_driver_class(driver, default=driver)
        args = list(args)
        args.insert(0, self)
        return driver.write_field(*args, **kwargs)


def get_field_property(field, name, strict=False):
    variable = field.dimension_map[name]['variable']
    bounds = field.dimension_map[name].get('bounds')
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
        if not isinstance(ret, CoordinateReferenceSystem) and ret is not None:
            ret.attrs.update(field.dimension_map[name]['attrs'])
            if bounds is not None:
                ret.set_bounds(field.get(bounds), force=True)
    return ret


def get_merged_dimension_map(dimension_map):
    dimension_map_template = deepcopy(_DIMENSION_MAP)
    # Merge incoming dimension map with the template.
    if dimension_map is not None:
        for k, v in list(dimension_map.items()):
            # Groups in dimension maps don't matter to the target field. Each field keeps its own copy.
            if k == 'groups':
                continue
            for k2, v2, in list(v.items()):
                if k2 == 'attrs':
                    dimension_map_template[k][k2].update(v2)
                else:
                    dimension_map_template[k][k2] = v2
    return dimension_map_template


def get_name_mapping(dimension_map):
    name_mapping = {}
    for k, v in list(dimension_map.items()):
        # Do not slice the coordinate system variable.
        if k == DimensionMapKeys.CRS:
            continue
        variable_name = v[DimensionMapKeys.VARIABLE]
        if variable_name is not None:
            dimension_names = dimension_map[k][DimensionMapKeys.NAMES]
            # Use the variable name if there are no dimension names available.
            if len(dimension_names) == 0:
                dimension_name = dimension_map[k][DimensionMapKeys.VARIABLE]
            else:
                dimension_name = dimension_names[0]

            variable_names = v[DimensionMapKeys.NAMES]
            if dimension_name not in variable_names:
                variable_names.append(dimension_name)
            name_mapping[k] = variable_names
    return name_mapping


def update_dimension_map_with_variable(dimension_map, key, variable, dimension):
    r = dimension_map[key]
    r[DimensionMapKeys.VARIABLE] = variable.name
    if variable.has_bounds:
        r[DimensionMapKeys.BOUNDS] = variable.bounds.name
    if dimension is not None:
        for dimension in get_iter(dimension, dtype=Dimension):
            r[DimensionMapKeys.NAMES].append(dimension.name)


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
    for key in [DimensionMapKeys.X, DimensionMapKeys.Y]:
        field.dimension_map[key][DimensionMapKeys.BOUNDS] = None
