from copy import deepcopy

import six

from ocgis.base import AbstractOcgisObject
from ocgis.base import get_dimension_names
from ocgis.base import get_variable_names
from ocgis.constants import DIMENSION_MAP_TEMPLATE, DMK, DEFAULT_DRIVER, GridAbstraction
from ocgis.exc import DimensionMapError, VariableNotInCollection
from ocgis.util.helpers import pprint_dict, get_or_create_dict, get_formatted_slice, is_xarray


class DimensionMap(AbstractOcgisObject):
    """
    A dimension map is used to link dimensions and variables with an explicit meaning. It is the main mapping produced
    by a driver and a request dataset's metadata. Dimension maps are used by fields to construct grids and geometries,
    perform subsetting, link bounds to parent variables, and manage coordinate systems.
    """

    _allowed_entry_keys = DMK.get_entry_keys()
    _allowed_element_keys = DMK.get_element_keys()
    _special_entry_keys = DMK.get_special_entry_keys()

    def __init__(self):
        self._storage = {}

    def __eq__(self, other):
        ret = True
        if type(other) != self.__class__:
            ret = False
        elif self._storage != other._storage:
            ret = False
        return ret

    @property
    def has_topology(self):
        """
        Return ``True`` if the dimension map has topology entries.

        :rtype: bool
        """
        return DMK.TOPOLOGY in self._storage

    def as_dict(self, curr=None):
        """
        Convert the the dimension map to a dictionary.
        
        :rtype: dict
        """
        if curr is None:
            curr = deepcopy(self._storage)
        if DMK.GROUPS in curr:
            for group_name, group_dmap in curr[DMK.GROUPS].items():
                curr[DMK.GROUPS][group_name] = group_dmap.as_dict(curr=group_dmap._storage)
        if DMK.TOPOLOGY in curr:
            for k, v in curr[DMK.TOPOLOGY].items():
                curr[DMK.TOPOLOGY][k] = v.as_dict(curr=v._storage)
        return curr

    @classmethod
    def from_dict(cls, dct):
        """
        Create a dimension map from a well-formed dictionary.
        
        :param dict dct: The input dimension map-like dictionary.
        :rtype: :class:`~ocgis.DimensionMap`
        """
        d = DimensionMap()
        dct = deepcopy(dct)
        has_groups = False
        for k, v in dct.items():
            if k == DMK.GROUPS:
                has_groups = True
            elif k == DMK.DRIVER:
                d.set_driver(v)
            else:
                try:
                    variable = v.pop(DMK.VARIABLE)
                except KeyError:
                    raise DimensionMapError(k, "No 'variable' is available.")
                if k == DMK.CRS:
                    d.set_crs(variable)
                elif k == DMK.SPATIAL_MASK:
                    if v != {}:
                        d.set_spatial_mask(variable)
                else:
                    d.set_variable(k, variable, **v)
        if has_groups:
            for group_name, group_dct in dct[DMK.GROUPS].items():
                d.set_group(group_name, cls.from_dict(group_dct))
        return d

    @classmethod
    def from_metadata(cls, driver, group_metadata, group_name=None, curr=None):
        """
        Create a dimension map from source metadata.
        
        :param driver: The driver to use for metadata interpretation.
        :type driver: :class:`~ocgis.driver.base.AbstractDriver`
        :param dict group_metadata: Source metadata for the target group to convert recursively. 
        :param str group_name: The current group name. 
        :rtype: :class:`~ocgis.DimensionMap`
        """
        dimension_map = driver.create_dimension_map(group_metadata)
        if curr is None:
            curr = dimension_map
        if group_name is not None:
            curr.set_group(group_name, dimension_map)
        if DMK.GROUPS in group_metadata:
            for group_name, sub_group_metadata in group_metadata[DMK.GROUPS].items():
                cls.from_metadata(driver, sub_group_metadata, curr=curr, group_name=group_name)

        return curr

    @classmethod
    def from_old_style_dimension_map(cls, odmap):
        """
        Convert an old-style dimension map (pre-v2.x) to a new-style dimension map.
        
        :param dict odmap: The old-style dimension map to convert. 
        :rtype: :class:`~ocgis.DimensionMap`
        """

        ret = cls()
        for k, v in odmap.items():
            entry_key = DMK.get_axis_mapping()[k]
            bounds = v.get(DMK.BOUNDS)
            dimension = v.get(DMK.DIMENSION)
            variable = v.get(DMK.VARIABLE)
            ret.set_variable(entry_key, variable, dimension=dimension, bounds=bounds)
        return ret

    def get_attrs(self, entry_key):
        """
        Get attributes for the dimension map entry ``entry_key``.
        
        :param str entry_key: See :class:`ocgis.constants.DimensionMapKey` for valid entry keys.
        :rtype: :class:`~collections.OrderedDict` 
        """
        return self._get_element_(entry_key, DMK.ATTRS, self._storage.__class__())

    def get_available_topologies(self):
        """
        Get a list of available topologies keys. Keys are of type :class:`ocgis.constants.Topology`. The returned tuple
        may be of zero length if no topologies are present on the dimension map.

        :rtype: tuple
        """
        if not self.has_topology:
            ret = tuple()
        else:
            topologies = self._storage.get(DMK.TOPOLOGY)
            ret = topologies.keys()
        return ret

    def get_bounds(self, entry_key):
        """
        Get the bounds variable name for the dimension map entry ``entry_key``.

        :param str entry_key: See :class:`ocgis.constants.DimensionMapKey` for valid entry keys.
        :rtype: str
        """
        return self._get_element_(entry_key, DMK.BOUNDS, None)

    def get_crs(self, parent=None, nullable=False):
        """
        Get the coordinate reference system variable name for the dimension map entry ``entry_key``.

        :rtype: str | :class:`~ocgis.CRS`
        """
        entry = self._get_entry_(DMK.CRS)
        ret = get_or_create_dict(entry, DMK.VARIABLE, None)
        if parent is not None:
            ret = get_variable_from_field(ret, parent, nullable)
        return ret

    def get_dimension(self, entry_key, dimensions=None):
        """
        Get the dimension names for the dimension map entry ``entry_key``.

        :param str entry_key: See :class:`ocgis.constants.DimensionMapKey` for valid entry keys.
        :param dict dimensions: A dictionary of dimension names (keys) and objects (values). If provided, a dimension
         object will be returned from this dictionary if the dimension name is present on the dimension map.
        :rtype: :class:`list` of :class:`str`
        """
        ret = self._get_element_(entry_key, DMK.DIMENSION, [])
        if dimensions is not None:
            if len(ret) > 0:
                ret = dimensions[ret[0]]
            else:
                ret = None
        return ret

    def get_driver(self, as_class=False):
        """
        Return the driver key or class associated with the dimension map.

        :param bool as_class: If ``True``, return the driver class instead of the driver string key.
        :rtype: str | :class:`ocgis.driver.base.AbstractDriver`
        """
        from ocgis.driver.registry import get_driver_class
        ret = self._storage.get(DMK.DRIVER, DEFAULT_DRIVER)
        if as_class:
            ret = get_driver_class(ret)
        return ret

    def get_grid_abstraction(self, default=GridAbstraction.AUTO):
        """
        Get the grid abstraction or, if absent on the dimension map, return ``default``.

        :param default: Default return value.
        :type default: :class:`ocgis.constants.GridAbstraction`
        :rtype: str | :class:`ocgis.constants.GridAbstraction`
        """
        return self._storage.get(DMK.GRID_ABSTRACTION, default)

    def get_group(self, group_key):
        """
        Get the dimension map for a group indexed by ``group_key`` starting from the root group.

        :param group_key: The group indexing key.
        :rtype: :class:`list` of :class:`str`
        """
        if DMK.GROUPS not in self._storage:
            self._storage[DMK.GROUPS] = {}
        try:
            return get_dmap_group(self, group_key)
        except KeyError:
            raise DimensionMapError(DMK.GROUPS, "Group key not found: {}".format(group_key))

    def get_property(self, key, default=None):
        """
        Return a dimension map property value.

        :param str key: The key name
        :param default: A default value to return if the key is not present
        :rtype: <varying>
        """
        return self._storage.get(key, default)

    def get_spatial_mask(self):
        """
        Get the spatial mask variable name.

        :rtype: str
        """

        entry = self._get_entry_(DMK.SPATIAL_MASK)
        return get_or_create_dict(entry, DMK.VARIABLE, None)

    def get_topology(self, topology, create=False):
        """
        Get a child dimension map for a given topology. If ``create`` is ``True``, the child dimension map will be
        created if it is not present on the dimension map. If create is ``False``, ``None`` will be returned if the
        topology does not exist.

        :param topology: The target topology to get or create.
        :type topology: :class:`ocgis.constants.Topology`
        :param bool create: Flag for creation behavior if the child dimension map does not exist.
        :rtype: :class:`~ocgis.DimensionMap` | ``None``
        """
        if create:
            default = self.__class__()
        else:
            default = None

        if DMK.TOPOLOGY not in self._storage:
            if create:
                self._storage[DMK.TOPOLOGY] = {}
                ret = self._storage[DMK.TOPOLOGY]
            else:
                ret = None
        else:
            ret = self._storage[DMK.TOPOLOGY]

        if ret is not None:
            ret = self._storage[DMK.TOPOLOGY].get(topology, default)
            self._storage[DMK.TOPOLOGY][topology] = ret

        # if ret is None:
        #     ret = GridAbstraction.AUTO
        return ret

    def get_variable(self, entry_key, parent=None, nullable=False):
        """
        Get the coordinate variable name for the dimension map entry ``entry_key``.

        :param str entry_key: See :class:`ocgis.constants.DimensionMapKey` for valid entry keys.
        :param parent: If present, use the returned variable name to return the variable object form ``parent``.
        :type parent: :class:`~ocgis.VariableCollection`
        :param bool nullable: If ``True`` and ``parent`` is not ``None``, return ``None`` if the variable is not found
         in ``parent``.
        :rtype: str | None
        """
        ret = self._get_element_(entry_key, DMK.VARIABLE, None)

        # If there is an entry and a parent is provided, get the variable from the parent.
        if ret is not None and parent is not None:
            to_remove = []
            base_variable = parent[ret]
            base_variable_ndim = base_variable.ndim
            has_sections = False
            for ii in [DMK.X, DMK.Y]:
                entry = self._get_entry_(ii)
                section = entry.get(DMK.SECTION)
                # Sections allow use to use a single variable as a source for multiple variables. In general, variables
                # are atomic in the dimension map (x-coordinate is one variable). However, some metadata formats put
                # both coordinates in a single variable (x/y-coordinate is one variable with the dimension name
                # determining what the values represent.
                if section is not None:
                    has_sections = True
                    section = get_formatted_slice(section, base_variable_ndim)
                    new_variable = base_variable[section]

                    if is_xarray(new_variable):
                        new_dimensions = [d for ii, d in enumerate(new_variable.dims) if new_variable.shape[ii] > 1]
                    else:
                        new_dimensions = [d for d in new_variable.dims if d.size > 1]

                    if is_xarray(new_variable):
                        new_variable = new_variable.squeeze()
                        new_variable.name = ii
                    else:
                        new_variable.reshape(new_dimensions)
                        ew_variable = new_variable.extract()
                        new_variable.set_name(ii)

                    entry[DMK.VARIABLE] = new_variable.name
                    entry.pop(DMK.SECTION)

                    parent.add_variable(new_variable)
                    if base_variable.name not in to_remove:
                        to_remove.append(base_variable.name)

            if has_sections:
                for tt in to_remove:
                    parent.remove_variable(tt)
                ret = self._get_element_(entry_key, DMK.VARIABLE, None)

        if ret is not None and parent is not None:
            # Check if the variable has bounds.
            bnds = self.get_bounds(entry_key)
            ret = get_variable_from_field(ret, parent, nullable)
            # Set the bounds on the outgoing variable if they are not already set by the object.
            try:
                if bnds is not None and not ret.has_bounds:
                    ret.set_bounds(get_variable_from_field(bnds, parent, False), force=True)
            except AttributeError:
                if not is_xarray(ret):
                    raise
                else:
                    pass
        return ret

    def inquire_is_xyz(self, variable):
        """
        Inquire the dimension map to identify a variable's spatial classification.

        :param variable: The target variable to identify.
        :type variable: str | :class:`~ocgis.Variable`
        :rtype: :class:`ocgis.constants.DimensionMapKey`
        """
        name = get_variable_names(variable)[0]
        x = self.get_variable(DMK.X)
        y = self.get_variable(DMK.Y)
        z = self.get_variable(DMK.LEVEL)
        poss = {x: DMK.X, y: DMK.Y, z: DMK.LEVEL}
        ret = poss.get(name)
        if ret is None and self.has_topology:
            poss = {}
            topologies = self.get_available_topologies()
            for t in topologies:
                curr = self.get_topology(t)
                x = curr.get_variable(DMK.X)
                y = curr.get_variable(DMK.Y)
                z = curr.get_variable(DMK.LEVEL)
                poss.update({x: DMK.X, y: DMK.Y, z: DMK.LEVEL})
            ret = poss.get(name)
        return ret

    def pprint(self, as_dict=False):
        """
        Pretty print the dimension map.

        :param bool as_dict: If ``True``, convert group dimension maps to dictionaries.
        """
        if as_dict:
            target = self.as_dict()
        else:
            target = self._storage
        pprint_dict(target)

    def set_bounds(self, entry_key, bounds):
        """
        Set the bounds variable name for ``entry_key``.

        :param str entry_key: See :class:`ocgis.constants.DimensionMapKey` for valid entry keys.
        :param bounds: :class:`str` | :class:`~ocgis.Variable`
        """
        name = get_variable_names(bounds)[0]
        entry = self._get_entry_(entry_key)
        if entry[DMK.VARIABLE] is None:
            raise DimensionMapError(entry_key, 'No variable set. Bounds may not be set.')
        entry[DMK.BOUNDS] = name

    def set_crs(self, variable):
        """
        Set the coordinate reference system variable name.

        :param variable: :class:`str` | :class:`~ocgis.Variable`
        """
        variable = get_variable_names(variable)[0]
        entry = self._get_entry_(DMK.CRS)
        entry[DMK.VARIABLE] = variable

    def set_driver(self, driver):
        from ocgis.driver.registry import get_driver_class

        klass = get_driver_class(driver)
        self._storage[DMK.DRIVER] = klass.key

    def set_grid_abstraction(self, abstraction):
        self._storage[DMK.GRID_ABSTRACTION] = abstraction

    def set_group(self, group_key, dimension_map):
        """
        Set the group dimension map for ``group_key``.

        :param group_key: See :meth:`~ocgis.DimensionMap.get_group`.
        :param dimension_map: The dimension map to insert.
        :type dimension_map: :class:`~ocgis.DimensionMap`
        """
        _ = get_dmap_group(self, group_key, create=True, last=dimension_map)

    def set_property(self, key, value):
        """
        Set a property on the dimension map.

        :param str key: The key name
        :param value: The property's value
        """
        assert key in DMK.get_special_entry_keys()
        self._storage[key] = value

    def set_spatial_mask(self, variable, attrs=None, default_attrs=None):
        """
        Set the spatial mask variable for the dimension map. If ``attrs`` is not ``None``, then ``attrs`` >
        ``variable.attrs`` (if ``variable`` is not a string) > default attributes.
        
        :param variable: The spatial mask variable.
        :param dict attrs: Attributes to associate with the spatial mask variable *in addition* to default attributes.
        :param dict default_attrs: If provided, use these attributes as default spatial mask attributes.
        :type variable: :class:`~ocgis.Variable` | :class:`str`
        """

        if default_attrs is None:
            default_attrs = deepcopy(DIMENSION_MAP_TEMPLATE[DMK.SPATIAL_MASK][DMK.ATTRS])

        try:
            vattrs = deepcopy(variable.attrs)
        except AttributeError:
            vattrs = {}

        if attrs is None:
            attrs = {}

        default_attrs.update(vattrs)
        default_attrs.update(attrs)

        variable = get_variable_names(variable)[0]
        entry = self._get_entry_(DMK.SPATIAL_MASK)
        entry[DMK.VARIABLE] = variable
        entry[DMK.ATTRS] = default_attrs

    def set_variable(self, entry_key, variable, dimension=None, bounds=None, attrs=None, pos=None, dimensionless=False,
                     section=None):
        """
        Set coordinate variable information for ``entry_key``.
        
        :param str entry_key: See :class:`ocgis.constants.DimensionMapKey` for valid entry keys.
        :param variable: The variable to set. Use a variable object to auto-fill additional fields if they are ``None``.
        :type variable: :class:`str` | :class:`~ocgis.Variable`
        :param dimension: A sequence of dimension names. If ``None``, they will be pulled from ``variable`` if it is a
         variable object.
        :param bounds: See :meth:`~ocgis.DimensionMap.set_bounds`.
        :param dict attrs: Default attributes for the coordinate variables. If ``None``, they will be pulled from 
         ``variable`` if it is a variable object.
        :param int pos: The representative dimension position in ``variable`` if ``variable`` has more than one
         dimension. For example, a latitude variable may have two dimensions ``(lon, lat)``. The mapper must determine
         which dimension position is representative for the latitude variable when slicing.
        :param section: A slice-like tuple used to extract the data out of its source variable into a single variable
         format.
        :type section: tuple

        >>> section = (None, 0)
        >>> # This will be converted to a slice.
        >>> [slice(None), slice(0, 1)]

        :param bool dimensionless: If ``True``, this variable has no canonical dimension.
        :raises: DimensionMapError
        """
        if entry_key in self._special_entry_keys:
            raise DimensionMapError(entry_key, "The entry '{}' has a special set method.".format(entry_key))
        if section is not None and (pos is None and dimension is None):
            raise DimensionMapError(entry_key, "If a section is provided, position or dimension must be defined.")

        entry = self._get_entry_(entry_key)

        if variable is None:
            self._storage.pop(entry_key)
            return

        try:
            if bounds is None:
                if is_xarray(variable):
                    bounds = getattr(variable, 'bounds', None)
                else:
                    bounds = variable.bounds
            if dimension is None:
                if variable.ndim > 1:
                    if pos is None and not dimensionless:
                        msg = "A position (pos) is required if no dimension is provided and target variable has " \
                              "greater than one dimension."
                        raise DimensionMapError(entry_key, msg)
                elif variable.ndim == 1:
                    pos = 0
                else:
                    pos = None
                # We can have scalar dimensions.
                if pos is not None and not dimensionless:
                    dimension = variable.dims[pos]
        except AttributeError:
            # Assume string type.
            if is_xarray(variable):
                raise
            else:
                pass

        value = get_variable_names(variable)[0]
        if bounds is not None:
            bounds = get_variable_names(bounds)[0]
        if dimension is None:
            dimension = []
        else:
            dimension = list(get_dimension_names(dimension))

        if attrs is None:
            try:
                attrs = self._storage.__class__(deepcopy(DIMENSION_MAP_TEMPLATE[entry_key][DMK.ATTRS]))
            except KeyError:
                # Default attributes are empty.
                attrs = self._storage.__class__()
        entry[DMK.VARIABLE] = value
        entry[DMK.BOUNDS] = bounds
        entry[DMK.DIMENSION] = dimension
        entry[DMK.ATTRS] = attrs
        if section is not None:
            entry[DMK.SECTION] = section

    def update(self, other):
        """
        Update this dimension map from another dimension map.

        :param other:
        :type other: :class:`~ocgis.DimensionMap`
        """
        for other_k, other_v in other._storage.items():
            if other_k == DMK.TOPOLOGY:
                if DMK.TOPOLOGY not in self._storage:
                    self._storage[DMK.TOPOLOGY] = {}
                self._storage[DMK.TOPOLOGY].update(other_v)
            else:
                self._storage[other_k] = other_v

    def update_dimensions_from_metadata(self, metadata):
        """
        Update dimension names for coordinate variables using a metadata dictionary.

        :param dict metadata: A metadata dictionary containing dimension names for variables.
        """
        to_update = (DMK.REALIZATION, DMK.TIME, DMK.LEVEL, DMK.Y, DMK.X)
        for k in to_update:
            variable_name = self.get_variable(k)
            dimension_names = self.get_dimension(k)
            if variable_name is not None and len(dimension_names) == 0:
                metadata_dimensions = metadata.get('variables', {}).get(variable_name, {}).get('dimensions', [])
                if len(metadata_dimensions) > 0:
                    self.set_variable(k, variable_name, dimension=metadata_dimensions)

    def _get_element_(self, entry_key, element_key, default):
        if entry_key == DMK.CRS:
            raise DimensionMapError(entry_key, "Use 'get_crs' to get the CRS variable name.")
        entry = self._get_entry_(entry_key)
        ret = get_or_create_dict(entry, element_key, default)
        return ret

    def _get_entry_(self, key):
        if key not in self._allowed_entry_keys:
            raise DimensionMapError(key, 'Entry not allowed.')
        else:
            return get_or_create_dict(self._storage, key, self._storage.__class__())


def get_variable_from_field(name, field, nullable):
    ret = None
    if name is None and nullable:
        pass
    elif field is not None:
        try:
            ret = field[name]
        except KeyError:
            raise VariableNotInCollection(name)
    return ret


def get_dmap_group(dmap, keyseq, create=False, last=None):
    keyseq = deepcopy(keyseq)
    if last is None:
        last = {}

    if keyseq is None:
        keyseq = [None]
    elif isinstance(keyseq, six.string_types):
        keyseq = [keyseq]

    if keyseq[0] is not None:
        keyseq.insert(0, None)

    curr = dmap
    len_keyseq = len(keyseq)
    for ctr, key in enumerate(keyseq, start=1):
        if key is None:
            continue
        else:
            try:
                curr = curr._storage[DMK.GROUPS][key]
            except KeyError:
                if create:
                    curr_dct = get_or_create_dict(curr._storage, DMK.GROUPS, {})
                    if ctr == len_keyseq:
                        default = last
                    else:
                        default = {}
                    curr = get_or_create_dict(curr_dct, key, default)
                else:
                    raise
    return curr


def has_bounds(target):
    # tdk: DOC
    try:
        ret = target.has_bounds
    except AttributeError:
        if is_xarray(target):
            ret = False
            if getattr(target, 'bounds', None) is not None:
                ret = True
        else:
            raise
    return ret


def is_bounded(dmap, key):
    # tdk: DOC
    return dmap.get_bounds(key) is not None