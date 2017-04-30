from copy import deepcopy

import six

from ocgis.base import AbstractOcgisObject, get_variables
from ocgis.base import get_dimension_names
from ocgis.base import get_variable_names
from ocgis.constants import DIMENSION_MAP_TEMPLATE, DMK
from ocgis.exc import DimensionMapError
from ocgis.util.helpers import pprint_dict, get_or_create_dict


class DimensionMap(AbstractOcgisObject):
    """
    A dimension map is used to link dimensions and variables with an explicit meaning. It is the main mapping produced
    by a driver and a request dataset's metadata. Dimension maps are used by field to construct grids and geometries,
    perform subsetting, link bounds to parent variables, and manage coordinate systems.
    """

    _allowed_entry_keys = (DMK.REALIZATION, DMK.TIME, DMK.LEVEL, DMK.Y, DMK.X, DMK.GEOM, DMK.CRS, DMK.GROUPS)
    _allowed_element_keys = (DMK.VARIABLE, DMK.DIMS, DMK.BOUNDS, DMK.ATTRS)

    def __init__(self):
        self._storage = {}

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
            else:
                try:
                    variable = v.pop(DMK.VARIABLE)
                except KeyError:
                    raise DimensionMapError(k, "No 'variable' is available.")
                if k == DMK.CRS:
                    d.set_crs(variable)
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
        dimension_map = driver.get_dimension_map(group_metadata)
        if curr is None:
            curr = dimension_map
        if group_name is not None:
            curr.set_group(group_name, dimension_map)
        if DMK.GROUPS in group_metadata:
            for group_name, sub_group_metadata in group_metadata[DMK.GROUPS].items():
                cls.from_metadata(driver, sub_group_metadata, curr=curr, group_name=group_name)

        return curr

    def get_attrs(self, entry_key):
        """
        Get attributes for the dimension map entry ``entry_key``.
        
        :param str entry_key: See :class:`ocgis.constants.DimensionMapKey` for valid entry keys.
        :rtype: :class:`~collections.OrderedDict` 
        """
        return self._get_element_(entry_key, DMK.ATTRS, self._storage.__class__())

    def get_bounds(self, entry_key):
        """
        Get the bounds variable name for the dimension map entry ``entry_key``.

        :param str entry_key: See :class:`ocgis.constants.DimensionMapKey` for valid entry keys.
        :rtype: str
        """
        return self._get_element_(entry_key, DMK.BOUNDS, None)

    def get_crs(self):
        """
        Get the coordinate reference system variable name for the dimension map entry ``entry_key``.

        :rtype: str
        """
        entry = self._get_entry_(DMK.CRS)
        return get_or_create_dict(entry, DMK.VARIABLE, None)

    def get_dimensions(self, entry_key):
        """
        Get the dimension names for the dimension map entry ``entry_key``.

        :param str entry_key: See :class:`ocgis.constants.DimensionMapKey` for valid entry keys.
        :rtype: :class:`list` of :class:`str`
        """
        return self._get_element_(entry_key, DMK.DIMS, [])

    def get_group(self, group_key):
        """
        Get the dimension map for a group indexed by ``group_key`` starting from the root group.
        
        :param group_key: The group indexing key.
        :rtype: :class:`list` of :class:`str`
        """
        if DMK.GROUPS not in self._storage:
            self._storage[DMK.GROUPS] = {}
        try:
            return _get_dmap_group_(self, group_key)
        except KeyError:
            raise DimensionMapError(DMK.GROUPS, "Group key not found: {}".format(group_key))

    def get_variable(self, entry_key):
        """
        Get the coordinate variable name for the dimension map entry ``entry_key``.

        :param str entry_key: See :class:`ocgis.constants.DimensionMapKey` for valid entry keys.
        :rtype: str
        """
        return self._get_element_(entry_key, DMK.VARIABLE, None)

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

    def set_group(self, group_key, dimension_map):
        """
        Set the group dimension map for ``group_key``.
        
        :param group_key: See :meth:`~ocgis.DimensionMap.get_group`.
        :param dimension_map: The dimension map to insert.
        :type dimension_map: :class:`~ocgis.DimensionMap`
        """
        _ = _get_dmap_group_(self, group_key, create=True, last=dimension_map)

    def set_variable(self, entry_key, variable, dimensions=None, bounds=None, attrs=None):
        """
        Set coordinate variable information for ``entry_key``.
        
        :param str entry_key: See :class:`ocgis.constants.DimensionMapKey` for valid entry keys.
        :param variable: The variable to set. Use a variable object to auto-fill additional fields if they are ``None``.
        :type variable: :class:`str` | :class:`~ocgis.Variable`
        :param dimensions: A sequence of dimension names. If ``None``, they will be pulled from ``variable`` if it is a
         variable object.
        :param bounds: See :meth:`~ocgis.DimensionMap.set_bounds`.
        :param dict attrs: Default attributes for the coordinate variables. If ``None``, they will be pulled from 
         ``variable`` if it is a variable object.
        """
        if entry_key == DMK.CRS:
            raise DimensionMapError(entry_key, "Use 'set_crs' to set CRS variable.")

        entry = self._get_entry_(entry_key)

        if variable is None:
            self._storage.pop(entry_key)
            return

        try:
            if bounds is None:
                bounds = variable.bounds
            if dimensions is None:
                dimensions = variable.dimensions
        except AttributeError:
            # Assume string type.
            pass

        value = get_variable_names(variable)[0]
        if bounds is not None:
            bounds = get_variable_names(bounds)[0]
        if dimensions is None:
            dimensions = []
        else:
            dimensions = list(get_dimension_names(dimensions))

        if attrs is None:
            attrs = self._storage.__class__(deepcopy(DIMENSION_MAP_TEMPLATE[entry_key][DMK.ATTRS]))
        entry[DMK.VARIABLE] = value
        entry[DMK.BOUNDS] = bounds
        entry[DMK.DIMS] = dimensions
        entry[DMK.ATTRS] = attrs

    def update_dimensions_from_field(self, field):
        """
        Update dimension names for coordinate variables using a ``field`` object.
        
        :param field: The field to pull data from.
        :type field: :class:`~ocgis.Field`
        :raises: ValueError
        """
        to_update = (DMK.REALIZATION, DMK.TIME, DMK.LEVEL, DMK.Y, DMK.X)
        for k in to_update:
            variable_name = self.get_variable(k)
            dimension_names = self.get_dimensions(k)
            if variable_name is not None and len(dimension_names) == 0:
                try:
                    vc_var = get_variables(variable_name, field)[0]
                except KeyError:
                    msg = "Variable '{}' list in dimension map, but it is not present in the field.".format(
                        variable_name)
                    raise ValueError(msg)
                if vc_var.ndim == 1:
                    dimension_names.append(vc_var.dimensions[0].name)

    def update_dimensions_from_metadata(self, metadata):
        """
        Update dimension names for coordinate variables using a metadata dictionary.

        :param dict metadata: A metadata dictionary containg dimension names for variables.
        """
        to_update = (DMK.REALIZATION, DMK.TIME, DMK.LEVEL, DMK.Y, DMK.X)
        for k in to_update:
            variable_name = self.get_variable(k)
            dimension_names = self.get_dimensions(k)
            if variable_name is not None and len(dimension_names) == 0:
                metadata_dimensions = metadata.get('variables', {}).get(variable_name, {}).get('dimensions', [])
                if len(metadata_dimensions) > 0:
                    self.set_variable(k, variable_name, dimensions=metadata_dimensions)

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


def _get_dmap_group_(dmap, keyseq, create=False, last=None):
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
