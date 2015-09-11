import abc
from collections import OrderedDict

import fiona
from shapely.geometry import mapping, MultiPoint, MultiPolygon
from shapely.geometry.base import BaseMultipartGeometry

from ocgis.util.helpers import get_ordered_dicts_from_records_array


class AbstractCollection(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._storage = OrderedDict()
        self._storage_id = []

    @property
    def _storage_id_next(self):
        try:
            ret = max(self._storage_id) + 1
        # max of an empty list
        except ValueError:
            if len(self._storage_id) == 0:
                ret = 1
            else:
                raise
        return ret

    def __contains__(self, item):
        return item in self._storage

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            ret = self.__dict__ == other.__dict__
        else:
            ret = False
        return ret

    def __iter__(self):
        for key in self.iterkeys():
            yield key

    def __getitem__(self, item):
        return self._storage[item]

    def __len__(self):
        return len(self._storage)

    def __setitem__(self, key, value):
        self._storage[key] = value

    def __repr__(self):
        ret = '{0}({1})'.format(self.__class__.__name__, [(k, v) for k, v in self.iteritems()])
        return ret

    def __str__(self):
        return self.__repr__()

    def first(self):
        for key in self.iterkeys():
            return self._storage[key]

    def items(self):
        return self._storage.items()

    def iteritems(self):
        for k, v in self._storage.iteritems():
            yield k, v

    def iterkeys(self):
        for k in self._storage.iterkeys():
            yield k

    def itervalues(self):
        for v in self._storage.itervalues():
            yield v

    def keys(self):
        return self._storage.keys()

    def pop(self, *args, **kwargs):
        return self._storage.pop(*args, **kwargs)

    def update(self, dictionary):
        self._storage.update(dictionary)

    def values(self):
        return self._storage.values()


class SpatialCollection(AbstractCollection):
    _multi_cast = {'Point': MultiPoint, 'Polygon': MultiPolygon}

    def __init__(self, meta=None, key=None, crs=None, headers=None, value_keys=None):
        super(SpatialCollection, self).__init__()

        self._crs = None

        self.meta = meta
        self.key = key
        self.crs = crs
        self.headers = headers
        self.value_keys = value_keys

        self.ugeom = OrderedDict()

    @property
    def crs(self):
        if self._crs is None:
            try:
                ret = self._archetype_field.spatial.crs
            # likely no field loaded into collection
            except IndexError:
                ret = None
            # likely no spatial dimension on the field
            except AttributeError:
                ret = None
        else:
            ret = self._crs
        return ret

    @crs.setter
    def crs(self, value):
        self._crs = value

    @property
    def geoms(self):

        def getter(v):
            return v.abstraction_geometry.value[0, 0]

        return get_ugeom_attribute(self.ugeom, getter)

    @property
    def properties(self):

        def getter(v):
            return v.properties

        return get_ugeom_attribute(self.ugeom, getter)

    @property
    def _archetype_field(self):
        ukey = self.keys()[0]
        fkey = self[ukey].keys()[0]
        return self[ukey][fkey]

    def add_field(self, field, ugeom=None, name=None):
        """
        Add a field to the collection.

        :param field: The field object to add. There are no checks if the field data is unique.
        :type field: :class:`~ocgis.Field`
        :param ugeom: The selection/user geometry associated with the field.
        :type ugeom: :class:`~ocgis.SpatialDimension`
        :param str name: The name associated with the field. If this is ``None``, use the field's name attribute.
        :raises: ValueError
        """

        # allow nonetype fields
        if field is not None:
            # all data must have the same coordinate system
            try:
                if field.spatial.crs != self.crs and len(self) > 0:
                    msg = 'Field and collection coordinate systems differ.'
                    raise ValueError(msg)
            except AttributeError:
                # likely no spatial dimension
                if field.spatial is not None:
                    raise

            # add a unique identifier to the field if it is not already present
            if field.uid is None:
                field.uid = self._storage_id_next
            self._storage_id.append(field.uid)

        # use the provided name or the name attached to the field
        name = name or field.name

        # pull out the geometry unique identifier
        if ugeom is not None:
            assert ugeom.shape == (1, 1)
            ugid = ugeom.uid[0, 0]
        else:
            ugid = 1

        # if there is no dictionary associated with the unique identifier, create one.
        if ugid not in self:
            self[ugid] = OrderedDict()
        # the name of the field associated with the geometry unique identifier must be unique.
        if name in self[ugid]:
            msg = 'A field with name "{0}" is already present for geometry with unique identifier "{1}".'.format(name,
                                                                                                                 ugid)
            raise ValueError(msg)

        # store the field and geometry data
        self[ugid][name] = field
        self.ugeom[ugid] = ugeom

    def get_iter_dict(self, use_upper_keys=False, conversion_map=None, melted=False):
        """
        :param bool use_upper_keys: If ``True``, capitalize all keys in the yielded data dictionary.
        :param dict conversion_map: If present, keys correspond to headers with values being the type to convert to.
        :param bool melted: If ``True``, yield in melted form with variables collected under the value header.
        :returns: A generator yielding tuples. If headers on the collection are not ``None``, these headers will be used
         to limit keys in the yielded data dictionary.
        :rtype: tuple(:class:`shapely.geometry.base.BaseGeometry`, dict)
        """

        if conversion_map is None or melted is False:
            use_conversion = False
        else:
            use_conversion = True
        for ugid, field_dict in self.iteritems():
            ugeom = self.ugeom.get(ugid)
            for field in field_dict.itervalues():
                for yld_geom, row in field.get_iter(value_keys=self.value_keys, melted=melted,
                                                    use_upper_keys=use_upper_keys, headers=self.headers, ugeom=ugeom):
                    if melted:
                        if use_conversion:
                            for k, v in conversion_map.iteritems():
                                row[k] = v(row[k])
                    yield yld_geom, row

    def get_iter_elements(self):
        for ugid, fields in self.iteritems():
            for field_alias, field in fields.iteritems():
                for var_alias, variable in field.variables.iteritems():
                    yield (ugid, field_alias, var_alias, variable)

    def get_iter_melted(self):
        for ugid, container in self.iteritems():
            for field_alias, field in container.iteritems():
                for variable_alias, variable in field.variables.iteritems():
                    yield (dict(ugid=ugid, field_alias=field_alias, field=field, variable_alias=variable_alias,
                                variable=variable))

    def gvu(self, ugid, alias_variable, alias_field=None):
        ref = self[ugid]
        if alias_field is None:
            field = ref.values()[0]
        else:
            field = ref[alias_field]
        return field.variables[alias_variable].value

    def write_ugeom(self, path=None, driver='ESRI Shapefile', fobject=None):
        """
        Write the user geometries to a ``fiona``-supported file format.

        :param str path: Full path of file to write. If ``None``, ``fobject`` is required.
        :param str driver: The ``fiona`` driver to use for writing. Ignored if ``fobject`` is provided.
        :param fobject: An open ``fiona`` file object to write to. If ``path`` is provided and this is not ``None``,
         then ``path`` will be ignored.
        :type fobject: :class:`fiona.collection.Collection`
        """
        from ocgis.conv.base import get_schema_from_numpy_dtype

        if fobject is None and self.crs is None:
            msg = 'A coordinate system is required when writing to Fiona formats.'
            raise ValueError(msg)

        build = True if fobject is None else False
        is_open = False
        needs_casting = False
        try:
            for ugid, geom in self.geoms.iteritems():
                if build:
                    # it is possible to end with a mix of singleton and multi-geometries
                    type_check = set()
                    for check_geom in self.geoms.itervalues():
                        type_check.update([check_geom.geom_type])
                    if len(type_check) > 1:
                        needs_casting = True
                        for xx in type_check:
                            if xx.startswith('Multi'):
                                geometry = xx
                            else:
                                cast_target_key = xx
                    else:
                        geometry = type_check.pop()

                    fiona_properties = get_schema_from_numpy_dtype(self.properties[ugid].dtype)
                    fiona_schema = {'geometry': geometry, 'properties': fiona_properties}
                    fiona_kwds = {'schema': fiona_schema, 'driver': driver, 'mode': 'w'}
                    if self.crs is not None:
                        fiona_kwds['crs'] = self.crs.value
                    fobject = fiona.open(path, **fiona_kwds)
                    is_open = True
                    build = False
                properties = get_ordered_dicts_from_records_array(self.properties[ugid])[0]
                if needs_casting:
                    if not isinstance(geom, BaseMultipartGeometry):
                        geom = self._multi_cast[cast_target_key]([geom])
                mapped_geom = mapping(geom)
                record = {'geometry': mapped_geom, 'properties': properties}
                fobject.write(record)
        finally:
            if is_open:
                fobject.close()


def get_ugeom_attribute(ugeom, getter):
    """
    :param dict ugeom: A dictionary with keys are unique identifiers (int) and values as spatial dimension objects.
    :param getter: A function taking a value that will be the target of attribute extraction.
    :type getter: function
    :returns: A dictionary with keys matching the unique identifiers from ``ugeom``. The values are the attributes
     extracted by ``getter``.
    :rtype: dict
    """

    ret = OrderedDict()
    for k, v in ugeom.iteritems():
        try:
            attr = getter(v)
        # likely a nonetype for the overview geometry
        except AttributeError:
            if v is None:
                attr = None
            else:
                raise
        ret[k] = attr
    return ret
