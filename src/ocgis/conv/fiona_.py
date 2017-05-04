import abc
import datetime
from types import NoneType

import fiona
import numpy as np

from ocgis.conv.base import AbstractTabularConverter


class AbstractFionaConverter(AbstractTabularConverter):
    __metaclass__ = abc.ABCMeta

    _add_ugeom = True
    _add_ugeom_nest = False
    _fiona_conversion = {np.int32: int,
                         np.int16: int,
                         np.int64: int,
                         np.float64: float,
                         np.float32: float,
                         np.float16: float,
                         datetime.datetime: str,
                         datetime.date: str}
    _fiona_type_mapping = {datetime.date: 'str',
                           datetime.datetime: 'str',
                           np.int64: 'int',
                           NoneType: None,
                           np.int32: 'int',
                           np.float64: 'float',
                           np.float32: 'float',
                           np.float16: 'float',
                           np.int16: 'int',
                           np.int32: 'int',
                           str: 'str',
                           np.dtype('int32'): 'int',
                           np.dtype('int64'): 'int',
                           np.dtype('float32'): 'float',
                           np.dtype('float64'): 'float'}

    @classmethod
    def get_field_type(cls, the_type, key=None, fiona_conversion=None):
        """
        :param the_type: The target type object to map to a Fiona field type.
        :type the_type: type
        :param key: The key to update the Fiona conversion map.
        :type key: str
        :param fiona_conversion: A dictionary used to convert Python values to Fiona-expected values.
        :type fiona_conversion: dict
        :returns: The appropriate ``fiona`` field type.
        :rtype: str or NoneType
        :raises: AttributeError
        """

        # bypass for string types...
        try:
            the_types_type = the_type.type
        except AttributeError:
            # likely not a numpy type
            pass
        else:
            if the_types_type == np.string_:
                length = the_type.str[2:]
                ret = 'str:{0}'.format(length)
                if key is not None:
                    fiona_conversion[key] = unicode
                return ret

        # this is for other types...
        ret = None
        for k, v in fiona.FIELD_TYPES_MAP.iteritems():
            if the_type == v:
                ret = k
                break
        if ret is None:
            ret = cls._fiona_type_mapping[the_type]

        try:
            if the_type in cls._fiona_conversion:
                fiona_conversion.update({key.lower(): cls._fiona_conversion[the_type]})
        except AttributeError:
            if fiona_conversion is not None:
                raise

        return ret

    def _finalize_(self, f):
        """
        Perform any final operations on file objects.

        :param dict f: A dictionary containing file-level metadata and potentially the file object itself.
        """

        f['fobject'].close()

    def _build_(self, coll):
        """
        :param coll: An archetypical spatial collection that will be written to file.
        :type coll: :class:`~ocgis.SpatialCollection`
        :returns: A dictionary with all the file object metadata and the file object itself.
        :rtype: dict
        """

        field = coll.first().itervalues().next()
        ugeom = coll.ugeom.itervalues().next()
        arch = field.get_iter(melted=self.melted, use_upper_keys=self._use_upper_keys, headers=coll.headers,
                              ugeom=ugeom).next()
        fdict = field.get_fiona_dict(field, arch[1])
        fdict['fobject'] = fiona.open(self.path, driver=self._driver, schema=fdict['schema'], crs=fdict['crs'],
                                      mode='w')
        return fdict

    def _write_coll_(self, f, coll):
        """
        Write a spatial collection using file information from ``f``.

        :param dict f: A dictionary containing all the necessary variables to write the spatial collection to a file
         object.
        :param coll: The spatial collection to write.
        :type coll: :class:`~ocgis.SpatialCollection`
        """

        for ugid, field_dict in coll.iteritems():
            ugeom = coll.ugeom[ugid]
            for field in field_dict.itervalues():
                fobject = f['fobject']
                field.write_fiona(melted=self.melted, fobject=fobject, use_upper_keys=self._use_upper_keys,
                                  headers=coll.headers, ugeom=ugeom)


class ShpConverter(AbstractFionaConverter):
    _ext = 'shp'
    _driver = 'ESRI Shapefile'


class GeoJsonConverter(AbstractFionaConverter):
    _ext = 'json'
    _driver = 'GeoJSON'
