import abc
from abc import abstractproperty

from ocgis.constants import KeywordArguments, HeaderNames
from ocgis.conv.base import AbstractTabularConverter
from ocgis.driver.vector import DriverVector
from ocgis.exc import DefinitionValidationError


class AbstractFionaConverter(AbstractTabularConverter):
    __metaclass__ = abc.ABCMeta

    _add_ugeom = True
    _add_ugeom_nest = False

    @abstractproperty
    def _driver(self):
        """Fiona driver string."""

    def _write_coll_(self, f, coll, add_geom_uid=True):
        """
        Write a spatial collection using file information from ``f``.

        :param dict f: A dictionary containing all the necessary variables to write the spatial collection to a file
         object.
        :param coll: The spatial collection to write.
        :type coll: :class:`~ocgis.new_interface.collection.SpatialCollection`
        """
        write_mode = f[KeywordArguments.WRITE_MODE]
        path = f[KeywordArguments.PATH]

        iter_kwargs = {'melted': self.melted}

        for field, container in coll.iter_fields(yield_container=True):
            # Try to load the geometry from the grid.
            set_ugid_as_data = False
            if len(field.data_variables) == 0:
                set_ugid_as_data = True
            field.set_abstraction_geom(create_ugid=True, set_ugid_as_data=set_ugid_as_data)

            if add_geom_uid and field.geom is not None and field.geom.ugid is None:
                field.geom.create_ugid_global(HeaderNames.ID_GEOMETRY)

            if container.geom is not None:
                repeater = [(self.geom_uid, container.geom.ugid.get_value()[0])]
            else:
                repeater = None
            iter_kwargs[KeywordArguments.REPEATERS] = repeater
            field.write(path, write_mode=write_mode, driver=DriverVector, fiona_driver=self._driver,
                        iter_kwargs=iter_kwargs)


class ShpConverter(AbstractFionaConverter):
    _ext = 'shp'
    _driver = 'ESRI Shapefile'


class GeoJsonConverter(AbstractFionaConverter):
    _ext = 'json'
    _driver = 'GeoJSON'

    @classmethod
    def validate_ops(cls, ops):
        from ocgis.calc.base import AbstractMultivariateFunction
        from ocgis.calc.eval_function import MultivariateEvalFunction
        from ocgis.ops.parms.definition import OutputFormat

        if len(list(ops.dataset)) > 1:
            should_raise = True
            if ops.calc is not None:
                for c in ops.calc:
                    if c['ref'] in [AbstractMultivariateFunction, MultivariateEvalFunction]:
                        should_raise = False
                        break
            if should_raise:
                msg = 'Only one request dataset may be written to GeoJSON.'
                raise DefinitionValidationError(OutputFormat.name, msg)
