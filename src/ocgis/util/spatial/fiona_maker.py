import os
import fiona
from shapely import wkt
from shapely.geometry import mapping
from ocgis import CoordinateReferenceSystem


class FionaMaker(object):

    def __init__(self, path, epsg=4326, driver='ESRI Shapefile', geometry='Polygon'):
        assert (not os.path.exists(path))
        self.path = path
        self.crs = CoordinateReferenceSystem(epsg=epsg).value
        self.properties = {'UGID': 'int', 'NAME': 'str'}
        self.geometry = geometry
        self.driver = driver
        self.schema = {'geometry': self.geometry,
                       'properties': self.properties}

    def __enter__(self):
        self._ugid = 1
        self._collection = fiona.open(self.path, 'w', driver=self.driver, schema=self.schema, crs=self.crs)
        return self

    def __exit__(self, *args, **kwargs):
        self._collection.close()

    def make_record(self, dct):
        properties = dct.copy()

        if 'wkt' in properties:
            geom = wkt.loads(properties.pop('wkt'))
        elif 'geom' in properties:
            geom = properties.pop('geom')
        else:
            raise NotImplementedError

        properties.update({'UGID': self._ugid})
        self._ugid += 1
        record = {'geometry': mapping(geom),
                  'properties': properties}
        return record

    def write(self, sequence_or_dct):
        if isinstance(sequence_or_dct, dict):
            itr = [sequence_or_dct]
        else:
            itr = sequence_or_dct
        for element in itr:
            record = self.make_record(element)
            self._collection.write(record)
