import os

import fiona
from shapely.geometry import Point

from ocgis import CoordinateReferenceSystem, GeomCabinetIterator
from ocgis.test.base import TestBase
from ocgis.util.spatial.fiona_maker import FionaMaker


class TestFionaMaker(TestBase):
    def get(self, **kwargs):
        path = os.path.join(self.current_dir_output, 'test.shp')
        fm = FionaMaker(path, **kwargs)
        return fm

    def test_init(self):
        fm = self.get()
        self.assertEqual(CoordinateReferenceSystem(epsg=4326), CoordinateReferenceSystem(value=fm.crs))

    def test_writing_point(self):
        """Test writing a point shapefile."""

        with self.get(geometry='Point') as source:
            source.write({'geom': Point(-130, 50), 'UGID': 1, 'NAME': 'the point'})
        with fiona.open(source.path) as source:
            crs = CoordinateReferenceSystem(value=source.crs)
            self.assertEqual(crs, CoordinateReferenceSystem(epsg=4326))

    def test_through_shpcabinet(self):
        """Test reading the shapefile into a spatial dimension object."""

        with self.get(geometry='Point') as source:
            source.write({'geom': Point(-130, 50), 'UGID': 1, 'NAME': 'the point'})
        sci = GeomCabinetIterator(path=source.path, as_spatial_dimension=True)
        sdims = list(sci)
        self.assertEqual(len(sdims), 1)
        sdim = sdims[0]
        point = sdim.geom.point.value[0, 0]
        self.assertTrue(point.almost_equals(Point(-130, 50)))
        self.assertEqual(CoordinateReferenceSystem(value=source.crs), sdim.crs)
