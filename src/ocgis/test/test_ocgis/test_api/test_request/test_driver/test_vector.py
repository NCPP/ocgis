from ocgis import constants
from ocgis import RequestDataset, ShpCabinet, ShpCabinetIterator
from ocgis.api.request.driver.base import AbstractDriver
from ocgis.api.request.driver.vector import DriverVector
from ocgis.interface.base.crs import WGS84
from ocgis.test.base import TestBase


class TestDriverVector(TestBase):
    def get_driver(self, **kwargs):
        rd = self.get_rd(**kwargs)
        driver = DriverVector(rd)
        return driver

    def get_rd(self, variable=None):
        uri = ShpCabinet().get_shp_path('state_boundaries')
        rd = RequestDataset(uri=uri, driver='vector', variable=variable)
        return rd

    def test_init(self):
        self.assertIsInstances(self.get_driver(), (DriverVector, AbstractDriver))

        actual = [constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH,
                  constants.OUTPUT_FORMAT_SHAPEFILE]
        self.assertAsSetEqual(actual, DriverVector.output_formats)

    def test_close(self):
        driver = self.get_driver()
        sci = driver.open()
        driver.close(sci)

    def test_get_crs(self):
        driver = self.get_driver()
        self.assertEqual(WGS84(), driver.get_crs())

    def test_get_dimensioned_variables(self):
        driver = self.get_driver()
        target = driver.get_dimensioned_variables()
        self.assertEqual(target, [u'UGID', u'STATE_FIPS', u'ID', u'STATE_NAME', u'STATE_ABBR'])

    def test_get_field(self):
        driver = self.get_driver()
        field = driver.get_field()
        self.assertIsNone(field.spatial.properties)
        self.assertEqual(len(field.variables), 5)
        for variable in field.variables.itervalues():
            self.assertEqual(variable.shape, (1, 1, 1, 1, 51))

        # test with a variable
        driver = self.get_driver(variable=['ID', 'STATE_NAME'])
        field = driver.get_field()
        self.assertIn('ID', field.variables)
        self.assertEqual(field.variables['ID'].shape, (1, 1, 1, 1, 51))

        # test an alias and name
        rd = self.get_rd(variable='ID')
        rd.alias = 'another'
        rd.name = 'something_else'
        driver = DriverVector(rd)
        field = driver.get_field()
        self.assertEqual(field.name, rd.name)
        self.assertIn('another', field.variables)

    def test_get_source_metadata(self):
        driver = self.get_driver()
        meta = driver.get_source_metadata()
        self.assertIsInstance(meta, dict)
        self.assertTrue(len(meta) > 2)

    def test_inspect(self):
        driver = self.get_driver()
        driver.inspect()

    def test_inspect_get_lines(self):
        driver = self.get_driver()
        lines = driver._inspect_get_lines_()
        self.assertTrue(len(lines) > 5)

    def test_open(self):
        driver = self.get_driver()
        sci = driver.open()
        self.assertIsInstance(sci, ShpCabinetIterator)
        self.assertFalse(sci.as_spatial_dimension)
