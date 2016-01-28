from ocgis import RequestDataset, GeomCabinet, GeomCabinetIterator
from ocgis import constants
from ocgis.api.request.driver.base import AbstractDriver
from ocgis.api.request.driver.vector import DriverVector
from ocgis.interface.base.crs import WGS84
from ocgis.test.base import TestBase
from ocgis.test.base import attr


class TestDriverVector(TestBase):
    def get_driver(self, **kwargs):
        rd = self.get_rd(**kwargs)
        driver = DriverVector(rd)
        return driver

    def get_rd(self, variable=None):
        uri = GeomCabinet().get_shp_path('state_boundaries')
        rd = RequestDataset(uri=uri, driver='vector', variable=variable)
        return rd

    @attr('data')
    def test_init(self):
        self.assertIsInstances(self.get_driver(), (DriverVector, AbstractDriver))

        actual = [constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH,
                  constants.OUTPUT_FORMAT_SHAPEFILE]
        self.assertAsSetEqual(actual, DriverVector.output_formats)

    @attr('data')
    def test_close(self):
        driver = self.get_driver()
        sci = driver.open()
        driver.close(sci)

    @attr('data')
    def test_get_crs(self):
        driver = self.get_driver()
        self.assertEqual(WGS84(), driver.get_crs())

    @attr('data')
    def test_get_dimensioned_variables(self):
        driver = self.get_driver()
        target = driver.get_dimensioned_variables()
        self.assertEqual(target, [u'UGID', u'STATE_FIPS', u'ID', u'STATE_NAME', u'STATE_ABBR'])

    @attr('data')
    def test_get_dump_report(self):
        driver = self.get_driver()
        lines = driver.get_dump_report()
        self.assertTrue(len(lines) > 5)

    @attr('data')
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

    @attr('data')
    def test_get_source_metadata(self):
        driver = self.get_driver()
        meta = driver.get_source_metadata()
        self.assertIsInstance(meta, dict)
        self.assertTrue(len(meta) > 2)

    @attr('data')
    def test_inspect(self):
        driver = self.get_driver()
        with self.print_scope() as ps:
            driver.inspect()
        self.assertTrue(len(ps.storage) >= 1)

    @attr('data')
    def test_open(self):
        driver = self.get_driver()
        sci = driver.open()
        self.assertIsInstance(sci, GeomCabinetIterator)
        self.assertFalse(sci.as_spatial_dimension)
