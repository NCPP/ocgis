from cfunits import Units
from ocgis import RequestDataset
from ocgis.test.test_simple.make_test_data import SimpleNcMultivariate
from ocgis.test.test_simple.test_simple import TestSimpleBase
import numpy as np


class TestSimpleMultivariate(TestSimpleBase):
    base_value = np.array([[1.0, 1.0, 2.0, 2.0],
                           [1.0, 1.0, 2.0, 2.0],
                           [3.0, 3.0, 4.0, 4.0],
                           [3.0, 3.0, 4.0, 4.0]])
    nc_factory = SimpleNcMultivariate
    fn = 'test_simple_multivariate_01.nc'
    var = ['foo', 'foo2']

    def test_variable_has_appropriate_units(self):
        """Test multiple variables loaded from a netCDF file are assigned the appropriate units."""

        field = RequestDataset(**self.get_dataset()).get()
        self.assertDictEqual({v.name: v.cfunits for v in field.variables.itervalues()},
                             {'foo': Units('K'), 'foo2': Units('mm/s')})