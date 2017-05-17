import os
from collections import OrderedDict

from ocgis.test.base import TestBase, nc_scope
from ocgis.variable.attributes import Attributes


class TestAttributes(TestBase):
    def get_attributes(self):
        attrs = {'a': 5, 'b': 6}
        a = Attributes(attrs=attrs)
        return a, attrs

    def test_init(self):
        a = Attributes()
        self.assertEqual(a.attrs, OrderedDict())

        a, attrs = self.get_attributes()
        self.assertIsInstance(a.attrs, OrderedDict)
        self.assertEqual(a.attrs, attrs)
        attrs['c'] = 'another'
        self.assertNotIn('c', a.attrs)

    def test_write_to_netcdf_object(self):
        path = os.path.join(self.current_dir_output, 'foo.nc')

        a, attrs = self.get_attributes()

        # Write to dataset object.
        with nc_scope(path, 'w') as ds:
            a.write_attributes_to_netcdf_object(ds)
        with nc_scope(path, 'r') as ds:
            self.assertDictEqual(ds.__dict__, a.attrs)

        # Write to variable object.
        with nc_scope(path, 'w') as ds:
            ds.createDimension('foo')
            var = ds.createVariable('foo', int, dimensions=('foo',))
            a.write_attributes_to_netcdf_object(var)
        with nc_scope(path, 'r') as ds:
            var = ds.variables['foo']
            self.assertDictEqual(var.__dict__, a.attrs)

        # Test strange unicode characters are handled.
        strange = u'\xfc'
        attrs = Attributes(attrs={'uni': strange, 'normal': 'attribute'})
        path = self.get_temporary_file_path('foo2.nc')
        with self.nc_scope(path, 'w') as ds:
            attrs.write_attributes_to_netcdf_object(ds)
