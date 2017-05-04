import os

from ocgis.interface.metadata import NcMetadata
from ocgis.test.base import TestBase
from ocgis.test.base import attr
from ocgis.test.test_simple.test_simple import nc_scope


class TestNcMetadata(TestBase):
    @property
    def rd(self):
        return self.test_data.get_rd('cancm4_tasmax_2001')

    @attr('data')
    def test_init(self):
        with nc_scope(self.rd.uri, 'r') as ds:
            ncm = NcMetadata(ds)
        self.assertEqual(set(ncm.keys()), set(['dataset', 'variables', 'dimensions', 'file_format']))

    def test_get_lines(self):
        # test with a unicode string
        path = os.path.join(self.current_dir_output, 'foo.nc')
        with nc_scope(path, 'w') as ds:
            ds.foo = u'a bad \u2013 unicode character'
            md = NcMetadata(rootgrp=ds)
            ds.sync()
            lines = md.get_lines()
        self.assertEqual(lines[4], '// global attributes:')
