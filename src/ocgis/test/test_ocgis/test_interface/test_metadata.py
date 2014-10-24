import unittest
import netCDF4 as nc
from ocgis.test.base import TestBase
from ocgis.interface.metadata import NcMetadata


class TestNcMetadata(TestBase):

    def setUp(self):
        uri = self.test_data.get_rd('cancm4_tasmax_2001').uri
        self.rootgrp = nc.Dataset(uri)

    def tearDown(self):
        self.rootgrp.close()

    def test_init(self):
        ncm = NcMetadata(self.rootgrp)
        self.assertEqual(set(ncm.keys()), set(['dataset', 'variables', 'dimensions', 'file_format']))
