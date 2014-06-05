from collections import OrderedDict
import os
import re
import unittest
import ocgis
from ocgis.exc import RequestValidationError
from ocgis.interface.metadata import NcMetadata
from ocgis.test.test_simple.make_test_data import SimpleNc
from ocgis.test.test_simple.test_simple import nc_scope, TestSimpleBase
from ocgis import Inspect, RequestDataset
import numpy as np
from ocgis.util.itester import itr_products_keywords


class TestInspect(TestSimpleBase):
    base_value = np.array([[1.0, 1.0, 2.0, 2.0],
                           [1.0, 1.0, 2.0, 2.0],
                           [3.0, 3.0, 4.0, 4.0],
                           [3.0, 3.0, 4.0, 4.0]])
    nc_factory = SimpleNc
    fn = 'test_simple_spatial_01.nc'

    def test_init(self):
        dataset = self.get_dataset()
        uri = dataset['uri']
        variable = dataset['variable']
        with nc_scope(uri) as ds:
            nc_metadata = NcMetadata(ds)

        keywords = dict(
            uri=[None, self.get_dataset()['uri']],
            variable=[None, self.get_dataset()['variable']],
            request_dataset=[None, RequestDataset(uri=uri, variable=variable)],
            meta=[None, nc_metadata])

        for k in itr_products_keywords(keywords, as_namedtuple=True):
            try:
                ip = Inspect(**k._asdict())
            except ValueError:
                if k.uri is None and k.request_dataset is None and k.meta is None:
                    continue
                else:
                    raise
            ret = ip.__repr__()
            search = re.search('URI = (.*)\n', ret).groups()[0]
            if k.uri is None and k.meta is not None and k.request_dataset is None:
                self.assertEqual(search, 'None')
            else:
                self.assertTrue(os.path.exists(search))

    def test_missing_calendar_attribute(self):
        # # path to the test data file
        out_nc = os.path.join(self._test_dir, self.fn)

        ## case of calendar being set to an empty string
        with nc_scope(out_nc, 'a') as ds:
            ds.variables['time'].calendar = ''
        rd = ocgis.RequestDataset(uri=out_nc, variable='foo')
        ## the default for the calendar overload is standard
        self.assertEqual(rd.t_calendar, None)

        ## case of a calendar being set a bad value but read anyway
        with nc_scope(out_nc, 'a') as ds:
            ds.variables['time'].calendar = 'foo'
        rd = ocgis.RequestDataset(uri=out_nc, variable='foo')
        field = rd.get()
        self.assertEqual(field.temporal.calendar, 'foo')
        ## calendar is only access when the float values are converted to datetime
        ## objects
        with self.assertRaises(ValueError):
            field.temporal.value_datetime
        ## now overload the value and ensure the field datetimes may be loaded
        rd = ocgis.RequestDataset(uri=out_nc, variable='foo', t_calendar='standard')
        self.assertEqual(rd.source_metadata['variables']['time']['attrs']['calendar'], 'foo')
        field = rd.get()
        self.assertEqual(field.temporal.calendar, 'standard')
        field.temporal.value_datetime

        ## case of a missing calendar attribute altogether
        with nc_scope(out_nc, 'a') as ds:
            ds.variables['time'].delncattr('calendar')
        rd = ocgis.RequestDataset(uri=out_nc, variable='foo')
        self.assertEqual(rd.t_calendar, None)
        self.assertIsInstance(rd.inspect_as_dct(), OrderedDict)
        self.assertEqual(rd.inspect_as_dct()['derived']['Calendar'],
                         'None (will assume "standard")')
        ## write the data to a netCDF and ensure the calendar is written.
        ret = ocgis.OcgOperations(dataset=rd, output_format='nc').execute()
        with nc_scope(ret) as ds:
            self.assertEqual(ds.variables['time'].calendar, 'standard')
            self.assertEqual(ds.variables['time_bnds'].calendar, 'standard')
        field = rd.get()
        ## the standard calendar name should be available at the dataset level
        self.assertEqual(field.temporal.calendar, 'standard')
        ## test different forms of inspect ensuring the standard calendar is
        ## correctly propagated
        ips = [ocgis.Inspect(request_dataset=rd), ocgis.Inspect(uri=out_nc, variable='foo')]
        for ip in ips:
            self.assertNotIn('calendar', ip.meta['variables']['time']['attrs'])
            self.assertTrue(ip.get_temporal_report()[2].endswith(('will assume "standard")')))
        ip = ocgis.Inspect(uri=out_nc)
        ## this method is only applicable when a variable is present
        with self.assertRaises(AttributeError):
            ip.get_report()
        self.assertIsInstance(ip.get_report_no_variable(), list)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()