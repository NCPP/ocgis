import os
import re
from collections import OrderedDict

import numpy as np

import ocgis
from ocgis import Inspect, RequestDataset
from ocgis.test.base import nc_scope, TestBase
from ocgis.test.test_simple.make_test_data import SimpleNc
from ocgis.test.test_simple.test_simple import TestSimpleBase
from ocgis.util.itester import itr_products_keywords


class TestInspect1(TestBase):
    def __iter__(self):
        variable = 'tas'
        novar = self.get_netcdf_path_no_dimensioned_variables()
        keywords = dict(uri=[self.uri, None],
                        variable=[variable, None],
                        request_dataset=[RequestDataset(uri=self.uri), None, RequestDataset(uri=novar)])
        for k in self.iter_product_keywords(keywords):
            yield k

    @property
    def uri(self):
        uri = self.test_data.get_uri('cancm4_tas')
        return uri

    def pprint(self, lines):
        for l in lines:
            print l

    def get(self):
        rd = RequestDataset(uri=self.uri)
        fai = Inspect(request_dataset=rd)
        return fai

    def test_init(self):
        for k in self:
            try:
                fai = Inspect(**k._asdict())
            except ValueError:
                self.assertIsNone(k.uri)
                self.assertIsNone(k.request_dataset)
                continue
            self.assertIsInstance(fai.request_dataset, RequestDataset)

    def test_append_dump_report(self):
        path = self.get_netcdf_path_no_dimensioned_variables()
        ip = Inspect(uri=path)
        target = []
        ip._append_dump_report_(target)
        self.assertTrue(len(target) > 5)

    def test_get_field_report(self):
        rd = self.test_data.get_rd('cancm4_tas')
        fai = Inspect(request_dataset=rd)
        target = fai.get_field_report()
        self.assertEqual(len(target), 25)

    def test_get_header(self):
        fai = self.get()
        h = fai.get_header()
        self.assertEqual(len(h), 2)

    def test_get_report(self):
        fai = self.get()
        target = fai.get_report()
        self.assertEqual(len(target), 109)

    def test_get_report_no_field(self):
        fai = self.get()
        target = fai.get_report_no_field()
        self.assertEqual(len(target), 84)

    def test_get_report_possible(self):
        uri = [self.test_data.get_uri('cancm4_tas'), self.get_netcdf_path_no_dimensioned_variables()]
        for u in uri:
            ip = Inspect(uri=u)
            target = ip.get_report_possible()
            self.assertTrue(len(target) > 10)

    def test_str(self):
        ret = str(Inspect(uri=self.uri))
        self.assertTrue(len(ret) > 4000)

        novar = self.get_netcdf_path_no_dimensioned_variables()
        ret = str(Inspect(uri=novar))
        self.assertTrue(len(ret) > 100)

    def test_variable(self):
        fai = Inspect(uri=self.uri, variable='tasmax')
        self.assertEqual(fai.variable, 'tasmax')

        fai = Inspect(uri=self.uri)
        self.assertEqual(fai.variable, 'tas')

        novar = self.get_netcdf_path_no_dimensioned_variables()
        fai = Inspect(uri=novar)
        self.assertIsNone(fai.variable)


class TestInspect2(TestSimpleBase):
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

        keywords = dict(
            uri=[None, self.get_dataset()['uri']],
            variable=[None, self.get_dataset()['variable']],
            request_dataset=[None, RequestDataset(uri=uri, variable=variable)])
        for k in itr_products_keywords(keywords, as_namedtuple=True):
            try:
                ip = Inspect(**k._asdict())
            except ValueError:
                if k.uri is None and k.request_dataset is None:
                    continue
                else:
                    raise
            ret = ip.__str__()
            search = re.search('URI = (.*)\n', ret).groups()[0]
            if k.uri is None and k.request_dataset is None:
                self.assertEqual(search, 'None')
            else:
                self.assertTrue(os.path.exists(search))

    def test_calendar_attribute_none(self):
        """Test that the empty string is correctly interpreted as None."""

        # path to the test data file
        out_nc = os.path.join(self.current_dir_output, self.fn)

        # case of calendar being set to an empty string
        with nc_scope(out_nc, 'a') as ds:
            ds.variables['time'].calendar = ''

        rd = ocgis.RequestDataset(uri=out_nc, variable='foo')
        # the default for the calendar overload is standard
        self.assertEqual(rd.t_calendar, None)

    def test_unknown_calendar_attribute(self):
        """Test a calendar attribute with an unknown calendar attribute."""

        # Path to the test data file.
        out_nc = os.path.join(self.current_dir_output, self.fn)

        # Case of a calendar being set a bad value but read anyway.
        with nc_scope(out_nc, 'a') as ds:
            ds.variables['time'].calendar = 'foo'

        rd = ocgis.RequestDataset(uri=out_nc, variable='foo')
        field = rd.get()
        self.assertEqual(field.temporal.calendar, 'foo')
        # Calendar is only accessed when the float values are converted to datetime objects.
        with self.assertRaises(ValueError):
            field.temporal.value_datetime
        # now overload the value and ensure the field datetimes may be loaded
        rd = ocgis.RequestDataset(uri=out_nc, variable='foo', t_calendar='standard')
        self.assertEqual(rd.source_metadata['variables']['time']['attrs']['calendar'], 'foo')
        field = rd.get()
        self.assertEqual(field.temporal.calendar, 'standard')
        field.temporal.value_datetime

        # Case of a missing calendar attribute.
        with nc_scope(out_nc, 'a') as ds:
            ds.variables['time'].delncattr('calendar')
        rd = ocgis.RequestDataset(uri=out_nc, variable='foo')
        self.assertEqual(rd.t_calendar, None)
        self.assertIsInstance(rd.inspect_as_dct(), OrderedDict)
        # Write the data to a netCDF and ensure the calendar is written.
        ret = ocgis.OcgOperations(dataset=rd, output_format='nc').execute()
        with nc_scope(ret) as ds:
            self.assertEqual(ds.variables['time'].calendar, 'standard')
            self.assertEqual(ds.variables['time_bnds'].calendar, 'standard')
        field = rd.get()
        # The standard calendar name should be available at the dataset level.
        self.assertEqual(field.temporal.calendar, 'standard')
