from ocgis.test.base import TestBase
import datetime
import ocgis
import fiona
import numpy as np


class Test(TestBase):
    
    def test_months_in_units(self):
        rd = self.test_data_nc.get_rd('clt_month_units')
        field = rd.get()
        self.assertEqual(field.temporal.units,'months since 1979-1-1 0')
        self.assertEqual(field.temporal.value_datetime[50],datetime.datetime(1983,3,16))
        self.assertEqual(field.temporal.bounds,None)
        self.assertEqual(field.temporal.shape,(120,))

    def test_months_in_units_time_range_subsets(self):
        rd = self.test_data_nc.get_rd('clt_month_units')
        field = rd.get()
        time_range = [field.temporal.value_datetime[0], field.temporal.value_datetime[0]]
        ops = ocgis.OcgOperations(dataset=rd, time_range=time_range)
        ret = ops.execute()
        self.assertEqual((1, 1, 1, 46, 72), ret[1]['clt'].shape)
    
    def test_months_in_units_convert_to_shapefile(self):
        uri = self.test_data_nc.get_uri('clt_month_units')
        variable = 'clt'
        ## select the month of may for two years
        time_region = {'month':[5],'year':[1982,1983]}
        rd = ocgis.RequestDataset(uri=uri,variable=variable,time_region=time_region)
        ## for fun, interpolate the spatial bounds from the point centroids.
        ## setting this the spatial bounds interpolate to false will write a point
        ## shapefile.
        ops = ocgis.OcgOperations(dataset=rd,output_format='shp',interpolate_spatial_bounds=True)
        ret = ops.execute()
        with fiona.open(ret,driver='ESRI Shapefile') as source:
            self.assertEqual(len(source),6624)

    def test_months_in_units_convert_to_netcdf(self):
        uri = self.test_data_nc.get_uri('clt_month_units')
        variable = 'clt'
        rd = ocgis.RequestDataset(uri=uri,variable=variable)
        ## subset the clt dataset by the state of nevada and write to netcdf
        ops = ocgis.OcgOperations(dataset=rd,output_format='nc',geom='state_boundaries',
                                  select_ugid=[23])
        ret = ops.execute()
        rd2 = ocgis.RequestDataset(uri=ret,variable=variable)
        field = rd.get()
        field2 = rd2.get()
        ## confirm raw values and datetime values are equivalent
        self.assertNumpyAll(field.temporal.value_datetime,field2.temporal.value_datetime)
        self.assertNumpyAll(field.temporal.value,field2.temporal.value)

    def test_months_in_units_calculation(self):
        rd = self.test_data_nc.get_rd('clt_month_units')
        calc = [{'func': 'mean', 'name': 'mean'}]
        calc_grouping = ['month']
        ops = ocgis.OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping)
        ret = ops.execute()
        # '[[datetime.datetime(1979, 1, 16, 0, 0) datetime.datetime(1988, 1, 16, 0, 0)]\n [datetime.datetime(1979, 2, 16, 0, 0) datetime.datetime(1988, 2, 16, 0, 0)]\n [datetime.datetime(1979, 3, 16, 0, 0) datetime.datetime(1988, 3, 16, 0, 0)]\n [datetime.datetime(1979, 4, 16, 0, 0) datetime.datetime(1988, 4, 16, 0, 0)]\n [datetime.datetime(1979, 5, 16, 0, 0) datetime.datetime(1988, 5, 16, 0, 0)]\n [datetime.datetime(1979, 6, 16, 0, 0) datetime.datetime(1988, 6, 16, 0, 0)]\n [datetime.datetime(1979, 7, 16, 0, 0) datetime.datetime(1988, 7, 16, 0, 0)]\n [datetime.datetime(1979, 8, 16, 0, 0) datetime.datetime(1988, 8, 16, 0, 0)]\n [datetime.datetime(1979, 9, 16, 0, 0) datetime.datetime(1988, 9, 16, 0, 0)]\n [datetime.datetime(1979, 10, 16, 0, 0)\n  datetime.datetime(1988, 10, 16, 0, 0)]\n [datetime.datetime(1979, 11, 16, 0, 0)\n  datetime.datetime(1988, 11, 16, 0, 0)]\n [datetime.datetime(1979, 12, 16, 0, 0)\n  datetime.datetime(1988, 12, 16, 0, 0)]]'
        actual = '\x80\x02cnumpy.core.multiarray\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x01K\x0cK\x02\x86cnumpy\ndtype\nq\x04U\x02O8K\x00K\x01\x87Rq\x05(K\x03U\x01|NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?tb\x89]q\x06(cdatetime\ndatetime\nq\x07U\n\x07\xbb\x01\x10\x00\x00\x00\x00\x00\x00\x85Rq\x08h\x07U\n\x07\xc4\x01\x10\x00\x00\x00\x00\x00\x00\x85Rq\th\x07U\n\x07\xbb\x02\x10\x00\x00\x00\x00\x00\x00\x85Rq\nh\x07U\n\x07\xc4\x02\x10\x00\x00\x00\x00\x00\x00\x85Rq\x0bh\x07U\n\x07\xbb\x03\x10\x00\x00\x00\x00\x00\x00\x85Rq\x0ch\x07U\n\x07\xc4\x03\x10\x00\x00\x00\x00\x00\x00\x85Rq\rh\x07U\n\x07\xbb\x04\x10\x00\x00\x00\x00\x00\x00\x85Rq\x0eh\x07U\n\x07\xc4\x04\x10\x00\x00\x00\x00\x00\x00\x85Rq\x0fh\x07U\n\x07\xbb\x05\x10\x00\x00\x00\x00\x00\x00\x85Rq\x10h\x07U\n\x07\xc4\x05\x10\x00\x00\x00\x00\x00\x00\x85Rq\x11h\x07U\n\x07\xbb\x06\x10\x00\x00\x00\x00\x00\x00\x85Rq\x12h\x07U\n\x07\xc4\x06\x10\x00\x00\x00\x00\x00\x00\x85Rq\x13h\x07U\n\x07\xbb\x07\x10\x00\x00\x00\x00\x00\x00\x85Rq\x14h\x07U\n\x07\xc4\x07\x10\x00\x00\x00\x00\x00\x00\x85Rq\x15h\x07U\n\x07\xbb\x08\x10\x00\x00\x00\x00\x00\x00\x85Rq\x16h\x07U\n\x07\xc4\x08\x10\x00\x00\x00\x00\x00\x00\x85Rq\x17h\x07U\n\x07\xbb\t\x10\x00\x00\x00\x00\x00\x00\x85Rq\x18h\x07U\n\x07\xc4\t\x10\x00\x00\x00\x00\x00\x00\x85Rq\x19h\x07U\n\x07\xbb\n\x10\x00\x00\x00\x00\x00\x00\x85Rq\x1ah\x07U\n\x07\xc4\n\x10\x00\x00\x00\x00\x00\x00\x85Rq\x1bh\x07U\n\x07\xbb\x0b\x10\x00\x00\x00\x00\x00\x00\x85Rq\x1ch\x07U\n\x07\xc4\x0b\x10\x00\x00\x00\x00\x00\x00\x85Rq\x1dh\x07U\n\x07\xbb\x0c\x10\x00\x00\x00\x00\x00\x00\x85Rq\x1eh\x07U\n\x07\xc4\x0c\x10\x00\x00\x00\x00\x00\x00\x85Rq\x1fetb.'
        actual = np.loads(actual)
        self.assertNumpyAll(ret[1]['clt'].temporal.bounds_datetime, actual)
