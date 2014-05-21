from ocgis.test.base import TestBase
import datetime
import ocgis
import fiona


class Test(TestBase):
    
    def test_months_in_units(self):
        rd = self.test_data.get_rd('clt_month_units')
        field = rd.get()
        self.assertEqual(field.temporal.units,'months since 1979-1-1 0')
        self.assertEqual(field.temporal.value_datetime[50],datetime.datetime(1983,3,16))
        self.assertEqual(field.temporal.bounds,None)
        self.assertEqual(field.temporal.shape,(120,))

    def test_months_in_units_time_range_subsets(self):
        rd = self.test_data.get_rd('clt_month_units')
        field = rd.get()
        time_range = [field.temporal.value_datetime[0], field.temporal.value_datetime[0]]
        ops = ocgis.OcgOperations(dataset=rd, time_range=time_range)
        ret = ops.execute()
        self.assertEqual((1, 1, 1, 46, 72), ret[1]['clt'].shape)
    
    def test_months_in_units_convert_to_shapefile(self):
        uri = self.test_data.get_uri('clt_month_units')
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
        uri = self.test_data.get_uri('clt_month_units')
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
        rd = self.test_data.get_rd('clt_month_units')
        calc = [{'func': 'mean', 'name': 'mean'}]
        calc_grouping = ['month']
        ops = ocgis.OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping)
        ret = ops.execute()
        self.assertEqual(str(ret[1]['clt'].temporal.bounds_datetime), '[[1979-01-16 00:00:00 1988-01-16 00:00:00]\n [1979-02-16 00:00:00 1988-02-16 00:00:00]\n [1979-03-16 00:00:00 1988-03-16 00:00:00]\n [1979-04-16 00:00:00 1988-04-16 00:00:00]\n [1979-05-16 00:00:00 1988-05-16 00:00:00]\n [1979-06-16 00:00:00 1988-06-16 00:00:00]\n [1979-07-16 00:00:00 1988-07-16 00:00:00]\n [1979-08-16 00:00:00 1988-08-16 00:00:00]\n [1979-09-16 00:00:00 1988-09-16 00:00:00]\n [1979-10-16 00:00:00 1988-10-16 00:00:00]\n [1979-11-16 00:00:00 1988-11-16 00:00:00]\n [1979-12-16 00:00:00 1988-12-16 00:00:00]]')
