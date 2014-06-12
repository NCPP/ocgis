import ocgis
from ocgis.test.base import TestBase
import itertools
from ocgis.api.operations import OcgOperations
from datetime import datetime as dt
from ocgis.exc import DefinitionValidationError, MaskedDataError, ExtentError, RequestValidationError
import numpy as np
import unittest
from ocgis.interface.base.crs import CFWGS84
import fiona
from csv import DictReader
from ocgis.api.request.base import RequestDataset
from ocgis.test.test_simple.test_simple import nc_scope
from copy import deepcopy
from ocgis.test.test_base import longrunning
from shapely.geometry.point import Point
from ocgis.util.shp_cabinet import ShpCabinetIterator
import os


class TestCMIP3Masking(TestBase):
    
    @longrunning
    def test_many_request_datasets(self):
        rd_base = self.test_data.get_rd('subset_test_Prcp')
        geom = [-74.0, 40.0, -72.0, 42.0]
        rds = [deepcopy(rd_base) for ii in range(500)]
        for rd in rds:
            ret = OcgOperations(dataset=rd,geom=geom).execute()
            self.assertEqual(ret[1]['Prcp'].variables['Prcp'].value.shape,(1,1800,1,1,1))
    
    def test(self):
        for key in ['subset_test_Prcp','subset_test_Tavg_sresa2','subset_test_Tavg']:
            ## test method to return a RequestDataset
            rd = self.test_data.get_rd(key)
            geoms = [[-74.0, 40.0, -72.0, 42.0],
                     [-74.0, 38.0, -72.0, 40.0]]
            for geom in geoms:
                try:
                    ## this will raise the exception from the 38/40 bounding box
                    OcgOperations(dataset=rd,output_format='shp',geom=geom,
                                  prefix=str(geom[1])+'_'+key,allow_empty=False).execute()
                except MaskedDataError:
                    if geom[1] == 38.0:
                        ## note all returned data is masked!
                        ret = OcgOperations(dataset=rd,output_format='numpy',geom=geom,
                                            prefix=str(geom[1])+'_'+key,allow_empty=True).execute()
                        self.assertTrue(ret[1][rd.alias].variables[rd.alias].value.mask.all())
                    else:
                        raise


class TestCnrmCerfacs(TestBase):

    @property
    def rd(self):
        return self.test_data.get_rd('rotated_pole_cnrm_cerfacs')

    def test_subset(self):
        """Test data may be subsetted and that coordinate transformations return the same value arrays."""

        ops = OcgOperations(dataset=self.rd, output_format='numpy', snippet=True, geom='world_countries', select_ugid=[69])
        ret = ops.execute()

        # assert some of the geometry values are masked
        self.assertTrue(ret[69]['pr'].spatial.get_mask().any())

        # perform the operations but change the output coordinate system. the value arrays should be equivalent
        # regardless of coordinate transformation
        ops2 = OcgOperations(dataset=self.rd, output_format='numpy', snippet=True, geom='world_countries', select_ugid=[69],
                             output_crs=CFWGS84())
        ret2 = ops2.execute()

        # value arrays should be the same
        self.assertNumpyAll(ret[69]['pr'].variables['pr'].value, ret2[69]['pr'].variables['pr'].value)
        # grid coordinates should not be the same
        self.assertNumpyNotAll(ret[69]['pr'].spatial.grid.value, ret2[69]['pr'].spatial.grid.value)

    def test_subset_shp(self):
        """Test conversion to shapefile."""

        for ii, output_crs in enumerate([None, CFWGS84()]):
            ops = OcgOperations(dataset=self.rd, output_format='shp', snippet=True, geom='world_countries',
                                select_ugid=[69], output_crs=output_crs, prefix=str(ii))
            ret = ops.execute()


            with fiona.open(ret) as source:
                records = list(source)

            self.assertEqual(len(records), 2375)


class Test(TestBase):
    
    def test_cccma_rotated_pole(self):
        ## with rotated pole, the uid mask was not being updated correctly following
        ## a transformation back to rotated pole. this needed to be updated explicitly
        ## in subset.py
        rd = self.test_data.get_rd('rotated_pole_cccma')
        geom = (5.87161922454834, 47.26985931396479, 15.03811264038086, 55.05652618408209)
        ops = ocgis.OcgOperations(dataset=rd,output_format='shp',geom=geom,
                                  select_ugid=[1],snippet=True)
        ret = ops.execute()

        with fiona.open(ret) as source:
            self.assertEqual(len(source),228)
            gid = [row['properties']['GID'] for row in source]
            for element in gid:
                self.assertTrue(element > 4000)
    
    def test_ichec_rotated_pole(self):
        ## this point is far outside the domain
        ocgis.env.OVERWRITE = True
        rd = self.test_data.get_rd('rotated_pole_ichec')
        for geom in [[-100.,45.],[-100,45,-99,46]]:
            ops = ocgis.OcgOperations(dataset=rd,output_format='nc',
                              calc=[{'func':'mean','name':'mean'}],
                              calc_grouping=['month'],
                              geom=geom)
            with self.assertRaises(ExtentError):
                ops.execute()
    
    def test_narccap_cancm4_point_subset_no_abstraction(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('narccap_tas_rcm3_gfdl')
        rd.alias = 'tas_narccap'
        rds = [rd,rd2]
        geom = [-105.2751,39.9782]
        ops = ocgis.OcgOperations(dataset=rds,geom=geom,output_format='csv+',
                                  prefix='ncar_point',add_auxiliary_files=True,output_crs=ocgis.crs.CFWGS84(),
                                  snippet=True)
        with self.assertRaises(ValueError):
            ops.execute()
            
    def test_narccap_cancm4_point_subset_with_abstraction(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('narccap_tas_rcm3_gfdl')
        rd2.alias = 'tas_narccap'
        rds = [
               rd,
               rd2
               ]
        geom = [-105.2751,39.9782]
        ops = ocgis.OcgOperations(dataset=rds,geom=geom,output_format='numpy',
                                  prefix='ncar_point',add_auxiliary_files=True,output_crs=ocgis.crs.CFWGS84(),
                                  snippet=True,abstraction='point')
        ret = ops.execute()
        
        ## ensure array is trimmed and masked tgeometries removed
        self.assertEqual(ret[2]['tas_narccap'].spatial.shape,(2,1))
        ## only two geometries returned
        self.assertEqual(ret[1]['tas'].spatial.shape,(1,2))
        ## different buffer radii should have unique identifiers
        self.assertEqual(ret.keys(),[1,2])
        ## the first buffer radius is larger
        self.assertTrue(ret.geoms[1].area > ret.geoms[2].area)
        
    def test_narccap_cancm4_point_subset_with_abstraction_to_csv_shp(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('narccap_tas_rcm3_gfdl')
        rd.alias = 'tas_narccap'
        rds = [
               rd,
               rd2
               ]
        geom = [-105.2751,39.9782]
        ops = ocgis.OcgOperations(dataset=rds,geom=geom,output_format='csv+',
                                  prefix='ncar_point',add_auxiliary_files=True,output_crs=ocgis.crs.CFWGS84(),
                                  snippet=True,abstraction='point')
        ret = ops.execute()
        ugid_shp_path = os.path.join(os.path.split(ret)[0],'shp',ops.prefix+'_ugid.shp')
        with fiona.open(ugid_shp_path) as ds:
            rows = list(ds)
        self.assertEqual(set([row['properties']['UGID'] for row in rows]),set([1,2]))
    
    def test_collection_field_geometries_equivalent(self):
        rd = self.test_data.get_rd('cancm4_tas',kwds=dict(time_region={'month':[6,7,8]}))
        geom = ['state_boundaries',[{'properties':{'UGID':16},'geom':Point([-99.80780059778753,41.52315831343389])}]]
        for vw,g in itertools.product([True,False],geom):
            ops = ocgis.OcgOperations(dataset=rd,select_ugid=[16,32],geom=g,
                                      aggregate=True,vector_wrap=vw,spatial_operation='clip')
            coll = ops.execute()
            coll_geom = coll.geoms[16]
            field_geom = coll[16]['tas'].spatial.geom.polygon.value[0,0]
            self.assertTrue(coll_geom.bounds,field_geom.bounds)
            self.assertTrue(coll_geom.area,field_geom.area)
    
    def test_empty_subset_multi_geometry_wrapping(self):
        ## adjacent state boundaries were causing an error with wrapping where
        ## a reference to the source field was being updated.
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd,geom='state_boundaries',select_ugid=[5,6,7])
        ret = ops.execute()
        self.assertEqual(set(ret.keys()),set([5,6,7]))
    
    def test_seasonal_calc(self):
        calc = [{'func':'mean','name':'my_mean'},{'func':'std','name':'my_std'}]
        calc_grouping = [[3,4,5]]
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                                  calc_sample_size=True,geom='state_boundaries',
                                  select_ugid=[23])
        ret = ops.execute()
        self.assertEqual(ret[23]['tas'].variables['n_my_std'].value.mean(),920.0)
        self.assertEqual(ret[23]['tas'].variables['my_std'].value.shape,(1,1,1,4,3))
        
        calc = [{'func':'mean','name':'my_mean'},{'func':'std','name':'my_std'}]
        calc_grouping = [[12,1,2],[3,4,5],[6,7,8],[9,10,11]]
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                                  calc_sample_size=True,geom='state_boundaries',
                                  select_ugid=[23])
        ret = ops.execute()
        self.assertEqual(ret[23]['tas'].variables['my_std'].value.shape,(1,4,1,4,3))
        self.assertNumpyAll(ret[23]['tas'].temporal.value,np.array([ 56955.,  56680.,  56771.,  56863.]))
        
        calc = [{'func':'mean','name':'my_mean'},{'func':'std','name':'my_std'}]
        calc_grouping = [[12,1],[2,3]]
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                                  calc_sample_size=True,geom='state_boundaries',
                                  select_ugid=[23])
        ret = ops.execute()
        self.assertEqual(ret[23]['tas'].variables['my_std'].value.shape,(1,2,1,4,3))
        self.assertNumpyAll(ret[23]['tas'].temporal.bounds,np.array([[ 55115.,  58765.],[ 55146.,  58490.]]))

    def test_seasonal_calc_dkp(self):        
        key = 'dynamic_kernel_percentile_threshold'
        calc = [{'func':key,'name':'dkp','kwds':{'operation':'lt','percentile':90,'width':5}}]
        calc_grouping = [[3,4,5]]
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                                  calc_sample_size=False,geom='state_boundaries',
                                  select_ugid=[23])
        ret = ops.execute()
        to_test = ret[23]['tas'].variables['dkp'].value
        reference = np.ma.array(data=[[[[[0,0,838],[831,829,834],[831,830,834],[831,835,830]]]]],
                                mask=[[[[[True,True,False],[False,False,False],[False,False,False],[False,False,False]]]]])
        self.assertNumpyAll(to_test,reference)
    
    def test_selecting_single_value(self):
        rd = self.test_data.get_rd('cancm4_tas')
        lat_index = 32
        lon_index = 97
        with nc_scope(rd.uri) as ds:
            lat_value = ds.variables['lat'][lat_index]
            lon_value = ds.variables['lon'][lon_index]
            data_values = ds.variables['tas'][:,lat_index,lon_index]
        
        ops = ocgis.OcgOperations(dataset=rd,geom=[lon_value,lat_value],search_radius_mult=0.1)
        ret = ops.execute()
        values = np.squeeze(ret[1]['tas'].variables['tas'].value)
        self.assertNumpyAll(data_values,values.data)
        self.assertFalse(np.any(values.mask))
        
        geom = Point(lon_value,lat_value).buffer(0.001)
        ops = ocgis.OcgOperations(dataset=rd,geom=geom)
        ret = ops.execute()
        values = np.squeeze(ret[1]['tas'].variables['tas'].value)
        self.assertNumpyAll(data_values,values.data)
        self.assertFalse(np.any(values.mask))
        
        geom = Point(lon_value-360.,lat_value).buffer(0.001)
        ops = ocgis.OcgOperations(dataset=rd,geom=geom)
        ret = ops.execute()
        values = np.squeeze(ret[1]['tas'].variables['tas'].value)
        self.assertNumpyAll(data_values,values.data)
        self.assertFalse(np.any(values.mask))
        
        geom = Point(lon_value-360.,lat_value).buffer(0.001)
        ops = ocgis.OcgOperations(dataset=rd,geom=geom,aggregate=True,spatial_operation='clip')
        ret = ops.execute()
        values = np.squeeze(ret[1]['tas'].variables['tas'].value)
        self.assertNumpyAll(data_values,values.data)
        self.assertFalse(np.any(values.mask))
        
        ops = ocgis.OcgOperations(dataset=rd,geom=[lon_value,lat_value],
                                  search_radius_mult=0.1,output_format='nc')
        ret = ops.execute()
        with nc_scope(ret) as ds:
            values = np.squeeze(ds.variables['tas'][:])
            self.assertNumpyAll(data_values,values)
    
    @longrunning
    def test_value_conversion(self):
        ## confirm value data types are properly converted
        ocgis.env.DIR_DATA = '/usr/local/climate_data'
        rd_maurer = ocgis.RequestDataset('Maurer02new_OBS_tasmax_daily.1971-2000.nc',
                                     'tasmax',
                                     alias='maurer_tasmax')

        ops = ocgis.OcgOperations(dataset=rd_maurer,output_format='shp',snippet=True,
                                  output_crs=ocgis.crs.WGS84(),geom='state_boundaries',
                                  select_ugid=[25])
        ops.execute()
    
    def test_qed_multifile(self):
        ddir = '/usr/local/climate_data/QED-2013/multifile'
        variable = 'txxmmedm'
        ocgis.env.DIR_DATA = ddir
        
        uri = ['maurer02v2_median_txxmmedm_january_1971-2000.nc',
               'maurer02v2_median_txxmmedm_february_1971-2000.nc',
               'maurer02v2_median_txxmmedm_march_1971-2000.nc']
        
        rd = ocgis.RequestDataset(uri,variable)
        field = rd.get()

    @longrunning
    def test_maurer_concatenated_shp(self):
        """Test Maurer concatenated data may be appropriately subsetted."""

        ocgis.env.DIR_DATA = '/usr/local/climate_data/maurer/2010-concatenated'
        # ocgis.env.VERBOSE = True
        # ocgis.env.DEBUG = True

        names = [
            # [u'Maurer02new_OBS_dtr_daily.1971-2000.nc'],
            [u'Maurer02new_OBS_tas_daily.1971-2000.nc'],
            [u'Maurer02new_OBS_tasmin_daily.1971-2000.nc'],
            [u'Maurer02new_OBS_pr_daily.1971-2000.nc'],
            [u'Maurer02new_OBS_tasmax_daily.1971-2000.nc']
        ]
        variables = [
            # u'dtr',
            u'tas',
            u'tasmin',
            u'pr',
            u'tasmax'
        ]
        #        time_range = [datetime.datetime(1971, 1, 1, 0, 0),datetime.datetime(2000, 12, 31, 0, 0)]

        # rd = RequestDataset(uri=names[0], variable='tas')
        # field = rd.get()
        # # ops = OcgOperations(dataset=rd, output_format='shp', snippet=True)
        # # print ops.execute()
        # import ipdb;ipdb.set_trace()

        time_range = None
        time_region = {'month': [6, 7, 8], 'year': None}
        rds = [ocgis.RequestDataset(name, variable, time_range=time_range,
                                    time_region=time_region) for name, variable in zip(names, variables)]

        ops = ocgis.OcgOperations(dataset=rds, calc=[{'name': 'Standard Deviation', 'func': 'std', 'kwds': {}}],
                                  calc_grouping=['month'], calc_raw=False, geom='us_counties', select_ugid=[286],
                                  output_format='shp',
                                  spatial_operation='clip',
                                  headers=['did', 'ugid', 'gid', 'year', 'month', 'day', 'variable', 'calc_key',
                                           'value'],
                                  abstraction=None)
        ret = ops.execute()

        with fiona.open(ret) as f:
            variables = set([row['properties']['VARIABLE'] for row in f])
        self.assertEqual(variables, set([u'pr', u'tasmax', u'tasmin', u'tas']))
    
    def test_point_shapefile_subset(self):
        _output_format = ['numpy','nc','csv','csv+']
        for output_format in _output_format:
            rd = self.test_data.get_rd('cancm4_tas')
            ops = OcgOperations(dataset=rd,geom='qed_city_centroids',output_format=output_format,
                                prefix=output_format)
            ret = ops.execute()
            if output_format == 'numpy':
                self.assertEqual(len(ret),4)
    
    @longrunning
    def test_maurer_concatenated_tasmax_region(self):
        ocgis.env.DIR_DATA = '/usr/local/climate_data/maurer/2010-concatenated'
        filename = 'Maurer02new_OBS_tasmax_daily.1971-2000.nc'
        variable = 'tasmax'
#        ocgis.env.VERBOSE = True
        
        rd = ocgis.RequestDataset(filename,variable)
        ops = ocgis.OcgOperations(dataset=rd,geom='us_counties',select_ugid=[2778],
                                  output_format='numpy')
        ret = ops.execute()
        ref = ret[2778]['tasmax']
        years = np.array([dt.year for dt in ret[2778]['tasmax'].temporal.value_datetime])
        months = np.array([dt.month for dt in ret[2778]['tasmax'].temporal.value_datetime])
        select = np.array([dt.month in (6,7,8) and dt.year in (1990,1991,1992,1993,1994,1995,1996,1997,1998,1999) for dt in ret[2778]['tasmax'].temporal.value_datetime])
        time_subset = ret[2778]['tasmax'].variables['tasmax'].value[:,select,:,:,:]
        time_values = ref.temporal.value[select]
        
        rd = ocgis.RequestDataset(filename,variable,time_region={'month':[6,7,8],'year':[1990,1991,1992,1993,1994,1995,1996,1997,1998,1999]})
        ops = ocgis.OcgOperations(dataset=rd,geom='us_counties',select_ugid=[2778],
                                  output_format='numpy')
        ret2 = ops.execute()
        ref2 = ret2[2778]['tasmax']
        
        self.assertEqual(time_values.shape,ref2.temporal.shape)
        self.assertEqual(time_subset.shape,ref2.variables['tasmax'].value.shape)
        self.assertNumpyAll(time_subset,ref2.variables['tasmax'].value)
        self.assertFalse(np.any(ref2.variables['tasmax'].value < 0))
    
    def test_time_region_subset(self):
        
        _month = [[6,7],[12],None,[1,3,8]]
        _year = [[2011],None,[2012],[2011,2013]]
        
        def run_test(month,year):
            rd = self.test_data.get_rd('cancm4_rhs',kwds={'time_region':{'month':month,'year':year}})
                        
            ops = ocgis.OcgOperations(dataset=rd,geom='state_boundaries',
                                      select_ugid=[25])
            ret = ops.execute()
            
            ret = ret[25]['rhs'].temporal.value_datetime
            
            years = [dt.year for dt in ret.flat]
            months = [dt.month for dt in ret.flat]
            
            if year is not None:
                self.assertEqual(set(years),set(year))
            if month is not None:
                self.assertEqual(set(months),set(month))
            
        for month,year in itertools.product(_month,_year):
            run_test(month,year)
            
    def test_time_range_time_region_subset(self):
        time_range = [dt(2013,1,1),dt(2015,12,31)]
        time_region = {'month':[6,7,8],'year':[2013,2014]}
        kwds = {'time_range':time_range,'time_region':time_region}
        rd = self.test_data.get_rd('cancm4_rhs',kwds=kwds)
        ops = ocgis.OcgOperations(dataset=rd,geom='state_boundaries',select_ugid=[25])
        ret = ops.execute()
        ref = ret[25]['rhs']
        years = set([obj.year for obj in ref.temporal.value_datetime])
        self.assertFalse(2015 in years)
        
    def test_time_range_time_region_do_not_overlap(self):
        time_range = [dt(2013,1,1),dt(2015,12,31)]
        time_region = {'month':[6,7,8],'year':[2013,2014,2018]}
        kwds = {'time_range':time_range,'time_region':time_region}
        with self.assertRaises(RequestValidationError):
            self.test_data.get_rd('cancm4_rhs',kwds=kwds)

    @longrunning
    def test_maurer_2010(self):
        ## inspect the multi-file maurer datasets
        keys = ['maurer_2010_pr','maurer_2010_tas','maurer_2010_tasmin','maurer_2010_tasmax']
        calc = [{'func':'mean','name':'mean'},{'func':'median','name':'median'}]
        calc_grouping = ['month']
        for key in keys:
            rd = self.test_data.get_rd(key)
            
            dct = rd.inspect_as_dct()
            self.assertEqual(dct['derived']['Count'],'102564')
            
            ops = ocgis.OcgOperations(dataset=rd,snippet=True,select_ugid=[10,15],
                   output_format='numpy',geom='state_boundaries')
            ret = ops.execute()
            self.assertTrue(ret.gvu(10,rd.variable).sum() != 0)
            self.assertTrue(ret.gvu(15,rd.variable).sum() != 0)
            
            ops = ocgis.OcgOperations(dataset=rd,snippet=False,select_ugid=[10,15],
                   output_format='numpy',geom='state_boundaries',calc=calc,
                   calc_grouping=calc_grouping)
            ret = ops.execute()
            for calc_name in ['mean','median']:
                self.assertEqual(ret[10][rd.alias].variables[calc_name].value.shape[1],12)
                
            ops = ocgis.OcgOperations(dataset=rd,snippet=False,select_ugid=[10,15],
                   output_format='csv+',geom='state_boundaries',calc=calc,
                   calc_grouping=calc_grouping,prefix=key)
            ret = ops.execute()
            
    def test_clip_aggregate(self):
        ## this geometry was hanging
        rd = self.test_data.get_rd('cancm4_tas',kwds={'time_region':{'year':[2003]}})
        ops = OcgOperations(dataset=rd,geom='state_boundaries',select_ugid=[14,16],
                            aggregate=False,spatial_operation='clip',output_format='csv+')
        ret = ops.execute()
    
    @longrunning
    def test_narccap_point_subset_small(self):
        rd = self.test_data.get_rd('narccap_pr_wrfg_ncep')
        geom = [-97.74278,30.26694]
#        ocgis.env.VERBOSE = True
#        ocgis.env.DEBUG = True
    
        calc = [{'func':'mean','name':'mean'},
                {'func':'median','name':'median'},
                {'func':'max','name':'max'},
                {'func':'min','name':'min'}]
        calc_grouping = ['month','year']
        ops = ocgis.OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                                  output_format='numpy',geom=geom,abstraction='point',
                                  snippet=False,allow_empty=False,output_crs=CFWGS84())
        ret = ops.execute()
        ref = ret[1]['pr']
        self.assertEqual(set(ref.variables.keys()),set(['mean', 'median', 'max', 'min']))
        
    def test_bad_time_dimension(self):
        ocgis.env.DIR_DATA = '/usr/local/climate_data'
        uri = 'seasonalbias.nc'
        variable = 'bias'
        for output_format in [
                              'numpy',
                              'csv',
                              'csv+','shp',
                              'nc'
                              ]:
            
            dataset = RequestDataset(uri=uri,variable=variable)
            ops = OcgOperations(dataset=dataset,output_format=output_format,
                                format_time=False,prefix=output_format)
            ret = ops.execute()
            
            if output_format == 'numpy':
                self.assertNumpyAll(ret[1]['bias'].temporal.value,
                                    np.array([-712208.5,-712117. ,-712025. ,-711933.5]))
                self.assertNumpyAll(ret[1]['bias'].temporal.bounds,
                                    np.array([[-712254.,-712163.],[-712163.,-712071.],[-712071.,-711979.],[-711979.,-711888.]]))
            
            if output_format == 'csv':
                with open(ret) as f:
                    reader = DictReader(f)
                    for row in reader:
                        self.assertTrue(all([row[k] == '' for k in ['YEAR','MONTH','DAY']]))
                        self.assertTrue(float(row['TIME']) < -50000)
                        
            if output_format == 'nc':
                self.assertNcEqual(dataset.uri,ret,check_types=False,ignore_attributes={'global': ['history']})
        
    def test_time_region_climatology(self):
        ocgis.env.DIR_DATA = '/usr/local/climate_data'
        
        uri = 'climatology_TNn_monthly_max.nc'
        variable = 'climatology_TNn_monthly_max'
        rd = ocgis.RequestDataset(uri,variable,time_region={'year':[1989],'month':[6]})
        ops = ocgis.OcgOperations(dataset=rd,geom='state_boundaries',select_ugid=[16])
        ret = ops.execute()
        ref = ret[16]['climatology_TNn_monthly_max']
        self.assertEqual(set([6]),set([dt.month for dt in ref.temporal.value_datetime]))
        self.assertNumpyAll(np.array([[   151.,  10774.]]),ref.temporal.bounds)
        
        uri = 'climatology_TNn_monthly_max.nc'
        variable = 'climatology_TNn_monthly_max'
        rd = ocgis.RequestDataset(uri,variable,time_region={'year':None,'month':[6]})
        ops = ocgis.OcgOperations(dataset=rd,geom='state_boundaries',select_ugid=[16])
        ret = ops.execute()
        ref = ret[16]['climatology_TNn_monthly_max']
        self.assertEqual(set([6]),set([dt.month for dt in ref.temporal.value_datetime]))
        
        rd = ocgis.RequestDataset('climatology_TNn_annual_min.nc','climatology_TNn_annual_min')
        ops = ocgis.OcgOperations(dataset=rd,geom='state_boundaries',select_ugid=[16])
        ret = ops.execute()
        ref = ret[16]['climatology_TNn_annual_min']
        
        rd = ocgis.RequestDataset('climatology_TasMin_seasonal_max_of_seasonal_means.nc','climatology_TasMin_seasonal_max_of_seasonal_means')#,time_region={'year':[1989]})
        ops = ocgis.OcgOperations(dataset=rd,geom='state_boundaries',select_ugid=[16])
        ret = ops.execute()
        ref = ret[16]['climatology_TasMin_seasonal_max_of_seasonal_means']
        
        uri = 'climatology_Tas_annual_max_of_annual_means.nc'
        variable = 'climatology_Tas_annual_max_of_annual_means'
        rd = ocgis.RequestDataset(uri,variable)
        ops = ocgis.OcgOperations(dataset=rd,geom='state_boundaries',select_ugid=[16])
        ret = ops.execute()
        ref = ret[16][variable]
        
    def test_mfdataset_to_nc(self):
        rd = self.test_data.get_rd('maurer_2010_pr')
        ops = OcgOperations(dataset=rd,output_format='nc',calc=[{'func':'mean','name':'my_mean'}],
                            calc_grouping=['year'],geom='state_boundaries',select_ugid=[23])
        ret = ops.execute()
        field = RequestDataset(ret,'my_mean').get()
        self.assertNumpyAll(field.temporal.value,np.array([ 18444.,  18809.]))

        
if __name__ == '__main__':
    unittest.main()
