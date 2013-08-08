import ocgis
from ocgis.test.base import TestBase
import itertools
from unittest.case import SkipTest
from ocgis.api.operations import OcgOperations
from ocgis.interface.nc.dataset import NcDataset
import webbrowser
from datetime import datetime as dt
from ocgis.exc import DefinitionValidationError
import numpy as np
import datetime


class Test(TestBase):
    
    def test_maurer_concatenated_shp(self):
        ocgis.env.DIR_DATA = '/usr/local/climate_data/maurer/2010-concatenated'
#        filename = 'Maurer02new_OBS_tasmax_daily.1971-2000.nc'
#        variable = 'tasmax'
        ocgis.env.VERBOSE = True
        
        names = [
#         [u'Maurer02new_OBS_dtr_daily.1971-2000.nc'],
         [u'Maurer02new_OBS_tas_daily.1971-2000.nc'], 
         [u'Maurer02new_OBS_tasmin_daily.1971-2000.nc'], 
         [u'Maurer02new_OBS_pr_daily.1971-2000.nc'], 
         [u'Maurer02new_OBS_tasmax_daily.1971-2000.nc']]
        variables = [
#                     u'dtr', 
                     u'tas', u'tasmin', u'pr', u'tasmax']
#        time_range = [datetime.datetime(1971, 1, 1, 0, 0),datetime.datetime(2000, 12, 31, 0, 0)]
        time_range = None
        time_region = {'month':[6,7,8],'year':None}
        rds = [ocgis.RequestDataset(name,variable,time_range=time_range,
            time_region=time_region) for name,variable in zip(names,variables)]
        
        ops = ocgis.OcgOperations(dataset=rds,calc=[{'name': 'Standard Deviation', 'func': 'std', 'kwds': {}}],
         calc_grouping=['month'],calc_raw=False,geom='us_counties',select_ugid=[286],output_format='shp',
         spatial_operation='clip',headers=['did','ugid','gid','year','month','day','variable','calc_name','value'],
         abstraction=None)
        ret = ops.execute()
    
    def test_point_shapefile_subset(self):
        _output_format = ['numpy','nc','csv','csv+']
        for output_format in _output_format:
            rd = self.test_data.get_rd('cancm4_tas')
            ops = OcgOperations(dataset=rd,geom='qed_city_centroids',output_format=output_format,
                                prefix=output_format)
            ret = ops.execute()
            if output_format == 'numpy':
                self.assertEqual(len(ret),3)
                
    def test_maurer_concatenated_tasmax_region(self):
        ocgis.env.DIR_DATA = '/usr/local/climate_data/maurer/2010-concatenated'
        filename = 'Maurer02new_OBS_tasmax_daily.1971-2000.nc'
        variable = 'tasmax'
#        ocgis.env.VERBOSE = True
        
        rd = ocgis.RequestDataset(filename,variable)
        ops = ocgis.OcgOperations(dataset=rd,geom='us_counties',select_ugid=[2778],
                                  output_format='numpy')
        ret = ops.execute()
        ref = ret[2778].variables['tasmax']
        years = np.array([dt.year for dt in ret[2778].variables['tasmax'].temporal.value])
        months = np.array([dt.month for dt in ret[2778].variables['tasmax'].temporal.value])
        select = np.array([dt.month in (6,7,8) and dt.year in (1990,1991,1992,1993,1994,1995,1996,1997,1998,1999) for dt in ret[2778].variables['tasmax'].temporal.value])
        time_subset = ret[2778].variables['tasmax'].value[select,:,:,:]
        time_values = ref.temporal.value[select]
        
        rd = ocgis.RequestDataset(filename,variable,time_region={'month':[6,7,8],'year':[1990,1991,1992,1993,1994,1995,1996,1997,1998,1999]})
        ops = ocgis.OcgOperations(dataset=rd,geom='us_counties',select_ugid=[2778],
                                  output_format='numpy')
        ret2 = ops.execute()
        ref2 = ret2[2778].variables['tasmax']
        
        self.assertEqual(time_values.shape,ref2.temporal.shape)
        self.assertEqual(time_subset.shape,ref2.value.shape)
        self.assertNumpyAll(time_subset,ref2.value)
        self.assertFalse(np.any(ref2.value < 0))
        
    
    def test_time_region_subset(self):
        
        _month = [[6,7],[12],None,[1,3,8]]
        _year = [[2011],None,[2012],[2011,2013]]
        
        def run_test(month,year):
            rd = self.test_data.get_rd('cancm4_rhs')
            time_region = {'month':month,'year':year}
            rd.time_region = time_region
            
            ops = ocgis.OcgOperations(dataset=rd,geom='state_boundaries',
                                      select_ugid=[25])
            ret = ops.execute()
            
            ret = ret[25].variables['rhs'].temporal.value
            
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
        ref = ret[25].variables['rhs']
        years = set([obj.year for obj in ref.temporal.value])
        self.assertFalse(2015 in years)
        
        time_range = [dt(2013,1,1),dt(2015,12,31)]
        time_region = {'month':[6,7,8],'year':[2013,2014,2018]}
        kwds = {'time_range':time_range,'time_region':time_region}
        with self.assertRaises(DefinitionValidationError):
            self.test_data.get_rd('cancm4_rhs',kwds=kwds)
            
    def test_maurer_2010(self):
        raise(SkipTest('dev'))
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
            self.assertTrue(ret[10].variables[rd.variable].value.sum() != 0)
            self.assertTrue(ret[15].variables[rd.variable].value.sum() != 0)
            
            ops = ocgis.OcgOperations(dataset=rd,snippet=False,select_ugid=[10,15],
                   output_format='numpy',geom='state_boundaries',calc=calc,
                   calc_grouping=calc_grouping)
            ret = ops.execute()
            for calc_name in ['mean','median','n']:
                self.assertEqual(ret[10].calc[rd.variable][calc_name].shape[0],12)
                
            ops = ocgis.OcgOperations(dataset=rd,snippet=False,select_ugid=[10,15],
                   output_format='csv+',geom='state_boundaries',calc=calc,
                   calc_grouping=calc_grouping,prefix=key)
            ret = ops.execute()
            
    def test_clip_aggregate(self):
        ## this geometry was hanging
#        ocgis.env.VERBOSE = True
#        ocgis.env.DEBUG = True
        rd = self.test_data.get_rd('cancm4_tas',kwds={'time_region':{'year':[2003]}})
        ops = OcgOperations(dataset=rd,geom='state_boundaries',select_ugid=[14,16],
                            aggregate=False,spatial_operation='clip',output_format='csv+')
        ret = ops.execute()
            
    def test_QED_2013(self):
        raise(SkipTest('data changes rapidly - for development purposes only'))
        variable = 'tasmax'
        ocgis.env.DIR_DATA = '/usr/local/climate_data/QED-2013'
        files = ['Maurer_11day_double_tasmax.nc','Maurer_11day_float_tasmax.nc']
        rds = [ocgis.RequestDataset(f,variable,alias=f) for f in files]
        calc = [{'func':'mean','name':'mean'},{'func':'median','name':'median'}]
        calc_grouping = ['month']
        
        for rd in rds:
            dct = rd.inspect_as_dct()
            
        ops = ocgis.OcgOperations(dataset=rds,snippet=False,select_ugid=[10,15],
                   output_format='csv+',geom='state_boundaries',calc=calc,
                   calc_grouping=calc_grouping)
        ret = ops.execute()
        
    def test_narccap_point_subset_small(self):
        rd = self.test_data.get_rd('narccap_pr_wrfg_ncep')
        geom = [-97.74278,30.26694]
        ocgis.env.WRITE_TO_REFERENCE_PROJECTION = True
        ocgis.env.VERBOSE = False
    
        calc = [{'func':'mean','name':'mean'},
                {'func':'median','name':'median'},
                {'func':'max','name':'max'},
                {'func':'min','name':'min'}]
        calc_grouping = ['month','year']
        ops = ocgis.OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                                  output_format='csv+',geom=geom,abstraction='point',
                                  snippet=False,allow_empty=False)
        ret = ops.execute()
    
    def test_narccap_point_subset_long(self):
        raise(SkipTest('dev'))
        import sys
        sys.path.append('/home/local/WX/ben.koziol/links/git/examples/')
        from narccap.co_watersheds_subset import parse_narccap_filenames
        snippet = False
        ## city center coordinate
        geom = [-97.74278,30.26694]
        ocgis.env.DIR_DATA = '/usr/local/climate_data/narccap'
        ocgis.env.WRITE_TO_REFERENCE_PROJECTION = True
        ocgis.env.VERBOSE = True
        
        rds = parse_narccap_filenames(ocgis.env.DIR_DATA)
        calc = [{'func':'mean','name':'mean'},
                {'func':'median','name':'median'},
                {'func':'max','name':'max'},
                {'func':'min','name':'min'}]
        calc_grouping = ['month','year']
        ops = ocgis.OcgOperations(dataset=rds,calc=calc,calc_grouping=calc_grouping,
                                  output_format='csv+',geom=geom,abstraction='point',
                                  snippet=snippet,allow_empty=False)
        ret = ops.execute()

    def test_qed_maurer_concatenated(self):
        raise(SkipTest('dev'))
        calc = [{'func':'freq_duration','name':'freq_duration','kwds':{'operation':'gt','threshold':15}}]
        ocgis.env.DIR_DATA = '/usr/local/climate_data/maurer/2010-concatenated'
        filename = 'Maurer02new_OBS_tasmax_daily.1971-2000.nc'
        variable = 'tasmax'
        rd = ocgis.RequestDataset(filename,variable)
        ocgis.env.VERBOSE = True
        ops = ocgis.OcgOperations(dataset=rd,geom='gg_city_centroids',select_ugid=None,
                                  calc=calc,calc_grouping=['month','year'],output_format='csv+')
        ret = ops.execute()
        webbrowser.open(ret)
        import ipdb;ipdb.set_trace()