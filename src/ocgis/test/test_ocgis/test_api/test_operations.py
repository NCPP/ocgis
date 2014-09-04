import unittest
from ocgis.test.base import TestBase
from ocgis.exc import DefinitionValidationError, DimensionNotFound, RequestValidationError
from ocgis.api.parms import definition
from ocgis import env, constants
import os
from datetime import datetime as dt
from ocgis.api.operations import OcgOperations
from ocgis.util.helpers import make_poly
import itertools
import ocgis
from ocgis.api.request.base import RequestDataset, RequestDatasetCollection
from ocgis.util.shp_cabinet import ShpCabinetIterator
import datetime
from numpy import dtype


class TestOcgOperations(TestBase):
    
    def setUp(self):
        TestBase.setUp(self)
        env.DIR_DATA = os.path.join(env.DIR_TEST_DATA,'CanCM4')
        
        ## data may need to be pulled from remote repository
        self.test_data.get_rd('cancm4_tasmin_2001')
        self.test_data.get_rd('cancm4_tasmax_2011')
        self.test_data.get_rd('cancm4_tas')
        
        uris = [
                'tasmin_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc',
                'tasmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',
                'tas_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc'
                ]
        vars = ['tasmin','tasmax','tas']
        time_range = [dt(2000,1,1),dt(2000,12,31)]
        level_range = [2,2]
        self.datasets = [{'uri':uri,'variable':var,'time_range':time_range,'level_range':level_range} for uri,var in zip(uris,vars)]
        self.datasets_no_range = [{'uri':uri,'variable':var} for uri,var in zip(uris,vars)]

    def test_conform_units_to(self):
        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tas')
        rd2.alias = 'foo'
        ops = OcgOperations(dataset=[rd1, rd2], conform_units_to='celsius')
        for ds in ops.dataset.itervalues():
            self.assertEqual(ds.conform_units_to, 'celsius')

        ## test that the conform argument is updated
        ops.conform_units_to = 'fahrenheit'
        for ds in ops.dataset.itervalues():
            self.assertEqual(ds.conform_units_to, 'fahrenheit')

    def test_conform_units_to_bad_units(self):
        rd = self.test_data.get_rd('cancm4_tas')
        with self.assertRaises(RequestValidationError):
            OcgOperations(dataset=rd, conform_units_to='crap')

    def test_no_calc_grouping_with_string_expression(self):
        calc = 'es=tas*3'
        calc_grouping = ['month']
        rd = self.test_data.get_rd('cancm4_tas')
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping)

    def test_time_range(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tas')
        rd.alias = 'foo'
        tr = [datetime.datetime(2002,1,1),datetime.datetime(2002,3,1)]
        ops = ocgis.OcgOperations(dataset=[rd,rd2],time_range=tr)
        for r in [rd,rd2]:
            self.assertEqual(r.time_range,None)
        for r in ops.dataset.itervalues():
            self.assertEqual(r.time_range,tuple(tr))
            
        tr = [datetime.datetime(2002,1,1),datetime.datetime(2003,3,1)]
        ops.time_range = tr
        for r in ops.dataset.itervalues():
            self.assertEqual(r.time_range,tuple(tr))
            
    def test_time_region(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tas')
        rd.alias = 'foo'
        tr = {'month':[6],'year':[2005]}
        ops = ocgis.OcgOperations(dataset=[rd,rd2],time_region=tr)
        for r in [rd,rd2]:
            self.assertEqual(r.time_region,None)
        for r in ops.dataset.itervalues():
            self.assertEqual(r.time_region,tr)
            
        tr = {'month':[6],'year':[2006]}
        ops.time_region = tr
        for r in ops.dataset.itervalues():
            self.assertEqual(r.time_region,tr)
            
    def test_level_range(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tas')
        rd.alias = 'foo'
        lr = [1,2]
        ops = ocgis.OcgOperations(dataset=[rd,rd2],level_range=lr)
        for r in [rd,rd2]:
            self.assertEqual(r.level_range,None)
        for r in ops.dataset.itervalues():
            self.assertEqual(r.level_range,tuple(lr))
            
        lr = [2,3]
        ops.level_range = lr
        for r in ops.dataset.itervalues():
            self.assertEqual(r.level_range,tuple(lr))

    def test_nc_package_validation_raised_first(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('rotated_pole_ichec',kwds={'alias':'tas2'})
        try:
            ocgis.OcgOperations(dataset=[rd,rd2],output_format='nc')
        except DefinitionValidationError as e:
            self.assertIn('Data packages (i.e. more than one RequestDataset) may not be written to netCDF.',
                          e.message)
            pass
    
    def test_with_callback(self):
        
        app = []
        def callback(perc,msg,app=app):
            app.append((perc,msg))
#            print(perc,msg)
            
        rd = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tasmax_2011')
        dataset = [rd,rd2]
        for ds in dataset:
            ds.time_region = {'month':[6]}
        ops = ocgis.OcgOperations(dataset=dataset,geom='state_boundaries',select_ugid=[16,17],
                                  calc_grouping=['month'],calc=[{'func':'mean','name':'mean'},{'func':'median','name':'median'}],
                                  callback=callback)
        ops.execute()
        
        self.assertTrue(len(app) > 15)
        self.assertEqual(app[-1][0],100.0)
    
    def test_get_base_request_size(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd)
        size = ops.get_base_request_size()
        self.assertEqual(size,{'variables': {'tas': {'level': {'kb': 0.0, 'shape': None, 'dtype': None}, 'temporal': {'kb': 28.515625, 'shape': (3650,), 'dtype': dtype('float64')}, 'value': {'kb': 116800.0, 'shape': (1, 3650, 1, 64, 128), 'dtype': dtype('float32')}, 'realization': {'kb': 0.0, 'shape': None, 'dtype': None}, 'col': {'kb': 1.0, 'shape': (128,), 'dtype': dtype('float64')}, 'row': {'kb': 0.5, 'shape': (64,), 'dtype': dtype('float64')}}}, 'total': 116830.015625})
        
    def test_get_base_request_size_with_geom(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd,geom='state_boundaries',select_ugid=[23])
        size = ops.get_base_request_size()
        self.assertEqual(size,{'variables': {'tas': {'level': {'kb': 0.0, 'shape': None, 'dtype': None}, 'temporal': {'kb': 28.515625, 'shape': (3650,), 'dtype': dtype('float64')}, 'value': {'kb': 171.09375, 'shape': (1, 3650, 1, 4, 3), 'dtype': dtype('float32')}, 'realization': {'kb': 0.0, 'shape': None, 'dtype': None}, 'col': {'kb': 0.0234375, 'shape': (3,), 'dtype': dtype('float64')}, 'row': {'kb': 0.03125, 'shape': (4,), 'dtype': dtype('float64')}}}, 'total': 199.6640625})
        
    def test_get_base_request_size_multifile(self):
        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('narccap_pr_wrfg_ncep')
        rds = [rd1,rd2]
        ops = OcgOperations(dataset=rds)
        size = ops.get_base_request_size()
        self.assertEqual({'variables': {'pr': {'level': {'kb': 0.0, 'shape': None, 'dtype': None}, 'temporal': {'kb': 228.25, 'shape': (29216,), 'dtype': dtype('float64')}, 'value': {'kb': 1666909.75, 'shape': (1, 29216, 1, 109, 134), 'dtype': dtype('float32')}, 'realization': {'kb': 0.0, 'shape': None, 'dtype': None}, 'col': {'kb': 1.046875, 'shape': (134,), 'dtype': dtype('float64')}, 'row': {'kb': 0.8515625, 'shape': (109,), 'dtype': dtype('float64')}}, 'tas': {'level': {'kb': 0.0, 'shape': None, 'dtype': None}, 'temporal': {'kb': 28.515625, 'shape': (3650,), 'dtype': dtype('float64')}, 'value': {'kb': 116800.0, 'shape': (1, 3650, 1, 64, 128), 'dtype': dtype('float32')}, 'realization': {'kb': 0.0, 'shape': None, 'dtype': None}, 'col': {'kb': 1.0, 'shape': (128,), 'dtype': dtype('float64')}, 'row': {'kb': 0.5, 'shape': (64,), 'dtype': dtype('float64')}}}, 'total': 1783969.9140625},size)
        
    def test_get_base_request_size_multifile_with_geom(self):
        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('narccap_pr_wrfg_ncep')
        rds = [rd1,rd2]
        ops = OcgOperations(dataset=rds,geom='state_boundaries',select_ugid=[23])
        size = ops.get_base_request_size()
        self.assertEqual(size,{'variables': {'pr': {'level': {'kb': 0.0, 'shape': None, 'dtype': None}, 'temporal': {'kb': 228.25, 'shape': (29216,), 'dtype': dtype('float64')}, 'value': {'kb': 21341.375, 'shape': (1, 29216, 1, 17, 11), 'dtype': dtype('float32')}, 'realization': {'kb': 0.0, 'shape': None, 'dtype': None}, 'col': {'kb': 0.0859375, 'shape': (11,), 'dtype': dtype('float64')}, 'row': {'kb': 0.1328125, 'shape': (17,), 'dtype': dtype('float64')}}, 'tas': {'level': {'kb': 0.0, 'shape': None, 'dtype': None}, 'temporal': {'kb': 28.515625, 'shape': (3650,), 'dtype': dtype('float64')}, 'value': {'kb': 171.09375, 'shape': (1, 3650, 1, 4, 3), 'dtype': dtype('float32')}, 'realization': {'kb': 0.0, 'shape': None, 'dtype': None}, 'col': {'kb': 0.0234375, 'shape': (3,), 'dtype': dtype('float64')}, 'row': {'kb': 0.03125, 'shape': (4,), 'dtype': dtype('float64')}}}, 'total': 21769.5078125})

    def test_get_base_request_size_test_data(self):
        for key in self.test_data.keys():
            rd = self.test_data.get_rd(key)
            try:
                ops = OcgOperations(dataset=rd)
            ## the project cmip data may raise an exception since projection is
            ## not associated with a variable
            except DimensionNotFound:
                rd = self.test_data.get_rd(key,kwds=dict(dimension_map={'R':'projection','T':'time','X':'longitude','Y':'latitude'}))
                ops = OcgOperations(dataset=rd)
            ret = ops.get_base_request_size()
            self.assertTrue(ret['total'] > 1)
            
    def test_get_base_request_size_with_calculation(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd,calc=[{'func':'mean','name':'mean'}],
                            calc_grouping=['month'])
        size = ops.get_base_request_size()
        self.assertEqual(size['variables']['tas']['temporal']['shape'][0],3650)

    def test_str(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd)
        ret = str(ops)
        self.assertTrue(str(ret).startswith('OcgOperations'))
        self.assertGreater(len(ret), 1000)

    def test_get_meta(self):
        ops = OcgOperations(dataset=self.datasets)
        meta = ops.get_meta()
        self.assertTrue(len(meta) > 100)
        self.assertTrue('\n' in meta)
        
        ops = OcgOperations(dataset=self.datasets,calc=[{'func':'mean','name':'my_mean'}],
                            calc_grouping=['month'])
        meta = ops.get_meta()
        self.assertTrue(len(meta) > 100)
        self.assertTrue('\n' in meta)

    def test_null_parms(self):
        ops = OcgOperations(dataset=self.datasets_no_range)
        self.assertEqual(ops.geom,None)
        self.assertEqual(len(ops.dataset),3)
        for ds in ops.dataset.itervalues():
            self.assertEqual(ds.time_range,None)
            self.assertEqual(ds.level_range,None)
        ops.__repr__()
    
    def test_aggregate(self):
        A = definition.Aggregate

        a = A(True)
        self.assertEqual(a.value,True)
        
        a = A(False)
        self.assertEqual(a.value,False)
        
        a = A('True')
        self.assertEqual(a.value,True)
    
    def test_geom_string(self):
        ops = OcgOperations(dataset=self.datasets,geom='state_boundaries')
        self.assertEqual(len(list(ops.geom)),51)
        ops.geom = None
        self.assertEqual(ops.geom,None)
        ops.geom = 'mi_watersheds'
        self.assertEqual(len(list(ops.geom)),60)
        ops.geom = [-120,40,-110,50]
        self.assertEqual(ops.geom[0].single.geom.bounds,(-120.0,40.0,-110.0,50.0))
        
    def test_geom(self):
        geom = make_poly((37.762,38.222),(-102.281,-101.754))
        g = definition.Geom(geom)
        self.assertEqual(type(g.value),tuple)
        self.assertEqual(g.value[0].single.geom.bounds,(-102.281, 37.762, -101.754, 38.222))
        
        g = definition.Geom(None)
        self.assertEqual(g.value,None)
        self.assertEqual(str(g),'geom=None')
        
        g = definition.Geom('mi_watersheds')
        self.assertEqual(str(g),'geom="mi_watersheds"')
        
        geoms = ShpCabinetIterator('mi_watersheds')
        g = definition.Geom(geoms)
        self.assertEqual(len(list(g.value)),60)
        self.assertEqual(g._shp_key,'mi_watersheds')
        
    def test_geom_having_changed_select_ugid(self):
        ops = OcgOperations(dataset=self.test_data.get_rd('cancm4_tas'),
                            geom='state_boundaries')
        self.assertEqual(len(list(ops.geom)),51)
        ops.select_ugid = [16,17]
        self.assertEqual(len(list(ops.geom)),2)
        
    def test_headers(self):
        headers = ['did','value']
        for htype in [list,tuple]:
            hvalue = htype(headers)
            hh = definition.Headers(hvalue)
            self.assertEqual(hh.value,tuple(constants.required_headers+['value']))
            
        headers = ['foo']
        with self.assertRaises(DefinitionValidationError):
            hh = definition.Headers(headers)
            
        headers = []
        hh = definition.Headers(headers)
        self.assertEqual(hh.value,tuple(constants.required_headers))
                
    def test_calc_grouping_none_date_parts(self):
        _cg = [
               None,
               ['day','month'],
               'day'
               ]
        
        for cg in _cg:
            if cg is not None:
                eq = tuple(cg)
            else:
                eq = cg
            obj = definition.CalcGrouping(cg)
            try:
                self.assertEqual(obj.value,eq)
            except AssertionError:
                self.assertEqual(obj.value,('day',))
        
        ## only month, year, and day combinations are currently supported
        rd = self.test_data.get_rd('cancm4_tas')
        calcs = [None,[{'func':'mean','name':'mean'}]]
        acceptable = ['day','month','year']
        for calc in calcs:
            for length in [1,2,3,4,5]:
                for combo in itertools.combinations(['day','month','year','hour','minute'],length):
                    try:
                        ops = OcgOperations(dataset=rd,calc=calc,calc_grouping=combo)
                    except DefinitionValidationError:
                        reraise = True
                        for c in combo:
                            if c not in acceptable:
                                reraise = False
                        if reraise:
                            raise
                        
    def test_calc_grouping_seasonal_with_year(self):
        calc_grouping = [[1,2,3],'year']
        calc = [{'func':'mean','name':'mean'}]
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                            geom='state_boundaries',select_ugid=[25])
        ret = ops.execute()
        self.assertEqual(ret[25]['tas'].shape,(1,10,1,5,4))
        
    def test_calc_grouping_seasonal_with_unique(self):
        calc_grouping = [[12,1,2],'unique']
        calc = [{'func':'mean','name':'mean'}]
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd,calc_grouping=calc_grouping,geom='state_boundaries',
                                  select_ugid=[27],output_format='nc',calc=calc)
        ret = ops.execute()
        rd2 = ocgis.RequestDataset(uri=ret,variable='mean')
        field = rd2.get()
        self.assertNotEqual(field.temporal.bounds,None)
        self.assertEqual(field.temporal.bounds_datetime.tolist(),[[datetime.datetime(2002, 1, 1, 12, 0), datetime.datetime(2002, 2, 28, 12, 0)], [datetime.datetime(2003, 1, 1, 12, 0), datetime.datetime(2003, 2, 28, 12, 0)], [datetime.datetime(2004, 1, 1, 12, 0), datetime.datetime(2004, 2, 28, 12, 0)], [datetime.datetime(2005, 1, 1, 12, 0), datetime.datetime(2005, 2, 28, 12, 0)], [datetime.datetime(2006, 1, 1, 12, 0), datetime.datetime(2006, 2, 28, 12, 0)], [datetime.datetime(2007, 1, 1, 12, 0), datetime.datetime(2007, 2, 28, 12, 0)], [datetime.datetime(2008, 1, 1, 12, 0), datetime.datetime(2008, 2, 28, 12, 0)], [datetime.datetime(2009, 1, 1, 12, 0), datetime.datetime(2009, 2, 28, 12, 0)], [datetime.datetime(2010, 1, 1, 12, 0), datetime.datetime(2010, 2, 28, 12, 0)]])
        self.assertEqual(field.shape,(1, 9, 1, 3, 3))
                
    def test_dataset(self):
        env.DIR_DATA = ocgis.env.DIR_TEST_DATA
        reference_rd = self.test_data.get_rd('cancm4_tas')
        rd = RequestDataset(reference_rd.uri,reference_rd.variable)
        ds = definition.Dataset(rd)
        self.assertEqual(ds.value,RequestDatasetCollection([rd]))
        
        dsa = {'uri':reference_rd.uri,'variable':reference_rd.variable}
        ds = definition.Dataset(dsa)
        
        reference_rd2 = self.test_data.get_rd('narccap_crcm')
        dsb = [dsa,{'uri':reference_rd2.uri,'variable':reference_rd2.variable,'alias':'knight'}]
        ds = definition.Dataset(dsb)
        
    def test_abstraction(self):
        K = definition.Abstraction
        
        k = K()
        self.assertEqual(k.value,None)
        self.assertEqual(str(k),'abstraction="None"')
        
        k = K('point')
        self.assertEqual(k.value,'point')
        
        with self.assertRaises(DefinitionValidationError):
            K('pt')
            
    def test_spatial_operation(self):
        values = (None,'clip','intersects')
        ast = ('intersects','clip','intersects')
        
        klass = definition.SpatialOperation
        for v,a in zip(values,ast):
            obj = klass(v)
            self.assertEqual(obj.value,a)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()