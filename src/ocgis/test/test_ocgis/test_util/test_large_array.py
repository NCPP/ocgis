from ocgis.test.base import TestBase
import ocgis
from ocgis.util.large_array import compute
import netCDF4 as nc
import numpy as np
from ocgis.calc import tile
from ocgis.api.request.base import RequestDatasetCollection
from ocgis.test.test_base import longrunning
from copy import deepcopy
import time


class Test(TestBase):
    
    @longrunning
    def test_timing_use_optimizations(self):
        n = range(10)
        t = {True:[],False:[]}
        
        for use_optimizations in [True,False]:
            for ii in n:
                t1 = time.time()
                rd = self.test_data.get_rd('cancm4_tas')
                ops = ocgis.OcgOperations(dataset=rd,calc=[{'func':'mean','name':'mean'}],
                                          calc_grouping=['month'],output_format='nc',
                                          geom='state_boundaries',
                                          select_ugid=[2,9,12,23,25],
                                          add_auxiliary_files=False,
                                          prefix=str(ii)+str(use_optimizations))
                compute(ops,5,verbose=False,use_optimizations=use_optimizations)
                t2 = time.time()
                t[use_optimizations].append(t2-t1)
        tmean = {k:{'mean':np.array(v).mean(),'stdev':np.array(v).std()} for k,v in t.iteritems()}
        self.assertTrue(tmean[True]['mean'] < tmean[False]['mean'])
    
    def test_multivariate_computation(self):
        rd = self.test_data.get_rd('cancm4_tas',kwds={'time_region':{'month':[3]}})
        rd2 = deepcopy(rd)
        rd2.alias = 'tas2'
        calc = [{'func':'divide','name':'ln','kwds':{'arr1':'tas','arr2':'tas2'}}]
        ops = ocgis.OcgOperations(dataset=[rd,rd2],calc=calc,
                                  calc_grouping=['month'],output_format='nc',
                                  geom='state_boundaries',
                                  select_ugid=[2,9,12,23,25],
                                  add_auxiliary_files=False)
        ret = compute(ops,5,verbose=False)
        
        ops.prefix = 'ocgis'
        ret_ocgis = ops.execute()
        self.assertNcEqual(ret,ret_ocgis,ignore_attributes={'global': ['history']})
    
    def test_with_no_calc_grouping(self):
        rd = self.test_data.get_rd('cancm4_tas',kwds={'time_region':{'month':[3]}})
        ops = ocgis.OcgOperations(dataset=rd,calc=[{'func':'ln','name':'ln'}],
                                  calc_grouping=None,output_format='nc',
                                  geom='state_boundaries',
                                  select_ugid=[2,9,12,23,25],
                                  add_auxiliary_files=False)
        ret = compute(ops,5,verbose=False)
        
        ops.prefix = 'ocgis'
        ret_ocgis = ops.execute()
        self.assertNcEqual(ret,ret_ocgis,ignore_attributes={'global': ['history']})
    
    def test_compute_with_time_region(self):
        rd = self.test_data.get_rd('cancm4_tas',kwds={'time_region':{'month':[3]}})
        ops = ocgis.OcgOperations(dataset=rd,calc=[{'func':'mean','name':'mean'}],
                                  calc_grouping=['month'],output_format='nc',
                                  geom='state_boundaries',
                                  select_ugid=[2,9,12,23,25],
                                  add_auxiliary_files=False)
        ret = compute(ops,5,verbose=False)
        
        ops.prefix = 'ocgis'
        ret_ocgis = ops.execute()
        self.assertNcEqual(ret,ret_ocgis,ignore_attributes={'global': ['history']})
    
    def test_compute_with_geom(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd,calc=[{'func':'mean','name':'mean'}],
                                  calc_grouping=['month'],output_format='nc',
                                  geom='state_boundaries',
                                  select_ugid=[2,9,12,23,25],
                                  add_auxiliary_files=False)
        ret = compute(ops,5,verbose=False)
        
        ops.prefix = 'ocgis'
        ret_ocgis = ops.execute()
        self.assertNcEqual(ret,ret_ocgis,ignore_attributes={'global': ['history']})
    
    def test_compute_small(self):
        rd = self.test_data.get_rd('cancm4_tas')
        
        ## use a smaller netCDF as target
        ops = ocgis.OcgOperations(dataset=rd,
                                  geom='state_boundaries',
                                  select_ugid=[2,9,12,23,25],
                                  output_format='nc',
                                  prefix='sub',
                                  add_auxiliary_files=False)
        sub = ops.execute()
        
        ## use the compute function
        rd_sub = ocgis.RequestDataset(sub,'tas')
        ops = ocgis.OcgOperations(dataset=rd_sub,calc=[{'func':'mean','name':'mean'}],
                                  calc_grouping=['month'],output_format='nc',
                                  add_auxiliary_files=False)
        ret_compute = compute(ops,5,verbose=False)
        
        ## now just run normally and ensure the answers are the same!
        ops.prefix = 'ocgis_compare'
        ops.add_auxiliary_files = False
        ret_ocgis = ops.execute()
        self.assertNcEqual(ret_compute,ret_ocgis,ignore_attributes={'global': ['history']})
    
    @longrunning
    def test_compute_large(self):
        """Test calculations using compute are equivalent with standard calculations."""

#        ocgis.env.VERBOSE = True
#        ocgis.env.DEBUG = True

        verbose = False
        n_tile_dimensions = 1
        tile_range = [100, 100]

        rd = RequestDatasetCollection(self.test_data.get_rd('cancm4_tasmax_2011'))
        
        calc = [{'func': 'mean', 'name': 'my_mean'},
                {'func': 'freq_perc', 'name': 'perc_90', 'kwds': {'percentile': 90}},
                {'func': 'freq_perc', 'name': 'perc_95', 'kwds': {'percentile': 95}},
                {'func': 'freq_perc', 'name': 'perc_99', 'kwds': {'percentile': 99}}]
        calc_grouping = ['month']
        
        ## construct the operational arguments to compute
        ops_compute = ocgis.OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, output_format='nc',
                                          prefix='tile')
        
        ## perform computations the standard way
        if verbose:
            print('computing standard file...')
        ops = ocgis.OcgOperations(dataset=rd, output_format='nc', calc=calc, calc_grouping=calc_grouping, prefix='std')
        std_file = ops.execute()
        if verbose:
            print('standard file is: {0}'.format(std_file))
        std_ds = nc.Dataset(std_file,'r')

        for ii in range(n_tile_dimensions):
            tile_dimension = np.random.random_integers(tile_range[0],tile_range[1])
            if verbose:
                print('tile dimension: {0}'.format(tile_dimension))
            ## perform computations using tiling
            tile_file = compute(ops_compute, tile_dimension, verbose=verbose)
            
            ## ensure output paths are different
            self.assertNotEqual(tile_file, std_file)
            
            self.assertNcEqual(std_file, tile_file, ignore_attributes={'global': ['history']})
            
            ## confirm each variable is identical
            tile_ds = nc.Dataset(tile_file, 'r')
            
            tile_ds.close()
        std_ds.close()
    
    def get_random_integer(self,low=1,high=100):
        return(int(np.random.random_integers(low,high)))

    def test_tile_get_tile_schema(self):
        schema = tile.get_tile_schema(5,5,2)
        self.assertEqual(len(schema),9)
        
        schema = tile.get_tile_schema(25,1,2)
        self.assertEqual(len(schema),13)
        
    def test_tile_sum(self):
        ntests = 1000
        for ii in range(ntests):
            nrow,ncol,tdim = [self.get_random_integer() for ii in range(3)]
            x = np.random.rand(nrow,ncol)
            y = np.empty((nrow,ncol),dtype=float)
            schema = tile.get_tile_schema(nrow,ncol,tdim)
            tidx = schema[0]
            row = tidx['row']
            col = tidx['col']
            self.assertTrue(np.all(x[row[0]:row[1],col[0]:col[1]] == x[0:tdim,0:tdim]))
            running_sum = 0.0
            for value in schema.itervalues():
                row,col = value['row'],value['col']
                slice = x[row[0]:row[1],col[0]:col[1]]
                y[row[0]:row[1],col[0]:col[1]] = slice
                running_sum += slice.sum()
            self.assertAlmostEqual(running_sum,x.sum())
            self.assertTrue(np.all(x == y))
