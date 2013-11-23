from ocgis.test.base import TestBase
import ocgis
from ocgis.util.large_array import compute
import netCDF4 as nc
import numpy as np
from ocgis.calc import tile
from ocgis.api.request.base import RequestDatasetCollection
from ocgis.test.test_base import longrunning


class Test(TestBase):
    
    @longrunning
    def test_compute(self):
#        ocgis.env.VERBOSE = True
#        ocgis.env.DEBUG = True

        verbose = False
        n_tile_dimensions = 1
        tile_range = [100,100]
        rd = RequestDatasetCollection(self.test_data.get_rd('cancm4_tasmax_2011'))
        
        calc = [{'func':'mean','name':'my_mean'},
                {'func':'freq_perc','name':'perc_90','kwds':{'percentile':90,}},
                {'func':'freq_perc','name':'perc_95','kwds':{'percentile':95,}},
                {'func':'freq_perc','name':'perc_99','kwds':{'percentile':99,}}
               ]
        calc_grouping = ['month']
        
        ## perform computations the standard way
        if verbose: print('computing standard file...')
        ops = ocgis.OcgOperations(dataset=rd,output_format='nc',calc=calc,
                                      calc_grouping=calc_grouping,prefix='std')
        std_file = ops.execute()
        if verbose: print('standard file is: {0}'.format(std_file))
        std_ds = nc.Dataset(std_file,'r')

        for ii in range(n_tile_dimensions):
            tile_dimension = np.random.random_integers(tile_range[0],tile_range[1])
            if verbose: print('tile dimension: {0}'.format(tile_dimension))
            ## perform computations using tiling
            tile_file = compute(rd,calc,calc_grouping,tile_dimension,verbose=verbose,
                                 prefix='tile')
            
            ## ensure output paths are different
            self.assertNotEqual(tile_file,std_file)
            
            self.assertNcEqual(std_file,tile_file)
            
            ## confirm each variable is identical
            tile_ds = nc.Dataset(tile_file,'r')
            
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
