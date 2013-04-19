from ocgis.test.base import TestBase
import ocgis
from ocgis.util.large_array import compute
import netCDF4 as nc
import numpy as np


class Test(TestBase):

    def test_compute(self):
        verbose = False
        n_tile_dimensions = 1
        tile_range = [100,100]
        rd = self.test_data.get_rd('cancm4_tasmax_2011')
        
        calc = [{'func':'mean','name':'my_mean'},
                {'func':'freq_perc','name':'perc_90','kwds':{'perc':90,}},
                {'func':'freq_perc','name':'perc_95','kwds':{'perc':95,}},
                {'func':'freq_perc','name':'perc_99','kwds':{'perc':99,}}
               ]
        calc_grouping = ['month']
        
        ## perform computations the standard way
        if verbose: print('computing standard file...')
        ops = ocgis.OcgOperations(dataset=rd,output_format='nc',calc=calc,
                                      calc_grouping=calc_grouping,prefix='std')
        std_file = ops.execute()
        if verbose: print('standard file is: {0}'.format(std_file))
        std_ds = nc.Dataset(std_file,'r')
        std_meta = ocgis.Inspect(std_file).meta
        
        for ii in range(n_tile_dimensions):
            tile_dimension = np.random.random_integers(tile_range[0],tile_range[1])
            if verbose: print('tile dimension: {0}'.format(tile_dimension))
            ## perform computations using tiling
            tile_file = compute(rd,calc,calc_grouping,tile_dimension,verbose=verbose,
                                 prefix='tile')
            
            ## ensure output paths are different
            self.assertNotEqual(tile_file,std_file)
            
            ## confirm each variable is identical
            tile_ds = nc.Dataset(tile_file,'r')
            
            ## compare calculated values
            for element in calc:
                tile_value,std_value = [ds.variables[element['name']][:] for ds in [tile_ds,std_ds]]
                cmp = tile_value == std_value
                self.assertTrue(cmp.all())
                
            ## compare meta dictionaries
            tile_meta = ocgis.Inspect(tile_file).meta
            for k in tile_meta.iterkeys():
                for k2,v2 in tile_meta[k].iteritems():
                    ref = std_meta[k][k2]
                    self.assertEqual(v2,ref)
            
            tile_ds.close()
        std_ds.close()