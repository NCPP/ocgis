import unittest
import numpy as np
from ocgis.calc import library, tile
from ocgis.api.operations import OcgOperations
from datetime import datetime as dt
from ocgis.api.dataset.collection.iterators import MeltedIterator, KeyedIterator
import ocgis
import datetime
from ocgis.test.base import TestBase
import time


class Test(TestBase):
    
    def test_HeatIndex(self):
        kwds = {'time_range':[dt(2011,1,1),dt(2011,12,31,23,59,59)]}
        ds = [self.test_data.get_rd('cancm4_tasmax_2011',kwds=kwds),self.test_data.get_rd('cancm4_rhsmax',kwds=kwds)]
        calc = [{'func':'heat_index','name':'heat_index','kwds':{'tas':'tasmax','rhs':'rhsmax','units':'k'}}]
        
        ops = OcgOperations(dataset=ds,calc=calc)
        self.assertEqual(ops.calc_grouping,None)
        ret = ops.execute()
        ref = ret[1]
        self.assertEqual(ref.variables.keys(),['tasmax','rhsmax','heat_index'])
        hi = ref.variables['heat_index']
        self.assertEqual(hi.value.shape,(365,1,64,128))
        it = MeltedIterator(ret[1],mode='calc')
        for ii,row in enumerate(it.iter_rows()):
            if ii == 0:
                self.assertEqual(row['value'],None)
            if ii < 1000:
                for key in ['vid','var_name','did','uri']:
                    self.assertEqual(row[key],None)
            else:
                break
        
        ops = OcgOperations(dataset=ds,calc=calc,output_format='numpy',snippet=True)
        ret = ops.execute()
        
    def test_HeatIndex_keyed_output(self):
        ds = [self.test_data.get_rd('cancm4_tasmax_2011'),self.test_data.get_rd('cancm4_rhsmax')]
        calc = [{'func':'heat_index','name':'heat_index','kwds':{'tas':'tasmax','rhs':'rhsmax','units':'k'}}]
        ops = OcgOperations(dataset=ds,calc=calc,snippet=False,output_format='numpy')
        self.assertEqual(ops.calc_grouping,None)
        ret = ops.execute()
        it = KeyedIterator(ret[1],mode='calc')
        for ii,row in enumerate(it.iter_rows(ret[1])):
            if ii < 1000:
                self.assertEqual(row['cid'],1)
                self.assertEqual(row['tgid'],None)
                self.assertNotEqual(row['tid'],None)
            else:
                break
            
        ops = OcgOperations(dataset=ds,calc=calc,snippet=True,output_format='keyed')
        ops.execute()

    def test_Mean(self):
        agg = True
        weights = None
        values = np.ones((36,2,4,4))
        values = np.ma.array(values,mask=False)
        
        on = np.ones(12,dtype=bool)
        off = np.zeros(12,dtype=bool)
        
        groups = []
        base_groups = [[on,off,off],[off,on,off],[off,off,on]]
        for bg in base_groups:
            groups.append(np.concatenate(bg))
        
        mean = library.Mean(values=values,agg=agg,weights=weights,groups=groups)
        ret = mean.calculate()
        
    def test_computational_nc_output(self):
        rd = self.test_data.get_rd('cancm4_tasmax_2011',kwds={'time_range':[datetime.datetime(2011,1,1),datetime.datetime(2011,12,31)]})
        calc = [{'func':'mean','name':'tasmax_mean'}]
        calc_grouping = ['month','year']
        ops = ocgis.OcgOperations(rd,calc=calc,calc_grouping=calc_grouping)
        ret = ops.execute()
        tasmax = ret[1].variables['tasmax']
        date_centroid = tasmax.temporal_group.date_centroid
        ops = ocgis.OcgOperations(rd,calc=calc,calc_grouping=calc_grouping,
                                  output_format='nc')
        ret = ops.execute()
        ip = ocgis.Inspect(ret,variable='n')
        
    def test_frequency_percentiles(self):
        ## data comes in as 4-dimensional array. (time,level,row,column)
        
        perc = 0.95
        round_method = 'ceil' #floor
        
        ## generate gaussian sequence
        np.random.seed(1)
        seq = np.random.normal(size=(31,1,2,2))
        seq = np.ma.array(seq,mask=False)
        ## sort the data
        cseq = seq.copy()
        cseq.sort(axis=0)
        ## reference the time vector length
        n = cseq.shape[0]
        ## calculate the index
        idx = getattr(np,round_method)(perc*n)
        ## get the percentiles
        ret = cseq[idx,:,:,:]
        self.assertAlmostEqual(7.2835104624617717,ret.sum())
        
        ## generate gaussian sequence
        np.random.seed(1)
        seq = np.random.normal(size=(31,1,2,2))
        mask = np.zeros((31,1,2,2))
        mask[:,:,1,1] = True
        seq = np.ma.array(seq,mask=mask)
        ## sort the data
        cseq = seq.copy()
        cseq.sort(axis=0)
        ## reference the time vector length
        n = cseq.shape[0]
        ## calculate the index
        idx = getattr(np,round_method)(perc*n)
        ## get the percentiles
        ret = cseq[idx,:,:,:]
        self.assertAlmostEqual(5.1832553259829295,ret.sum())


class TestTile(TestBase):
    
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
            
    def test_execute_fill(self):
        #215 minutes
        uri = '/tmp/gridded_obs.tasmax.OBS_125deg.daily.1950-1999.nc'
#        uri = '/usr/local/climate_data/daymet/tmax.nc'
        variable = 'tasmax'
#        variable = 'tmax'
        rd = ocgis.RequestDataset(uri,variable)
        import netCDF4 as nc
        ods = ocgis.api.dataset.dataset.OcgDataset(rd)
        shp = ods.i.spatial.shape
        print('getting schema...')
        schema = ocgis.calc.tile.get_tile_schema(shp[0],shp[1],25)
        calc = [{'func':'mean','name':'my_mean'},
                {'func':'freq_perc','name':'perc_90','kwds':{'perc':0.90,'round_method':'floor'}},
                {'func':'freq_perc','name':'perc_95','kwds':{'perc':0.95,'round_method':'floor'}},
                {'func':'freq_perc','name':'perc_99','kwds':{'perc':0.99,'round_method':'floor'}}
               ]
        print('getting fill file...')
        fill_file = ocgis.OcgOperations(dataset=rd,file_only=True,
                                      calc=calc,calc_grouping=['month'],
                                      output_format='nc').execute()
        print fill_file, len(schema)
        fds = nc.Dataset(fill_file,'a')
        t1 = time.time()
        for tile_id,indices in schema.iteritems():
            print tile_id
            row = indices['row']
            col = indices['col']
            ret = ocgis.OcgOperations(dataset=rd,slice_row=row,slice_column=col,
                                      calc=calc,calc_grouping=['month']).execute()
            ref = ret[1].variables[variable].calc_value
            for k,v in ref.iteritems():
                vref = fds.variables[k]
                if len(vref.shape) == 3:
                    vref[:,row[0]:row[1],col[0]:col[1]] = v
                elif len(vref.shape) == 4:
                    vref[:,:,row[0]:row[1],col[0]:col[1]] = v
                else:
                    raise(NotImplementedError)
                fds.sync()
        fds.close()
        print((time.time()-t1)/60.0)
        import ipdb;ipdb.set_trace()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()