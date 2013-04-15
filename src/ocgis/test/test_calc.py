import unittest
import numpy as np
from ocgis.calc import library
from ocgis.api.operations import OcgOperations
from datetime import datetime as dt
from ocgis.api.dataset.collection.iterators import MeltedIterator, KeyedIterator
import ocgis
import datetime
from ocgis.test.base import TestBase


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
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()