import unittest
import numpy as np
from ocgis.calc import library
from ocgis.api.operations import OcgOperations
from datetime import datetime as dt
import ocgis
import datetime
from ocgis.test.base import TestBase
from unittest.case import SkipTest
import netCDF4 as nc
import subprocess
import webbrowser


class Test(TestBase):
    
    def test_with_time_region(self):
        kwds = {'time_region':{'year':[2011]}}
        rd = self.test_data.get_rd('cancm4_tasmax_2011',kwds=kwds)
        calc = [{'func':'mean','name':'mean'}]
        calc_grouping = ['year','month']
        
        ops = ocgis.OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                                  geom='state_boundaries',select_ugid=[25])
        ret = ops.execute()
        
        tgroup = ret[25].variables['tasmax'].temporal.group.value
        self.assertEqual(set([2011]),set(tgroup[:,0]))
        self.assertEqual(tgroup[-1,1],12)
        
        kwds = {'time_region':{'year':[2011,2013],'month':[8]}}
        rd = self.test_data.get_rd('cancm4_tasmax_2011',kwds=kwds)
        calc = [{'func':'threshold','name':'threshold','kwds':{'threshold':0.0,'operation':'gte'}}]
        calc_grouping = ['month']
        aggregate = True
        calc_raw = True
        geom = 'us_counties'
        select_ugid = [2762]
        
        ops = ocgis.OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                                 aggregate=aggregate,calc_raw=calc_raw,geom=geom,
                                 select_ugid=select_ugid,output_format='numpy')
        ret = ops.execute()
        threshold = ret[2762].calc['tasmax']['threshold']
        self.assertEqual(threshold.flatten()[0],62)
    
    def test_HeatIndex(self):
        kwds = {'time_range':[dt(2011,1,1),dt(2011,12,31,23,59,59)]}
        ds = [self.test_data.get_rd('cancm4_tasmax_2011',kwds=kwds),self.test_data.get_rd('cancm4_rhsmax',kwds=kwds)]
        calc = [{'func':'heat_index','name':'heat_index','kwds':{'tas':'tasmax','rhs':'rhsmax','units':'k'}}]
        
        ## operations on entire data arrays
        ops = OcgOperations(dataset=ds,calc=calc)
        self.assertEqual(ops.calc_grouping,None)
        ret = ops.execute()
        ref = ret[1]
        self.assertEqual(ref.variables.keys(),['tasmax','rhsmax'])
        self.assertEqual(ref.calc.keys(),['heat_index'])
        hi = ref.calc['heat_index']
        self.assertEqual(hi.shape,(365,1,64,128))
        
        ## confirm no masked geometries
        self.assertFalse(ref._archetype.spatial.vector.geom.mask.any())
        ## confirm some masked data in calculation output
        self.assertTrue(hi.mask.any())
        
        ## snippet-based testing
        ops = OcgOperations(dataset=ds,calc=calc,snippet=True)
        ret = ops.execute()
        self.assertEqual(ret[1].calc['heat_index'].shape,(1,1,64,128))
        ops = OcgOperations(dataset=ds,calc=calc,snippet=True,output_format='csv')
        ret = ops.execute()
        
#        subprocess.check_call(['loffice',ret])
        
        # try temporal grouping
        ops = OcgOperations(dataset=ds,calc=calc,calc_grouping=['month'])
        ret = ops.execute()
        self.assertEqual(ret[1].calc['heat_index'].shape,(12,1,64,128))
        ret = OcgOperations(dataset=ds,calc=calc,calc_grouping=['month'],
                            output_format='csv',snippet=True).execute()
                            
#        subprocess.check_call(['loffice',ret])
        
    def test_HeatIndex_keyed_output(self):
        raise(SkipTest)
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
        ops = ocgis.OcgOperations(rd,calc=calc,calc_grouping=calc_grouping,
                                  output_format='nc')
        ret = ops.execute()
        ip = ocgis.Inspect(ret,variable='n')
        
        ds = nc.Dataset(ret,'r')
        ref = ds.variables['time'][:]
        self.assertEqual(len(ref),12)
        ds.close()
        
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
