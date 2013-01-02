import unittest
from ocgis.util.shp_cabinet import ShpCabinet
from ocgis.api.operations import OcgOperations
from itertools import izip


class Test(unittest.TestCase):
    maurer = {'uri':'/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/bin/climate_data/maurer/bccr_bcm2_0.1.sresa1b.monthly.Prcp.1950.nc','variable':'Prcp'}
    cancm4 = {'uri':'/usr/local/climate_data/CanCM4/tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc','variable':'tasmax'}
    
    @property
    def dataset(self):
        dataset = [
                   self.maurer,
                   self.cancm4
                   ]
        return(dataset)
    @property
    def california(self):
        sc = ShpCabinet()
        ret = sc.get_geoms('state_boundaries',{'ugid':[25]})
        return(ret)

    def get_ops(self,kwds={}):
        geom = self.california
        ops = OcgOperations(dataset=self.dataset,
                            snippet=True,
                            geom=geom,
                            output_format='numpy')
        for k,v in kwds.iteritems():
            setattr(ops,k,v)
        return(ops)
    
    def get_ref(self,kwds={}):
        ops = self.get_ops(kwds=kwds)
        ret = ops.execute()
        return(ret[25])
    
    def test_default(self):
        ops = self.get_ops()
        ret = ops.execute()
        
        self.assertEqual(['Prcp','tasmax'],ret[25].variables.keys())
        
        shapes = [(1,1,77,83),(1,1,5,4)]
        for v,shape in izip(ret[25].variables.itervalues(),shapes):
            self.assertEqual(v.value.shape,shape)
    
    def test_aggregate_clip(self):
        kwds = {'aggregate':True,'spatial_operation':'clip'}
        ref = self.get_ref(kwds)
        for v in ref.variables.itervalues():
            self.assertEqual(v.spatial.value.shape,(1,1))
            self.assertEqual(v.value.shape,(1,1,1,1))
    
    def test_calculation(self):
        calc = [{'func':'mean','name':'mean'},{'func':'std','name':'std'}]
        calc_grouping = ['year']
        kwds = {'aggregate':True,
                'spatial_operation':'clip',
                'calc':calc,
                'calc_grouping':calc_grouping,
                'output_format':'numpy',
                'geom':self.california,
                'dataset':self.dataset,
                'snippet':False}
        ops = OcgOperations(**kwds)
        ret = ops.execute()
        
        ref = ret[25].variables['Prcp'].calc_value
        self.assertEquals(ref.keys(),['n','mean','std'])
        for value in ref.itervalues():
            self.assertEqual(value.shape,(1,1,1,1))
            
        ref = ret[25].variables['tasmax'].calc_value
        self.assertEquals(ref.keys(),['n','mean','std'])
        for value in ref.itervalues():
            self.assertEqual(value.shape,(10,1,1,1))
            
    def test_same_variable_name(self):
        ds = [self.cancm4,self.cancm4]
        ops = OcgOperations(dataset=ds,snippet=True)
        ret = ops.execute()
        import ipdb;ipdb.set_trace()


if __name__ == "__main__":
    import sys;sys.argv = ['', 'Test.test_calculation']
    unittest.main()