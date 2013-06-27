import unittest
import numpy as np
import datetime
from ocg.api.interp.iocg.dataset.collection import OcgReference, OcgVariable,\
    OcgCollection
from ocg.calc.engine import OcgCalculationEngine


class TestCollection(unittest.TestCase):
    ntime = 60
    geomshape = [2,2]
    nlevel = 2
    nmask = 0
    start_date = datetime.datetime(2000,1,1)
    vars = ['foo','foo2']
    funcs = [{'func':'median','name':'median01'}]
    multi_func = [{'func':'foomulti'}]
    
    
    def make_cengine(self,ocg_reference,raw,agg,multi=False):
        if multi:
            funcs = self.multi_func
        else:
            funcs = self.funcs
        cengine = OcgCalculationEngine(['month'],
                                       ocg_reference.timevec,
                                       funcs,
                                       raw=raw,
                                       agg=agg,
                                       time_range=None)
        return(cengine)
    
    
    def make_reference(self,agg=False):
        tid = np.arange(1,self.ntime+1,dtype=int)
        if agg:
            gid = np.array([1])
        else:
            gid = np.arange(1,self.geomshape[0]*self.geomshape[1]+1,dtype=int).\
                     reshape(self.geomshape)
        geom = np.empty(gid.shape,dtype=object)
        geom[:] = 'GEOM'
        geom_mask = np.zeros(geom.shape,dtype=bool)
        weights = np.ones(geom.shape,dtype=float)
        
        td = datetime.timedelta(days=1)
        timevec = []
        d = self.start_date
        for ii in range(0,self.ntime):
            timevec.append(d)
            d += td
            
        ref = OcgReference(tid,gid,geom,geom_mask,timevec,weights)
        return(ref)
    
    def make_variable(self,name,agg=False):
        lid = np.arange(1,self.nlevel+1,dtype=int)
        levelvec = lid*10
        raw_value = np.random.rand(self.ntime,self.nlevel,self.geomshape[0],
                                   self.geomshape[1])
        raw_value = np.ma.array(raw_value,
                                mask=np.zeros(raw_value.shape,dtype=bool))
        var = OcgVariable(name,lid,levelvec,raw_value)
        if agg:
            agg_value = np.random.rand(self.ntime,self.nlevel,1)
            agg_value = np.ma.array(agg_value,
                                    mask=np.zeros(agg_value.shape,dtype=bool))
            var.agg_value = agg_value
        return(var)
    
    def make_collection(self,agg=False,ocg_reference=None):
        if ocg_reference is None:
            ocg_reference = self.make_reference(agg=agg)
        coll = OcgCollection(ocg_reference)
        for var in self.vars:
            coll.add_variable(self.make_variable(var,agg=agg))
        return(coll)
    
    def _test_iterator_(self,mode,agg):
        coll = self.make_collection(agg=agg)
        for ii in coll:
            print ii
            import ipdb;ipdb.set_trace()
    
    def test_raw_iterator(self):
        self._test_iterator_('raw',False)
#            
##    def test_agg_iterator(self):
##        self._test_iterator_('agg',True)
#        
#    def test_calc_iterator(self):
#        raw = True
#        agg = False
#        ocg_reference = self.make_reference(agg)
#        coll = self.make_collection(agg,ocg_reference)
#        cengine = self.make_cengine(ocg_reference,raw,agg)
#        ocg_reference.cengine = cengine
#        cengine.execute(coll)
#        iterator = coll.iter_rows('calc')
#        for ii in iterator:
#            self.assertEqual(len(ii),2)

#    def test_multi_iterator(self):
#        raw = True
#        agg = False
#        ocg_reference = self.make_reference(agg)
#        coll = self.make_collection(agg,ocg_reference)
#        cengine = self.make_cengine(ocg_reference,raw,agg,multi=True)
#        ocg_reference.cengine = cengine
#        cengine.execute(coll)
#        iterator = coll.iter_rows('multi')
#        print coll.get_headers('multi')
#        for ii in iterator:
#            print ii
#            self.assertEqual(len(ii),2)

    
if __name__ == '__main__':
    unittest.main()