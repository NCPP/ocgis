from ocgis.test.base import TestBase
from ocgis.calc.library.statistics import Mean
from ocgis.calc.engine import OcgCalculationEngine
import ocgis
from copy import deepcopy, copy
import numpy as np


class TestOcgCalculationEngine(TestBase):
    
    @property
    def funcs(self):
        return(deepcopy([{'ref':Mean,'name':'mean','kwds':{},'func':'mean'}]))
    
    @property
    def grouping(self):
        return(deepcopy(['month']))
        
    def get_engine(self,kwds=None):
        kwds = kwds or {}
        return(OcgCalculationEngine(self.grouping,self.funcs,**kwds))
    
    def test_constructor(self):
        self.get_engine()
        
    def test_execute(self):
        rd = self.test_data.get_rd('cancm4_tas')
        coll = ocgis.OcgOperations(dataset=rd,slice=[None,[0,700],None,[0,10],[0,10]]).execute()
        engine = self.get_engine()
        ret = engine.execute(coll)
        self.assertEqual(ret[1]['tas'].shape,(1, 12, 1, 10, 10))
    
    def test_execute_tgd(self):
        rd = self.test_data.get_rd('cancm4_tas')
        coll = ocgis.OcgOperations(dataset=rd,slice=[None,[0,700],None,[0,10],[0,10]],
                                   calc=self.funcs,calc_grouping=self.grouping).execute()
        coll_data = ocgis.OcgOperations(dataset=rd,slice=[None,[0,700],None,[0,10],[0,10]]).execute()
        tgds = {'tas':coll[1]['tas'].temporal}
        engine = self.get_engine()
        coll_engine = deepcopy(coll_data)
        engine.execute(coll_engine,tgds=tgds)
        self.assertNumpyAll(coll.gvu(1,'mean'),coll_engine.gvu(1,'mean'))
        self.assertFalse(np.may_share_memory(coll.gvu(1,'mean'),coll_engine.gvu(1,'mean')))

    def test_execute_tgd_malformed(self):
        rd = self.test_data.get_rd('cancm4_tas')
        coll = ocgis.OcgOperations(dataset=rd,slice=[None,[0,700],None,[0,10],[0,10]],
                                   calc=self.funcs,calc_grouping=['month','year']).execute()
        tgds = {'tas':coll[1]['tas'].temporal}
        engine = self.get_engine()
        ## should raise a value error as the passed tgd is not compatible when
        ## executing the calculation
        with self.assertRaises(ValueError):
            engine.execute(coll,tgds=tgds)
