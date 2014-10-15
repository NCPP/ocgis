from ocgis.test.base import TestBase
from ocgis.calc.library.statistics import Mean
from ocgis.calc.engine import OcgCalculationEngine
import ocgis
from copy import deepcopy
import numpy as np
from ocgis.util.logging_ocgis import ProgressOcgOperations
from ocgis.api.collection import SpatialCollection
from ocgis.interface.base.field import DerivedMultivariateField
from ocgis.calc.eval_function import EvalFunction


class TestOcgCalculationEngine(TestBase):
    
    @property
    def funcs(self):
        return(deepcopy([{'ref':Mean,'name':'mean','kwds':{},'func':'mean'}]))
    
    @property
    def grouping(self):
        return(deepcopy(['month']))
        
    def get_engine(self,kwds=None,funcs=None,grouping="None"):
        kwds = kwds or {}
        funcs = funcs or self.funcs
        if grouping == 'None':
            grouping = self.grouping
        return(OcgCalculationEngine(grouping,funcs,**kwds))
    
    def test_with_eval_function_one_variable(self):
        funcs = [{'func':'tas2=tas+4','ref':EvalFunction}]
        engine = self.get_engine(funcs=funcs,grouping=None)
        rd = self.test_data_nc.get_rd('cancm4_tas')
        coll = ocgis.OcgOperations(dataset=rd,slice=[None,[0,700],None,[0,10],[0,10]]).execute()
        to_test = deepcopy(coll)
        engine.execute(coll)
        self.assertNumpyAll(coll[1]['tas'].variables['tas2'].value,to_test[1]['tas'].variables['tas'].value+4)
        
    def test_with_eval_function_two_variables(self):
        funcs = [{'func':'tas_out=tas+tas2','ref':EvalFunction}]
        engine = self.get_engine(funcs=funcs,grouping=None)
        rd = self.test_data_nc.get_rd('cancm4_tas')
        rd2 = self.test_data_nc.get_rd('cancm4_tas')
        rd2.alias = 'tas2'
        field = rd.get()
        field2 = rd2.get()
        field.variables.add_variable(field2.variables['tas2'],assign_new_uid=True)
        field = field[:,0:100,:,0:10,0:10]
        coll = SpatialCollection()
        coll.add_field(1,None,field)
        to_test = deepcopy(coll)
        engine.execute(coll)
        self.assertIsInstance(coll[1]['tas'],DerivedMultivariateField)
        self.assertNumpyAll(coll[1]['tas'].variables['tas_out'].value,
                            to_test[1]['tas'].variables['tas'].value+to_test[1]['tas'].variables['tas2'].value)
    
    def test_constructor(self):
        for kwds in [None,{'progress':ProgressOcgOperations()}]:
            self.get_engine(kwds=kwds)
        
    def test_execute(self):
        rd = self.test_data_nc.get_rd('cancm4_tas')
        coll = ocgis.OcgOperations(dataset=rd,slice=[None,[0,700],None,[0,10],[0,10]]).execute()
        engine = self.get_engine()
        ret = engine.execute(coll)
        self.assertEqual(ret[1]['tas'].shape,(1, 12, 1, 10, 10))
    
    def test_execute_tgd(self):
        rd = self.test_data_nc.get_rd('cancm4_tas')
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
        rd = self.test_data_nc.get_rd('cancm4_tas')
        coll = ocgis.OcgOperations(dataset=rd,slice=[None,[0,700],None,[0,10],[0,10]],
                                   calc=self.funcs,calc_grouping=['month','year']).execute()
        tgds = {'tas':coll[1]['tas'].temporal}
        engine = self.get_engine()
        ## should raise a value error as the passed tgd is not compatible when
        ## executing the calculation
        with self.assertRaises(ValueError):
            engine.execute(coll,tgds=tgds)
