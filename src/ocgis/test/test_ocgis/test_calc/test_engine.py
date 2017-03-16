from copy import deepcopy

import numpy as np

import ocgis
from ocgis.base import orphaned
from ocgis.calc.engine import OcgCalculationEngine
from ocgis.calc.eval_function import EvalFunction
from ocgis.calc.library.statistics import Mean
from ocgis.collection.spatial import SpatialCollection
from ocgis.test.base import TestBase
from ocgis.test.base import attr
from ocgis.util.logging_ocgis import ProgressOcgOperations


class TestOcgCalculationEngine(TestBase):
    @property
    def funcs(self):
        return deepcopy([{'ref': Mean, 'name': 'mean', 'kwds': {}, 'func': 'mean'}])

    @property
    def grouping(self):
        return deepcopy(['month'])

    def get_engine(self, kwds=None, funcs=None, grouping="None"):
        """
        :type grouping: None or str
        """
        kwds = kwds or {}
        funcs = funcs or self.funcs
        if grouping == 'None':
            grouping = self.grouping
        return OcgCalculationEngine(grouping, funcs, **kwds)

    @attr('data')
    def test_with_eval_function_one_variable(self):
        funcs = [{'func': 'tas2=tas+4', 'ref': EvalFunction}]
        engine = self.get_engine(funcs=funcs, grouping=None)
        rd = self.test_data.get_rd('cancm4_tas')
        coll = ocgis.OcgOperations(dataset=rd, slice=[None, [0, 700], None, [0, 10], [0, 10]]).execute()
        to_test = deepcopy(coll)
        engine.execute(coll)

        actual = coll.get_element(variable_name='tas2').get_value()
        desired = to_test.get_element(variable_name='tas').get_value() + 4
        self.assertNumpyAll(actual, desired)

    @attr('data')
    def test_with_eval_function_two_variables(self):
        funcs = [{'func': 'tas_out=tas+tas2', 'ref': EvalFunction}]
        engine = self.get_engine(funcs=funcs, grouping=None)
        rd = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tas', kwds={'rename_variable': 'tas2'})
        field = rd.get()
        field2 = rd2.get()
        with orphaned(field2['tas2']):
            field.add_variable(field2['tas2'], is_data=True)
        field = field.get_field_slice({'time': slice(0, 100), 'y': slice(0, 10), 'x': slice(0, 10)})

        desired = SpatialCollection()
        desired.add_field(field, None)
        actual = deepcopy(desired)
        engine.execute(desired)

        tas_out = desired.get_element(variable_name='tas_out').get_value()
        tas = actual.get_element(variable_name='tas').get_value()
        tas2 = actual.get_element(variable_name='tas2').get_value()
        self.assertNumpyAll(tas_out, tas + tas2)

    def test_constructor(self):
        for kwds in [None, {'progress': ProgressOcgOperations()}]:
            self.get_engine(kwds=kwds)

    @attr('data')
    def test_execute(self):
        rd = self.test_data.get_rd('cancm4_tas')
        coll = ocgis.OcgOperations(dataset=rd, slice=[None, [0, 700], None, [0, 10], [0, 10]]).execute()
        engine = self.get_engine()
        ret = engine.execute(coll)

        actual = ret.get_element(variable_name='mean').shape
        desired = (12, 10, 10)
        self.assertEqual(actual, desired)

    @attr('data')
    def test_execute_tgd(self):
        rd = self.test_data.get_rd('cancm4_tas')
        coll = ocgis.OcgOperations(dataset=rd, slice=[None, [0, 700], None, [0, 10], [0, 10]],
                                   calc=self.funcs, calc_grouping=self.grouping).execute()
        coll_data = ocgis.OcgOperations(dataset=rd, slice=[None, [0, 700], None, [0, 10], [0, 10]]).execute()
        tgds = {'tas': coll.get_element().time.extract()}
        engine = self.get_engine()
        coll_engine = deepcopy(coll_data)
        engine.execute(coll_engine, tgds=tgds)
        desired = coll.get_element(variable_name='mean').get_value()
        actual = coll_engine.get_element(variable_name='mean').get_value()
        self.assertNumpyAll(actual, desired)
        self.assertFalse(np.may_share_memory(actual, desired))

    @attr('data')
    def test_execute_tgd_malformed(self):
        rd = self.test_data.get_rd('cancm4_tas')
        coll = ocgis.OcgOperations(dataset=rd, slice=[None, [0, 700], None, [0, 10], [0, 10]],
                                   calc=self.funcs, calc_grouping=['month', 'year']).execute()
        tgds = {'tas': coll.get_element().time.extract()}
        engine = self.get_engine()
        # Should raise a value error as the passed temporal group dimension is not compatible with the calculation.
        with self.assertRaises(ValueError):
            engine.execute(coll, tgds=tgds)
