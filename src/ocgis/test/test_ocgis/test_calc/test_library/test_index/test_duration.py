import csv

import numpy as np

from ocgis import constants
from ocgis.calc.library.index.duration import Duration, FrequencyDuration
from ocgis.exc import DefinitionValidationError
from ocgis.ops.core import OcgOperations
from ocgis.test.base import attr
from ocgis.test.test_ocgis.test_calc.test_calc_general import AbstractCalcBase


class TestDuration(AbstractCalcBase):
    def test_calculate(self):
        duration = Duration()

        # Three consecutive days over 3
        values = np.array([1, 2, 3, 3, 3, 1, 1], dtype=float)
        values = self.get_reshaped(values)
        ret = duration.calculate(values, 2, operation='gt', summary='max')
        self.assertEqual(3.0, ret.flatten()[0])

        # No duration over the threshold
        values = np.array([1, 2, 1, 2, 1, 2, 1], dtype=float)
        values = self.get_reshaped(values)
        ret = duration.calculate(values, 2, operation='gt', summary='max')
        self.assertEqual(0., ret.flatten()[0])

        # No duration over the threshold
        values = np.array([1, 2, 1, 2, 1, 2, 1], dtype=float)
        values = self.get_reshaped(values)
        ret = duration.calculate(values, 2, operation='gte', summary='max')
        self.assertEqual(1., ret.flatten()[0])

        # Average duration
        values = np.array([1, 5, 5, 2, 5, 5, 5], dtype=float)
        values = self.get_reshaped(values)
        ret = duration.calculate(values, 4, operation='gte', summary='mean')
        self.assertEqual(2.5, ret.flatten()[0])

        # Add some masked values
        values = np.array([1, 5, 5, 2, 5, 5, 5], dtype=float)
        mask = [0, 0, 0, 0, 0, 1, 0]
        values = np.ma.array(values, mask=mask)
        values = self.get_reshaped(values)
        ret = duration.calculate(values, 4, operation='gte', summary='max')
        self.assertEqual(2., ret.flatten()[0])

        # Test with an actual matrix
        values = np.array([1, 5, 5, 2, 5, 5, 5, 4, 4, 0, 2, 4, 4, 4, 3, 3, 5, 5, 6, 9], dtype=float)
        values = values.reshape(5, 2, 2)
        values = np.ma.array(values, mask=False)
        ret = duration.calculate(values, 4, operation='gte', summary='mean')
        self.assertNumpyAll(np.ma.array([4., 2., 1.5, 1.5], dtype=ret.dtype), ret.flatten())

    @attr('data')
    def test_system_standard_operations(self):
        ret = self.run_standard_operations(
            [{'func': 'duration', 'name': 'max_duration',
              'kwds': {'operation': 'gt', 'threshold': 2, 'summary': 'max'}}],
            capture=True)
        for cap in ret:
            reraise = True
            if isinstance(cap['exception'], DefinitionValidationError):
                if cap['parms']['calc_grouping'] in [['month'], 'all']:
                    reraise = False
            if reraise:
                raise cap


class TestFrequencyDuration(AbstractCalcBase):
    def test_init(self):
        FrequencyDuration()

    @attr('data')
    def test_calculate(self):
        fduration = FrequencyDuration()
        values = np.array([1, 2, 3, 3, 3, 1, 1, 3, 3, 3, 4, 4, 1, 4, 4, 1, 10, 10], dtype=float)
        values = self.get_reshaped(values)
        ret = fduration.calculate(values, threshold=2, operation='gt')
        self.assertEqual(ret.flatten()[0].dtype.names, ('duration', 'count'))
        self.assertNumpyAll(np.array([2, 3, 5]), ret.flatten()[0]['duration'])
        self.assertNumpyAll(np.array([2, 1, 1]), ret.flatten()[0]['count'])
