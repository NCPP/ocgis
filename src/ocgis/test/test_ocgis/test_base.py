from copy import deepcopy

from ocgis.base import renamed_dimensions, renamed_dimensions_on_variables
from ocgis.test.base import AbstractTestInterface
from ocgis.variable.base import VariableCollection, Variable
from ocgis.variable.dimension import Dimension


class Test(AbstractTestInterface):
    def test_renamed_dimensions(self):
        d = [Dimension('a', 5), Dimension('b', 6)]
        desired_after = deepcopy(d)
        name_mapping = {'time': ['b']}
        desired = [Dimension('a', 5), Dimension('time', 6)]
        with renamed_dimensions(d, name_mapping):
            for idx in range(len(d)):
                self.assertEqual(d[idx], desired[idx])
            self.assertEqual(d, desired)
        self.assertEqual(desired_after, d)

    def test_renamed_dimensions_on_variables(self):
        vc = VariableCollection()
        var1 = Variable(name='ugid', value=[1, 2, 3], dimensions='ocgis_geom')
        var2 = Variable(name='state', value=[20, 30, 40], dimensions='ocgis_geom')
        vc.add_variable(var1)
        vc.add_variable(var2)
        with renamed_dimensions_on_variables(vc, {'geom': ['ocgis_geom']}):
            for var in list(vc.values()):
                self.assertEqual(var.dimensions[0].name, 'geom')
        for var in list(vc.values()):
            self.assertEqual(var.dimensions[0].name, 'ocgis_geom')
