from copy import deepcopy

from shapely.geometry import Point

from ocgis import RequestDataset, OcgOperations
from ocgis.test.base import attr
from ocgis.test.test_simple.make_test_data import SimpleNcNoLevel
from ocgis.test.test_simple.test_simple import TestSimpleBase
from ocgis.util.units import get_units_object


@attr('optional')
class TestOptionalDependencies(TestSimpleBase):
    nc_factory = SimpleNcNoLevel
    fn = 'test_simple_spatial_no_level_01.nc'

    def test_cfunits(self):
        units = get_units_object('K')
        self.assertEqual(str(units), 'K')

    @attr('esmf')
    def test_esmf(self):
        rd1 = RequestDataset(**self.get_dataset())
        rd2 = deepcopy(rd1)
        ops = OcgOperations(dataset=rd1, regrid_destination=rd2, output_format='nc')
        ret = ops.execute()

        actual_value = RequestDataset(ret).get().data_variables[0].get_value()
        desired_value = rd1.get().data_variables[0].get_value()
        self.assertNumpyAllClose(actual_value, desired_value)

    @attr('icclim')
    def test_icclim(self):
        rd = RequestDataset(**self.get_dataset())
        calc = [{'func': 'icclim_TG', 'name': 'TG'}]
        calc_grouping = ['month', 'year']
        ret = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping).execute()
        self.assertEqual(ret.get_element(variable_name='TG').get_value().mean(), 2.5)

    def test_rtree(self):
        from ocgis.spatial.index import SpatialIndex

        geom_mapping = {1: Point(1, 2)}
        si = SpatialIndex()
        si.add(1, Point(1, 2))
        ret = list(si.iter_intersects(Point(1, 2), geom_mapping))
        self.assertEqual(ret, [1])
