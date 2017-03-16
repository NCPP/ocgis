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
        ignore_attributes = {'time_bnds': ['units', 'calendar'], 'global': ['history'], 'foo': ['grid_mapping']}
        ignore_variables = ['latitude_longitude']
        self.assertNcEqual(ret, rd1.uri, ignore_attributes=ignore_attributes, ignore_variables=ignore_variables)

    @attr('icclim')
    def test_icclim(self):
        rd = RequestDataset(**self.get_dataset())
        calc = [{'func': 'icclim_TG', 'name': 'TG'}]
        calc_grouping = ['month', 'year']
        ret = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping).execute()
        self.assertEqual(ret[1]['foo'].variables['TG'].value.mean(), 2.5)

    def test_rtree(self):
        from ocgis.spatial.index import SpatialIndex

        geom_mapping = {1: Point(1, 2)}
        si = SpatialIndex()
        si.add(1, Point(1, 2))
        ret = list(si.iter_intersects(Point(1, 2), geom_mapping))
        self.assertEqual(ret, [1])
