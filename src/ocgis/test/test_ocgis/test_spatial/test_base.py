from unittest import SkipTest

from ocgis.spatial.base import iter_spatial_decomposition, create_split_polygons
from ocgis.test.base import TestBase, create_gridxy_global, attr
from ocgis.variable.crs import Spherical
from ocgis.vmachine.core import vm
from shapely.geometry import box


class Test(TestBase):

    def test_create_split_polygons(self):
        bbox = box(180, 30, 270, 40)
        splits = (2, 3)
        polys = create_split_polygons(bbox, splits)
        actual = []
        for poly in polys:
            actual.append(poly.bounds)

        desired = [(180.0, 30.0, 210.0, 35.0), (210.0, 30.0, 240.0, 35.0), (240.0, 30.0, 270.0, 35.0),
                   (180.0, 35.0, 210.0, 40.0), (210.0, 35.0, 240.0, 40.0), (240.0, 35.0, 270.0, 40.0)]
        self.assertEqual(actual, desired)

    @attr('mpi')
    def test_iter_spatial_decomposition(self):
        self.add_barrier = False
        if vm.size not in [1, 4]:
            raise SkipTest('vm.size not in [1, 4]')

        grid = create_gridxy_global(resolution=10., wrapped=False, crs=Spherical())
        splits = (2, 3)
        actual = []
        for sub, slc in iter_spatial_decomposition(grid, splits, optimized_bbox_subset=True):
            root = vm.get_live_ranks_from_object(sub)[0]
            with vm.scoped_by_emptyable('test extent', sub):
                if vm.is_null:
                    extent_global = None
                else:
                    extent_global = sub.extent_global
            extent_global = vm.bcast(extent_global, root=root)
            actual.append(extent_global)

        desired = [(0.0, -90.0, 120.0, 0.0), (120.0, -90.0, 240.0, 0.0), (240.0, -90.0, 360.0, 0.0),
                   (0.0, 0.0, 120.0, 90.0), (120.0, 0.0, 240.0, 90.0), (240.0, 0.0, 360.0, 90.0)]
        self.assertEqual(actual, desired)
