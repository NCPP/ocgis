from shapely.geometry import Polygon, box

from ocgis import Field, Grid
from ocgis.constants import DMK, GridAbstraction
from ocgis.test import create_gridxy_global, create_exact_field
from ocgis.test.base import TestBase


class TestDriverXarray(TestBase):

    def test(self):
        # tdk: wish list for xarray:
        #      (1) some concept of a bounded variable

        # tdk: TEST: data variables on ocgis fiel

        import xarray as xr

        grid = create_gridxy_global()
        field = create_exact_field(grid, 'foo', ntime=31)
        path = self.get_temporary_file_path('foo.nc')
        field.write(path)
        xname = field.x.name
        yname = field.y.name

        m = Field()
        ds = xr.open_dataset(path, autoclose=True)
        m._storage = ds
        m.set_x(xname, 'dimx')
        m.set_y(yname, 'dimy')

        # tdk: FEATURE: bounds are configured manually on the dimension map
        m.dimension_map.set_bounds(DMK.X, 'xbounds')
        m.dimension_map.set_bounds(DMK.Y, 'ybounds')

        # tdk: FEATURE: dimension names are configured manually on the dimension map
        # m.dimension_map.pprint()

        grid1 = m.grid
        self.assertIsInstance(grid1, Grid)
        grid2 = m.grid
        self.assertIsInstance(grid2, Grid)

        # tdk: TEST: test a grid with a mask
        self.assertFalse(grid1.has_mask)

        self.assertIsNotNone(grid1.get_value_stacked())

        self.assertEqual(grid1.extent, (-180.0, -90.0, 180.0, 90.0))

        self.assertEqual(grid.abstraction, GridAbstraction.POLYGON)

        self.assertIsNone(m.geom)

        print(grid1.shape)
        self.assertEqual(grid1.shape, (180, 360))

        m.set_abstraction_geom()

        self.assertIsInstance(m.geom.values[0, 0], Polygon)

        bbox = box(*[-40, -30, 40, 40])
        res = m.grid.get_intersects(bbox)

        # tdk: FEATURE: re-work geometry variabl to not need xarray
        coords = m.geom.convert_to(pack=False)

        pass
