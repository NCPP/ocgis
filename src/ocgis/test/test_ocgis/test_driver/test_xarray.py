from mock import Mock, mock
from ocgis.driver.dxarray import DriverXarray
from ocgis.driver.nc import DriverNetcdfCF
from ocgis.util.addict import Dict
from ocgis.variable.crs import Spherical
from shapely.geometry import Polygon, box

from ocgis import Field, Grid, RequestDataset, DimensionMap
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
        # xname = field.x.name
        # yname = field.y.name

        # m = Field()
        ds = xr.open_dataset(path, autoclose=True)

        # tdk: FEATURE: no crs support in xarray yet
        ds['latitude_longitude'] = Spherical()
        # m._storage = ds
        # m.set_x(xname, 'dimx')
        # m.set_y(yname, 'dimy')

        xmeta = Dict()
        for dimname, dimsize in ds.dims.items():
            xmeta.dimensions[dimname] = {'name': dimname, 'size': dimsize}
        for varname, var in ds.variables.items():
            xmeta.variables[varname] = {'name': varname, 'dimensions': var.dims, 'attrs': var.attrs}

        # tdk: FEATURE: we should not have to pass an instance of request dataset to DimensionMap.from_metadata
        rd = RequestDataset(uri=path)
        xdimmap = DimensionMap.from_metadata(DriverNetcdfCF(rd), xmeta)

        f = Field(initial_data=ds, dimension_map=xdimmap, driver='xarray')
        self.assertIsInstance(f.x, xr.DataArray)
        self.assertIsInstance(f.time, xr.DataArray)

        # m.dimension_map.set_bounds(DMK.X, 'xbounds')
        # m.dimension_map.set_bounds(DMK.Y, 'ybounds')

        # m.dimension_map.pprint()

        grid1 = f.grid
        self.assertIsInstance(grid1, Grid)
        grid2 = f.grid
        self.assertIsInstance(grid2, Grid)

        # tdk: TEST: test a grid with a mask
        self.assertFalse(grid1.has_mask)

        self.assertIsNotNone(grid1.get_value_stacked())

        self.assertEqual(grid1.extent, (-180.0, -90.0, 180.0, 90.0))

        self.assertEqual(grid.abstraction, GridAbstraction.POLYGON)

        self.assertIsNone(f.geom)

        print(grid1.shape)
        self.assertEqual(grid1.shape, (180, 360))

        f.set_abstraction_geom()

        self.assertIsInstance(f.geom.values[0, 0], Polygon)

        bbox = box(*[-40, -30, 40, 40])
        res = f.grid.get_intersects(bbox)

        # tdk: FEATURE: re-work geometry variable to not need ocgis
        # coords = m.geom.convert_to(pack=False)

        # tdk: RESUME: continue testing with an unstructured grid

        pass

    def test_init(self):
        rd = mock.create_autospec(RequestDataset)
        xd = DriverXarray(rd)
        self.assertIsInstance(xd, DriverXarray)
