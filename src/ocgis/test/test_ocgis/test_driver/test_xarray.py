from copy import deepcopy

from mock import mock
from ocgis.driver.dxarray import DriverXarray, create_dimension_map, create_metadata_from_xarray
from ocgis.driver.nc import DriverNetcdfCF
from ocgis.driver.nc_esmf_unstruct import DriverESMFUnstruct
from ocgis.spatial.grid_chunker import GridChunker
from ocgis.variable.crs import Spherical, WGS84
from shapely.geometry import Polygon, box

from ocgis import Field, Grid, RequestDataset, GridUnstruct, GeometryVariable
from ocgis.constants import GridAbstraction, DriverKey, Topology
from ocgis.test import create_gridxy_global, create_exact_field
from ocgis.test.base import TestBase
import xarray as xr
import numpy as np


class TestDriverXarray(TestBase):

    def fixture_esmf_unstructured(self):
        rd = RequestDataset(self.path_state_boundaries)
        field = rd.get()
        gc = field.geom.convert_to(use_geometry_iterator=True, pack=False)
        path = self.get_temporary_file_path('unstruct.nc')
        gc.parent.write(path, driver=DriverKey.NETCDF_ESMF_UNSTRUCT)
        return path

    def test(self):
        # tdk: wish list for xarray:
        #      (1) some concept of a bounded variable

        # tdk: TEST: data variables on ocgis fiel

        grid = create_gridxy_global(resolution=2.0)
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

        xmeta = create_metadata_from_xarray(ds)

        xdimmap = create_dimension_map(xmeta, DriverNetcdfCF, path)

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
        self.assertEqual(grid1.shape, (90, 180))

        f.set_abstraction_geom()

        self.assertIsInstance(f.geom.values[0, 0], Polygon)

        bbox = box(*[-40, -30, 40, 40])
        res = f.grid.get_intersects(bbox)

        # tdk: FEATURE: re-work geometry variable to not need ocgis
        # coords = m.geom.convert_to(pack=False)

    def test_system_grid_chunking(self):
        # grid = create_gridxy_global(resolution=1.0)
        # field = create_exact_field(grid, 'foo', ntime=10)
        # path1 = self.get_temporary_file_path('foo.nc')
        # field.write(path1)
        #
        # grid = create_gridxy_global(resolution=1.5)
        # field = create_exact_field(grid, 'foo', ntime=10)
        # path2 = self.get_temporary_file_path('foo2.nc')
        # field.write(path2)

        # path1 = r'C:\Users\benko\Dropbox\dtmp\gpw-v4-population-density-rev10_2020_1_deg_tif\gpw_v4_population_density_rev10_2020_1_deg.tif'
        path1 = '/home/benkoziol/Dropbox/dtmp/gpw-v4-population-density-rev10_2020_1_deg_tif/gpw_v4_population_density_rev10_2020_1_deg.tif'
        path2 = path1

        ds1 = xr.open_rasterio(path1, parse_coordinates=True)
        ds1 = xr.Dataset(data_vars={'data': ds1})

        ds2 = xr.open_rasterio(path2, parse_coordinates=True)
        ds2 = xr.Dataset(data_vars={'data': ds2})

        xmeta1 = create_metadata_from_xarray(ds1)
        xdimmap1 = create_dimension_map(xmeta1, DriverNetcdfCF) # tdk: FEATURE: need to get the coordinate system somehow
        xdimmap1.set_crs(None)

        xdimmap2 = deepcopy(xdimmap1)

        f1 = Field(initial_data=ds1, dimension_map=xdimmap1, crs=WGS84())
        f1.grid.set_extrapolated_bounds('xbounds', 'ybounds', 'bounds')
        global_indices = f1.grid._gc_create_global_indices_(f1.grid.shape)
        f1.add_variable(xr.DataArray(global_indices, dims=f1.grid.dims, name='ESMF_Index'))
        f2 = Field(initial_data=ds2, dimension_map=xdimmap2, crs=WGS84())
        f2.grid.set_extrapolated_bounds('xbounds', 'ybounds', 'bounds')
        global_indices = f2.grid._gc_create_global_indices_(f1.grid.shape)
        f1.add_variable(xr.DataArray(global_indices, dims=f2.grid.dims, name='ESMF_Index'))

        gc = GridChunker(f1, f2, nchunks_dst=(5, 5))
        for res in gc.iter_src_grid_subsets(yield_dst=True):
            print(res[0].extent)
            print(res)
            print(res[0].parent.to_xarray())
            print(res[2].parent.to_xarray())
            arch = res[0].parent.to_xarray()

    def test_system_masking(self):
        """Test masks are created and maintained when using an xarray backend."""

        grid = create_gridxy_global(resolution=2.0)
        field = create_exact_field(grid, 'foo', ntime=31)
        path = self.get_temporary_file_path('foo.nc')
        field.write(path)

        ds = xr.open_dataset(path, autoclose=True)

        # tdk: FEATURE: need ocgis data_variables support using xarray somehow...
        # tdk: FEATURE: grid expansion needs to use xarray

        # tdk: FEATURE: no crs support in xarray yet
        ds['latitude_longitude'] = Spherical()

        xmeta = create_metadata_from_xarray(ds)
        xdimmap = create_dimension_map(xmeta, DriverNetcdfCF)

        f1 = Field(initial_data=ds, dimension_map=xdimmap, driver='xarray')
        f1.add_variable(f1['foo'], is_data=True, force=True)

        poly = Polygon([[270, 40], [230, 20], [310, 40]])
        poly = GeometryVariable(name='subset', value=poly, crs=Spherical(), dimensions='ngeom')
        poly.wrap()

        self.assertIsNone(f1.grid.get_mask())

        # tdk: TEST: add loop with create=[False, True]
        # tdk: TEST: test with check_value in get_mask?

        mask = f1.grid.get_mask(create=True)
        self.assertFalse(mask.any())
        mask = f1.grid.get_mask()
        self.assertFalse(mask.any())

        sub = f1.grid.get_intersects(poly)

        # Assert spatially masked values are set to NaNs in the data variable.
        self.assertGreater(np.sum(np.isnan(sub.parent['foo'])), 0)

        sub.expand()
        mx = np.ma.array(sub.x, mask=sub.get_mask())
        tkk

    def test_system_unstructured_grid(self):
        path = self.fixture_esmf_unstructured()
        # tdk: FIX: the load step is incredibly slow; what is it doing?
        ds = xr.open_dataset(path, autoclose=True)
        xmeta = create_metadata_from_xarray(ds)
        xdimmap = create_dimension_map(xmeta, DriverESMFUnstruct, path)
        f = Field(initial_data=ds, dimension_map=xdimmap, driver=DriverESMFUnstruct)
        self.assertIsInstance(f.grid, GridUnstruct)
        self.assertEqual(f.grid.abstraction, Topology.POLYGON)

        # tdk: TEST: add test for field properties with unstructured grids
        # tdk: RESUME: continue working with unstructured grids; eventual goal is conversion to ESMF

        self.assertIsInstance(f.x, xr.DataArray)

    def test_init(self):
        # tdk: ORDER
        xd = DriverXarray()
        self.assertIsInstance(xd, DriverXarray)

    def test_create_varlike(self):
        name = 'moon'
        vl = DriverXarray.create_varlike(None, name=name)
        self.assertIsInstance(vl, xr.DataArray)
