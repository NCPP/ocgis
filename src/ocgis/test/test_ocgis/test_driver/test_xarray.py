from copy import deepcopy

from mock import mock
from ocgis.driver.dimension_map import create_dimension_map
from ocgis.driver.dxarray import DriverXarray, create_metadata_from_xarray
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
        # tdk:RESUME: work on removing the tdks and don't try and get chunker working
        grid = create_gridxy_global(resolution=2.0)
        field = create_exact_field(grid, 'foo', ntime=31)
        path = self.get_temporary_file_path('foo.nc')
        field.write(path)

        # m = Field()
        ds = xr.open_dataset(path, autoclose=True)

        # tdk:FEAT: no crs support in xarray yet
        ds['latitude_longitude'] = Spherical()

        xmeta = create_metadata_from_xarray(ds)
        xdimmap = create_dimension_map(xmeta, DriverNetcdfCF)
        data_vars = DriverXarray.get_data_variable_names(xmeta, xdimmap)
        f = Field(initial_data=ds, dimension_map=xdimmap, driver='xarray', is_data=data_vars)

        self.assertIsInstance(f.x, xr.DataArray)
        self.assertIsInstance(f.time, xr.DataArray)
        self.assertEqual(f.data_variables[0].name, 'foo')

        grid1 = f.grid
        self.assertIsInstance(grid1, Grid)
        grid2 = f.grid
        self.assertIsInstance(grid2, Grid)

        # tdk:TEST: test a grid with a mask
        self.assertFalse(grid1.has_mask)

        self.assertIsNotNone(grid1.get_value_stacked())
        self.assertEqual(grid1.extent, (-180.0, -90.0, 180.0, 90.0))
        self.assertEqual(grid.abstraction, GridAbstraction.POLYGON)
        self.assertIsNone(f.geom)
        self.assertEqual(grid1.shape, (90, 180))

        f.set_abstraction_geom()

        self.assertIsInstance(f.geom.values[0, 0], Polygon)

        bbox = box(*[-40, -30, 40, 40])
        res = f.grid.get_intersects(bbox)

        self.assertEqual(res.extent, (-40.0, -30.0, 40.0, 40.0))

        # tdk:FEAT: re-work geometry variable to not need ocgis
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

        path1 = r'C:\Users\benko\Dropbox\dtmp\gpw-v4-population-density-rev10_2020_1_deg_tif\gpw_v4_population_density_rev10_2020_1_deg.tif'
        # path1 = '/home/benkoziol/Dropbox/dtmp/gpw-v4-population-density-rev10_2020_1_deg_tif/gpw_v4_population_density_rev10_2020_1_deg.tif'
        path2 = path1

        ds1 = xr.open_rasterio(path1, parse_coordinates=True, chunks={'y': 5, 'x': 5})
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

        # tdk: RESUME: run this example but use dask chunks and "get_source_subset"

        gc = GridChunker(f1, f2, nchunks_dst=(5, 5))
        for res in gc.iter_src_grid_subsets(yield_dst=True):
            print(res[0].extent)
            print(res)
            print(res[0].parent.to_xarray())
            print(res[2].parent.to_xarray())
            arch = res[0].parent.to_xarray()

    def test_system_masking(self):
        """Test masks are created and maintained when using an xarray backend."""

        grid = create_gridxy_global(resolution=2.0, wrapped=True)
        field = create_exact_field(grid, 'foo', ntime=31)
        path = self.get_temporary_file_path('foo.nc')
        field.write(path)

        ds = xr.open_dataset(path, autoclose=True)

        # tdk: FEATURE: grid expansion needs to use xarray

        # tdk: FEATURE: no crs support in xarray yet
        ds['latitude_longitude'] = Spherical()

        xmeta = create_metadata_from_xarray(ds)
        xdimmap = create_dimension_map(xmeta, DriverNetcdfCF)
        # tdk: FEATURE: need ocgis data_variables support using xarray somehow. it seems like this should happen automatically
        # tdk: FEATURE:   during field creation since everything else springs from metadata...
        is_data = DriverXarray.get_data_variable_names(xmeta, xdimmap)

        f1 = Field(initial_data=ds, dimension_map=xdimmap, driver='xarray', is_data=is_data)
        self.assertEqual(f1.data_variables[0].name, 'foo')

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

        # Ensure we can expand the grid.
        sub.expand()

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

    def test_has_bounds(self):
        # Assert we get the bounds variable from the container.
        da = xr.DataArray([4, 5, 6], attrs={'bounds': 'the_bounds'})
        dab = xr.DataArray([7, 8, 9])
        ds = xr.Dataset(data_vars={'foo': da, 'the_bounds': dab})
        self.assertTrue(DriverXarray.has_bounds(da, ds))

        # Assert the object does not have bounds if the bounds data is not actually in the dataset.
        da = xr.DataArray([4, 5, 6], attrs={'bounds': 'not_here'})
        ds = xr.Dataset(data_vars={'foo': da})
        self.assertFalse(DriverXarray.has_bounds(da, ds))
