from copy import deepcopy

from mock import Mock, mock
from ocgis.driver.dxarray import DriverXarray
from ocgis.driver.nc import DriverNetcdfCF
from ocgis.driver.nc_esmf_unstruct import DriverESMFUnstruct
from ocgis.driver.nc_ugrid import DriverNetcdfUGRID
from ocgis.driver.registry import get_driver_class
from ocgis.spatial.grid_chunker import GridChunker
from ocgis.util.addict import Dict
from ocgis.variable.crs import Spherical, CFSpherical
from shapely.geometry import Polygon, box

from ocgis import Field, Grid, RequestDataset, DimensionMap, GridUnstruct, env
from ocgis.constants import DMK, GridAbstraction, DriverKey, Topology
from ocgis.test import create_gridxy_global, create_exact_field
from ocgis.test.base import TestBase
import xarray as xr


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
        xdimmap1 = create_dimension_map(xmeta1, DriverNetcdfCF)
        xdimmap1.set_crs(None)

        xdimmap2 = deepcopy(xdimmap1)

        f1 = Field(initial_data=ds1, dimension_map=xdimmap1, crs=CFSpherical())
        f1.grid.set_extrapolated_bounds('xbounds', 'ybounds', 'bounds')
        f2 = Field(initial_data=ds2, dimension_map=xdimmap2, crs=CFSpherical())
        f2.grid.set_extrapolated_bounds('xbounds', 'ybounds', 'bounds')

        gc = GridChunker(f1, f2, nchunks_dst=(5, 5))
        for res in gc.iter_src_grid_subsets(yield_dst=True):
            print(res[0].extent)
            print(res)
            print(res[0].parent.to_xarray())
            print(res[2].parent.to_xarray())
            arch = res[0].parent.to_xarray()

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
        rd = mock.create_autospec(RequestDataset)
        xd = DriverXarray(rd)
        self.assertIsInstance(xd, DriverXarray)


def create_dimension_map(meta, driver):
    # tdk: DOC
    # Check if this is a class or an instance. If it is a class, convert to instance for dimension map
    # creation.
    if isinstance(driver, type):
        driver = driver()
    dimmap = DimensionMap.from_metadata(driver, meta)
    return dimmap


def create_metadata_from_xarray(ds):
    # tdk: DOC
    xmeta = Dict()
    for dimname, dimsize in ds.dims.items():
        xmeta.dimensions[dimname] = {'name': dimname, 'size': dimsize}
    for varname, var in ds.variables.items():
        xmeta.variables[varname] = {'name': varname, 'dimensions': var.dims, 'attrs': var.attrs}
    xmeta = dict(xmeta)
    return xmeta
