import os

import ocgis
from ocgis import vm
from ocgis.constants import GridAbstraction
from ocgis.driver.nc_ugrid import DriverNetcdfUGRID
from ocgis.driver.request.core import RequestDataset
from ocgis.spatial.grid_splitter import GridSplitter
from ocgis.test.base import create_gridxy_global, create_exact_field


IN_PATH = '/home/ubuntu/data/UGRID_1km-merge-10min_HYDRO1K-merge-nomask_c130402.nc'
OUTDIR = '/home/ubuntu/htmp/ugrid_splits'

BBOX = [10., 5., 12., 7.]
XVAR = 'landmesh_node_x'
YVAR = 'landmesh_node_y'
FACE_CENTER_X = 'landmesh_face_x'
FACE_CENTER_Y = 'landmesh_face_y'
FACE_NODE = 'landmesh_face_node'
DIM_FACE_COUNT = 'nlandmesh_face'
MESHVAR = 'landmesh'

# Do not set units on bounds variables by default.
ocgis.env.CLOBBER_UNITS_ON_BOUNDS = False


def assert_weight_file_is_rational(weight_filename):
    wf = RequestDataset(weight_filename).get()
    row = wf['row'].get_value()
    S = wf['S']
    # col = wf['col'].get_value()
    for idx in xrange(1, row.size + 1):
        print 'current row:', idx
        row_idx = row == idx
        curr_S = S[row_idx].get_value()
        print 'current S sum:', curr_S.sum()
        print '============================='


def create_grid_splitter():
    paths = {'wd': OUTDIR}
    resolution = 1. / 111.
    dst_grid = create_gridxy_global(resolution=resolution, wrapped=False, crs=ocgis.crs.Spherical())
    field = create_exact_field(dst_grid, 'exact', ntime=3, fill_data_var=False, crs=ocgis.crs.Spherical())
    field.write(os.path.join(OUTDIR, 'dst_field_1km.nc'))
    src_grid = RequestDataset(uri=IN_PATH, driver=DriverNetcdfUGRID, grid_abstraction=GridAbstraction.POINT).get().grid
    gs = GridSplitter(src_grid, dst_grid, (10, 10), paths=paths, src_grid_resolution=0.167, check_contains=False)
    return gs


def main(write_subsets=False, merge_weight_files=False):
    gs = create_grid_splitter()
    if write_subsets:
        gs.write_subsets()
    if merge_weight_files:
        mwfn = os.path.join(OUTDIR, '01-merged_weights.nc')
        gs.create_merged_weight_file(mwfn)
    vm.barrier()


def test():
    weight_filename = os.path.join(OUTDIR, '01-merged_weights.nc')
    assert_weight_file_is_rational(weight_filename)


if __name__ == '__main__':
    main(write_subsets=True, merge_weight_files=False)
    # test()
