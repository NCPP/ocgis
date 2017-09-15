import os
import tempfile

import ocgis
from ocgis import vm
from ocgis.constants import GridAbstraction
from ocgis.driver.nc_ugrid import DriverNetcdfUGRID
from ocgis.driver.request.core import RequestDataset
from ocgis.spatial.grid_splitter import GridSplitter
from ocgis.test.base import create_gridxy_global, create_exact_field

WD = tempfile.gettempdir()
IN_PATH = '/home/ubuntu/data/UGRID_1km-merge-10min_HYDRO1K-merge-nomask_c130402.nc'
# IN_PATH = '/media/benkoziol/Extra Drive 1/data/ocgis/ugrid-cesm-subsetting/UGRID_1km-merge-10min_HYDRO1K-merge-nomask_c130402.nc'
# OUT_PATH = '/tmp/ugrid_subset.nc'

# OUTDIR = '/home/benkoziol/htmp/ugrid_splits'
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
    resolution = 1. / 111.
    # resolution = 1.
    dst_grid = create_gridxy_global(resolution=resolution, wrapped=False, crs=ocgis.crs.Spherical())
    field = create_exact_field(dst_grid, 'exact', ntime=3, fill_data_var=False, crs=ocgis.crs.Spherical())
    field.write(os.path.join(OUTDIR, 'dst_field_1km.nc'))
    src_grid = RequestDataset(uri=IN_PATH, driver=DriverNetcdfUGRID, grid_abstraction=GridAbstraction.POINT).get().grid
    gs = GridSplitter(src_grid, dst_grid, (10, 10), src_grid_resolution=0.167, check_contains=False)
    return gs


def main(write_subsets=False, merge_weight_files=False):
    gs = create_grid_splitter()

    index_path = os.path.join(OUTDIR, 'gs_index.nc')
    if write_subsets:
        src_template = os.path.join(OUTDIR, 'src_subset_{}.nc')
        dst_template = os.path.join(OUTDIR, 'dst_subset_{}.nc')
        wgt_template = os.path.join('esmf_weights_{}.nc')
        gs.write_subsets(src_template, dst_template, wgt_template, index_path)

    if merge_weight_files:
        mwfn = os.path.join(OUTDIR, '01-merged_weights.nc')
        gs.create_merged_weight_file(index_path, mwfn, OUTDIR, split_grids_directory=OUTDIR)

    vm.barrier()


def test():
    # for ii in range(1, 101):
    #     src_template = os.path.join(OUTDIR, 'src_subset_{}.nc'.format(ii))
    #     if os.path.exists(src_template):
    #         dst_template = os.path.join(OUTDIR, 'dst_subset_{}.nc'.format(ii))
    #         fsrc = RequestDataset(src_template, driver=DriverNetcdfUGRID, grid_abstraction='polygon').get()
    #         fdst = RequestDataset(dst_template).get()
    #
    #         print 'src extent:', fsrc.grid.extent
    #         sbox = box(*fsrc.grid.extent)
    #         print 'dst extent:', fdst.grid.extent
    #         dbox = box(*fdst.grid.extent)
    #         print sbox.intersects(dbox)
    #         time.sleep(0.5)

    # for ii in range(1, 101):
    #     template = os.path.join(OUTDIR, 'esmf_weights_{}.nc'.format(ii))
    #     if os.path.exists(template):
    #         field = RequestDataset(template).get()
    #         S = field['S'].get_value()
    #         print 'esmf weights:', template
    #         print 'min:', S.min()
    #         print 'max:', S.max()
    #         if (S.max() - 1.0) > 1e6:
    #             raise ValueError('S is too high')
    #         if (S.min()) < 0.0:
    #             raise ValueError('S is less then zero')

    weight_filename = os.path.join(OUTDIR, '01-merged_weights.nc')
    assert_weight_file_is_rational(weight_filename)


if __name__ == '__main__':
    # main(write_subsets=False, merge_weight_files=True)
    test()
