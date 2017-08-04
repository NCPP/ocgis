import os
import tempfile
from copy import deepcopy

import numpy as np
from shapely.geometry import box

import ocgis
from ocgis import vm
from ocgis.spatial.grid_splitter import GridSplitter
from ocgis.test.base import create_gridxy_global, create_exact_field
from ocgis.util.helpers import arange_from_dimension
from ocgis.variable.dimension import create_distributed_dimension

WD = tempfile.gettempdir()
IN_PATH = '/home/ubuntu/data/UGRID_1km-merge-10min_HYDRO1K-merge-nomask_c130402.nc'
# IN_PATH = '/media/benkoziol/Extra Drive 1/data/ocgis/ugrid-cesm-subsetting/UGRID_1km-merge-10min_HYDRO1K-merge-nomask_c130402.nc'
OUT_PATH = '/tmp/ugrid_subset.nc'

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


def get_subset(bbox, subset_filename, buffer_width, rhs_tol=10.):
    rd = ocgis.RequestDataset(uri=IN_PATH)
    rd.metadata['dimensions']['nlandmesh_face']['dist'] = True
    vc = rd.get_variable_collection()

    # ------------------------------------------------------------------------------------------------------------------
    # Subset the face centers and accumulate the indices of face centers occurring inside the bounding box selection.

    start_index = vc[MESHVAR].attrs.get('start_index', 0)

    # Stores indices of faces contained in the bounding box.
    px = vc[FACE_CENTER_X].extract().get_value()
    py = vc[FACE_CENTER_Y].extract().get_value()

    # Handle bounding box wrapping. This requires creating two bounding boxes to capture the left and right side of the
    # sphere.
    buffered_bbox = box(*bbox).buffer(buffer_width).envelope.bounds
    if buffered_bbox[0] < 0:
        bbox_rhs = list(deepcopy(buffered_bbox))
        bbox_rhs[0] = buffered_bbox[0] + 360.
        bbox_rhs[2] = 360. + rhs_tol

        bboxes = [buffered_bbox, bbox_rhs]
    else:
        bboxes = [buffered_bbox]

    initial = None
    for ctr, curr_bbox in enumerate(bboxes):
        select = create_boolean_select_array(curr_bbox, px, py, initial=initial)
        initial = select

    # ------------------------------------------------------------------------------------------------------------------
    # Use the selection criteria to extract associated nodes and reindex the new coordinate arrays.

    from ocgis.vmachine.mpi import rank_print
    # Retrieve the live ranks following the subset.
    has_select = ocgis.vm.gather(select.any())
    if ocgis.vm.rank == 0:
        live_ranks = np.array(ocgis.vm.ranks)[has_select]
    else:
        live_ranks = None
    live_ranks = ocgis.vm.bcast(live_ranks)

    with ocgis.vm.scoped('live ranks', live_ranks):
        if not ocgis.vm.is_null:
            has_subset = True

            rank_print('live ranks:', ocgis.vm.ranks)

            sub = vc[FACE_NODE].get_distributed_slice([select, slice(None)]).parent

            cindex = sub[FACE_NODE]
            cindex_original_shape = cindex.shape
            cindex_value = cindex.get_value().flatten()

            if start_index > 0:
                cindex_value -= start_index
            vc_coords = vc[XVAR][cindex_value].parent

            archetype_dim = vc_coords[XVAR].dimensions[0]
            arange_dimension = create_distributed_dimension(cindex_value.shape[0], name='arange_dim')
            new_cindex_value = arange_from_dimension(arange_dimension, start=start_index)

            cindex.set_value(new_cindex_value.reshape(*cindex_original_shape))

            new_vc_coords_dimension = create_distributed_dimension(vc_coords[XVAR].shape[0], name=archetype_dim.name,
                                                                   src_idx=archetype_dim._src_idx)
            vc_coords.dimensions[archetype_dim.name] = new_vc_coords_dimension

            # ------------------------------------------------------------------------------------------------------------------
            # Format the new variable collection and write out the new data file.

            # Remove old coordinate variables.
            for to_modify in [XVAR, YVAR]:
                sub[to_modify].extract(clean_break=True)

            for to_add in [XVAR, YVAR]:
                var_to_add = vc_coords[to_add].extract()
                sub.add_variable(var_to_add)

            rank_print('start sub.write')
            sub.write(subset_filename)
            rank_print('finished sub.write')

            if ocgis.vm.rank == 0:
                print 'subset x extent:', sub[FACE_CENTER_X].extent
                print 'subset y extent:', sub[FACE_CENTER_Y].extent
                ocgis.RequestDataset(subset_filename).inspect()
        else:
            has_subset = False

    return has_subset


def create_boolean_select_array(bbox, px, py, initial=None):
    select_x1 = px >= bbox[0]
    select_x2 = px <= bbox[2]
    select_x = np.logical_and(select_x1, select_x2)
    select_y1 = py >= bbox[1]
    select_y2 = py <= bbox[3]
    select_y = np.logical_and(select_y1, select_y2)
    select = np.logical_and(select_x, select_y)
    if initial is not None:
        select = np.logical_or(select, initial)
    return select


if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------------------------
    # Grid splitter implementation

    resolution = 1. / 111.
    # resolution = 1.
    grid = create_gridxy_global(resolution=resolution, wrapped=False, crs=ocgis.crs.Spherical())
    field = create_exact_field(grid, 'exact', ntime=3, fill_data_var=False, crs=ocgis.crs.Spherical())

    field.write(os.path.join(OUTDIR, 'dst_field_1km.nc'))

    gs = GridSplitter(grid, grid, (10, 10))

    ctr = 1
    for grid_sub in gs.iter_dst_grid_subsets():
        subset_filename = os.path.join(OUTDIR, 'src_subset_{}.nc'.format(ctr))

        dst_subset_filename = os.path.join(OUTDIR, 'dst_subset_{}.nc'.format(ctr))

        if vm.rank == 0:
            print 'creating subset:', subset_filename

        with vm.scoped_by_emptyable('grid subset', grid_sub):
            if not vm.is_null:
                extent_global = grid_sub.extent_global
                if vm.rank == 0:
                    root = vm.rank_global
            else:
                extent_global = None

        live_ranks = vm.get_live_ranks_from_object(grid_sub)
        bbox = vm.bcast(extent_global, root=live_ranks[0])

        vm.barrier()
        if vm.rank == 0:
            print 'starting bbox subset:', bbox
        vm.barrier()

        has_subset = get_subset(bbox, subset_filename, 1)

        vm.barrier()
        if vm.rank == 0:
            print 'finished bbox subset:', bbox
        vm.barrier()

        has_subset = vm.gather(has_subset)
        if vm.rank == 0:
            if any(has_subset):
                has_subset = True
                ctr += 1
            else:
                has_subset = False
        ctr = vm.bcast(ctr)
        has_subset = vm.bcast(has_subset)

        if has_subset:
            with vm.scoped_by_emptyable('dst subset write', grid_sub):
                if not vm.is_null:
                    grid_sub.parent.write(dst_subset_filename)
