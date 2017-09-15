import tempfile

from shapely.geometry import box

import ocgis
from ocgis import vm
from ocgis.constants import GridAbstraction
from ocgis.driver.nc_ugrid import DriverNetcdfUGRID
from ocgis.driver.request.core import RequestDataset
from ocgis.vmachine.mpi import rank_print

WD = tempfile.gettempdir()
# IN_PATH = '/home/ubuntu/data/UGRID_1km-merge-10min_HYDRO1K-merge-nomask_c130402.nc'
IN_PATH = '/media/benkoziol/Extra Drive 1/data/ocgis/ugrid-cesm-subsetting/UGRID_1km-merge-10min_HYDRO1K-merge-nomask_c130402.nc'
OUT_PATH = '/tmp/ugrid_subset.nc'

OUTDIR = '/home/benkoziol/htmp/ugrid_splits'
# OUTDIR = '/home/ubuntu/htmp/ugrid_splits'

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


def main():
    rd = RequestDataset(IN_PATH, driver=DriverNetcdfUGRID, grid_abstraction=GridAbstraction.POINT)
    field = rd.get()
    foo = '/tmp/foo.nc'
    # assert field.grid.cindex is not None
    # print field.grid.archetype
    # tkk
    print field.shapes
    sub = field.grid.get_intersects(box(*BBOX), optimized_bbox_subset=True).parent
    with vm.scoped_by_emptyable('reduce global', sub):
        if not vm.is_null:
            sub.grid_abstraction = GridAbstraction.POLYGON
            # rank_print('sub.grid.abstraction', sub.grid.abstraction)
            # rank_print('sub.grid._abstraction', sub.grid._abstraction)
            # rank_print('archetype', sub.grid.archetype)
            # rank_print(sub.grid.extent)
            rank_print('sub', sub.grid.cindex.get_value())
            subr = sub.grid.reduce_global().parent
            rank_print('sub', subr.grid.cindex.get_value())
            # rank_print(subr.x.name)
            # rank_print(subr.x.get_value().min())
            rank_print(subr.grid.extent)
            # rank_print(subr.grid.cindex.get_value())
            # rank_print(subr.shapes)
            # subr.write(foo)
    # if vm.rank == 0:
    #     RequestDataset(foo).inspect()
    vm.barrier()
    # print sub.shapes


if __name__ == '__main__':
    main()
