import logging
import os

import ocgis
from ocgis import env, RequestDataset, Variable, constants, GeomCabinetIterator
from ocgis.constants import DriverKey, WrappedState
from ocgis.driver.nc_esmf_unstruct import DriverESMFUnstruct
from ocgis.spatial.grid_chunker import GridChunker
from ocgis.test import create_gridxy_global, create_exact_field
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.variable.crs import Spherical, WGS84

WD = "/home/benkoziol/l/data/esmf/hdma-catchments-cesm-20200729/chunked-regridding"
DSTPATH = "/home/benkoziol/l/data/esmf/hdma-catchments-cesm-20200729/hdma_global_catch_v2.gpkg"

if ocgis.vm.rank == 0:
    os.mkdir(WD)
ocgis.vm.Barrier()

ocgis_lh.configure(to_stream=True, level=logging.INFO)
env.CLOBBER_UNITS_ON_BOUNDS = False

# Create the source field --------------------------------------------------------------------------------------

src_grid = create_gridxy_global(crs=Spherical())
src_field = create_exact_field(src_grid, 'foo')

source = os.path.join(WD, "source.nc")
src_field.write(source)

# Call into the CLI --------------------------------------------------------------------------------------------

srcrd = RequestDataset(uri=source)

# dst = RequestDataset(uri=self.path_state_boundaries, driver=DriverKey.VECTOR).create_field().geom
# dst.set_ugid(dst.parent["UGID"])

dst = RequestDataset(uri=DSTPATH,
                     driver=DriverKey.VECTOR).create_field().geom
dst.set_ugid(dst.parent["hruid"])
dst._wrapped_state = WrappedState.WRAPPED

dst.parent.set_crs(Spherical())

chunkdir = os.path.join(WD, 'chunks')
if ocgis.vm.rank == 0:
    os.mkdir(chunkdir)
ocgis.vm.Barrier()


def iter_dst_grid_subsets(gc, yield_slice=False, yield_idx=None):
    gitr = GeomCabinetIterator(path=DSTPATH, as_field=True)
    nelements = len(gitr)
    # field = RequestDataset(self.path_state_boundaries, driver="vector").create_field()
    # state_names = field["STATE_NAME"].v().tolist()
    for ii, field in enumerate(gitr, start=0):
        if ii % 100 == 0:
            ocgis_lh(msg="{} of {}".format(ii, nelements), logger="iter_dst")
        ocgis_lh(msg="hruid={}".format(field["hruid"].v()[0]), logger="iter_dst")
        # field = RequestDataset(self.path_state_boundaries, driver="vector").create_field()
        # select = field["STATE_NAME"].v() == state_name
        # assert (select.any())
        # field = field["STATE_NAME"][select].parent
        import ipdb;ipdb.set_trace() #tdk:rm

        field.set_crs(WGS84())

        sub = field.geom.convert_to(use_geometry_iterator=False, pack=False,
                                    node_threshold=5000, split_interiors=False,
                                    remove_self_intersects=False,
                                    to_crs=Spherical(),
                                    allow_splitting_excs=False,
                                    add_center_coords=True)

        guidx = Variable(name=constants.GridChunkerConstants.IndexFile.NAME_DSTIDX_GUID,
                         dimensions=['elementCount'],
                         value=field["hruid"].v())
        sub.parent.add_variable(guidx)

        with ocgis.vm.scoped("sub write", [0]):
            if not ocgis.vm.is_null:
                sub.parent.write(os.path.join(gc.paths['wd'], gc.paths['dst_template'].format(ii + 1)),
                                 driver=DriverKey.NETCDF_ESMF_UNSTRUCT)
        ocgis.vm.Barrier()
        # Indicate the file is actually written as a ESMF unstructured file. Otherwise the default mesh file
        # format will be UGRID.
        sub.parent.set_driver(DriverESMFUnstruct)

        if yield_slice:
            yield sub, None
        else:
            yield sub

        tdk


gc = GridChunker(source=srcrd, destination=dst, paths={'wd': chunkdir},
                 src_grid_resolution=src_field.grid.resolution_max,
                 dst_grid_resolution=src_field.grid.resolution_max,
                 genweights=True, debug=False, iter_dst=iter_dst_grid_subsets,
                 esmf_kwargs={"ignore_degenerate": True})
gc.write_chunks()
gc.create_merged_weight_file(os.path.join(WD, "merged_weights.nc"))
