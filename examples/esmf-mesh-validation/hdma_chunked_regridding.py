import logging
import os

import ocgis
from ocgis import env, RequestDataset, Variable, constants, GeomCabinetIterator
from ocgis.constants import DriverKey, WrappedState
from ocgis.driver.nc_esmf_unstruct import DriverESMFUnstruct
from ocgis.spatial.grid_chunker import GridChunker
from ocgis.test import create_gridxy_global, create_exact_field
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.variable import stack
from ocgis.variable.crs import Spherical, WGS84

WD = "/home/ubuntu/scratch/hdma-chunks"
DSTPATH = "/home/ubuntu/data/hdma_global_catch_v2.gpkg"

if ocgis.vm.rank == 0:
    os.mkdir(WD)
ocgis.vm.Barrier()

ocgis_lh.configure(to_stream=False, to_file="ocgis-{}.log".format(ocgis.vm.rank), level=logging.INFO)
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
                     driver=DriverKey.VECTOR,
                     driver_kwargs={'feature_class': 'Catchment'}).create_field().geom
dst.set_ugid(dst.parent["hruid"])
dst._wrapped_state = WrappedState.WRAPPED

dst.parent.set_crs(Spherical())

chunkdir = os.path.join(WD, 'chunks')
if ocgis.vm.rank == 0:
    os.mkdir(chunkdir)
ocgis.vm.Barrier()


def iter_dst_grid_subsets(gc, yield_slice=False, yield_idx=None):
    problem_hruids_ocgis = [1074868, 1074869, 1033010, 1027959, 1035146, 1011816, 1021343, 1031035, 1024388, 1060208,
                            1059676, 1014166, 1048959, 1029493, 1074935, 1008801, 1055663, 1074875, 1074878, 1056130,
                            1038612, 1032965, 1026209, 1074930, 1074931, 1074932, 1038453, ]
    gitr = GeomCabinetIterator(path=DSTPATH, as_field=True)
    nelements = len(gitr)
    # accum_count = 1
    accum_count = 10000
    accum_ctr = 0
    accum_group = 0
    fields = []
    for ii, field in enumerate(gitr, start=0):
        hruid = field["hruid"].v()[0]
        if accum_ctr < accum_count or ii == nelements-1:
            if hruid in problem_hruids_ocgis:
                continue
            fields.append(field)
            accum_ctr += 1
        if accum_ctr % accum_count == 0 or ii == nelements-1:
            field = stack(fields, "ocgis_ngeom")

            # Reset variables in the scope
            accum_ctr = 0
            fields = []

            # ocgis_lh(msg="{} of {}, hruid={}".format(ii, nelements, hruid), logger="iter_dst")
            ocgis_lh(msg="{} of {}".format(ii, nelements), logger="iter_dst")

            field.set_crs(WGS84())
            # try:
            ocgis_lh(msg="starting conversion", logger="iter_dst")
            sub = field.geom.convert_to(use_geometry_iterator=False, pack=False,
                                        node_threshold=None, split_interiors=False,
                                        remove_self_intersects=False,
                                        to_crs=Spherical(),
                                        allow_splitting_excs=False,
                                        add_center_coords=True)
            # except:
            #     ocgis_lh(msg="error during conversion", logger="iter_dst", level=logging.ERROR)
            #     continue

            guidx = Variable(name=constants.GridChunkerConstants.IndexFile.NAME_DSTIDX_GUID,
                             dimensions=['elementCount'],
                             value=field["hruid"].v())
            sub.parent.add_variable(guidx)

            ocgis_lh(msg="starting write", logger="iter_dst")
            with ocgis.vm.scoped("sub write", [0]):
                if not ocgis.vm.is_null:
                    sub.parent.write(os.path.join(gc.paths['wd'], gc.paths['dst_template'].format(accum_group + 1)),
                                     driver=DriverKey.NETCDF_ESMF_UNSTRUCT)
            ocgis.vm.Barrier()
            accum_group += 1

            # Indicate the file is actually written as a ESMF unstructured file. Otherwise the default mesh file
            # format will be UGRID.
            sub.parent.set_driver(DriverESMFUnstruct)

            ocgis_lh(msg="yielding", logger="iter_dst")
            if yield_slice:
                yield sub, None
            else:
                yield sub


gc = GridChunker(source=srcrd, destination=dst, paths={'wd': chunkdir},
                 src_grid_resolution=src_field.grid.resolution_max,
                 dst_grid_resolution=src_field.grid.resolution_max,
                 genweights=True, debug=False, iter_dst=iter_dst_grid_subsets,
                 esmf_kwargs={"ignore_degenerate": True})
gc.write_chunks()
gc.create_merged_weight_file(os.path.join(WD, "merged_weights.nc"))
