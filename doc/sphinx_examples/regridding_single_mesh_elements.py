import os

import ESMF
import numpy as np
import ocgis
from ocgis.regrid import RegridOperation
from ocgis.test import create_gridxy_global, create_exact_field

# Path to the shapefile containing polygons defining the destination ESMF mesh
IN_SHAPEFILE_PATH = '/home/benkoziol/Dropbox/dtmp/basin_set_full_res/HCDN_nhru_final_671.shp'

# If true, create the weight file and do not perform a sparse matrix multiplication
WEIGHTS_ONLY = False

# Resolution of the structured source grid in degrees
SRC_RESOLUTION = 1.0

# Maximum error allowed for the test assert when applying weights
MAX_ERROR = 0.084

# Tell ESMPy to do logging
ESMF.Manager(debug=True)

# ----------------------------------------------------------------------------------------------------------------------

# Reference the OCGIS geometry variable. This contains an array of Shapely geometry objects eventually. These are the
# regrid targets or destinations.
geoms = ocgis.RequestDataset(IN_SHAPEFILE_PATH).create_field().geom

# Create an exact source field to use for evaluating regridding errors. Give it three timesteps to make sure we are
# iterating appropriately over the time dimension.
src_grid = create_gridxy_global(resolution=SRC_RESOLUTION, crs=ocgis.crs.Spherical())
src_field = create_exact_field(src_grid, 'exact', ntime=3)

# If we are applying weights, this will contain the difference between the regridded value and exact field value
# calculated for the polygon centroid.
max_errors = []

# Iterate over the geometries. This loop should be unwrapped for a parallel execution.
for ii in range(geoms.size):
    print('Weighting {} of {}'.format(ii + 1, geoms.size))

    # If we are writing weights, create the output weights filename
    if WEIGHTS_ONLY:
        weights_out = 'geom_weights_{}.nc'.format(ii)
    else:
        weights_out = None

    # Slice the geometry variable to get the current geometry.
    sub = geoms[ii]
    # The source data is probably not in spherical coordinate system so update it. If it is already spherical, this will
    # just pass through.
    sub.update_crs(ocgis.crs.Spherical())

    # Convert the Shapely geometry (OCGIS geometry variable) object to an in-memory coordinate representation. In other
    # words, create an unstructured grid with a UGRID-like schema. This is needed to create an ESMPy mesh. When pack
    # is False, we do not de-duplicate coordinates. This is okay since this is useful for topologies and with a single
    # element, it does not matter. The node threshold will limit the number of nodes per geometry which makes ESMF
    # triangulation faster and the conversion to coordinates a little slower.
    print(' Converting to coordinates...')
    coords = sub.convert_to(start_index=0, pack=False, node_threshold=500)

    print(' Subsetting...')
    # Buffer the subset geometry to make sure we have enough spatial halo for a conservative regridding operation
    subset_geometry = sub.envelope.buffer(SRC_RESOLUTION).envelope
    # Subset the source grid and reference its parent field
    sub_src_field = src_field.grid.get_intersects(subset_geometry, optimized_bbox_subset=True).parent

    # Execute the regridding operation applying weights or just writing them to file. When split=False, we do a bulk
    # operation as opposed to iterating over the timesteps which is much faster but consumes more memory.
    regrid_options = {'split': False, 'weights_out': weights_out, 'weights_only': WEIGHTS_ONLY}
    ro = RegridOperation(sub_src_field, coords.parent, regrid_options=regrid_options)

    print(' Regridding...')
    regridded = ro.execute()

    if not WEIGHTS_ONLY:
        # Add the regridded data variable and its time component to the source geometry field. This will preserve
        # attributes from the shapefile.
        for dv in regridded.data_variables:
            sub.parent.add_variable(dv.extract(), is_data=True)
        sub.parent.set_time(regridded.time.extract())

        # Compute the max error using the exact field and timestep adjustments
        sgeom = sub.v()[0]
        desired = ocgis.util.helpers.create_exact_field_value([sgeom.centroid.x], [sgeom.centroid.y])[0]
        desired = [(10 * (ii + 1)) + desired for ii in range(regridded.time.size)]
        desired = np.array(desired).reshape(regridded.data_variables[0].shape)
        diff = np.abs(desired - regridded.data_variables[0].v())
        max_errors.append(diff.max())

        print(' Current Max Error: {}'.format(np.max(max_errors)))

        # Convert the output regridded field to an xarray representation
        x = sub.parent.to_xarray()
    else:
        assert os.path.exists(weights_out)

if not WEIGHTS_ONLY:
    assert np.max(max_errors) < MAX_ERROR

print('Success!')
