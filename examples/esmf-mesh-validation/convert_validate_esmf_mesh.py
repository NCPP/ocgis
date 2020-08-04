"""
Test conversion of GeoPackage GIS data to ESMF Unstructured validating each mesh element along the way.
"""
import os
import random
import shutil
import subprocess
import sys
import tempfile
from subprocess import CalledProcessError

import numpy as np

import ocgis

# Path to the source GIS data assuming a GeoPackage
# GEOPKG = 'hdma_global_catch_0.gpkg'
GEOPKG = '/home/benkoziol/l/data/esmf/hdma-catchments-cesm-20200729/hdma_global_catch_0.gpkg'
# This is the template for catchment-specific directories. If the test operation is successful, this will deleted. If it
# is not successful, the directory will be left behind.
OUTDIR_TEMPLATE = os.path.join(tempfile.gettempdir(), 'individual-element-nc', 'hruid-tmp-{}')
# Debugging to simulate errors
DEBUG = False
# Number of nodes for each virtual polygon
NODE_THRESHOLD = 5000
# De-duplicate nodes (serial only). Leave this off for initial testing since there is only one node coordinates array
PACK = False
# Whether to split holes/interiors. Start with False just to use exteriors
SPLIT_INTERIORS = False


def do_esmf(ncpath, exedir):
    # Will return True if the operation is successful
    try:
        subprocess.check_call([sys.executable, os.path.join(exedir, 'run_esmf_mesh_test.py'), ncpath])
        was_successful = True
    except CalledProcessError:
        was_successful = False
        print('ERROR: ESMF read/regridding problem with file path {}'.format(ncpath))
    return was_successful


def do_record_test(exedir, record):
    # The record's unique HRU identifier
    hruid = record['properties']['hruid']
    # The current output directory
    curr_outdir = OUTDIR_TEMPLATE.format(hruid)
    # Create the directory
    os.makedirs(curr_outdir, exist_ok=True)
    # Make that the current working directory
    os.chdir(curr_outdir)
    # We need to transform the coordinate system from WGS84 to Spherical for ESMF
    crs = ocgis.crs.CoordinateReferenceSystem(value=record['meta']['crs'])
    field = ocgis.Field.from_records([record], crs=crs)
    field.update_crs(ocgis.crs.Spherical())
    # Convert the field geometry to an unstructured grid format based on the UGRID spec.
    gc = field.geom.convert_to(
        pack=PACK,
        node_threshold=NODE_THRESHOLD,
        split_interiors=SPLIT_INTERIORS,
        remove_self_intersects=False,
        allow_splitting_excs=False
    )
    # Path to the output netCDF file for the current element
    out_element_nc = os.path.join(curr_outdir, "esmf-element_hruid-{}.nc".format(hruid))
    # Add the center coordinate to make ESMF happy (even though we are not using it)
    if DEBUG and random.random() > 0.9:
        pass  # Purposefully make an error in the file
    else:
        centerCoords = np.array([field.geom.v()[0].centroid.x, field.geom.v()[0].centroid.y]).reshape(1, 2)
        ocgis.Variable(name='centerCoords', value=centerCoords, dimensions=['elementCount', 'coordDim'], attrs={'units': 'degrees'}, parent=gc.parent)
    # When writing the data to file, convert to ESMF unstructured format.
    gc.parent.write(out_element_nc, driver='netcdf-esmf-unstruct')
    # Run the simple regridding test
    success = do_esmf(out_element_nc, exedir)
    if success:
        # If successful, remove the directory
        assert 'hruid-tmp-' in curr_outdir
        shutil.rmtree(curr_outdir)
    else:
        # If it's not successful, leave the directory. Write the shapefile so it's easy to look at. Also send the record
        # string to a file.
        field.geom.write_vector('02-problem-hruid-{}.shp'.format(hruid))
        record_out = '01-problem-record-hruid-{}.out'.format(hruid)
        with open(record_out, 'w') as f:
            f.write(str(record))
    # Change back to the execution directory
    os.chdir(exedir)


def main(check_start_index):
    # The execution directory
    exedir = os.getcwd()
    # An iterator over the geometries in the GeoPackage
    gc = ocgis.GeomCabinetIterator(path=GEOPKG, driver_kwargs={'feature_class': 'Catchment'})
    # Number of records in the GeoPackage
    len_gc = len(gc)
    for ctr, record in enumerate(gc):
        if ctr < check_start_index:
            continue
        # Print an update every 100 iterations
        if ctr % 10 == 0:
            print('INFO: Index {} of {}'.format(ctr, len_gc))
        do_record_test(exedir, record)


if __name__ == "__main__":
    main(int(sys.argv[1]))
