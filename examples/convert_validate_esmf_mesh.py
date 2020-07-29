"""
Test conversion of GeoPackage GIS data to ESMF Unstructured validating each mesh element along the way.
"""
import os
import random
import shutil
import tempfile

import ESMF
import numpy as np

import ocgis

ESMF.Manager(debug=True)

# Path to the source GIS data assuming a GeoPackage
GEOPKG = 'hdma_global_catch_0.gpkg'
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


def do_destroy(to_destroy):
    for t in to_destroy:
        try:
            t.destroy()
        except:
            pass


def do_esmf(ncpath, exedir, dst_logdir):
    # Will return True if the operation is successful
    was_successful = False
    # Destroy the ESMF objects so deleting them if there is a failure is smoother
    mesh, src, dst, regrid = [None]*4
    try:
        # Fake debug error simulation
        if DEBUG and random.random() > 0.5:
            raise ValueError('a fake debug error for testing')
        # Create the ESMF mesh
        mesh = ESMF.Mesh(filename=ncpath, filetype=ESMF.constants.FileFormat.ESMFMESH)
        # Create the source
        src = ESMF.Field(mesh, ndbounds=np.array([1, 1]), meshloc=ESMF.constants.MeshLoc.ELEMENT)
        # Create the destination
        dst = ESMF.Field(mesh, ndbounds=np.array([1, 1]), meshloc=ESMF.constants.MeshLoc.ELEMENT)
        # This will create the route handle and return some weights
        regrid = ESMF.Regrid(srcfield=src, dstfield=dst, regrid_method=ESMF.constants.RegridMethod.CONSERVE, factors=True)
        factors = regrid.get_weights_dict(deep_copy=True)
        assert factors is not None
        was_successful = True
    except Exception as e:
        print('ERROR: {}: ESMF read/regridding problem with file path {}'.format(str(e), ncpath))
        # Copy over the ESMF log file
        srcpath = os.path.join(exedir, 'PET0.ESMF_LogFile')
        dstpath = os.path.join(dst_logdir, 'PET0.ESMF_LogFile')
        if os.path.exists(srcpath):
            shutil.copy2(srcpath, dstpath)
        raise
    finally:
        do_destroy([mesh, src, dst, regrid])
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
    centerCoords = np.array([field.geom.v()[0].centroid.x, field.geom.v()[0].centroid.y]).reshape(1, 2)
    ocgis.Variable(name='centerCoords', value=centerCoords, dimensions=['elementCount', 'coordDim'], attrs={'units': 'degrees'}, parent=gc.parent)
    # When writing the data to file, convert to ESMF unstructured format.
    gc.parent.write(out_element_nc, driver='netcdf-esmf-unstruct')
    # Run the simple regridding test
    success = do_esmf(out_element_nc, exedir, curr_outdir)
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
    # Try to remove the log file if it exists.
    if os.path.exists('PET0.ESMF_LogFile'):
        os.remove('PET0.ESMF_LogFile')


def main():
    # The execution directory
    exedir = os.getcwd()
    # An iterator over the geometries in the GeoPackage
    gc = ocgis.GeomCabinetIterator(path=GEOPKG, driver_kwargs={'feature_class': 'Catchment'})
    # Number of records in the GeoPackage
    len_gc = len(gc)
    for ctr, record in enumerate(gc):
        # Print an update every 100 iterations
        if ctr % 100 == 0:
            print('INFO: Index {} of {}'.format(ctr, len_gc))
        do_record_test(exedir, record)


if __name__ == "__main__":
    main()
