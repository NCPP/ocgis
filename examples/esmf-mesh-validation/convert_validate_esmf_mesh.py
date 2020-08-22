"""
Test conversion of GeoPackage GIS data to ESMF Unstructured validating each mesh element along the way.
"""
import os
import random
import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy
from subprocess import CalledProcessError

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

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


def itr_polygon(geom):
    if is_multipolygon(geom):
        itr = geom
    else:
        itr = [geom]
    for g in itr:
        yield g


def is_multipolygon(geom):
    return isinstance(geom, MultiPolygon)


def shift_coords(geom):
    is_multi = is_multipolygon(geom)
    new_geom = []
    for g in itr_polygon(geom):
        coords = np.array(g.exterior.coords)
        select = coords[:, 0] >= 180.0
        adjusted = False
        if np.any(select):
            adjusted = True
            # coords[select, 0] = 180.0 - 1e-6
            # import ipdb;ipdb.set_trace() #tdk:rm
            # with open('/tmp/coords.csv', 'w') as f:
            #     for i in range(coords.shape[0]):
            #         f.write('{},{}\n'.format(coords[i, 0], coords[i, 1]))
            coords[select, 0] = 180.0 - 1e-6
        if adjusted:
            new_geom.append(Polygon(coords).buffer(0.0))
        else:
            # pass
            new_geom.append(g) #tdk:enable
    if is_multi:
        ret = MultiPolygon(new_geom)
    else:
        ret = new_geom[0]
    return ret


def assign_longitude(src, dst):
    is_multi = is_multipolygon(src)
    new_geom = []
    for s, d in zip(itr_polygon(src), itr_polygon(dst)):
        src_coords = np.array(s.exterior.coords)
        dst_coords = np.array(d.exterior.coords)
        dst_coords[:, 0] = src_coords[:, 0]
        new_geom.append(Polygon(dst_coords).buffer(0.0))
    if is_multi:
        ret = MultiPolygon(new_geom)
    else:
        ret = new_geom[0]
    return ret


def check_crs_transform(original, transformed):
    """
    Return True if the transform is okay.
    """
    diff = np.abs(original.area - transformed.area)
    return diff < 1.0


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
    # Check if the CRS transformation is valid
    original_geom = ofield.geom.v().flatten()[0]
    crs_transform_valid = check_crs_transform(original_geom, field.geom.v().flatten()[0])
    # If the transform is not valid, then adjust the coordinate system of the original and update the CRS again.
    if not crs_transform_valid:
        shifted = shift_coords(original_geom)
        shifted_field = deepcopy(ofield)
        shifted_field.geom.v().flatten()[0] = shifted
        shifted_field.update_crs(ocgis.crs.Spherical())
        # If it is still not valid, swap out the longitudes since only the latitude is transformed.
        if not check_crs_transform(shifted, shifted_field.geom.one()):
            assigned = assign_longitude(shifted, shifted_field.geom.one())
            shifted_field.geom.v()[0] = assigned
        field = shifted_field
    # Convert the field geometry to an unstructured grid format based on the UGRID spec.
    try:
        gc = field.geom.convert_to(
            pack=PACK,
            node_threshold=NODE_THRESHOLD,
            split_interiors=SPLIT_INTERIORS,
            remove_self_intersects=False,
            allow_splitting_excs=False
        )
    except Exception as e:
        print('ERROR: OCGIS conversion problem for hruid={}'.format(hruid))
        success = False
    else:
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
        # if record['properties']['hruid'] != 1033010:
        #     continue
        # Print an update every 100 iterations
        if ctr % 10 == 0:
            print('INFO: Index {} of {}'.format(ctr, len_gc))
        do_record_test(exedir, record)


if __name__ == "__main__":
    main(int(sys.argv[1]))
