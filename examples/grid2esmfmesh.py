import itertools

import numpy as np

import ocgis
from ocgis.test import create_gridxy_global
from ocgis.util.helpers import get_esmf_corners_from_ocgis_corners

# Spatial resolution of the output grid in degrees.
RES = 2
RESOLUTION = 1/2**RES
# Path to the output netCDF file.
OUTFILE = "ll1x2e"+str(RES)+'deg.esmf.nc'

def format_corner_indices(corner_indices):
    """
    Convert corner indices into a flat array with the correct rotation.
    """
    formatted_indices = np.zeros(4, dtype=np.int32)
    formatted_indices[0] = corner_indices[0, 0]
    formatted_indices[1] = corner_indices[1, 0]
    formatted_indices[2] = corner_indices[1, 1]
    formatted_indices[3] = corner_indices[0, 1]
    return formatted_indices


def get_corner_indices(ii, jj, indexing):
    """
    Get corner indices from the global indexing array for an element indexed by ii, jj
    """
    slice_row = slice(ii-1, ii+1)
    slice_col = slice(jj-1, jj+1)
    corner_indices = indexing[slice_row, slice_col]
    try:
        assert(corner_indices.size == 4)
    except AssertionError:
        print(ii, jj)
        raise
    return corner_indices


# Create the grid object
grid = create_gridxy_global(resolution=RESOLUTION, with_bounds=True, wrapped=False)
# Grid should not be vectorized
grid.expand()
# Convert ocgis corners to esmf corners
esmf_corners_x = get_esmf_corners_from_ocgis_corners(grid.x.bounds.v())
esmf_corners_y = get_esmf_corners_from_ocgis_corners(grid.y.bounds.v())
# Create one-based global index
indexing = np.arange(1, esmf_corners_x.size + 1, dtype=np.int32).reshape(esmf_corners_x.shape)

# Number of elements
elementCount = grid.archetype.size
# Maximum number of nodes per element
maxNodePElement = 4
# Will hold element connectivity indexing
elementConn = np.zeros((elementCount, maxNodePElement), dtype=np.int32)
# Convert the indices to a master connectivity array
insert = 0
for ii, jj in itertools.product(range(1, esmf_corners_x.shape[0]), range(1, esmf_corners_x.shape[1])):
    corner_indices = get_corner_indices(ii, jj, indexing)
    formatted_indices = format_corner_indices(corner_indices)
    elementConn[insert] = formatted_indices
    insert += 1

# Number of nodes
nodeCount = esmf_corners_x.size
# Will hold node coordinates
nodeCoords = np.zeros((nodeCount, 2), dtype=np.float32)
# Insert the node coordinates
nodeCoords[:, 0] = esmf_corners_x.flatten()
nodeCoords[:, 1] = esmf_corners_y.flatten()

# Get the center coordinates
centerCoords = np.zeros((elementCount, 2), dtype=np.float32)
centerCoords[:, 0] = grid.x.v().flatten()
centerCoords[:, 1] = grid.y.v().flatten()

# Number of nodes per element. Always four corners in this case
numElementConn = np.ones(elementCount, dtype=np.int32) * 4

# Write out the data
vc = ocgis.VariableCollection()
nodeCoordsV = ocgis.Variable(name='nodeCoords', value=nodeCoords, dimensions=['nodeCount', 'coordDim'], attrs={'units': 'degrees'}, parent=vc)
elementConnV = ocgis.Variable(name='elementConn', value=elementConn, dimensions=['elementCount', 'maxNodePElement'], attrs={'long_name': 'Node indices that define the element connectivity'}, parent=vc)
numElementConnV = ocgis.Variable(name='numElementConn', value=numElementConn, dimensions=['elementCount'], attrs={'long_name': 'Number of nodes per element'}, parent=vc)
centerCoordsV = ocgis.Variable(name='centerCoords', value=centerCoords, dimensions=['elementCount', 'coordDim'], attrs={'units': 'degrees'}, parent=vc)
vc.attrs['gridType'] = 'unstructured mesh'
vc.attrs['version'] = 0.9
vc.attrs['title'] = 'Lat/lon '+str(RESOLUTION)+" degree grid - ESMFMESH format"

vc.write(OUTFILE)
