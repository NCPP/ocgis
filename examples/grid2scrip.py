import numpy as np

import ocgis
from ocgis.test import create_gridxy_global
from ocgis.util.helpers import get_esmf_corners_from_ocgis_corners

# Spatial resolution of the output grid in degrees.
RES = 4
RESOLUTION = 1/2**RES
# Path to the output netCDF file.
OUTFILE = "ll1x2e"+str(RES)+'deg.scrip.nc'

# Create the grid object
grid = create_gridxy_global(resolution=RESOLUTION, with_bounds=True, wrapped=False)
# Grid should not be vectorized
grid.expand()

# Should I be using ESMF corners?
# # Convert ocgis corners to esmf corners
# esmf_corners_x = get_esmf_corners_from_ocgis_corners(grid.x.bounds.v())
# esmf_corners_y = get_esmf_corners_from_ocgis_corners(grid.y.bounds.v())
# # Create one-based global index
# indexing = np.arange(1, esmf_corners_x.size + 1, dtype=np.int32).reshape(esmf_corners_x.shape)


# Write out the data
grid_size = grid.archetype.size
lat, lon = grid.x.shape
grid_corners = grid.x.bounds.shape[2]

vc = ocgis.VariableCollection()
grid_dims = ocgis.Variable(name='grid_dims', value=grid.x.shape, dimensions=['grid_rank'], parent=vc)
latitude = ocgis.Variable(name='grid_center_lat', value=grid.y.v().flatten(), dimensions=['grid_size'], attrs={'units': 'degrees'}, parent=vc)
longitude = ocgis.Variable(name='grid_center_lon', value=grid.x.v().flatten(), dimensions=['grid_size'], attrs={'units': 'degrees'}, parent=vc)

gxbf = np.zeros([grid_size, grid_corners])
for i in range(lat):
    gxbf[i*lon:(i+1)*lon,:] = grid.x.bounds.v()[i,:,:]

gybf = np.zeros([grid_size, grid_corners])
for i in range(lat):
    gybf[i*lon:(i+1)*lon,:] = grid.y.bounds.v()[i,:,:]

latitude_bounds = ocgis.Variable(name='grid_corner_lat', value=gybf, dimensions=['grid_size', 'grid_corners'], attrs={'units': 'degrees'}, parent=vc)
longitude_bounds = ocgis.Variable(name='grid_corner_lon', value=gxbf, dimensions=['grid_size', 'grid_corners'], attrs={'units': 'degrees'}, parent=vc)
vc.attrs['title'] = 'Lat/lon '+str(RESOLUTION)+" degree grid - SCRIP format"

vc.write(OUTFILE)
