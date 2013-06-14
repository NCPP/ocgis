import ocgis
from ESMF import *
import numpy as np


ocgis.env.DIR_DATA = '/usr/local/climate_data'
ocgis.env.OVERWRITE = True


rd = ocgis.RequestDataset(uri='tas_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc',
                          variable='tas')


#ops = ocgis.OcgOperations(dataset=rd,snippet=True,prefix='overview',output_format='shp')
#ret = ops.execute()


#ops = ocgis.OcgOperations(dataset=rd,snippet=True,prefix='output',output_format='shp',
#                          select_ugid=[1],geom='state_boundaries')
#ret = ops.execute()

ops = ocgis.OcgOperations(dataset=rd,snippet=True,prefix='output',output_format='numpy',
                          select_ugid=[1],geom='state_boundaries')
ret = ops.execute()

tas = ret[1].variables['tas']
dims = tas.spatial.grid.shape
row = tas.spatial.grid.row.value
row_bounds = tas.spatial.grid.row.bounds
col = tas.spatial.grid.column.value
col_bounds = tas.spatial.grid.column.bounds

x,y = np.meshgrid(col,row)

grid = Grid(np.array(dims),num_peri_dims=0,coord_sys=CoordSys.CART,
            staggerloc=[StaggerLoc.CENTER])
x_centers = grid.get_coords(0)
y_centers = grid.get_coords(1)
#x_corners = grid.get_coords(0,staggerloc=StaggerLoc.CORNER)
#y_corners = grid.get_coords(1,staggerloc=StaggerLoc.CORNER)

def set_coords(src, target):
    for i in range(x_centers.shape[0]):
        for j in range(y_centers.shape[1]):
            target[i,j] = src[i,j]
            target[i,j] = src[i,j]
            
set_coords(x, x_centers)
set_coords(y, y_centers)

def fill_bounds(arr,target,dim=0):
    u = np.unique(arr)
    for idx in range(u.shape[0]):
        if dim == 0:
            set_coords(u[idx], target[idx,:])
        else:
            set_coords(u[idx], target[:,idx])
        
#fill_bounds(col_bounds,x_corners,dim=1)
#fill_bounds(row_bounds,y_corners,dim=0)


tgrid = Grid(np.array(dims),num_peri_dims=0,coord_sys=CoordSys.CART,
             staggerloc=[StaggerLoc.CENTER])
ix_centers = grid.get_coords(0)
iy_centers = grid.get_coords(1)
#ix_corners = grid.get_coords(0,staggerloc=StaggerLoc.CORNER)
#iy_corners = grid.get_coords(1,staggerloc=StaggerLoc.CORNER)

set_coords(x, ix_centers)
set_coords(y, iy_centers)
#set_coords(x_corners, ix_corners)
#set_coords(y_corners, iy_corners)

ix_centers = ix_centers + 5
iy_centers = iy_centers + 5
#ix_corners = ix_corners + 5
#iy_corners = iy_corners + 5

src_field = Field(grid,'source')
dst_field = Field(tgrid, 'destination')    
exact_field = Field(tgrid, 'exact')

src_field[:] = 42
exact_field[:] = 42
dst_field[:] = 0



import ipdb;ipdb.set_trace()
regrid_S2D = Regrid(src_field, dst_field)

dst_field = regrid_S2D(src_field, dst_field)

print dst_field - exact_field


