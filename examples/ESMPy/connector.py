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

x_centers[:] = x
y_centers[:] = y

'''
def fill_bounds(arr,target,dim=0):
    u = np.unique(arr)
    for idx in range(u.shape[0]):
        if dim == 0:
            target[idx,:] = u[idx]
        else:
            target[:,idx] = u[idx]
        
fill_bounds(col_bounds,x_corners,dim=1)
fill_bounds(row_bounds,y_corners,dim=0)
'''
grid.dump_ESMF_coords(stagger=StaggerLoc.CENTER)

tgrid = Grid(np.array(dims),num_peri_dims=0,coord_sys=CoordSys.CART,
             staggerloc=[StaggerLoc.CENTER])
ix_centers = grid.get_coords(0)
iy_centers = grid.get_coords(1)
#ix_corners = grid.get_coords(0,staggerloc=StaggerLoc.CORNER)
#iy_corners = grid.get_coords(1,staggerloc=StaggerLoc.CORNER)

ix_centers[:] = x + 5
iy_centers[:] = y + 5

# FIELDS

src_field = Field(grid,'source')
dst_field = Field(tgrid, 'destination')    
exact_field = Field(tgrid, 'exact')

src_field[:] = 42.
exact_field[:] = 42.
dst_field[:] = 7.

print('src')
src_field.dump_ESMF_coords()
print('exact')
exact_field.dump_ESMF_coords()

regrid_S2D = Regrid(src_field, dst_field, unmapped_action=UnmappedAction.IGNORE)

dst_field = regrid_S2D(src_field, dst_field)

print('dst')
dst_field.dump_ESMF_coords()


