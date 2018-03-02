import logging
from math import sin, cos

from ESMF import *
from ocgis.interface.projection import WGS84

import ocgis

# short-term, useful, clearly defined, relate back to previous work

logging.basicConfig(level=logging.INFO)

# ocgis.env.DIR_DATA = '/Users/ryan.okuinghttons/netCDFfiles/climate_data'
ocgis.env.DIR_DATA = '/usr/local/climate_data'
# ocgis.env.DIR_SHPCABINET = '/Users/ryan.okuinghttons/netCDFfiles/shapefiles/ocgis_data/shp'
ocgis.env.OVERWRITE = True
ocgis.env.WRITE_TO_REFERENCE_PROJECTION = True

# rd = ocgis.RequestDataset(uri='tas_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc',
#                          variable='tas')
logging.info('creating request datasets')
crcm = ocgis.RequestDataset(uri='pr_CRCM_ccsm_1981010103.nc', variable='pr', alias='crcm')
wrfg = ocgis.RequestDataset(uri='pr_WRFG_ncep_1981010103.nc', variable='pr', alias='wrfg')

logging.info('generating destination grid')
dest_x, dest_y = np.meshgrid(np.arange(0, 360), np.arange(-90, 90))

# ops = ocgis.OcgOperations(dataset=rd,snippet=True,prefix='overview',output_format='shp')
# ret = ops.execute()


# ops = ocgis.OcgOperations(dataset=rd,snippet=True,prefix='output',output_format='shp',
#                          select_ugid=[1],geom='state_boundaries')
# ret = ops.execute()

logging.info('retrieving source grids')
ops = ocgis.OcgOperations(dataset=[crcm, wrfg], snippet=True, prefix='output',
                          output_format='numpy', geom='state_boundaries', agg_selection=True)
ret = ops.execute()

logging.info('projecting source grids')
for key in ['crcm', 'wrfg']:
    ret[1].variables[key].project(WGS84())

import ipdb;

ipdb.set_trace()

tas = ret[1].variables['tas']
dims = tas.spatial.grid.shape
row = tas.spatial.grid.row.value
row_bounds = tas.spatial.grid.row.bounds
col = tas.spatial.grid.column.value
col_bounds = tas.spatial.grid.column.bounds

x, y = np.meshgrid(col, row)

grid = Grid(np.array(dims), num_peri_dims=0, coord_sys=CoordSys.CART,
            staggerloc=[StaggerLoc.CENTER])
x_centers = grid.get_coords(0)
y_centers = grid.get_coords(1)
# x_corners = grid.get_coords(0,staggerloc=StaggerLoc.CORNER)
# y_corners = grid.get_coords(1,staggerloc=StaggerLoc.CORNER)

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

tgrid = Grid(np.array(dims), num_peri_dims=0, coord_sys=CoordSys.CART,
             staggerloc=[StaggerLoc.CENTER])
ix_centers = tgrid.get_coords(0)
iy_centers = tgrid.get_coords(1)
# ix_corners = grid.get_coords(0,staggerloc=StaggerLoc.CORNER)
# iy_corners = grid.get_coords(1,staggerloc=StaggerLoc.CORNER)

ix_centers[:] = x + 2.793
iy_centers[:] = y + 2.793

# FIELDS

src_field = Field(grid, 'source')
dst_field = Field(tgrid, 'destination')
exact_field = Field(tgrid, 'exact')

for i in range(x_centers.shape[0]):
    for j in range(y_centers.shape[1]):
        src_field[i, j] = 2 + sin(i ** 2) + cos(j) ** 2

for i in range(ix_centers.shape[0]):
    for j in range(iy_centers.shape[1]):
        exact_field[:] = 2. + sin(i ** 2) + cos(j) ** 2
        dst_field[:] = 2. + sin(i ** 2) + cos(j) ** 2

regrid_S2D = Regrid(src_field, dst_field, unmapped_action=UnmappedAction.IGNORE)

dst_field = regrid_S2D(src_field, dst_field)

# error analysis

exact = np.array(exact_field.data)
dst = np.array(dst_field.data)

# import pdb; pdb.set_trace()

check = (exact - dst) / exact
if np.any(check > .1):
    print "Relative error is greater than 10 percent: {0}". \
        format(check[check > .1])

min = np.min(check)
max = np.max(check)
rel = np.sum(check)

print "Minimum error  = {0}".format(min)
print "Maximum error  = {0}".format(max)
print "Relative error = {0}".format(rel)

'''
print('grid')
grid.dump_ESMF_coords(stagger=StaggerLoc.CENTER)
print('tgrid')
tgrid.dump_ESMF_coords(stagger=StaggerLoc.CENTER)
'''

print('src')
src_field.dump_ESMF_coords()
print('exact')
exact_field.dump_ESMF_coords()

print('dst')
dst_field.dump_ESMF_coords()
