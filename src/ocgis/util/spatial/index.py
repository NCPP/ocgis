import numpy as np
from shapely.geometry.multipolygon import MultiPolygon
from ocgis.util.helpers import make_poly

    
def shapely_grid(dim,rtup,ctup,target=None):

    if dim is None:
        ## construct an average of 10 polygons
        row_dim = np.ceil(np.abs(rtup[0]-rtup[1])/5.0)
        col_dim = np.ceil(np.abs(ctup[0]-ctup[1])/5.0)
        dim = np.mean([row_dim,col_dim])
        
    row_bounds = np.arange(rtup[0],rtup[1]+dim,dim)
    min_row = row_bounds[0:-1]
    max_row = row_bounds[1:]
    row_bounds = np.hstack((min_row.reshape(-1,1),max_row.reshape(-1,1)))
    col_bounds = np.arange(ctup[0],ctup[1]+dim,dim)
    min_col = col_bounds[0:-1]
    max_col = col_bounds[1:]
    col_bounds = np.hstack((min_col.reshape(-1,1),max_col.reshape(-1,1)))
    polygons = []
    for ii in range(row_bounds.shape[0]):
        rtup = (row_bounds[ii,0],row_bounds[ii,1])
        for jj in range(col_bounds.shape[0]):
            ctup = (col_bounds[jj,0],col_bounds[jj,1])
            polygon = make_poly(rtup,ctup)
            if target is not None and keep(target,polygon):
                polygons.append(polygon)
            elif target is None:
                polygons.append(polygon)
    
    return(MultiPolygon(polygons))


def build_index_grid(dim,target):
    bounds = target.bounds
    rtup = (bounds[1],bounds[3])
    ctup = (bounds[0],bounds[2])
    dim = dim if dim is None else float(dim)
    grid = shapely_grid(dim,rtup,ctup,target=target)
    return(grid)

def build_index(target,grid):
    tree = {}
    for ii,polygon in enumerate(grid):
        if keep(target,polygon):
            tree.update({ii:{'box':polygon,'geom':target}})
    return(tree)

def keep(target,selection):
    if selection.intersects(target) and not selection.touches(target):
        ret = True
    else:
        ret = False
    return(ret)

def index_intersects(target,index):
    ret = False
    for value in index.itervalues():
        if target.intersects(value['box']):
            if keep(target,value['geom']):
                ret = True
                break
    return(ret)

################################################################################

#sc = ShpCabinet()

#geom = sc.get_geom_dict('state_boundaries',{'id':[16]})[0]['geom']
#geom = sc.get_geom_dict('world_countries')
#geom = union_geom_dicts(geom)[0]['geom']
#
##target = Point(-99.77,41.22)
#target = make_poly((40,41),(-99,-98))
#
#dims = np.arange(10,100,10)
#build_times = []
#int_times = []
#for dim in dims.flat:
#    print(dim)
#    
#    t1 = time.time()
#    grid = build_index_grid(dim,geom)
#    index = build_index(geom,grid)
#    t2 = time.time()
#    build_times.append(t2-t1)
#    
#    t1 = time.time()
#    index_intersects(target,index)
#    t2 = time.time()
#    int_times.append(t2-t1)
#
#plt.figure(1)
#plt.subplot(211)
#plt.plot(dims,build_times)
#plt.title('build times')
#
#plt.subplot(212)
#plt.plot(dims,int_times)
#plt.title('intersects times')
#
#plt.show()

#print index_intersects(pt,index)

#import ipdb;ipdb.set_trace()

################################################################################

#rtup = (40.0,50.0)
#ctup = (-120.0,-100.0)
#dim = 5.0
#
#tag = str(datetime.now())
#target = make_poly(rtup,ctup)
#grid = shapely_grid(dim,rtup,ctup)
#index = build_index(target,grid)
#test_in = make_poly((42.17,43.31),(-118.72,-117.52))
#test_out = make_poly((41.10,42.81),(-125.56,-123.58))
#
#print('in')
#print(index_intersects(test_in,index))
#print('out')
#print(index_intersects(test_out,index))
#
#import ipdb;ipdb.set_trace()
#
#shapely_to_shp(target,'target_'+tag)
#shapely_to_shp(grid,'grid_'+tag)
#shapely_to_shp(test_in,'test_in_'+tag)
#shapely_to_shp(test_out,'test_out_'+tag)

#sc = ShpCabinet()
#geom_dict = sc.get_geom_dict('state_boundaries',{'ugid':[25]})
#geom = geom_dict[0]['geom']
#bounds = geom.bounds
#rtup = (bounds[1],bounds[3])
#ctup = (bounds[0],bounds[2])
#grid = shapely_grid(1.0,rtup,ctup,target=geom)
#index = build_index(geom,grid)
#
#import ipdb;ipdb.set_trace()
#
#shapely_to_shp(geom,'geom_'+tag)
#shapely_to_shp(grid,'grid_'+tag)