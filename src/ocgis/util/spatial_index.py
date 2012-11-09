import numpy as np
from shapely.geometry.multipolygon import MultiPolygon
from shapely import prepared
from ocgis.util.helpers import make_poly

    
def shapely_grid(dim,rtup,ctup,target=None):
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
    return(MultiPolygon(polygons))


def build_index_grid(dim,target):
    bounds = target.bounds
    rtup = (bounds[1],bounds[3])
    ctup = (bounds[0],bounds[2])
    grid = shapely_grid(float(dim),rtup,ctup,target=target)
    return(grid)

def build_index(target,grid):
    tree = {}
    for ii,polygon in enumerate(grid):
        if keep(target,polygon):
            tree.update({ii:{'box':polygon,'geom':target.intersection(polygon)}})
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
        if keep(target,value['box']):
            if keep(target,value['geom']):
                ret = True
                break
    return(ret)
            
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