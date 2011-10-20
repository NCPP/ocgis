import os
from netCDF4 import Dataset
import itertools
from shapely.geometry.multipoint import MultiPoint
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
import copy
import datetime
from netcdftime.netcdftime import num2date
from collections import namedtuple
import pp
from shapely import iterops, wkt
import geojson
from numpy.ma.core import MaskedConstant
import subprocess

from shapely.ops import cascaded_union


#NC = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data/wcrp_cmip3/pcmdi.ipcc4.bccr_bcm2_0.1pctto2x.run1.monthly.cl_A1_1.nc'
NC = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data/maurer/bccr_bcm2_0.1.sresa1b.monthly.Prcp.1950.nc'
## all
#POLYINT = Polygon(((-99,39),(-94,38),(-94,40),(-100,39)))
## great lakes
POLYINT = [Polygon(((-90.35,40.55),(-80.80,40.55),(-80.80,49.87),(-90.35,49.87)))]
## two areas
#POLYINT = [wkt.loads('POLYGON ((-85.324076923076916 44.028020242914977,-84.280765182186229 44.16008502024291,-84.003429149797569 43.301663967611333,-83.607234817813762 42.91867611336032,-84.227939271255053 42.060255060728736,-84.941089068825903 41.307485829959511,-85.931574898785414 41.624441295546553,-85.588206477732783 43.011121457489871,-85.324076923076916 44.028020242914977))'),
#           wkt.loads('POLYGON ((-89.24640080971659 46.061817813765174,-88.942651821862341 46.378773279352224,-88.454012145748976 46.431599190283393,-87.952165991902831 46.11464372469635,-88.163469635627521 45.190190283400803,-88.889825910931165 44.503453441295541,-88.770967611336033 43.552587044534405,-88.942651821862341 42.786611336032379,-89.774659919028338 42.760198380566798,-90.038789473684204 43.777097165991897,-89.735040485829956 45.097744939271251,-89.24640080971659 46.061817813765174))')]
TIMEINT = [datetime.datetime(1950,2,1),datetime.datetime(1950,4,30)]
AGGREGATE = True
CLIP = True
VAR = 'Prcp'

def make_poly(rtup,ctup):
    return Polygon(((ctup[0],rtup[0]),
                    (ctup[0],rtup[1]),
                    (ctup[1],rtup[1]),
                    (ctup[1],rtup[0])))

def is_masked(arg):
    if isinstance(arg,MaskedConstant):
        return None
    else:
        return arg
    
def itr_id(start=1):
    while True:
        try:
            yield start
        finally:
            start += 1

#def get_numpy_data(self,time_indices=[],x_indices=[],y_indices=[]):
def get_numpy_data(variable,idxtime,idxrow,idxcol):
    """ Returns multi-dimensional NumPy array extracted from a NC."""

#    def _f(idx):
#        return range(min(idx),max(idx)+1)
#
#    idxtime = _f(idxtime)
#    idxrow = _f(idxrow)
#    idxcol = _f(idxcol)
      
    data = variable[idxtime,idxrow,idxcol]
    col_grid,row_grid = np.meshgrid(idxcol,idxrow)
#    row_grid = np.flipud(row_grid)
    
    return(data,row_grid,col_grid)

#def itr_mask(mask):
#    ix = xrange(mask.shape[0])
#    jx = xrange(mask.shape[1])
#    for ii,jj in itertools.product(ix,jx):
#        if mask[ii,jj]:
#            yield ii,jj
            
def itr_array(a):
    ix = a.shape[0]
    jx = a.shape[1]
    for ii,jj in itertools.product(xrange(ix),xrange(jx)):
        yield ii,jj

#def weight_select(nitrs,igrid,npd,row_grid,col_grid):
#    for l in igrid:
#        row_select = row_grid == l['row']
#        col_select = col_grid == l['col']
#        select = row_select & col_select
#        for ii in xrange(len(nitrs)):
#            if len(idxtime) == 1:
#                dd = npd[:,:]
#            else:
#                dd = npd[ii,:,:]
#            v = float(dd[select])
#            if AGGREGATE:
#                v = v*l['weight']
#            l['time'].update({timevec[ii]:v})
#    return(igrid)
    
#class OcgPolygon(Polygon):
#    
#    def __init__(self,coords,row=None,col=None):
#        self.orow = row
#        self.ocol = col
#        super(OcgPolygon,self).__init__(coords)

## return the rectangular boundary of the polygon    
#polyenv = Polygon(list(POLYINT.envelope.exterior.coords))
## get the bounding coords
#min_col,min_row,max_col,max_row = polyenv.bounds

def dump(polygon):
    ds = Dataset(NC,'r')
    v = ds.variables[VAR]
    timevec = num2date(ds.variables['time'][:],'days since 1950-01-01 00:00:00','proleptic_gregorian')
    
    print('retrieving data...')
    row = ds.variables['latitude'][:]
    col = ds.variables['longitude'][:]
    row_bnds = ds.variables['bounds_latitude'][:]
    col_bnds = ds.variables['bounds_longitude'][:]
    
    #x,y = np.meshgrid(col,row)
    #plt.plot(x,y)
    #plt.show()
    #tdk
    
    print('making arrays...')
    min_col,min_row = np.meshgrid(col_bnds[:,0],row_bnds[:,0])
    max_col,max_row = np.meshgrid(col_bnds[:,1],row_bnds[:,1])
    real_col,real_row = np.meshgrid(np.arange(0,len(col)),np.arange(0,len(row)))
    
    ## make the meshgrid bounded by the envelope
    #idx1 = row_bnds[:,0] >= min_row
    #idx2 = row_bnds[:,1] <= max_row
    #srow = row[idx1 & idx2]
    #
    #idx1 = col_bnds[:,0] >= min_col
    #idx2 = col_bnds[:,1] <= max_col
    #scol = col[idx1 & idx2]
    #
    #gcol,grow = np.meshgrid(scol,srow)
    
    #Docg = namedtuple('Docg',['weight','row','col','geom'])
    
    #class Docg(object):
    #    
    #    def __init__(self,weight=None,row=None,col=None,geom=None):
    #        self.weight = weight
    #        self.row = row
    #        self.col = col
    #        self.geom = geom
    
    #grid_pt = MultiPoint([(c,r) for r,c in itertools.product(row,col)])
    #print('spatial representation...')
    #grid = MultiPolygon([make_poly(ii,jj) for ii,jj in itertools.product(row_bnds,col_bnds)])
    #grid_pt = MultiPoint([(c,r) for r,c in itertools.product(row,col)])
    #print('  intersects operation...')
    #igrid = MultiPolygon(list(iterops.intersects(POLYINT,grid,True)))
    #igrid_pt = MultiPoint(list(iterops.intersects(polyenv,grid_pt,True)))
    #geoms = np.empty(min_row.shape,dtype=object)
    #for ii,jj in itr_array(min_row):
    #    geoms[ii,jj] = make_poly((min_row[ii,jj],max_row[ii,jj]),(min_col[ii,jj],max_col[ii,jj]))
    
    print('overlay...')
    igrid = np.empty(min_row.shape,dtype=object)
    weights = np.empty(min_row.shape)
    for ii,jj in itr_array(min_row):
        g = make_poly((min_row[ii,jj],max_row[ii,jj]),(min_col[ii,jj],max_col[ii,jj]))
        if g.intersects(polygon):
            prearea = g.area
            if CLIP:
                ng = g.intersection(polygon)
            else:
                ng = g
            w = ng.area/prearea
            if w > 0:
                igrid[ii,jj] = ng
                weights[ii,jj] = w
    mask = weights > 0
    weights = weights/weights.sum()
    #for ii,jj in itertools.product(xrange(len(row_bnds)),xrange(len(col_bnds))):
    #    for jj in xrange(len(col_bnds)):
    #        grid.append(Docg(weight=None,row=ii,col=jj,geom=make_poly(row_bnds[ii],col_bnds[jj])))
    #    grid.append(dict(time={},weight=None,row=ii,col=jj,geom=make_poly(row_bnds[ii],col_bnds[jj])))
    #grid_poly = MultiPolygon([make_poly(r,c) for r,c in itertools.product(row_bnds,col_bnds)])
    
    #igrid = [p for p in grid if p['geom'].intersects(POLYINT)]
    #tdk
    #print('intersection...')
    #igrid = []
    #for l in grid: 
    #    if l.geom.intersects(POLYINT):
    #        prearea = l.geom.area
    #        l.geom = l.geom.intersection(POLYINT)
    ##        import ipdb;ipdb.set_trace()
    ##        l.update(dict(weight=l['geom'].area/prearea))
    #        l.weight = l.geom.area/prearea
    ##        igrid.append(l)
    #    if l['geom'].intersects(POLYINT):
    #        prearea = l['geom'].area
    #        l['geom'] = l['geom'].intersection(POLYINT)
    ##        import ipdb;ipdb.set_trace()
    ##        l.update(dict(weight=l['geom'].area/prearea))
    #        w = l['geom'].area/prearea
    #        if w > 0:
    #            l['weight'] = w
    #            igrid.append(l)
    
    print('getting numpy data...')
    idxtime = np.arange(0,len(timevec))[(timevec>=TIMEINT[0])*(timevec<=TIMEINT[1])]
    def u(arg):
        un = np.unique(arg)
        return(np.arange(un.min(),un.max()+1))
    idxrow = u(real_row[mask])
    idxcol = u(real_col[mask])
    
    def sub(idxrow,idxcol,arg):
        return arg[idxrow.min():idxrow.max()+1,idxcol.min():idxcol.max()+1]
    mask = sub(idxrow,idxcol,mask)
    weights = sub(idxrow,idxcol,weights)
    igrid = sub(idxrow,idxcol,igrid)
    
    #idxrow = np.unique(real_row[mask])
    #idxcol = np.unique(real_col[mask])
    #for ii,jj in itr_mask(mask):
    #    idxrow.append(ii)
    #    idxcol.append(jj)
    npd,row_grid,col_grid = get_numpy_data(v,idxtime,idxrow,idxcol)
    
    ### make weights array
    #weights = np.empty((row_grid.shape))
    #geoms = np.empty((row_grid.shape),dtype=object)
    #for l in igrid:
    #    weights[l['row'],l['col']] = l['weight']
    #    geoms[l['row'],l['col']] = l['geom']
    ### make mask
    #mask = weights > 0
    
    ## apply the mask
    #mnpd = npd*mask
    
    print('extracting data...')
    features = []
    ids = itr_id()
    if AGGREGATE:
        ## provide the unioned geometry
        geoms = igrid[mask]
        unioned = geoms[0]
        for ii,geom in enumerate(geoms):
            if ii == 0: continue
            unioned = unioned.union(geom)
        ## weight the data by area
        weighted = npd*weights
    #    tdata = dict(zip([timevec[it] for it in idxtime],[weighted[ii,:,:].sum() for ii in range(weighted.shape[0])]))
        for kk in range(len(idxtime)):
            if kk == 0:
                print('unioning geometry...')
                ## need to remove geometries that have masked data
                lyr = weighted[kk,:,:]
                geoms = igrid[mask*np.invert(lyr.mask)]
                unioned = cascaded_union([p for p in geoms])
#                unioned = geoms[0]
#                for ii,geom in enumerate(geoms):
#                    if ii == 0: continue
#                    unioned = unioned.union(geom)
            ## generate the feature
            feature = geojson.Feature(id=ids.next(),
                                      geometry=unioned,
                                      properties=dict({VAR:float(weighted[kk,:,:].sum()),
                                                       'timestamp':str(timevec[idxtime[kk]])}))
            features.append(feature)
    else:
        for ii,jj in itr_array(row_grid):
            if mask[ii,jj] == True:
                data = npd[:,ii,jj]
                data = [is_masked(da) for da in data]
    #            tdata = dict(zip([timevec[it] for it in idxtime],data))
    #            geom = igrid[ii,jj]
                for kk in range(len(data)):
                    if data[kk] == None: continue
                    feature = geojson.Feature(id=ids.next(),
                                              geometry=igrid[ii,jj],
                                              properties=dict({VAR:float(data[kk]),
                                                               'timestamp':str(timevec[idxtime[kk]])}))
                    features.append(feature)
    return(features)


features = []
for polygon in POLYINT:
    features += dump(polygon)

print('dumping...')
fc = geojson.FeatureCollection(features)
with open('/tmp/out.geojson','w') as f:
    f.write(geojson.dumps(fc))

args = ['ogr2ogr','-overwrite','-f','ESRI Shapefile', '/tmp/out.shp','/tmp/out.geojson','OGRGeoJSON']
subprocess.call(args)

        
        

## apply the weight dictionary list. this serves the dual purpose of 
## removing unneeded values included in the block netCDF query.
#print('getting actual data...')
#ctr = 1
#for l in igrid:
#    print('  {0} of {1} igrid...'.format(ctr,len(igrid)))
#    ctr += 1
##    row_select = row_grid == l['row']
##    col_select = col_grid == l['col']
##    select = row_select & col_select
#    for ii in xrange(len(idxtime)):
#        if len(idxtime) == 1:
#            dd = npd[:,:]
#        else:
#            dd = npd[ii,:,:]
#        v = float(dd[select])
#        if AGGREGATE:
#            v = v*l['weight']
#        l['time'].update({timevec[ii]:v})
        


#jobs = []
#job_server = pp.Server(3)
##for rid in rids:
#job = job_server.submit(weight_select,(3,igrid,npd,row_grid,col_grid),(),("numpy",))
#jobs.append(job)
#for job in jobs:
#    print job()
#    log.info(job())
#log.info('success.')
    

#igrid_pt = MultiPoint([p for p in grid_pt if p.intersects(POLYINT)])
#igrid_poly = MultiPolygon([p for p in grid_poly if p.intersects(POLYINT)])
#
#itgrid_poly = MultiPolygon([p.intersection(POLYINT) for p in igrid_poly])

#poly = Polygon(((col_bnds[ii,0],row_bnds[jj,0]),
#                               (col_bnds[ii,0],row_bnds[jj,1]),
#                               (col_bnds[ii,1],row_bnds[jj,1]),
#                               (col_bnds[ii,1],row_bnds[jj,0])))

#gnp = np.array(igrid_pt)
#plt.scatter(gnp[:,0],gnp[:,1])
#plt.show()