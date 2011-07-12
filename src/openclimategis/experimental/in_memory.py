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


#NC = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data/wcrp_cmip3/pcmdi.ipcc4.bccr_bcm2_0.1pctto2x.run1.monthly.cl_A1_1.nc'
NC = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data/maurer/bccr_bcm2_0.1.sresa1b.monthly.Prcp.1950.nc'
POLYINT = Polygon(((-99,30),(-70,25),(-70,50),(-99,50)))
TIMEINT = [datetime.datetime(1950,2,1),datetime.datetime(1950,4,30)]
AGGREGATE = True

def make_poly(rtup,ctup):
    return Polygon(((ctup[0],rtup[0]),
                    (ctup[0],rtup[1]),
                    (ctup[1],rtup[1]),
                    (ctup[1],rtup[0])))
    
#def get_numpy_data(self,time_indices=[],x_indices=[],y_indices=[]):
def get_numpy_data(variable,idxtime,idxrow,idxcol):
    """ Returns multi-dimensional NumPy array extracted from a NC."""

    def _f(idx):
        return range(min(idx),max(idx)+1)

    idxtime = _f(idxtime)
    idxrow = _f(idxrow)
    idxcol = _f(idxcol)
      
    data = variable[idxtime,idxrow,idxcol]
    col_grid,row_grid = np.meshgrid(idxcol,idxrow)
    row_grid = np.flipud(row_grid)
    
    return(data,row_grid,col_grid)

def weight_select(nitrs,igrid,npd,row_grid,col_grid):
    for l in igrid:
        row_select = row_grid == l['row']
        col_select = col_grid == l['col']
        select = row_select & col_select
        for ii in xrange(len(nitrs)):
            if len(idxtime) == 1:
                dd = npd[:,:]
            else:
                dd = npd[ii,:,:]
            v = float(dd[select])
            if AGGREGATE:
                v = v*l['weight']
            l['time'].update({timevec[ii]:v})
    return(igrid)
    
#class OcgPolygon(Polygon):
#    
#    def __init__(self,coords,row=None,col=None):
#        self.orow = row
#        self.ocol = col
#        super(OcgPolygon,self).__init__(coords)
    

d = Dataset(NC,'r')
v = d.variables['Prcp']
timevec = num2date(d.variables['time'][:],'days since 1950-01-01 00:00:00','proleptic_gregorian')

print('retrieving data...')
row = d.variables['latitude'][:]
col = d.variables['longitude'][:]
row_bnds = d.variables['bounds_latitude'][:]
col_bnds = d.variables['bounds_longitude'][:]

#Docg = namedtuple('Docg',['weight','row','col','geom'])

#class Docg(object):
#    
#    def __init__(self,weight=None,row=None,col=None,geom=None):
#        self.weight = weight
#        self.row = row
#        self.col = col
#        self.geom = geom

#grid_pt = MultiPoint([(c,r) for r,c in itertools.product(row,col)])
print('spatial representation...')
grid = []
for ii in xrange(len(row_bnds)):
    for jj in xrange(len(col_bnds)):
#        grid.append(Docg(weight=None,row=ii,col=jj,geom=make_poly(row_bnds[ii],col_bnds[jj])))
        grid.append(dict(time={},weight=None,row=ii,col=jj,geom=make_poly(row_bnds[ii],col_bnds[jj])))
#grid_poly = MultiPolygon([make_poly(r,c) for r,c in itertools.product(row_bnds,col_bnds)])

#igrid = [p for p in grid if p['geom'].intersects(POLYINT)]

print('intersection...')
igrid = []
for l in grid: 
#    if l.geom.intersects(POLYINT):
#        prearea = l.geom.area
#        l.geom = l.geom.intersection(POLYINT)
##        import ipdb;ipdb.set_trace()
##        l.update(dict(weight=l['geom'].area/prearea))
#        l.weight = l.geom.area/prearea
#        igrid.append(l)
    if l['geom'].intersects(POLYINT):
        prearea = l['geom'].area
        l['geom'] = l['geom'].intersection(POLYINT)
#        import ipdb;ipdb.set_trace()
#        l.update(dict(weight=l['geom'].area/prearea))
        w = l['geom'].area/prearea
        if w > 0:
            l['weight'] = w
            igrid.append(l)

print('getting numpy data...')
idxtime = np.arange(0,len(timevec))[(timevec>=TIMEINT[0])*(timevec<=TIMEINT[1])]
idxrow = [r['row'] for r in grid]
idxcol = [r['col'] for r in grid]
npd,row_grid,col_grid = get_numpy_data(v,idxtime,idxrow,idxcol)

## apply the weight dictionary list. this serves the dual purpose of 
## removing unneeded values included in the block netCDF query.
print('getting actual data...')

#ctr = 1
#for l in igrid:
#    print('  {0} of {1} igrid...'.format(ctr,len(igrid)))
#    ctr += 1
#    row_select = row_grid == l['row']
#    col_select = col_grid == l['col']
#    select = row_select & col_select
#    for ii in xrange(len(idxtime)):
#        if len(idxtime) == 1:
#            dd = npd[:,:]
#        else:
#            dd = npd[ii,:,:]
#        v = float(dd[select])
#        if AGGREGATE:
#            v = v*l['weight']
#        l['time'].update({timevec[ii]:v})
        


jobs = []
job_server = pp.Server(3)
#for rid in rids:
job = job_server.submit(weight_select,(3,igrid,npd,row_grid,col_grid),(),("numpy",))
jobs.append(job)
for job in jobs:
    print job()
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