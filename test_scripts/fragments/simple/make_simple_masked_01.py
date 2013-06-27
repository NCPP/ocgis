'''
script to generate test nc and other data files.
'''

from shapely.geometry.point import Point
import datetime
import numpy as np
import os.path
import netCDF4 as nc


SEED = 1 #: random number seeding for missing data
OUTDIR = os.path.split(__file__)[0] #: location to write
OUTNAME = 'test_simple_spatial_masked_01.nc' #: name of the file to write
RES = 1 #: resolution of the grid
ORIGIN = Point(-105,40) #: center coordinate of upper left cell
DIM = [4,4] #: number of cells [dimx,dimy]
## the scalar value to fill the cells. set to None for manual values. will use
## VAL_MAN variable instead.
VAL = None
VAR = ['foo','foo2'] #: name of the data variable
## any relevant variables for the time construction
TIME = {'origin':datetime.datetime(2000,1,1,0,0,0),
        'end':datetime.datetime(2001,12,31,0,0,0),
        'calendar':'proleptic_gregorian',
        'units':'days since 2000-01-01 00:00:00',
        'name':'time'}
## any relevant variables for the spatial construction
SPACE = {'row_bnds':'bounds_latitude',
         'col_bnds':'bounds_longitude'}
## any relevant variables for the level construction
LEVEL = {'name':'level',
         'n':2}
## variables for masked data
MASK = {'n':4,
        'value':float(1e20)}

## GENERATE BASE ARRAYS ########################################################

## make time vector
delta = datetime.timedelta(1)
start = TIME['origin']
timevec = []
while start <= TIME['end']:
    timevec.append(start)
    start += delta
timevec = np.array(timevec)
## make the level vector
levelvec = np.arange(1,LEVEL['n']+1)*100
    
## make centroids
col_coords = -np.arange(abs(ORIGIN.x)-RES*(DIM[0]-1),abs(ORIGIN.x)+RES,RES)
col_coords = col_coords[::-1]
row_coords = np.arange(ORIGIN.y-RES*(DIM[1]-1),ORIGIN.y+RES,RES)
row_coords = row_coords[::-1]
col,row = np.meshgrid(col_coords,row_coords)

## using points from |coords| adjust by |res| to provide bounds
def make_bounds(coords,res):
    bnds = []
    for g in coords.flat:
        bnds.append([g-res*0.5,g+res*0.5])
    return(np.array(bnds))

## create bounds arrays
col_bnds = make_bounds(col_coords,RES)
row_bnds = make_bounds(row_coords,RES)

## MANUAL VALUE SETTING ########################################################

d1l1 = np.array([[1,1,2,2],
                 [1,1,2,2],
                 [3,3,4,4],
                 [3,3,4,4]])
VAL_MAN = np.ones((len(timevec),len(levelvec),d1l1.shape[0],d1l1.shape[1]))
VAL_MAN[:,:,:,:] = d1l1

################################################################################

## make value array
if VAL is not None:
    val = np.ones((len(timevec),LEVEL['n'],DIM[0],DIM[1]),dtype=float)*VAL
else:
    val = VAL_MAN
## set up the mask if requested
mask = np.zeros(val.shape,dtype=bool)
np.random.seed(SEED)
# set the random masked cells to None
rxidx = np.random.permutation(DIM[0])
ryidx = np.random.permutation(DIM[1])
for ii in range(0,MASK['n']):
    rx = rxidx[ii]
    ry = ryidx[ii]
    mask[:,:,rx,ry] = True
val = np.ma.array(val,dtype=float,mask=mask,fill_value=MASK['value'])

## WRITE THE NC FILE ###########################################################

## initialize the output file
rootgrp = nc.Dataset(os.path.join(OUTDIR,OUTNAME),'w',format='NETCDF4')
## create the dimensions
level = rootgrp.createDimension(LEVEL['name'],size=LEVEL['n'])
time = rootgrp.createDimension(TIME['name'],size=len(timevec))
lat = rootgrp.createDimension('lat',size=len(row_coords))
lon = rootgrp.createDimension('lon',size=len(col_coords))
bound = rootgrp.createDimension('bound',size=2)
## create the variables
times = rootgrp.createVariable(TIME['name'],'f8',('time',))
levels = rootgrp.createVariable(LEVEL['name'],'i4',('level',))
cols = rootgrp.createVariable('longitude','f8',('lon',))
rows = rootgrp.createVariable('latitude','f8',('lat',))
bounds_col = rootgrp.createVariable(SPACE['col_bnds'],'f8',('lon','bound'))
bounds_row = rootgrp.createVariable(SPACE['row_bnds'],'f8',('lat','bound'))
for var in VAR:
    value = rootgrp.createVariable(var,'f8',('time','level','lat','lon'))
    value[:,:,:,:] = val
    value.missing_value = MASK['value']
## fill variables
times.units = TIME['units']
times.calendar = TIME['calendar']
times[:] = nc.date2num(timevec,units=times.units,calendar=times.calendar)
levels[:] = levelvec
cols[:] = col_coords
rows[:] = row_coords
bounds_col[:,:] = col_bnds
bounds_row[:,:] = row_bnds

rootgrp.close()