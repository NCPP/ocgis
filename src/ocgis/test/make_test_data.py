from shapely.geometry.point import Point
import datetime
import numpy as np
import os.path
import netCDF4 as nc
from ocgis.util.helpers import iter_array


## using points from |coords| adjust by |res| to provide bounds
def make_bounds(coords,res):
    bnds = []
    for g in coords.flat:
        bnds.append([g-res*0.5,g+res*0.5])
    return(np.array(bnds))

def make_simple():
    SEED = 1 #: random number seeding for missing data
    OUTDIR = '/tmp' #: location to write
    OUTNAME = 'test_simple_spatial_01.nc' #: name of the file to write
    RES = 1 #: resolution of the grid
    ORIGIN = Point(-105,40) #: center coordinate of upper left cell
    DIM = [4,4] #: number of cells [dimx,dimy]
    VAR = 'foo' #: name of the data variable
    ## any relevant variables for the time construction
    TIME = {'origin':datetime.datetime(2000,3,1,12,0,0),
            'end':datetime.datetime(2000,4,30,12,0,0),
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
    MASK = {'n':0,
            'value':float(1e20)}
    
    ## MANUAL VALUE SETTING ########################################################
    
    d1l1 = [[1,1,2,2],
            [1,1,2,2],
            [3,3,4,4],
            [3,3,4,4]]
    VAL = np.array(d1l1).reshape(1,1,4,4)
    VAL_MAN = None
    
    ################################################################################
    
    ## GENERATE BASE ARRAYS ##
    
    ## make time vector
    delta = datetime.timedelta(1)
    start = TIME['origin']
    timevec = []
    while start <= TIME['end']:
        timevec.append(start)
        start += delta
    timevec = np.array(timevec)
    ## make vector time bounds
    timevec_bnds = np.empty((len(timevec),2),dtype=object)
    delta = datetime.timedelta(hours=12)
    for idx,tv in iter_array(timevec,return_value=True):
        timevec_bnds[idx,0] = tv - delta
        timevec_bnds[idx,1] = tv + delta
    
    ## make the level vector
    levelvec = np.array([50,150])
    levelvec_bounds = np.array([[0,100],[100,200]])

    ## make centroids
    col_coords = -np.arange(abs(ORIGIN.x)-RES*(DIM[0]-1),abs(ORIGIN.x)+RES,RES)
    col_coords = col_coords[::-1]
    row_coords = np.arange(ORIGIN.y-RES*(DIM[1]-1),ORIGIN.y+RES,RES)
    #row_coords = row_coords[::-1]
    #col,row = np.meshgrid(col_coords,row_coords)
    
    ## create bounds arrays
    col_bnds = make_bounds(col_coords,RES)
    row_bnds = make_bounds(row_coords,RES)

    ## make value array
    if VAL is not None:
        val = np.ones((len(timevec),LEVEL['n'],DIM[0],DIM[1]),dtype=float)*VAL
    else:
        val = VAL_MAN
    ## set up the mask if requested
    mask = np.zeros(val.shape,dtype=bool)
    np.random.seed(SEED)
    # set the random masked cells to None
    for ii in range(0,MASK['n']):
        rx = np.random.randint(0,DIM[0])
        ry = np.random.randint(0,DIM[1])
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
    bounds_times = rootgrp.createVariable('time_bnds','f8',('time','bound'))
    levels = rootgrp.createVariable(LEVEL['name'],'i4',('level',))
    bounds_levels = rootgrp.createVariable('level_bnds','i4',('level','bound'))
    cols = rootgrp.createVariable('longitude','f8',('lon',))
    rows = rootgrp.createVariable('latitude','f8',('lat',))
    bounds_col = rootgrp.createVariable(SPACE['col_bnds'],'f8',('lon','bound'))
    bounds_row = rootgrp.createVariable(SPACE['row_bnds'],'f8',('lat','bound'))
    value = rootgrp.createVariable(VAR,'f8',('time','level','lat','lon'))
    ## fill variables
    times.units = TIME['units']
    times.calendar = TIME['calendar']
    times[:] = nc.date2num(timevec,units=times.units,calendar=times.calendar)
    bounds_times[:] = nc.date2num(timevec_bnds,units=times.units,calendar=times.calendar)
    levels[:] = levelvec
    bounds_levels[:] = levelvec_bounds
    cols[:] = col_coords
    rows[:] = row_coords
    bounds_col[:,:] = col_bnds
    bounds_row[:,:] = row_bnds
    value[:,:,:,:] = val
    value.missing_value = MASK['value']
    
    rootgrp.close()
    
def make_simple_mask():
    SEED = 1 #: random number seeding for missing data
    OUTDIR = '/tmp' #: location to write
    OUTNAME = 'test_simple_mask_spatial_01.nc' #: name of the file to write
    RES = 1 #: resolution of the grid
    ORIGIN = Point(-105,40) #: center coordinate of upper left cell
    DIM = [4,4] #: number of cells [dimx,dimy]
    VAR = 'foo' #: name of the data variable
    ## any relevant variables for the time construction
    TIME = {'origin':datetime.datetime(2000,3,1,0,0,0),
            'end':datetime.datetime(2000,4,30,0,0,0),
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
    MASK = {'n':5,
            'value':float(1e20)}
    
    ## MANUAL VALUE SETTING ########################################################
    
    d1l1 = [[1,1,2,2],
            [1,1,2,2],
            [3,3,4,4],
            [3,3,4,4]]
    VAL = np.array(d1l1).reshape(1,1,4,4)
    VAL_MAN = None
    
    ################################################################################
    
    ## GENERATE BASE ARRAYS ##
    
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
    #row_coords = row_coords[::-1]
    #col,row = np.meshgrid(col_coords,row_coords)
    
    ## create bounds arrays
    col_bnds = make_bounds(col_coords,RES)
    row_bnds = make_bounds(row_coords,RES)

    ## make value array
    if VAL is not None:
        val = np.ones((len(timevec),LEVEL['n'],DIM[0],DIM[1]),dtype=float)*VAL
    else:
        val = VAL_MAN
    ## set up the mask if requested
    mask = np.zeros(val.shape,dtype=bool)
    np.random.seed(SEED)
    # set the random masked cells to None
    for ii in range(0,MASK['n']):
        rx = np.random.randint(0,DIM[0])
        ry = np.random.randint(0,DIM[1])
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
    value = rootgrp.createVariable(VAR,'f8',('time','level','lat','lon'))
    ## fill variables
    times.units = TIME['units']
    times.calendar = TIME['calendar']
    times[:] = nc.date2num(timevec,units=times.units,calendar=times.calendar)
    levels[:] = levelvec
    cols[:] = col_coords
    rows[:] = row_coords
    bounds_col[:,:] = col_bnds
    bounds_row[:,:] = row_bnds
    value[:,:,:,:] = val
    value.missing_value = MASK['value']
    
    rootgrp.close()
    
def make_simple_360():
    SEED = 1 #: random number seeding for missing data
    OUTDIR = '/tmp' #: location to write
    OUTNAME = 'test_simple_360_01.nc' #: name of the file to write
    RES = 1 #: resolution of the grid
    ORIGIN = Point(270,40) #: center coordinate of upper left cell
    DIM = [4,4] #: number of cells [dimx,dimy]
    VAR = 'foo' #: name of the data variable
    ## any relevant variables for the time construction
    TIME = {'origin':datetime.datetime(2000,3,1,0,0,0),
            'end':datetime.datetime(2000,4,30,0,0,0),
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
    MASK = {'n':0,
            'value':float(1e20)}
    
    ## MANUAL VALUE SETTING ########################################################
    
    d1l1 = [[1,1,2,2],
            [1,1,2,2],
            [3,3,4,4],
            [3,3,4,4]]
    VAL = np.array(d1l1).reshape(1,1,4,4)
    VAL_MAN = None
    
    ################################################################################
    
    ## GENERATE BASE ARRAYS ##
    
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
    col_coords = np.arange(abs(ORIGIN.x)-RES*(DIM[0]-1),abs(ORIGIN.x)+RES,RES)
    col_coords = col_coords[::-1]
    row_coords = np.arange(ORIGIN.y-RES*(DIM[1]-1),ORIGIN.y+RES,RES)
    #row_coords = row_coords[::-1]
    #col,row = np.meshgrid(col_coords,row_coords)
    
    ## create bounds arrays
    col_bnds = make_bounds(col_coords,RES)
    row_bnds = make_bounds(row_coords,RES)

    ## make value array
    if VAL is not None:
        val = np.ones((len(timevec),LEVEL['n'],DIM[0],DIM[1]),dtype=float)*VAL
    else:
        val = VAL_MAN
    ## set up the mask if requested
    mask = np.zeros(val.shape,dtype=bool)
    np.random.seed(SEED)
    # set the random masked cells to None
    for ii in range(0,MASK['n']):
        rx = np.random.randint(0,DIM[0])
        ry = np.random.randint(0,DIM[1])
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
    value = rootgrp.createVariable(VAR,'f8',('time','level','lat','lon'))
    ## fill variables
    times.units = TIME['units']
    times.calendar = TIME['calendar']
    times[:] = nc.date2num(timevec,units=times.units,calendar=times.calendar)
    levels[:] = levelvec
    cols[:] = col_coords
    rows[:] = row_coords
    bounds_col[:,:] = col_bnds
    bounds_row[:,:] = row_bnds
    value[:,:,:,:] = val
    value.missing_value = MASK['value']
    
    rootgrp.close()
    
    
if __name__ == '__main__':
    make_simple()