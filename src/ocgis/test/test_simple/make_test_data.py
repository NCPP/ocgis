from shapely.geometry.point import Point
import datetime
import numpy as np
import os.path
import netCDF4 as nc
from ocgis.util.helpers import iter_array, project_shapely_geometry
from ocgis import env
from abc import ABCMeta, abstractproperty, abstractmethod
from ocgis.interface.base.crs import CoordinateReferenceSystem


class NcFactory(object):
    __metaclass__ = ABCMeta
    
    @abstractproperty
    def filename(self): pass
    
    @property
    def outdir(self):
        return(env.DIR_OUTPUT)
    
    @abstractmethod
    def write(self): pass
    
    def make_bounds(self,coords,res):
        '''using points from |coords| adjust by |res| to provide bounds'''
        bnds = []
        for g in coords.flat:
            bnds.append([g-res*0.5,g+res*0.5])
        return(np.array(bnds))


class SimpleNcNoLevel(NcFactory):
    
    @property
    def filename(self):
        return('test_simple_spatial_no_level_01.nc')
    
    def write(self):
        SEED = 1 #: random number seeding for missing data
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
        
#        ## make the level vector
#        levelvec = np.array([50,150])
#        levelvec_bounds = np.array([[0,100],[100,200]])
    
        ## make centroids
        col_coords = -np.arange(abs(ORIGIN.x)-RES*(DIM[0]-1),abs(ORIGIN.x)+RES,RES)
        col_coords = col_coords[::-1]
        row_coords = np.arange(ORIGIN.y-RES*(DIM[1]-1),ORIGIN.y+RES,RES)
        #row_coords = row_coords[::-1]
        #col,row = np.meshgrid(col_coords,row_coords)
        
        ## create bounds arrays
        col_bnds = self.make_bounds(col_coords,RES)
        row_bnds = self.make_bounds(row_coords,RES)
    
        ## make value array
        if VAL is not None:
            val = np.ones((len(timevec),DIM[0],DIM[1]),dtype=float)*VAL
        else:
            val = VAL_MAN
        ## set up the mask if requested
        mask = np.zeros(val.shape,dtype=bool)
        np.random.seed(SEED)
        # set the random masked cells to None
        for ii in range(0,MASK['n']):
            rx = np.random.randint(0,DIM[0])
            ry = np.random.randint(0,DIM[1])
            mask[:,rx,ry] = True
        val = np.ma.array(val,dtype=float,mask=mask,fill_value=MASK['value'])
        
        ## WRITE THE NC FILE ###########################################################
        
        ## initialize the output file
        rootgrp = nc.Dataset(os.path.join(self.outdir,self.filename),'w',format='NETCDF4')
        ## create the dimensions
        time = rootgrp.createDimension(TIME['name'],size=len(timevec))
        lat = rootgrp.createDimension('lat',size=len(row_coords))
        lon = rootgrp.createDimension('lon',size=len(col_coords))
        bound = rootgrp.createDimension('bound',size=2)
        ## create the variables
        times = rootgrp.createVariable(TIME['name'],'f8',('time',))
        times.axis = 'T'
        bounds_times = rootgrp.createVariable('time_bnds','f8',('time','bound'))
        cols = rootgrp.createVariable('longitude','f8',('lon',))
        cols.axis = 'X'
        rows = rootgrp.createVariable('latitude','f8',('lat',))
        rows.axis = 'Y'
        bounds_col = rootgrp.createVariable(SPACE['col_bnds'],'f8',('lon','bound'))
        bounds_row = rootgrp.createVariable(SPACE['row_bnds'],'f8',('lat','bound'))
        value = rootgrp.createVariable(VAR,'f8',('time','lat','lon'),fill_value=1e20)
        ## fill variables
        times.units = TIME['units']
        times.calendar = TIME['calendar']
        times[:] = nc.date2num(timevec,units=times.units,calendar=times.calendar)
        bounds_times[:] = nc.date2num(timevec_bnds,units=times.units,calendar=times.calendar)
        cols[:] = col_coords
        rows[:] = row_coords
        bounds_col[:,:] = col_bnds
        bounds_row[:,:] = row_bnds
        value[:,:,:] = val
        value.missing_value = MASK['value']
        value.standard_name = 'foo'
        value.long_name = 'foo_foo'
        value.units = 'huge'

        # add bounds attributes
        times.bounds = bounds_times._name
        rows.bounds = bounds_row._name
        cols.bounds = bounds_col._name

        rootgrp.close()


class SimpleNcNoBounds(NcFactory):
    
    @property
    def filename(self):
        return('test_simple_spatial_no_bounds_01.nc')
    
    def write(self):
        SEED = 1 #: random number seeding for missing data
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
#        ## any relevant variables for the spatial construction
#        SPACE = {'row_bnds':'bounds_latitude',
#                 'col_bnds':'bounds_longitude'}
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
        
#        ## make vector time bounds
#        timevec_bnds = np.empty((len(timevec),2),dtype=object)
#        delta = datetime.timedelta(hours=12)
#        for idx,tv in iter_array(timevec,return_value=True):
#            timevec_bnds[idx,0] = tv - delta
#            timevec_bnds[idx,1] = tv + delta
        
        ## make the level vector
        levelvec = np.array([50,150])
#        levelvec_bounds = np.array([[0,100],[100,200]])
    
        ## make centroids
        col_coords = -np.arange(abs(ORIGIN.x)-RES*(DIM[0]-1),abs(ORIGIN.x)+RES,RES)
        col_coords = col_coords[::-1]
        row_coords = np.arange(ORIGIN.y-RES*(DIM[1]-1),ORIGIN.y+RES,RES)
        #row_coords = row_coords[::-1]
        #col,row = np.meshgrid(col_coords,row_coords)
        
#        ## create bounds arrays
#        col_bnds = self.make_bounds(col_coords,RES)
#        row_bnds = self.make_bounds(row_coords,RES)
    
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
        rootgrp = nc.Dataset(os.path.join(self.outdir,self.filename),'w',format='NETCDF4')
        ## create the dimensions
        level = rootgrp.createDimension(LEVEL['name'],size=LEVEL['n'])
        time = rootgrp.createDimension(TIME['name'],size=len(timevec))
        lat = rootgrp.createDimension('lat',size=len(row_coords))
        lon = rootgrp.createDimension('lon',size=len(col_coords))
        bound = rootgrp.createDimension('bound',size=2)
        ## create the variables
        times = rootgrp.createVariable(TIME['name'],'f8',('time',))
#        bounds_times = rootgrp.createVariable('time_bnds','f8',('time','bound'))
        levels = rootgrp.createVariable(LEVEL['name'],'i4',('level',))
        bounds_levels = rootgrp.createVariable('level_bnds','i4',('level','bound'))
        cols = rootgrp.createVariable('longitude','f8',('lon',))
        rows = rootgrp.createVariable('latitude','f8',('lat',))
#        bounds_col = rootgrp.createVariable(SPACE['col_bnds'],'f8',('lon','bound'))
#        bounds_row = rootgrp.createVariable(SPACE['row_bnds'],'f8',('lat','bound'))
        value = rootgrp.createVariable(VAR,'f8',('time','level','lat','lon'),fill_value=1e20)
        ## fill variables
        times.units = TIME['units']
        times.calendar = TIME['calendar']
        times[:] = nc.date2num(timevec,units=times.units,calendar=times.calendar)
#        bounds_times[:] = nc.date2num(timevec_bnds,units=times.units,calendar=times.calendar)
        levels[:] = levelvec
#        bounds_levels[:] = levelvec_bounds
        cols[:] = col_coords
        rows[:] = row_coords
#        bounds_col[:,:] = col_bnds
#        bounds_row[:,:] = row_bnds
        value[:,:,:,:] = val
        value.missing_value = MASK['value']
        value.standard_name = 'foo'
        value.long_name = 'foo_foo'
        value.units = 'huge'
        
        rootgrp.close()


class SimpleNcProjection(NcFactory):
    
    @property
    def filename(self):
        return('test_simple_spatial_projected_01.nc')
    
    def write(self):
        SEED = 1 #: random number seeding for missing data
        RES = 1 #: resolution of the grid
        ORIGIN = Point(-105,40) #: center coordinate of upper left cell
        from_sr = CoordinateReferenceSystem(epsg=4326).sr
        to_sr = CoordinateReferenceSystem(epsg=2163).sr
        ORIGIN = project_shapely_geometry(ORIGIN,from_sr,to_sr)
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
        col_bnds = self.make_bounds(col_coords,RES)
        row_bnds = self.make_bounds(row_coords,RES)
    
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
        rootgrp = nc.Dataset(os.path.join(self.outdir,self.filename),'w',format='NETCDF4')
        ## create the dimensions
        level = rootgrp.createDimension(LEVEL['name'],size=LEVEL['n'])
        time = rootgrp.createDimension(TIME['name'],size=len(timevec))
        lat = rootgrp.createDimension('lat',size=len(row_coords))
        lon = rootgrp.createDimension('lon',size=len(col_coords))
        bound = rootgrp.createDimension('bound',size=2)
        ## create the variables
        times = rootgrp.createVariable(TIME['name'],'f8',('time',))
        times.axis = 'T'
        bounds_times = rootgrp.createVariable('time_bnds','f8',('time','bound'))
        levels = rootgrp.createVariable(LEVEL['name'],'i4',('level',))
        levels.axis = 'Z'
        bounds_levels = rootgrp.createVariable('level_bnds','i4',('level','bound'))
        cols = rootgrp.createVariable('longitude','f8',('lon',))
        cols.axis = 'X'
        cols.standard_name = 'projection_x_coordinate'
        rows = rootgrp.createVariable('latitude','f8',('lat',))
        rows.axis = 'Y'
        rows.standard_name = 'projection_y_coordinate'
        bounds_col = rootgrp.createVariable(SPACE['col_bnds'],'f8',('lon','bound'))
        bounds_row = rootgrp.createVariable(SPACE['row_bnds'],'f8',('lat','bound'))
        value = rootgrp.createVariable(VAR,'f8',('time','level','lat','lon'),fill_value=1e20)
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
        value.standard_name = 'foo'
        value.long_name = 'foo_foo'
        value.units = 'huge'
        value.grid_mapping = 'crs'

        grid_mapping = rootgrp.createVariable('crs','c')
        grid_mapping.grid_mapping_name = "lambert_conformal_conic"
        grid_mapping.standard_parallel = [30., 60.]
        grid_mapping.longitude_of_central_meridian = -97.
        grid_mapping.latitude_of_projection_origin = 47.5
        grid_mapping.false_easting = 3325000.
        grid_mapping.false_northing = 2700000.

        # add bounds attributes
        times.bounds = bounds_times._name
        rows.bounds = bounds_row._name
        cols.bounds = bounds_col._name
        levels.bounds = bounds_levels._name
        
        rootgrp.close()


class SimpleNc(NcFactory):
    
    @property
    def filename(self):
        return('test_simple_spatial_01.nc')

    def write2(self):
        rootgrp = nc.Dataset(os.path.join(self.outdir,self.filename),'w',format='NETCDF4')
        # rootgrp.createDimension('time', size=5)
        rootgrp.createVariable('time', int)
        # rootgrp.variables['time'][:] = [1, 2, 3, 4, 5]
        rootgrp.variables['time'].calendar = 'foo1'
        rootgrp.close()
    
    def write(self):
        SEED = 1 #: random number seeding for missing data
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
        col_bnds = self.make_bounds(col_coords,RES)
        row_bnds = self.make_bounds(row_coords,RES)
    
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
        rootgrp = nc.Dataset(os.path.join(self.outdir,self.filename),'w',format='NETCDF4')
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
        value = rootgrp.createVariable(VAR,'f8',('time','level','lat','lon'),fill_value=1e20)
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
        value.standard_name = 'Maximum Temperature Foo'
        value.long_name = 'foo_foo'
        value.units = 'K'

        # add bounds attributes
        times.bounds = bounds_times._name
        rows.bounds = bounds_row._name
        cols.bounds = bounds_col._name
        levels.bounds = bounds_levels._name
        
        rootgrp.close()


class SimpleNcNoSpatialBounds(NcFactory):
    
    @property
    def filename(self):
        return('test_simple_spatial_no_bounds_01.nc')
    
    def write(self):
        SEED = 1 #: random number seeding for missing data
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
#        ## any relevant variables for the spatial construction
#        SPACE = {'row_bnds':'bounds_latitude',
#                 'col_bnds':'bounds_longitude'}
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
        
#        ## create bounds arrays
#        col_bnds = self.make_bounds(col_coords,RES)
#        row_bnds = self.make_bounds(row_coords,RES)
    
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
        rootgrp = nc.Dataset(os.path.join(self.outdir,self.filename),'w',format='NETCDF4')

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
#        bounds_col = rootgrp.createVariable(SPACE['col_bnds'],'f8',('lon','bound'))
#        bounds_row = rootgrp.createVariable(SPACE['row_bnds'],'f8',('lat','bound'))
        value = rootgrp.createVariable(VAR,'f8',('time','level','lat','lon'),fill_value=1e20)
        ## fill variables
        times.units = TIME['units']
        times.calendar = TIME['calendar']
        times[:] = nc.date2num(timevec,units=times.units,calendar=times.calendar)
        bounds_times[:] = nc.date2num(timevec_bnds,units=times.units,calendar=times.calendar)
        levels[:] = levelvec
        bounds_levels[:] = levelvec_bounds
        cols[:] = col_coords
        rows[:] = row_coords
#        bounds_col[:,:] = col_bnds
#        bounds_row[:,:] = row_bnds
        value[:,:,:,:] = val
        value.missing_value = MASK['value']
        value.standard_name = 'foo'
        value.long_name = 'foo_foo'
        value.units = 'huge'

        # add bounds attributes
        times.bounds = bounds_times._name

        rootgrp.close()


class SimpleMaskNc(NcFactory):
    
    @property
    def filename(self):
        return('test_simple_mask_spatial_01.nc')
    
    def write(self):
        SEED = 1 #: random number seeding for missing data
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
        col_bnds = self.make_bounds(col_coords,RES)
        row_bnds = self.make_bounds(row_coords,RES)
    
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
        rootgrp = nc.Dataset(os.path.join(self.outdir,self.filename),'w',format='NETCDF4')
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

        # add bounds attributes
        rows.bounds = bounds_row._name
        cols.bounds = bounds_col._name
        
        rootgrp.close()


class SimpleNcMultivariate(NcFactory):

    @property
    def filename(self):
        return('test_simple_multivariate_01.nc')

    def write(self):
        SEED = 1 #: random number seeding for missing data
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
        col_bnds = self.make_bounds(col_coords,RES)
        row_bnds = self.make_bounds(row_coords,RES)

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
        rootgrp = nc.Dataset(os.path.join(self.outdir,self.filename),'w',format='NETCDF4')
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
        value = rootgrp.createVariable(VAR,'f8',('time','level','lat','lon'),fill_value=1e20)
        value2 = rootgrp.createVariable(VAR+'2','f8',('time','level','lat','lon'),fill_value=1e20)
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
        value.standard_name = 'Maximum Temperature Foo'
        value.long_name = 'foo_foo'
        value.units = 'K'

        value2[:,:,:,:] = val + 3.0
        value2.missing_value = MASK['value']
        value2.standard_name = 'Precipitation Foo'
        value2.long_name = 'foo_foo_pr'
        value2.units = 'mm/s'

        # add bounds attributes
        times.bounds = bounds_times._name
        rows.bounds = bounds_row._name
        cols.bounds = bounds_col._name

        rootgrp.close()


class SimpleNc360(NcFactory):
    
    @property
    def filename(self):
        return('test_simple_360_01.nc')
    
    def write(self):
        SEED = 1 #: random number seeding for missing data
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
        col_bnds = self.make_bounds(col_coords,RES)
        row_bnds = self.make_bounds(row_coords,RES)
    
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
        rootgrp = nc.Dataset(os.path.join(self.outdir,self.filename),'w',format='NETCDF4')
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

        # add bounds attributes
        rows.bounds = bounds_row._name
        cols.bounds = bounds_col._name
        
        rootgrp.close()
    

#if __name__ == '__main__':
#    make_simple()