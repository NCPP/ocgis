from django.core import serializers
import os
import climatedata
from climatedata.models import SpatialGridCell, SpatialGrid, TemporalGridCell
import numpy as np
from netCDF4 import date2num, Dataset
import datetime





#PATH = os.path.join(os.path.split(climatedata.__file__)[0],'fixtures','trivial_grid.json')
#
#
### collect data objects
#cells = []
#with open(PATH,'r') as f:
#    for obj in serializers.deserialize("json",f):
#        if isinstance(obj.object,SpatialGridCell):
#            cells.append(obj.object)
#        elif isinstance(obj.object,SpatialGrid):
#            grid = obj.object
#        else:
#            raise ValueError('db model not recognized')
        
## organize cells into lat/lon objects

class NcWrite(object):
    
    def __init__(self,var,units_var,spatial_grid=None,temporal_grid=None):
        self.var = var
        self.units_var = units_var
        
        ## return the polygon cell centroids
        if spatial_grid != None:
            pass
        else:
            self.centroids = SpatialGridCell.objects.all().centroid()
        ## get dimension coordinate vectors
        self.dim_x = self._get_dim_(0)
        self.dim_y = self._get_dim_(1)
        ## get time dimension
        if temporal_grid != None:
            pass
        else:
            self.units_time = 'days since 1800-01-01 00:00:00 0:00'
            self.calendar = 'gregorian'
            self.date_ref = datetime.datetime(1800,1,1)
#            self.dates = TemporalGridCell.objects.all().order_by('date')
            self.dates = [datetime.datetime(2011,1,16),
                          datetime.datetime(2011,2,15),
                          datetime.datetime(2011,3,16)]
            self.dim_time = date2num(self.dates,self.units_time,self.calendar) 
            
    def write(self,path,close=True,seed=2):
        np.random.seed(seed)
        
        rootgrp = Dataset(path,'w')
        
        rootgrp.createDimension('time',len(self.dim_time))
        rootgrp.createDimension('lon',len(self.dim_x))
        rootgrp.createDimension('lat',len(self.dim_y))
        
        times = rootgrp.createVariable('time','f8',('time',))
#        levels = rootgrp.createVariable('level','i4',('level',))
        latitudes = rootgrp.createVariable('latitude','f4',('lat',))
        longitudes = rootgrp.createVariable('longitude','f4',('lon',))
        var = rootgrp.createVariable(self.var,'f4',('time','lat','lon'))
        
        times.units = self.units_time
        times.calendar = self.calendar
        var.units = self.units_var
        
        latitudes[:] = self.dim_y
        longitudes[:] = self.dim_x
        times[:] = self.dim_time
        var[:,:,:] = np.random.uniform(size=(len(self.dim_time),len(self.dim_y),len(self.dim_x)))
        
        if close:
            rootgrp.close()
            ret = None
        else:
            ret = rootgrp
        
        return ret
        
    def _get_dim_(self,idx):
        dim = np.unique(np.array([c.centroid[idx] for c in self.centroids]))
        dim.sort()
        return dim
        
#        self.centroids = qs.centroid()
#        import ipdb;ipdb.set_trace()