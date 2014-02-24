from ocgis.interface.base.dimension.temporal import TemporalDimension
from ocgis.interface.nc.dimension import NcVectorDimension
import numpy as np
import netCDF4 as nc
import datetime
from ocgis.util.helpers import iter_array, get_none_or_slice


class NcTemporalDimension(NcVectorDimension,TemporalDimension):
    _attrs_slice = ('uid','_value','_src_idx','_value_datetime')
    
    def __init__(self,*args,**kwds):
        self.calendar = kwds.pop('calendar')
        self.format_time = kwds.pop('format_time',True)
        self._value_datetime = kwds.pop('value_datetime',None)
        self._bounds_datetime = kwds.pop('bounds_datetime',None)
        
        NcVectorDimension.__init__(self,*args,**kwds)
        
        assert(self.units != None)
        assert(self.calendar != None)
        
    @property
    def bounds_datetime(self):
        if self.bounds is not None:
            if self._bounds_datetime is None:
                self._bounds_datetime = np.atleast_2d(self.get_datetime(self.bounds))
        return(self._bounds_datetime)
    @bounds_datetime.setter
    def bounds_datetime(self,value):
        if value is None:
            new = None
        else:
            new = np.atleast_2d(value).reshape(-1,2)
        self._bounds_datetime = new
        
    @property
    def extent_datetime(self):
        return(tuple(self.get_datetime(self.extent)))
        
    @property
    def value_datetime(self):
        if self._value_datetime is None:
            self._value_datetime = np.atleast_1d(self.get_datetime(self.value))
        return(self._value_datetime)
    
    def get_between(self,lower,upper,return_indices=False):
        lower,upper = tuple(self.get_nc_time([lower,upper]))
        return(NcVectorDimension.get_between(self,lower,upper,return_indices=return_indices))
        
    def get_datetime(self,arr):
        arr = np.atleast_1d(nc.num2date(arr,self.units,calendar=self.calendar))
        dt = datetime.datetime
        for idx,t in iter_array(arr,return_value=True):
            ## attempt to convert times to datetime objects
            try:
                arr[idx] = dt(t.year,t.month,t.day,
                              t.hour,t.minute,t.second)
            ## this may fail for some calendars, in that case maintain the instance
            ## object returned from netcdftime see:
            ## http://netcdf4-python.googlecode.com/svn/trunk/docs/netcdftime.netcdftime.datetime-class.html
            except ValueError:
                arr[idx] = arr[idx]
        return(arr)
    
    def get_nc_time(self,values):
        ret = np.atleast_1d(nc.date2num(values,self.units,calendar=self.calendar))
        return(ret)
    
    def _format_slice_state_(self,state,slc):
        state = NcVectorDimension._format_slice_state_(self,state,slc)
        state.bounds_datetime = get_none_or_slice(state._bounds_datetime,(slc,slice(None)))
        return(state)
    
    def _get_datetime_bounds_(self):
        if self.format_time:
            ret = self.bounds_datetime
        else:
            ret = self.bounds
        return(ret)
    
    def _get_datetime_value_(self):
        if self.format_time:
            ret = self.value_datetime
        else:
            ret = self.value
        return(ret)
    
    def _get_temporal_group_dimension_(self,*args,**kwds):
        kwds['calendar'] = self.calendar
        kwds['units'] = self.units
        value = kwds.pop('value')
        bounds = kwds.pop('bounds')
        kwds['value'] = self.get_nc_time(value)
        kwds['bounds'] = self.get_nc_time(bounds)
        kwds['value_datetime'] = value
        kwds['bounds_datetime'] = bounds
        return(NcTemporalGroupDimension(*args,**kwds))
    
    def _set_date_parts_(self,yld,value):
        if self.format_time:
            TemporalDimension._set_date_parts_(self,yld,value)
        else:
            yld['year'],yld['month'],yld['day'] = None,None,None
    
    
class NcTemporalGroupDimension(NcTemporalDimension):
    
    def __init__(self,*args,**kwds):
        self.grouping = kwds.pop('grouping')
        self.dgroups = kwds.pop('dgroups')
        self.date_parts = kwds.pop('date_parts')
                
        NcTemporalDimension.__init__(self,*args,**kwds)
