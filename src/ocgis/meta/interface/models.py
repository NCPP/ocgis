from element import VariableElement, AttributeElement, ElementNotFound
import netCDF4 as nc
import numpy as np
import datetime
from warnings import warn


class RowBounds(VariableElement):
    _names = ['bounds_latitude',
              'bnds_latitude',
              'latitude_bounds',
              'lat_bnds']
    _ocg_name = 'latitude_bounds'
    
    
class ColumnBounds(VariableElement):
    _names = ['bounds_longitude',
              'bnds_longitude',
              'longitude_bounds',
              'lon_bnds']
    _ocg_name = 'longitude_bounds'
    
    
class Row(VariableElement):
    _names = ['latitude','lat']
    _ocg_name = 'latitude'
    
    
class Column(VariableElement):
    _names = ['longitude','lon']
    _ocg_name = 'longitude'


class Calendar(AttributeElement):
    _names = ['calendar','time_Convention']
    _ocg_name = 'calendar'
    
    def __init__(self,*args,**kwds):
        self._mode = 'local'
        super(Calendar,self).__init__(*args,**kwds)
    
    def _get_name_(self,dataset):
        try:
            ret = super(Calendar,self)._get_name_(dataset)
        except ElementNotFound:
            self._mode = 'global'
            ret = super(Calendar,self)._get_name_(dataset)
        return(ret)
    
    def _possible_(self,dataset):
        if self._mode == 'local':
            ret = dataset.variables[self._parent.name].ncattrs()
        else:
            ret = dataset.ncattrs()
        return(ret)
    
    def _get_value_(self,dataset):
        if self._mode == 'local':
            ret = getattr(dataset.variables[self._parent.name],self.name)
        else:
            ret = getattr(dataset,self.name)
        return(ret)
    
    
class TimeUnits(AttributeElement):
    _names = ['units']
    _ocg_name = 'units'


class Time(VariableElement):
    _names = ['time']
    _ocg_name = 'time'
    _AttributeElements = [Calendar,TimeUnits]
    _calendar_map = {'Calandar is no leap':'noleap',
                     'Calandar is actual':'noleap'}
    
    def _format_(self,timevec):
        time_units = self.units.value
        calendar = self.calendar.value
        try:
            ret = nc.num2date(timevec,units=time_units,calendar=calendar)
        except ValueError as e:
            try:
                new_calendar = self._calendar_map[calendar]
            except KeyError:
                raise(e)
            warn('calendar name "{0}" remapped to "{1}"'.format(calendar,
                                                                new_calendar))
            ret = nc.num2date(timevec,units=time_units,calendar=new_calendar)
        if not isinstance(ret[0],datetime.datetime):
            reformat_timevec = np.empty(ret.shape,dtype=object)
            for ii,t in enumerate(ret):
                reformat_timevec[ii] = datetime.datetime(t.year,t.month,t.day,
                                                         t.hour,t.minute,t.second)
            ret = reformat_timevec
        return(ret)


class Level(VariableElement):
    _names = ['level','lvl','levels','lvls']
    _ocg_name = 'level'