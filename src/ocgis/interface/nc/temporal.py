from ocgis.interface.base.dimension.temporal import TemporalDimension
from ocgis.interface.nc.dimension import NcVectorDimension
import numpy as np
import netCDF4 as nc
import datetime
from ocgis.util.helpers import iter_array, get_none_or_slice
from ocgis import constants


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
        
        ## test if the units are the special case with months in the time units
        if self.units.startswith('months'):
            self._has_months_units = True
        else:
            self._has_months_units = False
        
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
        ## if there are month units, call the special procedure to convert those
        ## to datetime objects
        if self._has_months_units == False:
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
        else:
            arr = get_datetime_from_months_time_units(arr,self.units,month_centroid=constants.calc_month_centroid)
        return(arr)
    
    def get_nc_time(self,values):
        try:
            ret = np.atleast_1d(nc.date2num(values,self.units,calendar=self.calendar))
        except ValueError:
            ## special behavior for conversion of time units with months
            if self._has_months_units:
                ret = get_num_from_months_time_units(values, self.units, dtype=None)
            else:
                raise
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

        try:
            kwds['bounds'] = self.get_nc_time(bounds)
        ## this may happen if the data has months in the time units. the functions that compute the datetime-numeric
        ## conversions did not anticipate bounds.
        except AttributeError:
            if self._has_months_units:
                bounds_fill = np.empty(bounds.shape)
                bounds_fill[:,0] = self.get_nc_time(bounds[:,0])
                bounds_fill[:,1] = self.get_nc_time(bounds[:,1])
                kwds['bounds'] = bounds_fill
            else:
                raise

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
        
        
def get_origin_datetime_from_months_units(units):
    '''
    Get the origin Python :class:``datetime.datetime`` object from a month
    string.
    
    :param str units: Source units to parse.
    :returns: :class:``datetime.datetime``
    
    >>> units = "months since 1978-12"
    >>> get_origin_datetime_from_months_units(units)
    datetime.datetime(1978, 12, 1, 0, 0)
    '''
    origin = ' '.join(units.split(' ')[2:])
    to_try = ['%Y-%m','%Y-%m-%d %H']
    converted = False
    for tt in to_try:
        try:
            origin = datetime.datetime.strptime(origin,tt)
            converted = True
            break
        except ValueError as e:
            continue
    if converted == False:
        raise(e)
    return(origin)

def get_datetime_from_months_time_units(vec,units,month_centroid=16):
    '''
    Convert a vector of months offsets into :class:``datetime.datetime`` objects.
    
    :param vec: Vector of integer month offsets.
    :type vec: :class:``np.ndarray``
    :param str units: Source units to parse.
    :param month_centroid: The center day of the month to use when creating the
     :class:``datetime.datetime`` objects.
    
    >>> units = "months since 1978-12"
    >>> vec = np.array([0,1,2,3])
    >>> get_datetime_from_months_time_units(vec,units)
    array([1978-12-16 00:00:00, 1979-01-16 00:00:00, 1979-02-16 00:00:00,
           1979-03-16 00:00:00], dtype=object)
    '''
    ## only work with integer inputs
    vec = np.array(vec,dtype=int)
      
    def _get_datetime_(current_year,origin_month,offset_month,current_month_correction,month_centroid):
        return(datetime.datetime(current_year,(origin_month+offset_month)-current_month_correction,month_centroid))
    
    origin = get_origin_datetime_from_months_units(units)
    origin_month = origin.month
    current_year = origin.year
    current_month_correction = 0
    ret = np.ones(len(vec),dtype=object)
    for ii,offset_month in enumerate(vec):
        try:
            fill = _get_datetime_(current_year,origin_month,offset_month,current_month_correction,month_centroid)
        except ValueError:
            current_month_correction += 12
            current_year += 1
            fill = _get_datetime_(current_year,origin_month,offset_month,current_month_correction,month_centroid)
        ret[ii] = fill
    return(ret)

def get_difference_in_months(origin,target):
    '''
    Get the integer difference in months between an origin and target datetime.
    
    :param :class:``datetime.datetime`` origin: The origin datetime object.
    :param :class:``datetime.datetime`` target: The target datetime object.
    
    >>> get_difference_in_months(datetime.datetime(1978,12,1),datetime.datetime(1979,3,1))
    3
    >>> get_difference_in_months(datetime.datetime(1978,12,1),datetime.datetime(1978,7,1))
    -5
    '''
            
    def _count_(start_month,stop_month,start_year,stop_year,direction):
        count = 0
        curr_month = start_month
        curr_year = start_year
        while True:
            if curr_month == stop_month and curr_year == stop_year:
                break
            else:
                pass
            
            if direction == 'forward':
                curr_month += 1
            elif direction == 'backward':
                curr_month -= 1
            else:
                raise(NotImplementedError(direction))
                
            if curr_month == 13:
                curr_month = 1
                curr_year += 1
            if curr_month == 0:
                curr_month = 12
                curr_year -= 1
            
            if direction == 'forward':
                count += 1
            else:
                count -= 1
            
        return(count)
    
    origin_month,origin_year = origin.month,origin.year
    target_month,target_year = target.month,target.year
    
    if origin <= target:
        direction = 'forward'
    else:
        direction = 'backward'
        
    diff_months = _count_(origin_month,target_month,origin_year,target_year,direction)
    return(diff_months)

def get_num_from_months_time_units(vec,units,dtype=None):
    '''
    Convert a vector of :class:``datetime.datetime`` objects into an integer
    vector.
    
    :param vec: Input vector to convert.
    :type vec: :class:``np.ndarray``
    :param str units: Source units to parse.
    :param type dtype: Output vector array type.
    
    >>> units = "months since 1978-12"
    >>> vec = np.array([datetime.datetime(1978,12,1),datetime.datetime(1979,1,1)])
    >>> get_num_from_months_time_units(vec,units)
    array([0, 1])
    '''
    origin = get_origin_datetime_from_months_units(units)
    ret = [get_difference_in_months(origin,target) for target in vec]
    return(np.array(ret,dtype=dtype))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
