import numpy as np
from ocgis.calc.base import AbstractParameterizedFunction, \
    AbstractUnivariateSetFunction
from ocgis import constants
from collections import OrderedDict, defaultdict
from datetime import datetime
import calendar

########### begin: utility functions for get_daily_percentile()

# def num2date(num, calend, units):
#     '''
#     Converts numerical date to datetime object.
#
#     :param num: numerical date
#     :type num: float
#     :param calend: calendar attribute of variable "time" in netCDF file
#     :type calend: str
#     :param units: units of variable "time" in netCDF file
#     :type units: str
#
#     :rtype: datetime object
#     '''
#     t = utime(units, calend)
#     dt = t.num2date(num)
#     return dt


def get_dict_caldays(dt_arr):
    '''
    Create a dictionary of calendar days, where keys=months, values=days.
    
    :param dt_arr: time steps vector
    :type dt_arr: np.ndarray (1D) of datetime objects
    
    :rtype: dict
    
    '''

    dic = defaultdict(list)

    for dt in dt_arr:
        dic[dt.month].append(dt.day)

    for key in dic.keys():
        dic[key] = list(set(dic[key])) 
    
    return dic



    
def get_masked(current_date, month, day, hour, window_width, only_leap_years): 

    '''
    Returns "True" if "current_date" is not in the window centered on the given calendar day (month-day).
    Returns "False", if it enters in the window.
    
    :param current_date: current date
    :type current_date: datetime object 
    :param month: month of the corresponding calendar day
    :type month: int
    :param day: day of the corresponding calendar day
    :type day: int
    :param hour: hour of the current day
    :type hour int
    :param window_width: window width, must be odd
    :type window_width: int
    :param only_leap_years: option for February 29th 
    :type only_leap_years: bool

    :rtype: bool (if True, the date will be masked)
    
    '''
    
    yyyy = current_date.year

    if (day==29 and month==02):
        if calendar.isleap(yyyy):
            dt1 = datetime(yyyy,month,day,hour)
            diff = abs(current_date-dt1).days
            toReturn = diff > window_width/2
        else:
            if only_leap_years:
                toReturn=True
            else:                
                dt1 = datetime(yyyy,02,28,hour)
                diff = (current_date-dt1).days
                toReturn = (diff < (-(window_width/2) + 1)) or (diff > window_width/2)
    else:
        d1 = datetime(yyyy,month,day, hour)
        
        # In the case the current date is in December and calendar day (day-month) is at the beginning of year.
        # For example we are looking for dates around January 2nd, and the current date is 31 Dec 1999,
        # we will compare it with 02 Jan 2000 (1999 + 1)
        d2 = datetime(yyyy+1,month,day, hour)
        
        # In the case the current date is in January and calendar day (day-month) is at the end of year.
        # For example we are looking for dates around December 31st, and the current date is 02 Jan 2003,
        # we will compare it with 01 Jan 2002 (2003 - 1) 
        d3 = datetime(yyyy-1,month,day, hour)
        
        diff=min(abs(current_date-d1).days,abs(current_date-d2).days,abs(current_date-d3).days)
        toReturn = diff > window_width/2
        
    return toReturn


def get_mask_dt_arr(dt_arr, month, day, dt_hour, window_width, only_leap_years):
    '''
    Creates a binary mask for a datetime vector for a given calendar day (month-day).
    
    :param dt_arr: time steps vector
    :type dt_arr: numpy.ndarray (1D) of datetime objects
    :param month: month of a calendar day
    :type month: int
    :param day: day of a calendar day
    :type day: int
    :param window_width: window width, must be odd
    :type window_width: int
    :param only_leap_years: option for February 29th 
    :type only_leap_years: bool
    
    :param window_width: window width, must be odd
    :type window_width: int
    
    rtype: numpy.ndarray (1D)   
    ''' 
    mask = np.array([get_masked(dt, month, day, dt_hour, window_width, only_leap_years) for dt in dt_arr])
    return mask


def get_year_list(dt_arr):
    '''
    Just to get a list of all years conteining in time steps vector (dt_arr).
    '''

    year_list = []
    for dt in dt_arr:
        year_list.append(dt.year)
        
    year_list = list(set(year_list))
    
    return year_list


########### end: utility functions for get_daily_percentile()




class DynamicDailyKernelPercentileThreshold(AbstractUnivariateSetFunction,AbstractParameterizedFunction):
    key = 'dynamic_kernel_percentile_threshold'
    parms_definition = {'operation':str,'percentile':float,'daily_percentile':None,'width':int}
    dtype = constants.np_int
    description = 'Implementation of moving window percentile threshold calculations similar to ECA indices: http://eca.knmi.nl/documents/atbd.pdf'
    
    def __init__(self,*args,**kwds):
        self._daily_percentile = {}
        AbstractUnivariateSetFunction.__init__(self,*args,**kwds)
    
    def calculate(self,values,operation=None,percentile=None,daily_percentile=None,width=None):
        '''
        :param operation: One of 'gt', 'lt', 'lte', or 'gte'.
        :type operation: str
        :param percentile: Percentile threshold to use in the `numpy.percentile` calculation.
        :type percentile: int or float from 0 to 100
        :param width: Width of kernel (in days) to use for the moving window percentile.
        :type width: int, oddly-numbered, at least 3 or greater
        :param daily_percentile: Optional structure array as returned by :meth:`~ocgis.calc.library.DynamicDailyKernelPercentileThreshold.get_daily_percentile`. 
         If this is passed, it will not be calculated internally by the object.
        :type daily_percentile: `numpy.ndarray`
        '''
        
        ## if the daily percentile structured array is not passed, calculate it
        if daily_percentile is None:
            try:
                daily_percentile = self._daily_percentile[self._curr_variable.alias]
            ## likely has not been calculated
            except KeyError:
                dp = self.get_daily_percentile(self._curr_variable.value,
                                               self.field.temporal.value_datetime,
                                               percentile,
                                               width)
                self._daily_percentile.update({self._curr_variable.alias:dp})
                daily_percentile = self._daily_percentile[self._curr_variable.alias]
                
            
        ## extract the corresponding dates
        dates = self.field.temporal.value_datetime[self._curr_group]
        
        ## match each date to it's percentile
        # dp_indices = np.zeros(dates.shape)
        # for ii,date in enumerate(dates):
            # month,day = date.month,date.day
            # dp_indices[ii] = daily_percentile[month, day]
            # dp_indices[ii] = daily_percentile['index'][np.logical_and(daily_percentile['month'] == month,
            #                                                           daily_percentile['day'] == day)]

        ## construct the the comparison array
        b = np.empty_like(values)
        # for ii in range(b.shape[0]):
            # b[ii] = daily_percentile[dp_indices[ii]]['percentile']
        for ii, date in enumerate(dates):
            b[ii] = daily_percentile[date.month, date.day]
        
        ## perform requested logical operation
        if operation == 'gt':
            ret = values > b
        elif operation == 'lt':
            ret = values < b
        elif operation == 'gte':
            ret = values >= b
        elif operation == 'lte':
            ret = values <= b
            
        ## we want to maintain the same mask as the input
        ret = np.ma.array(ret,mask=values.mask)
        
        ## do the temporal aggregation
        ret = np.ma.sum(ret,axis=0)
        
        return(ret)
        
#    @property
#    def daily_percentile(self):
#        if self._daily_percentile is None:
#            self._daily_percentile = self.get_daily_percentile(self.dataset.value,
#                                                               self.dataset.temporal.value_datetime,
#                                                               self.kwds['percentile'],
#                                                               self.kwds['width'])
#        return(self._daily_percentile)
    
    # @staticmethod
#    def get_daily_percentile(all_values,temporal,percentile,width):
#        '''
#        :param all_values: Array holding all values to use for base percentile calculations.
#        :type all_values: numpy.MaskedArray
#        :param temporal: Vector holding `datetime.datetime` objects with same length as the time axis of `all_values`.
#        :type temporal: numpy.ndarray
#        :param percentile: Percentile threshold to use in the `numpy.percentile` calculation.
#        :type percentile: int or float from 0 to 100
#        :param width: Width of kernel to use for the moving window percentile.
#        :type width: int, oddly-number, at least 3 or greater
#        :returns: A structure array with four fields: month, day, index, and percentile.
#        :rtype: numpy.ndarray
#        '''
#        assert(len(all_values.shape) == 5)
#        
#        ## collect the days and months uniquely to use for the calendar attribution
#        dmap = {}
#        years = set()
#        for dt in temporal.flat:
#            day,month,year = dt.day,dt.month,dt.year
#            years.update([year])
#            if month not in dmap:
#                dmap[month] = []
#            if day not in dmap[month]:
#                dmap[month].append(day)
#        
#        ## this is the structure array storing date parts and percentile arrays
#        cday_length = sum([len(v) for v in dmap.itervalues()])
#        cday = np.zeros(cday_length,dtype=[('month',int),('day',int),('index',int),('percentile',object)])
#        idx = 0
#        for month in sorted(dmap):
#            for day in sorted(dmap[month]):
#                cday[idx] = (month,day,idx,None)
#                idx += 1
#        
#        ## loop for each calendar day and calculate the percentile value for its
#        ## data window
#        r_cday_index = cday['index']
#        r_logical_and = np.logical_and
#        cday_shape = cday.shape[0]
#        for target_cday_index in range(cday_shape):
#            select = np.zeros(all_values.shape[1],dtype=bool)
#            
#            ## this function returns the calendar days part of the target calendar
#            ## day's window. these are then used to subset the calendar day structure
#            ## array.
#            window_days = DynamicDailyKernelPercentileThreshold._get_calendar_day_window_(r_cday_index,target_cday_index,width)
#            cday_select = np.zeros(cday_shape,dtype=bool)
#            for wd in window_days:
#                cday_select = np.logical_or(cday_select,r_cday_index == wd)
#            cday_sub = cday[cday_select]
#            
#            ## a tight loop used to determine if a date is a member of a particular
#            ## window.
#            ## TODO: optimize
#            for ii,dt in enumerate(temporal.flat):
#                r_month,r_day = dt.month,dt.day
#                dt_cday = cday['index'][r_logical_and(cday['day'] == r_day,cday['month'] == r_month)]
#                select[ii] = dt_cday in cday_sub['index']
#            
#            ## calculate the percentile value for the window
#            percentile_subset = all_values[:,select]
##            assert(percentile_subset.shape[0] == len(years)*width)
#            cday['percentile'][target_cday_index] = np.percentile(percentile_subset,percentile,axis=1)
#            
#        return(cday)

    @staticmethod
    def get_daily_percentile(arr, dt_arr, percentile, window_width, only_leap_years=False):
        
        '''
        Creates a dictionary with keys=calendar day (month,day) and values=numpy.ndarray (2D)
        Example - to get the 2D percentile array corresponding to the 15th Mai: percentile_dict[5,15]
        
        :param arr: array of values
        :type arr: numpy.ndarray (3D) of float
        :param dt_arr: corresponding time steps vector (base period: usually 1961-1990)
        :type dt_arr: numpy.ndarray (1D) of datetime objects
        :param percentile: percentile to compute which must be between 0 and 100 inclusive
        :type percentile: int
        :param window_width: window width, must be odd
        :type window_width: int
        :param only_leap_years: option for February 29th (default: False)
        :type only_leap_years: bool
        
        :rtype: dict
    
        '''
        # we reduce the number of dimensions
        arr = arr.squeeze()
        dt_arr = dt_arr.squeeze()
        
        # step1: creation of the dictionary with all calendar days:
        dic_caldays = get_dict_caldays(dt_arr)
    
        percentile_dict = OrderedDict()
        
        dt_hour = dt_arr[0].hour # (we get hour of a date only one time, because usually the hour is the same for all dates in input dt_arr)
        
        for month in dic_caldays.keys():
            for day in dic_caldays[month]:
                
                # step2: we do a mask for the datetime vector for current calendar day (day/month)
                dt_arr_mask = get_mask_dt_arr(dt_arr, month, day, dt_hour, window_width, only_leap_years)
    
                # step3: we are looking for the indices of non-masked dates (i.e. where dt_arr_mask==False) 
                indices_non_masked = np.where(dt_arr_mask==False)[0]
                
                # step4: we subset our arr
                arr_subset = arr[indices_non_masked, :, :].squeeze()
                
                # step5: we compute the percentile for current arr_subset
                ############## WARNING: type(arr_subset) = numpy.ndarray. Numpy.percentile does not work with masked arrays,
                ############## so if arr_subset has aberrant values like 999999 or 1e+20, the result will be wrong.
                ############## Check with numpy.nanpercentile (Numpy version 1.9) !!!            
                arr_percentille_current_calday = np.percentile(arr_subset, percentile, axis=0)
                
                # step6: we add to the dictionnary...
                percentile_dict[month,day] = arr_percentille_current_calday
                
            # print 'Creating percentile dictionary: month ', month, '---> OK'
        
        # print 'Percentile dictionary is created.'
        
        return percentile_dict
        
            
    # @staticmethod
    # def _get_calendar_day_window_(cday_index,target_cday_index,width):
    #     width = int(width)
    #     try:
    #         assert(width >= 3)
    #         assert(width%2 != 0)
    #     except AssertionError:
    #         ocgis_lh(exc=ValueError('Kernel widths must be >= 3 and be oddly numbered.'),logger='calc.library')
    #
    #     stride_dim = (width-1)/2
    #     axis_length = cday_index.shape[0]
    #
    #     lower_idx = target_cday_index - stride_dim
    #     upper_idx = target_cday_index + stride_dim + 1
    #
    #     if lower_idx < 0:
    #         a = cday_index[lower_idx:]
    #         b = cday_index[0:target_cday_index]
    #         lower = np.append(a,b)
    #     else:
    #         lower = cday_index[lower_idx:target_cday_index]
    #
    #     if upper_idx > axis_length:
    #         a = cday_index[0:upper_idx-axis_length]
    #         b = cday_index[target_cday_index+1:upper_idx]
    #         upper = np.append(a,b)
    #     else:
    #         upper = cday_index[target_cday_index+1:upper_idx]
    #
    #     ret = np.append(cday_index[target_cday_index],np.append(lower,upper))
    #
    #     return(ret)