from ocgis.util.logging_ocgis import ocgis_lh
import numpy as np
from ocgis.calc.base import AbstractParameterizedFunction, \
    AbstractUnivariateSetFunction
from ocgis import constants


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
        dp_indices = np.zeros(dates.shape)
        for ii,date in enumerate(dates):
            month,day = date.month,date.day
            dp_indices[ii] = daily_percentile['index'][np.logical_and(daily_percentile['month'] == month,
                                                                      daily_percentile['day'] == day)]
        ## construct the the comparison array
        b = np.empty_like(values)
        for ii in range(b.shape[0]):
            b[ii] = daily_percentile[dp_indices[ii]]['percentile']
        
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
    
    @staticmethod
    def get_daily_percentile(all_values,temporal,percentile,width):
        '''
        :param all_values: Array holding all values to use for base percentile calculations.
        :type all_values: numpy.MaskedArray
        :param temporal: Vector holding `datetime.datetime` objects with same length as the time axis of `all_values`.
        :type temporal: numpy.ndarray
        :param percentile: Percentile threshold to use in the `numpy.percentile` calculation.
        :type percentile: int or float from 0 to 100
        :param width: Width of kernel to use for the moving window percentile.
        :type width: int, oddly-number, at least 3 or greater
        :returns: A structure array with four fields: month, day, index, and percentile.
        :rtype: numpy.ndarray
        '''
        assert(len(all_values.shape) == 5)
        
        ## collect the days and months uniquely to use for the calendar attribution
        dmap = {}
        years = set()
        for dt in temporal.flat:
            day,month,year = dt.day,dt.month,dt.year
            years.update([year])
            if month not in dmap:
                dmap[month] = []
            if day not in dmap[month]:
                dmap[month].append(day)
        
        ## this is the structure array storing date parts and percentile arrays
        cday_length = sum([len(v) for v in dmap.itervalues()])
        cday = np.zeros(cday_length,dtype=[('month',int),('day',int),('index',int),('percentile',object)])
        idx = 0
        for month in sorted(dmap):
            for day in sorted(dmap[month]):
                cday[idx] = (month,day,idx,None)
                idx += 1
        
        ## loop for each calendar day and calculate the percentile value for its
        ## data window
        r_cday_index = cday['index']
        r_logical_and = np.logical_and
        cday_shape = cday.shape[0]
        for target_cday_index in range(cday_shape):
            select = np.zeros(all_values.shape[1],dtype=bool)
            
            ## this function returns the calendar days part of the target calendar
            ## day's window. these are then used to subset the calendar day structure
            ## array.
            window_days = DynamicDailyKernelPercentileThreshold._get_calendar_day_window_(r_cday_index,target_cday_index,width)
            cday_select = np.zeros(cday_shape,dtype=bool)
            for wd in window_days:
                cday_select = np.logical_or(cday_select,r_cday_index == wd)
            cday_sub = cday[cday_select]
            
            ## a tight loop used to determine if a date is a member of a particular
            ## window.
            ## TODO: optimize
            for ii,dt in enumerate(temporal.flat):
                r_month,r_day = dt.month,dt.day
                dt_cday = cday['index'][r_logical_and(cday['day'] == r_day,cday['month'] == r_month)]
                select[ii] = dt_cday in cday_sub['index']
            
            ## calculate the percentile value for the window
            percentile_subset = all_values[:,select]
#            assert(percentile_subset.shape[0] == len(years)*width)
            cday['percentile'][target_cday_index] = np.percentile(percentile_subset,percentile,axis=1)
            
        return(cday)        
            
    @staticmethod
    def _get_calendar_day_window_(cday_index,target_cday_index,width):
        width = int(width)
        try:
            assert(width >= 3)
            assert(width%2 != 0)
        except AssertionError:
            ocgis_lh(exc=ValueError('Kernel widths must be >= 3 and be oddly numbered.'),logger='calc.library')
        
        stride_dim = (width-1)/2
        axis_length = cday_index.shape[0]
        
        lower_idx = target_cday_index - stride_dim
        upper_idx = target_cday_index + stride_dim + 1
        
        if lower_idx < 0:
            a = cday_index[lower_idx:]
            b = cday_index[0:target_cday_index]
            lower = np.append(a,b)
        else:
            lower = cday_index[lower_idx:target_cday_index]
            
        if upper_idx > axis_length:
            a = cday_index[0:upper_idx-axis_length]
            b = cday_index[target_cday_index+1:upper_idx]
            upper = np.append(a,b)
        else:
            upper = cday_index[target_cday_index+1:upper_idx]
            
        ret = np.append(cday_index[target_cday_index],np.append(lower,upper))
        
        return(ret)