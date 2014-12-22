import calendar
from collections import OrderedDict, defaultdict
import itertools

import numpy as np

from datetime import datetime
from ocgis.calc import base
from ocgis import constants
from ocgis.calc.base import AbstractUnivariateFunction, AbstractParameterizedFunction


class MovingWindow(AbstractUnivariateFunction, AbstractParameterizedFunction):
    key = 'moving_window'
    parms_definition = {'k': int, 'mode': str, 'operation': str}
    description = ()
    dtype = constants.np_float
    standard_name = 'moving_window'
    long_name = 'Moving Window Operation'

    _potential_operations = ('mean', 'min', 'max', 'median', 'var', 'std')

    def calculate(self, values, k=None, operation=None, mode='valid'):
        """
        Calculate ``operation`` for the set of values with window of width ``k`` centered on time coordinate `t`. The
        ``mode`` may either be ``'valid'`` or ``'same'`` following the definition here: http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html.
        The window width ``k`` must be an odd number and >= 3. Supported operations are: mean, min, max, median, var,
        and std.

        :param values: Array containing variable values.
        :type values: :class:`numpy.ma.core.MaskedArray`
        :param k: The width of the moving window. ``k`` must be odd and greater than three.
        :type k: int
        :param operation: The NumPy-based array operation to perform on the set of window values.
        :type operation: str in ('mean', 'min', 'max', 'median', 'var', 'std')
        :param str mode: See: http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html. The output mode
         ``full`` is not supported.
        :rtype: :class:`numpy.ma.core.MaskedArray`
        :raises: AssertionError, NotImplementedError
        """

        # 'full' is not supported as this would add dates to the temporal dimension
        assert(mode in ('same', 'valid'))
        assert(values.ndim == 5)
        assert(operation in self._potential_operations)

        operation = getattr(np, operation)

        fill = values.copy()

        # perform the moving average on the time axis
        axes = [0, 2]
        itrs = [range(values.shape[axis]) for axis in axes]
        for ie, il in itertools.product(*itrs):
            values_slice = values[ie, :, il, :, :]
            build = True
            for origin, values_kernel in self._iter_kernel_values_(values_slice, k, mode=mode):
                if build:
                    # if only the valid region is returned, this index will determine where the start index for the
                    # field/fill slice is
                    idx_start = origin
                    build = False
                fill[ie, origin, il, :, :] = operation(values_kernel, axis=0)

        if mode == 'valid':
            # slice the field and fill arrays
            self.field = self.field[:, idx_start:origin+1, :, :, :]
            fill = fill[:, idx_start:origin+1, :, :, :]
        elif mode == 'same':
            pass
        else:
            raise NotImplementedError(mode)

        return fill

    @staticmethod
    def _iter_kernel_values_(values, k, mode='valid'):
        """
        :param values: The three-dimensional array from which to extract window values.
        :type values: :class:`numpy.core.multiarray.ndarray` axes = (time, row, column)
        :param int k: The width of window. Must be odd and greater than 3.
        :param str mode: If ``valid``, return only values with a full window overlap. If ``same``, return all values
         regardless of window overlap.
        :returns: tuple(int, :class:`numpy.core.multiarray.ndarray`)
        :raises: AssertionError, NotImplementedError
        """

        assert(k % 2 != 0)
        assert(k >= 3)
        assert(values.ndim == 3)

        # used to track the current value for the centered window.
        origin = 0
        # size of one side of the window used to determine the slice for the kernel
        shift = (k - 1)/2
        # reference for the length of the value array
        shape_values = values.shape[0]

        # will return only values with a full window overlap
        if mode == 'valid':
            while True:
                start = origin - shift
                # skip slices without a full window
                if start < 0:
                    origin += 1
                    continue
                stop = origin + shift + 1
                # if the end index is greater than the length of the value array end iteration
                if stop > shape_values:
                    raise StopIteration
                yield origin, values[start:stop, :, :]
                origin += 1
        elif mode == 'same':
            while True:
                start = origin - shift
                stop = origin + shift + 1
                # return values regardless of window overlap. always start at the beginning of the array
                if start < 0:
                    start = 0
                yield origin, values[start:stop, :, :]
                origin += 1
                # stop when we've used the last array value
                if origin == shape_values:
                    raise StopIteration
        else:
            raise NotImplementedError(mode)


class DailyPercentile(base.AbstractUnivariateFunction, base.AbstractParameterizedFunction):
    key = 'daily_perc'
    parms_definition = {'percentile': float, 'window_width': int, 'only_leap_years': bool}
    description = ''
    dtype = constants.np_float
    standard_name = 'daily_percentile'
    long_name = 'Daily Percentile'

    def __init__(self, *args, **kwargs):
        super(DailyPercentile, self).__init__(*args, **kwargs)

        if self.file_only:
            self.tgd = self.field.temporal.get_grouping(['month', 'day'])
            self.field.temporal = self.tgd

    def calculate(self, values, percentile=None, window_width=None, only_leap_years=False):
        assert(values.shape[0] == 1)
        assert(values.shape[2] == 1)
        # assert(self.tgd is not None)
        # dtype = [('month', int), ('day', int), ('value', object)]
        arr = values[0, :, 0, :, :]
        assert(arr.ndim == 3)
        dt_arr = self.field.temporal.value_datetime
        dp = self.get_daily_percentile(arr, dt_arr, percentile, window_width, only_leap_years=only_leap_years)
        shape_fill = list(values.shape)
        shape_fill[1] = len(dp)
        fill = np.zeros(shape_fill, dtype=self.dtype)
        fill = np.ma.array(fill, mask=False)
        tgd = self.field.temporal.get_grouping(['month', 'day'])
        month_day_map = {(dt.month, dt.day): ii for ii, dt in enumerate(tgd.value_datetime)}
        for key, value in dp.iteritems():
            fill[0, month_day_map[key], 0, :, :] = value
        self.field.temporal = tgd
        for idx in range(fill.shape[1]):
            fill.mask[0, idx, 0, :, :] = values.mask[0, 0, 0, :, :]
        return fill

    @staticmethod
    def get_daily_percentile_from_request_dataset(rd, alias=None):
        ret = {}
        alias = alias or rd.alias
        field = rd.get()
        dt = field.temporal.value_datetime
        value = field.variables[alias].value
        for idx in range(len(dt)):
            curr = dt[idx]
            ret[(curr.month, curr.day)] = value[0, idx, 0, :, :]
        return ret

    def get_daily_percentile(self, arr, dt_arr, percentile, window_width, only_leap_years=False):
        """
        Creates a dictionary with keys=calendar day (month,day) and values=numpy.ndarray (2D)
        Example - to get the 2D percentile array corresponding to the 15th May: percentile_dict[5,15]

        :param arr: array of values
        :type arr: :class:`numpy.ndarray` (3D) of float
        :param dt_arr: Corresponding time steps vector (base period: usually 1961-1990).
        :type dt_arr: :class:`numpy.ndarray` (1D) of :class:`datetime.datetime` objects
        :param percentile: Percentile to compute which must be between 0 and 100 inclusive.
        :type percentile: int
        :param window_width: Window width - must be odd.
        :type window_width: int
        :param only_leap_years: Option for February 29th. If ``True``, use only leap years when computing the basis.
        :type only_leap_years: bool
        :rtype: dict
        """

        # we reduce the number of dimensions
        if arr.ndim == 5:
            arr = arr[0, :, 0, :, :]
        elif arr.ndim == 3:
            pass
        else:
            raise NotImplementedError(arr.ndim)
        dt_arr = dt_arr.squeeze()

        # step1: creation of the dictionary with all calendar days:
        dic_caldays = self.get_dict_caldays(dt_arr)

        percentile_dict = OrderedDict()

        dt_hour = dt_arr[0].hour # (we get hour of a date only one time, because usually the hour is the same for all dates in input dt_arr)

        for month in dic_caldays.keys():
            for day in dic_caldays[month]:

                # step2: we do a mask for the datetime vector for current calendar day (day/month)
                dt_arr_mask = self.get_mask_dt_arr(dt_arr, month, day, dt_hour, window_width, only_leap_years)

                # step3: we are looking for the indices of non-masked dates (i.e. where dt_arr_mask==False)
                indices_non_masked = np.where(dt_arr_mask==False)[0]

                # step4: we subset our arr
                arr_subset = arr[indices_non_masked, :, :]

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

    @staticmethod
    def get_dict_caldays(dt_arr):
        """
        Create a dictionary of calendar days, where keys=months, values=days.

        :param dt_arr: time steps vector
        :type dt_arr: :class:`numpy.core.multiarray.ndarray` (1D) of :class:`datetime.datetime` objects
        :rtype: dict
        """

        dic = defaultdict(list)

        for dt in dt_arr:
            dic[dt.month].append(dt.day)

        for key in dic.keys():
            dic[key] = list(set(dic[key]))

        return dic

    @staticmethod
    def get_masked(current_date, month, day, hour, window_width, only_leap_years):
        """
        Returns ``True`` if ``current_date`` is not in the window centered on the given calendar day (month-day). Returns
        ``False``, if it enters in the window.

        :param current_date: The date to check for inclusion in a given window.
        :type current_date: :class:`datetime.datetime`
        :param month: Month of the corresponding calendar day.
        :type month: int
        :param day: Day of the corresponding calendar day.
        :type day: int
        :param hour: Hour of the current day.
        :type hour: int
        :param window_width: Window width - must be odd.
        :type window_width: int
        :param only_leap_years: Option for February 29th. If ``True``, use only date from other leap years when constructing
         the comparison basis.
        :type only_leap_years: bool
        :rtype: bool (if ``True``, the date will be masked)
        """

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

    def get_mask_dt_arr(self, dt_arr, month, day, dt_hour, window_width, only_leap_years):
        """
        Creates a binary mask for a datetime vector for a given calendar day (month-day).

        :param dt_arr: Time steps vector.
        :type dt_arr: :class:`numpy.ndarray` (1D) of :class:`datetime.datetime` objects
        :param month: Month of a calendar day.
        :type month: int
        :param day: Day of a calendar day.
        :type day: int
        :param window_width: Window width - must be odd.
        :type window_width: int
        :param only_leap_years: Option for February 29th. If ``True``, use only leap years when constructing the basis.
        :type only_leap_years: bool
        :rtype: :class:`numpy.ndarray` (1D)
        """

        mask = np.array([self.get_masked(dt, month, day, dt_hour, window_width, only_leap_years) for dt in dt_arr])
        return mask

    @staticmethod
    def get_year_list(dt_arr):
        """
        Just to get a list of all years conteining in time steps vector (dt_arr).
        """

        year_list = []
        for dt in dt_arr:
            year_list.append(dt.year)

        year_list = list(set(year_list))

        return year_list



class FrequencyPercentile(base.AbstractUnivariateSetFunction,base.AbstractParameterizedFunction):
    key = 'freq_perc'
    parms_definition = {'percentile':float}
    description = 'The percentile value along the time axis. See: http://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.percentile.html.'
    dtype = constants.np_float
    standard_name = 'frequency_percentile'
    long_name = 'Frequency Percentile'
    
    def calculate(self,values,percentile=None):
        '''
        :param percentile: Percentile to compute.
        :type percentile: float on the interval [0,100]
        '''
        ret = np.percentile(values,percentile,axis=0)
        return(ret)


class Max(base.AbstractUnivariateSetFunction):
    description = 'Max value for the series.'
    key = 'max'
    dtype = constants.np_float
    standard_name = 'max'
    long_name = 'max'
    
    def calculate(self,values):
        return(np.ma.max(values,axis=0))


class Min(base.AbstractUnivariateSetFunction):
    description = 'Min value for the series.'
    key = 'min'
    dtype = constants.np_float
    standard_name = 'min'
    long_name = 'Min'
    
    def calculate(self,values):
        return(np.ma.min(values,axis=0))

    
class Mean(base.AbstractUnivariateSetFunction):
    description = 'Compute mean value of the set.'
    key = 'mean'
    dtype = constants.np_float
    standard_name = 'mean'
    long_name = 'Mean'
    
    def calculate(self,values):
        return(np.ma.mean(values,axis=0))
    
    
class Median(base.AbstractUnivariateSetFunction):
    description = 'Compute median value of the set.'
    key = 'median'
    dtype = constants.np_float
    standard_name = 'median'
    long_name = 'median'
    
    def calculate(self,values):
        return(np.ma.median(values,axis=0))
    
    
class StandardDeviation(base.AbstractUnivariateSetFunction):
    description = 'Compute standard deviation of the set.'
    key = 'std'
    dtype = constants.np_float
    standard_name = 'standard_deviation'
    long_name = 'Standard Deviation'
    
    def calculate(self,values):
        return(np.ma.std(values,axis=0))
