import itertools
import numpy as np
from ocgis.calc import base
from ocgis import constants
from ocgis.calc.base import AbstractUnivariateFunction, AbstractParameterizedFunction
from ocgis.calc.library.math import Convolve1D
from ocgis.util.helpers import iter_array


class MovingWindow(AbstractUnivariateFunction, AbstractParameterizedFunction):
    key = 'moving_window'
    parms_definition = {'k': int, 'mode': str, 'operation': str}
    description = ()
    dtype = constants.np_float

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
