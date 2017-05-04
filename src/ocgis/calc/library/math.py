import numpy as np

from ocgis.calc import base
from ocgis.util.helpers import iter_array


class Divide(base.AbstractMultivariateFunction):
    key = 'divide'
    description = 'Divide arr1 by arr2.'
    required_variables = ['arr1', 'arr2']

    standard_name = 'divide'
    long_name = 'Divide'

    def calculate(self, arr1=None, arr2=None):
        return (arr1 / arr2)


class NaturalLogarithm(base.AbstractUnivariateFunction):
    key = 'ln'
    description = 'Compute the natural logarithm.'

    standard_name = 'natural_logarithm'
    long_name = 'Natural Logarithm'

    def calculate(self, values):
        return (np.ma.log(values))

    def get_output_units(self, *args, **kwds):
        return (None)


class Sum(base.AbstractUnivariateSetFunction):
    key = 'sum'
    description = 'Compute the algebraic sum of a series.'

    standard_name = 'sum'
    long_name = 'Sum'

    def calculate(self, values):
        return np.ma.sum(values, axis=0)

    def aggregate_spatial(self, values, weights):
        # All element values contribute in their entirety. Weights are not applied.
        return np.ma.sum(values)


class Convolve1D(base.AbstractUnivariateFunction, base.AbstractParameterizedFunction):
    key = 'convolve_1d'
    parms_definition = {'v': np.ndarray, 'mode': str}
    description = 'Perform a one-dimensional convolution for each grid element along the time axis. See: http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html'

    standard_name = 'convolve_1d'
    long_name = 'Convolution along the Time Dimension'

    def calculate(self, values, v=None, mode='same'):
        """
        :param values: Array containing variable values.
        :type values: :class:`numpy.ma.core.MaskedArray`
        :param v: The one-dimensional array to convolve with ``values``.
        :type v: :class:`numpy.core.multiarray.ndarray`
        :param str mode: The convolution mode. See: http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html.
         The output mode ``full`` is not supported.
        :rtype: :class:`numpy.ma.core.MaskedArray`
        :raises: AssertionError
        """

        # 'full' is not supported as this would add dates to the temporal dimension
        assert (mode in ('same', 'valid'))
        assert (len(values.shape) == 5)

        # just to be safe, convert the second array to the same input data types as the values
        v = v.astype(values.dtype)

        # valid will have less values than the input as this checks if the two convolved arrays completely overlap
        shape_fill = list(values.shape)
        if mode == 'valid':
            shape_fill[1] = max(values.shape[1], v.shape[0]) - min(values.shape[1], v.shape[0]) + 1
        fill = np.zeros(shape_fill)

        # perform the convolution on the time axis
        itr = iter_array(values)
        for ie, it, il, ir, ic in itr:
            a = values[ie, :, il, ir, ic]
            fill[ie, :, il, ir, ic] = np.convolve(a, v, mode=mode)

        if mode == 'valid':
            # generate the mask for the output data and convert the output to a masked array
            mask = np.empty(fill.shape, dtype=bool)
            mask[...] = values.mask[0, 0, 0, :, :]
            fill = np.ma.array(fill, mask=mask)

            # identify where the two arrays completely overlap and collect the indices to subset the field object
            # attached to the calculation object
            self.field = self.field[:, slice(0, 0 - (v.shape[0] - 1)), :, :, :]
        else:
            # same does not modify the output array size
            fill = np.ma.array(fill, mask=values.mask)

        return fill
