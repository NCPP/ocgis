from collections import OrderedDict


class Attributes(object):
    """
    Adds an ``attrs`` attribute and writes to an open netCDF object.

    :param dict attrs: A dictionary of arbitrary attributes to write to a netCDF object.
    """

    def __init__(self, attrs=None):
        self.attrs = attrs

    @property
    def attrs(self):
        return self._attrs

    @attrs.setter
    def attrs(self, value):
        if value is None:
            self._attrs = OrderedDict()
        else:
            self._attrs = OrderedDict(value)

    def write_attributes_to_netcdf_object(self, target):
        """
        :param target: A netCDF data object to write attributes to.
        :type target: :class:`netCDF4.Variable` or :class:`netCDF4.Dataset`
        """

        for k, v in self.attrs.iteritems():
            setattr(target, k, v)
