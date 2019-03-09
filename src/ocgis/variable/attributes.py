from collections import OrderedDict
from warnings import warn

import six
from ocgis.base import AbstractOcgisObject
from ocgis.exc import OcgWarning


class Attributes(AbstractOcgisObject):
    """
    Adds an ``attrs`` dictionary. Always converts dictionaries to :class:`collections.OrderedDict` objects.

    :param dict attrs: A dictionary of attribute name/value pairs.
    """

    def __init__(self, attrs=None):
        self._attrs = None
        self.attrs = attrs

    @property
    def attrs(self):
        """
        Get or set the attributes dictionary.
        
        :param dict value: The dictionary of attributes. Always converted to an :class:`collections.OrderedDict`.
        :rtype: :class:`collections.OrderedDict`
        """
        if self._attrs is None:
            self._attrs = self._get_attrs_()
        return self._attrs

    @attrs.setter
    def attrs(self, value):
        if value is not None:
            value = OrderedDict(value)
        self._attrs = value

    def write_attributes_to_netcdf_object(self, target):
        """
        :param target: The attribute write target.
        :type target: :class:`netCDF4.Variable` | :class:`netCDF4.Dataset`
        """

        for k, v in self.attrs.items():
            if k.startswith('_') or v is None:
                # Do not write private/protected attributes used by netCDF or None values.
                continue
            try:
                if isinstance(v, six.string_types):
                    v = str(v)
                if k == 'axis' and isinstance(v, six.string_types):
                    # HACK: Axis writing was causing a strange netCDF failure.
                    target.axis = str(v)
                else:
                    target.setncattr(str(k), v)
            except UnicodeError:
                # Just write the attribute if we encounter a unicode error.
                msg = "UnicodeError encountered when converting the value of attribute with name '{}' to a string. " \
                      "Sending the value to the netCDF API".format(k)
                warn(OcgWarning(msg))
                target.setncattr(k, v)

    def _get_attrs_(self):
        return OrderedDict()
