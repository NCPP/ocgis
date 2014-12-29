import abc


class AbstractDriver(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, rd):
        self.rd = rd

    def __eq__(self, other):
        return self.key == other.key

    def __str__(self):
        return '"{0}"'.format(self.key)

    @abc.abstractproperty
    def extensions(self):
        """
        :returns: A list of acceptable extensions for this driver.
        :rtype: (str,)
        """

    @abc.abstractproperty
    def key(self):
        str

    @abc.abstractmethod
    def close(self, obj):
        pass

    @abc.abstractmethod
    def get_crs(self):
        return object

    @abc.abstractmethod
    def get_dimensioned_variables(self):
        return tuple(str, )

    def get_field(self, **kwargs):
        field = self._get_field_(**kwargs)
        # if this is a source grid for regridding, ensure the flag is updated
        if self.rd.regrid_source:
            field._should_regrid = True
        # update the assigned coordinate system flag
        if self.rd._has_assigned_coordinate_system:
            field._has_assigned_coordinate_system = True
        return field

    @abc.abstractmethod
    def _get_field_(self, **kwargs):
        """Return :class:`ocgis.interface.base.field.Field`"""

    @abc.abstractmethod
    def get_source_metadata(self):
        return dict

    @abc.abstractmethod
    def open(self):
        return object

    @abc.abstractmethod
    def inspect(self):
        pass
