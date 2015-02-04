import abc

from ocgis.exc import DefinitionValidationError


class AbstractDriver(object):
    """
    :param rd: The input request dataset object.
    :type rd: :class:`~ocgis.RequestDataset`
    """

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
        :returns: A sequence of regular expressions used to match appropriate URIs.
        :rtype: (str,)

        >>> ('.*\.shp',)
        """

    @abc.abstractproperty
    def key(self):
        str

    @abc.abstractproperty
    def output_formats(self):
        """
        :returns: A list of acceptable output formats for the driver. If this is `'all'`, then the driver's data may be
         converted to all output formats.
        :rtype: [str, ...]
        """

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
    def get_source_metadata(self):
        return dict

    @abc.abstractmethod
    def inspect(self):
        pass

    @abc.abstractmethod
    def open(self):
        return object

    @classmethod
    def validate_ops(cls, ops):
        """
        :param ops: An operation object to validate.
        :type ops: :class:`~ocgis.OcgOperations`
        :raises: DefinitionValidationError
        """

        if cls.output_formats != 'all':
            if ops.output_format not in cls.output_formats:
                msg = 'Output format not supported for driver "{0}". Supported output formats are: {1}'.format(cls.key, cls.output_formats)
                raise DefinitionValidationError('output_format', msg)

    @abc.abstractmethod
    def _get_field_(self, **kwargs):
        """Return :class:`ocgis.interface.base.field.Field`"""
