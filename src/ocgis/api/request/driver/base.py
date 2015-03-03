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
        """:rtype: str"""

    @abc.abstractproperty
    def output_formats(self):
        """
        :returns: A list of acceptable output formats for the driver. If this is `'all'`, then the driver's data may be
         converted to all output formats.
        :rtype: list[str, ...]
        """

    @abc.abstractmethod
    def close(self, obj):
        """
        Close and finalize the open file object.
        """

    @abc.abstractmethod
    def get_crs(self):
        """
        :rtype: :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
        """

    @abc.abstractmethod
    def get_dimensioned_variables(self):
        """:rtype: tuple(str, ...)"""

    @abc.abstractmethod
    def get_dump_report(self):
        """
        :returns: A sequence of strings containing the metadata dump from the source request dataset.
        :rtype: list[str, ...]
        """

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
        """
        :rtype: dict
        """

    def inspect(self):
        """
        Inspect the request dataset printing information to stdout.
        """

        from ocgis.util.inspect import Inspect

        for line in Inspect(request_dataset=self.rd).get_report_possible():
            print line

    @abc.abstractmethod
    def open(self):
        """
        :rtype: object
        """

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
        """:rtype: :class:`ocgis.interface.base.field.Field`"""
