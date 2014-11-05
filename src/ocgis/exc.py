class OcgException(Exception):
    """Base class for all OCGIS exceptions."""

    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return self.message


class MultipleElementsFound(OcgException):
    """
    Raised when multiple elements are encountered in a :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
    object.

    :param sdim: The incoming spatial dimension object.
    :type sdim: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
    """

    def __init__(self, sdim):
        self.sdim = sdim

    def __str__(self):
        msg = 'Shape of the spatial dimension object is: {0}'.format(self.sdim.shape)
        return msg


class ShapeError(OcgException):
    """
    Raised when an array has an incompatible shape with an operation.
    """


class SingleElementError(ShapeError):
    """
    Raised when an operation requires more than a single element.
    """


class CalculationException(OcgException):

    def __init__(self, function_klass, message=None):
        self.function_klass = function_klass
        OcgException.__init__(self, message=message)

    def __str__(self):
        msg = 'The function class "{0}" raised an exception with message: "{1}"'.format(self.function_klass.__name__,
                                                                                        self.message)
        return (msg)


class VariableInCollectionError(OcgException):
    def __init__(self, variable):
        self.variable = variable

    def __str__(self):
        msg = 'Variable alias already in collection: {0}'.format(self.variable.alias)
        return (msg)


class SampleSizeNotImplemented(CalculationException):
    pass


class InterpreterException(OcgException):
    pass


class InterpreterNotRecognized(InterpreterException):
    pass


class EmptyIterationError(OcgException):
    def __init__(self, obj):
        self.message = 'Iteration on the object "{0}" requested, but the object is empty.'.format(obj)


class CFException(OcgException):
    pass


class ProjectionCoordinateNotFound(CFException):
    def __init__(self, target):
        self.message = 'The projection coordinate "{0}" was not found in the dataset.'.format(target)


class ProjectionDoesNotMatch(CFException):
    pass


class DimensionNotFound(CFException):

    def __init__(self, axis):
        self.axis = axis

    def __str__(self):
        msg = 'Dimension data not found for axis: {0}'.format(self.axis)
        return msg



class DefinitionValidationError(OcgException):
    """Raised when validation fails on :class:`~ocgis.OcgOperations`.
    
    :param ocg_argument: The origin of the exception.
    :type ocg_argument: :class:`ocgis.api.definition.OcgParameter`, str
    :param msg: The message related to the exception to display in the exception's template.
    :type msg: str
    """

    def __init__(self, ocg_argument, msg):
        self.ocg_argument = ocg_argument

        fmt = ('OcgOperations validation raised an exception on the argument/operation '
               '"{0}" with the message: {1}')
        try:
            msg = fmt.format(ocg_argument.name, msg)
        except AttributeError:
            try:
                msg = fmt.format(ocg_argument._name, msg)
            except AttributeError:
                msg = fmt.format(ocg_argument, msg)

        self.message = msg


class ParameterFormattingError(OcgException):
    pass


class UniqueIdNotFound(OcgException):
    def __init__(self):
        self.message = 'No unique ids found.'


class DummyDimensionEncountered(OcgException):
    pass


class ResolutionError(OcgException):
    pass


class SubsetException(OcgException):
    """Base class for all subset exceptions."""
    pass


class OcgisEnvironmentError(OcgException):
    def __init__(self, env_parm, msg):
        self.env_parm = env_parm
        self.msg = msg

    def __str__(self):
        new_msg = 'Error when setting the ocgis.env variable {0}. The message is: {1}'.format(self.env_parm.name,
                                                                                              self.msg)
        return (new_msg)


class SpatialWrappingError(OcgException):
    """Raised for errors related to wrapping or unwrapping a geographic coordinate system."""
    pass


class MaskedDataError(SubsetException):
    def __init__(self):
        self.message = 'Geometric intersection returned all masked values.'


class ExtentError(SubsetException):
    def __init__(self, message=None):
        self.message = message or 'Geometry intersection is empty.'


class TemporalExtentError(SubsetException):
    def __init__(self):
        self.message = 'Temporal subset returned empty.'


class EmptyDataNotAllowed(SubsetException):
    """Raised when the empty set for a geometry is returned and ``allow_empty`` is ``False``."""

    def __init__(self):
        self.message = 'Intersection returned empty, but empty data not allowed.'


class EmptyData(SubsetException):
    def __init__(self, message=None, origin=None):
        self.message = message or 'Empty data returned.'
        self.origin = origin


class EmptySubsetError(SubsetException):
    def __init__(self, origin=None):
        self.origin = origin

    def __str__(self):
        msg = 'A subset operation on dimension "{0}" returned empty.'.format(self.origin)
        return (msg)


class NoUnitsError(OcgException):
    """Raised when a :class:`cfunits.Units` object is constructed from ``None``."""

    def __init__(self, variable, message=None):
        self.variable = variable
        super(NoUnitsError, self).__init__(message=message)

    def __str__(self):
        if self.message is None:
            msg = 'Variable "{0}" has not been assigned units in the source metadata. Set the "units" attribute to continue.'
            msg = msg.format(self.variable)
        else:
            msg = self.message
        return (msg)


class UnitsValidationError(OcgException):
    """Raised when units validation fails."""

    def __init__(self, variable, required_units, calculation_key):
        self.variable = variable
        self.required_units = required_units
        self.calculation_key = calculation_key

    def __str__(self):
        msg = ('There was an error in units validation for calculation with key "{3}". The units on variable "{0}" '
               '(units="{2}") do not match the required units "{1}". The units should be conformed or overloaded if '
               'incorrectly attributed.').format(self.variable.alias, self.required_units, self.variable.units,
                                                 self.calculation_key)
        return msg


class IncompleteSeasonError(OcgException):
    def __init__(self, season, year=None, month=None):
        self.season = season
        self.year = year
        self.month = month

    def __str__(self):
        if self.year is not None:
            msg = 'The season specification "{0}" is missing the year "{1}".'.format(self.season, self.year + 1)
        if self.month is not None:
            msg = 'The season specification "{0}" is missing the month "{1}".'.format(self.season, self.month)
        return msg


class VariableNotFoundError(OcgException):
    """Raised when a variable name is not found in the target dataset."""

    def __init__(self, uri, variable):
        self.uri = uri
        self.variable = variable

    def __str__(self):
        msg = 'The variable "{0}" was not found in the dataset with URI: {1}'.format(self.variable, self.uri)
        return msg


class RegriddingError(OcgException):
    """Raised for exceptions related to ESMPy-enabled regridding."""
    pass


class CornersInconsistentError(RegriddingError):
    """Raised when corners are not available on all sources and/or destination fields."""
    pass


class RequestValidationError(OcgException):
    """Raised when validation fails on a parameter when creating a :class:`~ocgis.RequestDataset` object."""

    def __init__(self, keyword, message):
        self.keyword = keyword
        self.message = message

    def __str__(self):
        message = 'Validation failed on the keyword parameter "{0}" with the message: {1}'.format(self.keyword,
                                                                                                  self.message)
        return message
