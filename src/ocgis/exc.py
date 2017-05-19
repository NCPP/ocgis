from ocgis import messages


class OcgException(Exception):
    """Base class for all OCGIS exceptions."""

    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return self.message


class OcgWarning(Warning):
    """Base class for all OCGIS warnings."""


########################################################################################################################


class BoundsAlreadyAvailableError(OcgException):
    """Raised when an attempt is made to extrapolate bounds and they are already present."""

    def __str__(self):
        msg = 'Bounds/corners already available.'
        return msg


class CannotFormatTimeError(OcgException):
    """
    Raised when datetime objects from numeric are blocked by "format_time".
    """

    def __init__(self, property_name):
        self.property_name = property_name

    def __str__(self):
        msg = 'Attempted to retrieve datetime values from "{0}" with "format_time" as "False". Set "format_time" to "True".'.format(
            self.property_name)
        return msg


class MaskedDataFound(OcgException):
    """Raised when data is masked."""

    def __str__(self):
        msg = 'Data is masked.'
        return msg


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
    def __init__(self, variable_or_name):
        try:
            name = variable_or_name.name
        except AttributeError:
            name = variable_or_name
        self.name = name

    def __str__(self):
        msg = 'Variable name already in collection: {0}'.format(self.name)
        return msg


class VariableShapeMismatch(OcgException):
    def __init__(self, variable, collection_shape):
        self.variable = variable
        self.collection_shape = collection_shape

    def __str__(self):
        msg = 'Variable with alias "{0}" has shape {1}. Collection shape is {2}'.format(self.variable.alias,
                                                                                        self.variable.shape,
                                                                                        self.collection_shape)
        return msg


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
    :type ocg_argument: :class:`ocgis.driver.definition.AbstractParameter`, str
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
        msg = 'A subset operation on variable "{0}" returned empty.'.format(self.origin)
        return msg


class AllElementsMaskedError(OcgException):
    """Raised when all elements are masked."""

    def __str__(self):
        return "All elements are masked."


class PayloadProtectedError(OcgException):
    def __init__(self, name):
        self.name = name
        super(PayloadProtectedError, self).__init__()

    def __str__(self):
        return 'Payload with name "{}" may not be loaded from source until "protected" is False.'.format(self.name)


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
        message = 'Validation failed on the parameter "{0}" with the message: {1}'.format(self.keyword, self.message)
        return message


class NoDataVariablesFound(RequestValidationError):
    """Raised when no data variables are found in the target dataset."""

    def __init__(self):
        super(NoDataVariablesFound, self).__init__('variable', messages.M1)


class GridDeficientError(OcgException):
    """Raised when a grid is missing parameters necessary to create a geometry array."""


class DimensionMismatchError(OcgException):
    """Raised when a variable's dimensions do not match those in the existing collection."""

    def __init__(self, dim_name, vc_name, message=None):
        self.dim_name = dim_name
        self.vc_name = vc_name

        super(DimensionMismatchError, self).__init__(message)

    def __str__(self):
        msg = 'The dimension "{}" does not match the dimension in variable collection "{}".'.format(self.dim_name,
                                                                                                    self.vc_name)
        return msg


class DimensionsRequiredError(OcgException):
    """Raised when a variable requires dimensions."""

    def __init__(self, message=None):
        if message is None:
            message = "Variables with dimension count greater than 0 (ndim > 0) require dimensions. Initialize the " \
                      "variable with dimensions or call 'create_dimensions' before size inquiries (i.e. ndim, shape)."
        super(DimensionsRequiredError, self).__init__(message=message)


class OcgDistError(OcgException):
    """Raised for MPI-related exceptions."""


class EmptyObjectError(OcgException):
    """Raised when an empty object is not allowed."""


class SubcommNotFoundError(OcgDistError):
    """Raised when a subcommunicator is not found."""

    def __init__(self, name):
        message = "Subcommunicator '{}' not found.".format(name)
        super(SubcommNotFoundError, self).__init__(message=message)


class SubcommAlreadyCreatedError(OcgDistError):
    """Raised when a subcommunicator name already exists."""

    def __init__(self, name):
        message = "Subcommunicator '{}' already created.".format(name)
        super(SubcommAlreadyCreatedError, self).__init__(message=message)


class CRSNotEquivalenError(OcgException):
    """Raised when coordinate systems are not equivalent (not compatible for transform)."""

    def __init__(self, lhs, rhs):
        msg = '{} is not equivalent to {}.'.format(lhs, rhs)
        super(CRSNotEquivalenError, self).__init__(message=msg)


class DimensionMapError(OcgException):
    """Raised when there is an issue with a dimension map entry."""

    def __init__(self, entry_key, message):
        msg = "Error with entry key '{}': {}".format(entry_key, message)
        super(DimensionMapError, self).__init__(message=msg)


class VariableMissingMetadataError(OcgException):
    """Raised when variable metadata cannot be found."""

    def __init__(self, variable_name):
        msg = 'Variable is missing metadata: {}'.format(variable_name)
        super(VariableMissingMetadataError, self).__init__(message=msg)
