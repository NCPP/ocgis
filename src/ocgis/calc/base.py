import abc
import itertools
import logging
from collections import OrderedDict
from copy import deepcopy

import numpy as np

from ocgis import constants
from ocgis import env
from ocgis.base import get_variables
from ocgis.constants import TagNames, DimensionMapKeys, HeaderNames
from ocgis.exc import SampleSizeNotImplemented, DefinitionValidationError, UnitsValidationError
from ocgis.util.conformer import conform_array_by_dimension_names
from ocgis.util.helpers import get_default_or_apply, get_iter
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.util.units import get_are_units_equal_by_string_or_cfunits
from ocgis.variable.base import Variable, VariableCollection

# Standard dimension order for data arrays.
STANDARD_DIMENSIONS = (DimensionMapKeys.REALIZATION, DimensionMapKeys.TIME, DimensionMapKeys.LEVEL,
                       DimensionMapKeys.Y, DimensionMapKeys.X)


class AbstractFunction(object):
    """
    Required class attributes to overload:

    * **description** (str): A arbitrary length string describing the calculation.
    * **key** (str): The function's unique string identifier.
    * **standard_name** (str): Standard name to store in output metadata.
    * **long_name** (str): Long name description to store in output metadata.

    :param alias: The string identifier to use for the calculation.
    :type alias: str
    :param dtype: The output data type. Set this to ``'int'`` or ``'float'`` to use the default datatype for the output
     format and NumPy installation (recommended). If a specific NumPy type is needed, provide the string representation
     of the type (i.e. ``'int32'``).
    :type dtype: str or :class:`numpy.core.multiarray.dtype`
    :param field: The field object over which the calculation is applied.
    :type field: :class:`ocgis.interface.base.Field`
    :param file_only: If ``True`` pass through but compute output sizes, etc.
    :type file_only: bool
    :param vc: The :class:`ocgis.interface.base.variable.VariableCollection` to append output calculation arrays to.
     If ``None`` a new collection will be created.
    :type vc: :class:`ocgis.interface.base.variable.VariableCollection`
    :param parms: A dictionary of parameter values. The includes any parameters for the calculation.
    :type parms: dict
    :param tgd: An instance of :class:`ocgis.interface.base.dimension.temporal.TemporalGroupDimension`.
    :type tgd: :class:`ocgis.interface.base.dimension.temporal.TemporalGroupDimension`
    :param calc_sample_size: If ``True``, also compute sample sizes for the calculation.
    :type calc_sample_size: bool
    :param meta_attrs: Contains overloads for variable and/or field attribute values.
    :type meta_attrs: :class:`ocgis.driver.parms.definition_helpers.MetadataAttributes`
    :param str tag: The tag to use for variable iteration on the source field (the source variables for calculation).
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def description(self):
        pass

    Group = None

    @abc.abstractproperty
    def key(self):
        pass

    #: The calculation's long name.
    @abc.abstractproperty
    def long_name(self):
        pass

    #: The calculation's standard name.
    @abc.abstractproperty
    def standard_name(self):
        pass

    #: The output data type is NumPy's default float representation. This may be overloaded by subclasses.
    dtype_default = 'float'

    #: The calculation's output units. Modify :meth:`get_output_units` for more complex units calculations. If the units
    #: are left as the default '_input_' then the input variable units are maintained. Otherwise, they will be set to
    #: units attribute value. The string flag is used to allow ``None`` units to be applied.
    units = '_input_'

    # standard empty dictionary to use for calculation outputs when the operation is file only
    _empty_fill = {'fill': None, 'sample_size': None}

    def __init__(self, alias=None, dtype=None, field=None, file_only=False, vc=None, parms=None, tgd=None,
                 calc_sample_size=False, fill_value=None, meta_attrs=None, tag=TagNames.DATA_VARIABLES,
                 spatial_aggregation=False):

        self._curr_variable = None
        self._current_conformed_array = None
        self._dtype = dtype

        self.alias = alias or self.key
        self.fill_value = fill_value
        self.vc = vc or VariableCollection()
        self.field = field
        self.file_only = file_only
        self.parms = get_default_or_apply(parms, self._format_parms_, default={})
        self.tgd = tgd
        self.calc_sample_size = calc_sample_size
        self.meta_attrs = deepcopy(meta_attrs)
        self.tag = tag
        self.spatial_aggregation = spatial_aggregation

    @property
    def dtype(self):
        return self._dtype

    def aggregate_spatial(self, values, weights):
        """
        Optional method overload for spatial aggregation.
        :param values: The input array with dimensions `(m, n)`.
        :type values: :class:`numpy.ma.core.MaskedArray`
        :param weights: The input weights array with dimension matching ``value``.
        :type weights: :class:`numpy.core.multiarray.ndarray`
        :rtype: :class:`numpy.ma.core.MaskedArray`
        """

        ret = np.ma.average(np.squeeze(values), weights=weights)
        return ret

    def aggregate_temporal(self, values, **kwargs):
        """
        Optional method to overload for temporal aggregation.

        :param values: The input five-dimensional array.
        :type values: :class:`numpy.ma.core.MaskedArray`
        """

        return np.ma.mean(values, axis=0)

    @abc.abstractmethod
    def calculate(self, values, **kwargs):
        """
        The calculation method to overload. Values are explicitly passed to avoid dereferencing. Reducing along the
        time axis is required (i.e. axis=0).

        :param values: A three-dimensional array with dimensions (time, row, column).
        :type values: :class:`numpy.ma.MaskedArray`
        :param kwargs: Any keyword parameters for the function.
        :rtype: :class:`numpy.ma.MaskedArray`
        """

        pass

    def execute(self):
        """
        Execute the computation over the input field.

        :rtype: :class:`ocgis.interface.base.variable.VariableCollection`
        """

        # call the subclass execute method
        self._execute_()
        # allow the field metadata to be modified
        self.set_field_metadata()
        return self.vc

    @classmethod
    def get_default_dtype(cls):
        """
        :returns: The data type for the calculation.
        :rtype: type
        """

        cls_dtype = cls.dtype_default
        if cls_dtype == 'int':
            ret = env.NP_INT
        elif cls_dtype == 'float':
            ret = env.NP_FLOAT
        else:
            ret = cls_dtype

        return ret

    def get_fill_variable(self, archetype, name, dimensions, file_only=False, dtype=None, add_repeat_record=True,
                          add_repeat_record_archetype_name=True, variable_value=None):
        """
        Initialize a return variable for a calculation.

        :param archetype: An archetypical variable to use for the creation of the output variable.
        :type archetype: :class:`ocgis.Variable`
        :param str name: Name of the output variable.
        :param dimensions: Dimension tuple for the variable creation. The dimensions from `archetype` or not used
         because output dimensions are often different. Temporal grouping is an example of this.
        :type dimensions: tuple(:class:`ocgis.Dimension`, ...)
        :param bool file_only: If `True`, this is a file-only operation and no value should be allocated.
        :param type dtype: The data type for the output variable.
        :param bool add_repeat_record: If `True`, add a repeat record to the variable containing the calculation key.
        :param add_repeat_record_archetype_name: If `True`, add the `archetype` name repeat record.
        :param variable_value: If not `None`, use this as the variable value during initialization.
        :return: :class:`ocgis.Variable`
        """
        # If a default data type was provided at initialization, use this value otherwise use the data type from the
        # input value.
        if dtype is None:
            if self.dtype is None:
                dtype = archetype.dtype
            else:
                dtype = self.get_default_dtype()

        if self.fill_value is None:
            fill_value = archetype.fill_value
        else:
            fill_value = self.fill_value

        attrs = OrderedDict()
        attrs['standard_name'] = self.standard_name
        attrs['long_name'] = self.long_name
        units = self.get_output_units(archetype)

        if add_repeat_record:
            repeat_record = [(HeaderNames.CALCULATION_KEY, self.key)]
            if add_repeat_record_archetype_name:
                repeat_record.append((HeaderNames.CALCULATION_SOURCE_VARIABLE, archetype.name))
        else:
            repeat_record = None

        fill = Variable(name=name, dimensions=dimensions, dtype=dtype, fill_value=fill_value, attrs=attrs, units=units,
                        repeat_record=repeat_record, value=variable_value)

        if not file_only and variable_value is None:
            fill.allocate_value()

        return fill

    @staticmethod
    def get_fill_sample_size_variable(archetype, file_only):
        attrs = OrderedDict()
        attrs['standard_name'] = constants.DEFAULT_SAMPLE_SIZE_STANDARD_NAME
        attrs['long_name'] = constants.DEFAULT_SAMPLE_SIZE_LONG_NAME
        fill_sample_size = Variable(name='n_{}'.format(archetype.name), dimensions=archetype.dimensions,
                                    attrs=attrs, dtype=np.int32)
        if not file_only:
            fill_sample_size.allocate_value()

        return fill_sample_size

    def get_function_definition(self):
        """
        Return a dictionary representation of the function definition.

        :rtype: dict
        """

        ret = {'key': self.key, 'alias': self.alias, 'parms': self.parms}
        return ret

    def get_output_units(self, variable):
        """
        Get the output units.

        :type variable: :class:`ocgis.interface.base.variable.Variable`
        :rtype: str
        """

        if self.units == '_input_':
            ret = variable.units
        else:
            ret = self.units

        return ret

    def get_sample_size(self, values):
        """
        Calculate the sample size of the temporal group based on the mask.

        :type values: :class:`numpy.ma.core.MaskedArray`
        :rtype: :class:`numpy.ma.core.MaskedArray`
        """

        to_sum = np.invert(values.mask)
        ret = np.ma.sum(to_sum, axis=0)
        ret = np.ma.array(ret, mask=values.mask[0, :, :])
        return ret

    @staticmethod
    def get_variable_value(variable):
        """
        Select the appropriate value to use for the calculation. This will return the raw or aggregated values. If the
        data is not spatially aggregated, then this will have no effect.

        :param variable: :class:`ocgis.interface.base.variable.Variable`
        :rtype: :class:`numpy.ma.core.MaskedArray`
        """

        return variable.masked_value

    def iter_calculation_targets(self, variable_names=None, yield_calculation_name=True, validate_units=True):
        if variable_names is None:
            seq = list(self.field.get_by_tag(self.tag))
        else:
            seq = get_variables(variable_names, self.field)

        if len(seq) == 0:
            raise ValueError('No data variables available on the field.')

        for variable in seq:
            if validate_units:
                self.validate_units(variable)
            self._curr_variable = variable
            if yield_calculation_name:
                # Append the variable to the calculation if there are more than one to calculate across.
                if len(seq) > 1:
                    calculation_name = '{}_{}'.format(self.alias, variable.name)
                else:
                    calculation_name = self.alias
            else:
                calculation_name = None

            if yield_calculation_name:
                yield variable, calculation_name
            else:
                yield variable

    def set_field_metadata(self):
        """
        Modify the :class:~`ocgis.interface.base.field.Field` metadata dictionary.
        """

        pass

    def set_variable_metadata(self, variable):
        """
        Set variable level metadata. If units are to be updated, this must be done on the "units" attribute of the
        variable as this value is read directly from the variable object during conversion.
        """
        pass

    @classmethod
    def validate(cls, ops):
        """
        Optional method to overload that validates the input :class:`ocgis.OcgOperations`.

        :type ops: :class:`ocgis.driver.operations.OcgOperations`
        :raises: :class:`ocgis.exc.DefinitionValidationError`
        """

    @classmethod
    def validate_definition(cls, definition):
        """
        Method to validate calculation definitions passed to operations.

        :param dict definition: The dictionary definition for the function as returned from
         :attr:`~ocgis.driver.parms.definition.Calc.value`.
        :raises: :class:`ocgis.exc.DefinitionValidationError`
        """

    def validate_units(self, *args, **kwargs):
        """Optional method to overload for units validation at the calculation level."""

    def _add_to_collection_(self, value):
        """
        :param str units: The units for the derived variable.

        >>> units = 'kelvin'

        :param value: The value for the derived variable.
        :type value: :class:`numpy.ma.core.MaskedArray` or dict

        >>> import numpy as np
        >>> value = np.zeros((2, 3, 4, 5, 6))
        >>> value = np.ma.array(value)

        *or*

        >>> sample_size = value.copy()
        >>> sample_size[:] = 5
        >>> value = {'fill': value, 'sample_size': sample_size}

        :param parent_variables: A variable collection containing variable data used to derive the current output.
        :type parent_variables: :class:`ocgis.interface.base.variable.VariableCollection`
        :param str alias: The alias of the derived variable.
        :param type dtype: The type of the derived variable.
        :param fill_value: The mask fill value of the derived variable.
        """
        dv = value['fill']
        sample_size = value.get('sample_size', None)

        # allow more complex manipulations of metadata
        self.set_variable_metadata(dv)
        # overload the metadata attributes with any provided
        if self.meta_attrs is not None:
            dv.attrs.update(self.meta_attrs.value['variable'])
            self.field.attrs.update(self.meta_attrs.value['field'])
        self.vc.add_variable(dv)

        # add the sample size if it is present in the fill dictionary
        if sample_size is not None:
            self.vc.add_variable(sample_size)

    @abc.abstractmethod
    def _execute_(self):
        pass

    def _format_parms_(self, values):
        return values

    def _get_dimension_crosswalk_(self, variable):
        crosswalk = []
        for dim in variable.dimensions:
            found = False
            for k, v in self.field.dimension_map.items():
                names = v.get(DimensionMapKeys.NAMES, [])
                if v.get(DimensionMapKeys.VARIABLE) is not None and dim.name in names:
                    crosswalk.append(k)
                    found = True
                    break
            if not found:
                crosswalk.append(dim.name)

        return crosswalk

    def _get_extra_indices_itr_and_src_names_(self, crosswalk, variable_shape):
        extra = [dn for dn in crosswalk if dn not in STANDARD_DIMENSIONS]
        extra = {e: {'index': crosswalk.index(e)} for e in extra}
        for v in extra.values():
            v['size'] = variable_shape[v['index']]

        # Remove the extra dimensions. The only dimension names remaining are standard field names.
        src_names_extra_removed = [dn for dn in crosswalk if dn not in extra]
        # Handles extra dimensions in the outer loop.
        izl = [itertools.izip_longest([v['index']], range(v['size']), fillvalue=v['index']) for v in extra.values()]
        itr = itertools.product(*izl)
        return itr, src_names_extra_removed

    def _get_parms_(self):
        return self.parms

    def _get_temporal_agg_fill_(self, variable, name, file_only, f=None, parms=None,
                                add_repeat_record_archetype_name=True):
        # Depending on the computational class we may be just aggregating temporally or actually executing the
        # calculation.
        f = f or self.calculate

        # Allow parameter overloading.
        parms = parms or self.parms

        # Variable dimension names remapped to standard field dimension names.
        crosswalk = self._get_dimension_crosswalk_(variable)

        # Create the fill variable.
        time_axis = crosswalk.index(DimensionMapKeys.TIME)
        fill_dimensions = list(variable.dimensions)
        fill_dimensions[time_axis] = self.tgd.dimensions[0]
        fill = self.get_fill_variable(variable, name, fill_dimensions, file_only,
                                      add_repeat_record_archetype_name=add_repeat_record_archetype_name)

        # Create the sample size variable.
        if self.calc_sample_size:
            fill_sample_size = self.get_fill_sample_size_variable(fill, file_only)
        else:
            fill_sample_size = None

        if not file_only:
            # Get value arrays.
            arr = self.get_variable_value(variable)
            arr_fill = self.get_variable_value(fill)
            if self.calc_sample_size:
                arr_fill_sample_size = self.get_variable_value(fill_sample_size)
            else:
                arr_fill_sample_size = None

            # Extra dimensions are not standard field dimensions.
            for yld in self._iter_conformed_arrays_(crosswalk, variable.shape, arr, arr_fill, arr_fill_sample_size):
                if not self.calc_sample_size:
                    carr, carr_fill = yld
                    carr_fill_sample_size = None
                else:
                    carr, carr_fill, carr_fill_sample_size = yld

                # Some variables need access to the entire 5d conformed array.
                self._current_conformed_array = carr

                # Standard field dimension iterators.
                standard_itrs = [range(carr.shape[ii]) for ii in [0, 2]]
                standard_itrs.append(range(self.tgd.shape[0]))

                # Execute the calculation.
                for ir, il, it in itertools.product(*standard_itrs):
                    self._curr_group = self.tgd.dgroups[it]
                    calculation_value = carr[ir, self._curr_group, il, :, :]
                    assert calculation_value.ndim == 3
                    res = f(calculation_value, **parms)
                    if self.spatial_aggregation:
                        # Weights are not currently conformed so should not be used.
                        res = self.aggregate_spatial(res, None)
                        carr_fill.data[ir, it, il, :, :] = res
                    else:
                        try:
                            carr_fill.data[ir, it, il, :, :] = res.data
                        except ValueError:
                            if not hasattr(res, 'mask'):
                                raise ValueError('Array return from calculation is not a masked array.')
                        else:
                            carr_fill.mask[ir, it, il, :, :] = res.mask

                    if self.calc_sample_size:
                        ss = self.get_sample_size(calculation_value)
                        carr_fill_sample_size.data[ir, it, il, :, :] = ss.data
                        carr_fill_sample_size.mask[ir, it, il, :, :] = ss.mask

            # Setting the values ensures the mask is updated on the output variables.
            fill.set_value(arr_fill)
            if self.calc_sample_size:
                fill_sample_size.set_value(arr_fill_sample_size)
                fill_sample_size.set_mask(fill.get_mask())

        return {'fill': fill, 'sample_size': fill_sample_size}

    def _iter_conformed_arrays_(self, crosswalk, variable_shape, arr, arr_fill, arr_fill_sample_size):
        # Allow sample size array to be set to None.
        if arr_fill_sample_size is None:
            calc_sample_size = False
        else:
            calc_sample_size = self.calc_sample_size

        itr_extra_indices, src_names_extra_removed = self._get_extra_indices_itr_and_src_names_(crosswalk,
                                                                                                variable_shape)

        # Loop for the extra dimensions.
        for indices in itr_extra_indices:
            # Slice for the extra dimensions.
            slc = [slice(None)] * arr.ndim
            for ii in indices:
                slc[ii[0]] = ii[1]
            extras_removed = arr.__getitem__(slc)
            extras_removed_fill = arr_fill.__getitem__(slc)
            if calc_sample_size:
                extras_removed_fill_sample_size = arr_fill_sample_size.__getitem__(slc)

            # Swap axes for the calculation values, the fill array for the calculation result, and (potentially) the
            # sample size.
            carr, carr_fill = [conform_array_by_dimension_names(t, src_names_extra_removed, STANDARD_DIMENSIONS)
                               for t in [extras_removed, extras_removed_fill]]
            if calc_sample_size:
                carr_fill_sample_size = conform_array_by_dimension_names(extras_removed_fill_sample_size,
                                                                         src_names_extra_removed,
                                                                         STANDARD_DIMENSIONS)

            if not calc_sample_size:
                yld = (carr, carr_fill)
            else:
                yld = (carr, carr_fill, carr_fill_sample_size)

            yield yld

    def _set_derived_variable_alias_(self, dv, parent_variables):
        """
        Set the alias of the derived variable.
        """

        if len(self.field.variables) > 1:
            original_alias = dv.alias
            dv.alias = '{0}_{1}'.format(dv.alias, parent_variables[0].alias)
            msg = 'Alias updated to maintain uniqueness. Changing "{0}" to "{1}".'.format(original_alias, dv.alias)
            ocgis_lh(logger='calc.base', level=logging.WARNING, msg=msg)


class AbstractFieldFunction(AbstractFunction):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def calculate(self, **kwargs):
        """
        Add variables to the internal variable collection. These variables will be treated as data variables in the
        output field. Clients are responsible for creating well-formed variables with appropriate units, etc. as this
        function type bypasses most output variable preparation steps.

        :param kwargs: Typically keyword arguments to multivariate and/or parameterized functions.
        """

    def _execute_(self):
        return self.calculate(**self.parms)


class AbstractUnivariateFunction(AbstractFunction):
    """
    Base class for functions accepting a single univariate input.
    """

    # Some calculations use a temporal group dimension but do not want any temporal aggregation to occur after the
    # calculation.
    should_temporally_aggregate = True

    __metaclass__ = abc.ABCMeta
    #: Optional sequence of acceptable string units definitions for input variables. If this is set to ``None``, no unit
    #: validation will occur.
    required_units = None

    def __init__(self, *args, **kwargs):
        super(AbstractUnivariateFunction, self).__init__(*args, **kwargs)

        if self.calc_sample_size and self.tgd is None:
            msg = 'Sample sizes not relevant for scalar transforms with no temporal grouping. Setting to False.'
            ocgis_lh(msg=msg, logger='calc.base', level=logging.WARN)
            self.calc_sample_size = False

    def validate_units(self, variable):
        if self.required_units is not None:
            matches = [get_are_units_equal_by_string_or_cfunits(variable.units, target, try_cfunits=True) \
                       for target in self.required_units]
            if not any(matches):
                raise UnitsValidationError(variable, self.required_units, self.key)

    def _execute_(self):
        for variable, calculation_name in self.iter_calculation_targets():
            crosswalk = self._get_dimension_crosswalk_(variable)
            fill_dimensions = variable.dimensions

            fill = self.get_fill_variable(variable, calculation_name, fill_dimensions, self.file_only)
            arr = self.get_variable_value(variable)
            arr_fill = self.get_variable_value(fill)

            for yld in self._iter_conformed_arrays_(crosswalk, variable.shape, arr, arr_fill, None):
                carr, carr_fill = yld
                res_calculation = self.calculate(carr, **self.parms)
                carr_fill.data[:] = res_calculation.data
                carr_fill.mask[:] = res_calculation.mask

            if not self.file_only:
                # Setting the values ensures the mask is updated on the output variables.
                fill.set_value(arr_fill)

            if self.tgd is not None and self.should_temporally_aggregate:
                fill = self._get_temporal_agg_fill_(fill, calculation_name, self.file_only, f=self.aggregate_temporal,
                                                    parms={})
            else:
                fill = {'fill': fill}

            self._add_to_collection_(fill)


class AbstractParameterizedFunction(AbstractFunction):
    """
    Base class for functions accepting parameters.
    """
    __metaclass__ = abc.ABCMeta

    #: Set to a tuple containing keys of required parameters. The keys correspond to keys in ``parms_definition``.
    parms_required = None

    def __init__(self, **kwargs):
        super(AbstractParameterizedFunction, self).__init__(**kwargs)
        self.parms = self._format_parms_(self.parms)

    @abc.abstractproperty
    def parms_definition(self):
        """
        A dictionary describing the input parameters with keys corresponding to parameter names and values to their
        types. Set the type to `None` for no type checking.

        >>> {'threshold': float, 'operation': str, 'basis': None}
        """
        dict

    @classmethod
    def validate_definition(cls, definition):
        AbstractFunction.validate_definition(definition)

        assert isinstance(definition, dict)
        from ocgis.ops.parms.definition import Calc

        key = constants.CALC_KEY_KEYWORDS
        if key not in definition:
            msg = 'Keyword arguments are required using the "{0}" key: {1}'.format(key, cls.parms_definition)
            raise DefinitionValidationError(Calc, msg)
        else:
            kwds = definition[key]
            try:
                required = cls.required_variables
            except AttributeError:
                # this function likely does not have required variables and is not a multivariate function
                assert not issubclass(cls, AbstractMultivariateFunction)
            else:
                kwds = kwds.copy()
                for r in required:
                    kwds.pop(r, None)

            if not set(kwds.keys()).issubset(cls.parms_definition.keys()):
                msg = 'Keyword arguments incorrect. Correct keyword arguments are: {0}'.format(cls.parms_definition)
                raise DefinitionValidationError(Calc, msg)

        if cls.parms_required is not None:
            for k in cls.parms_required:
                if k not in kwds:
                    msg = 'The keyword parameter "{0}" is required.'.format(k)
                    raise DefinitionValidationError(Calc, msg)

    def _format_parms_(self, values):
        """
        :param values: A dictionary containing the parameter values to check.
        :type values: dict[str, type]
        """

        ret = {}
        for k, v in values.iteritems():
            try:
                if isinstance(v, self.parms_definition[k]):
                    formatted = v
                else:
                    formatted = self.parms_definition[k](v)
            # likely a nonetype
            except TypeError as e:
                if self.parms_definition[k] is None:
                    formatted = v
                else:
                    ocgis_lh(exc=e, logger='calc.base')
            # likely a required variable for a multivariate calculation
            except KeyError as e:
                if k in self.required_variables:
                    formatted = values[k]
                else:
                    ocgis_lh(exc=e, logger='calc.base')
            ret.update({k: formatted})
        return ret


class AbstractUnivariateSetFunction(AbstractUnivariateFunction):
    """
    Base class for functions operating on a single variable but always reducing input data along the time dimension.
    """

    __metaclass__ = abc.ABCMeta

    def aggregate_temporal(self, *args, **kwargs):
        """
        This operations is always implicit to :meth:`~ocgis.calc.base.AbstractFunction.calculate`.
        """

        raise NotImplementedError('aggregation implicit to calculate method')

    def _execute_(self):
        for variable, calculation_name in self.iter_calculation_targets():
            # These executes a calculation with a temporal aggregation.
            fill = self._get_temporal_agg_fill_(variable, calculation_name, self.file_only)
            # Add the output to the variable collection
            self._add_to_collection_(value=fill)

    @classmethod
    def validate(cls, ops):
        if ops.calc_grouping is None:
            from ocgis.ops.parms.definition import Calc

            msg = 'Set functions must have a temporal grouping.'
            ocgis_lh(exc=DefinitionValidationError(Calc, msg), logger='calc.base')


class AbstractMultivariateFunction(AbstractFunction):
    """
    Base class for functions operating on multivariate inputs.
    """

    __metaclass__ = abc.ABCMeta
    # : Optional dictionary mapping unit definitions for required variables.
    #: For example: required_units = {'tas':'fahrenheit','rhs':'percent'}
    required_units = None
    #: If True, time aggregation is external to the calculation and will require running the standard time aggregation
    #: methods.
    time_aggregation_external = True

    def __init__(self, *args, **kwargs):
        if kwargs.get('calc_sample_size') is True:
            exc = SampleSizeNotImplemented(self.__class__,
                                           'Multivariate functions do not calculate sample size at this time.')
            ocgis_lh(exc=exc, logger='calc.base')
        else:
            AbstractFunction.__init__(self, *args, **kwargs)

    @abc.abstractproperty
    def required_variables(self):
        """
        Required property/attribute containing the list of input variables expected by the function.
        
        >>> ('tas', 'rhs')
        """

    def get_output_units(self, *args, **kwargs):
        return None

    def _get_slice_and_calculation_(self, f, ir, il, parms, value=None):
        if self.time_aggregation_external:
            ret = AbstractFunction._get_slice_and_calculation_(self, f, ir, il, parms, value=value)
        else:
            new_parms = {}
            for k, v in parms.iteritems():
                if k in self.required_variables:
                    new_parms[k] = v[ir, self._curr_group, il, :, :]
                else:
                    new_parms[k] = v
            cc = f(**new_parms)
            ret = (cc, None)
        return ret

    def _execute_(self):
        # Multivariate unit validation requires both variables as opposed to individual unit validation that typically
        # occurs in the calculation target iteration.
        self.validate_units()

        try:
            variable_names = [self.parms[r] for r in self.required_variables]
        # Try again without the parms dictionary assuming the variables are named appropriately with the parameter
        # dictionary.
        except KeyError:
            variable_names = self.required_variables

        # Get the variable calculation targets out of the field.
        itr = self.iter_calculation_targets(variable_names=variable_names, yield_calculation_name=False,
                                            validate_units=False)
        calculation_targets = {self.required_variables[idx]: var for idx, var in enumerate(itr)}

        # Synchronize the parameters that only contain the variable mappings with the other parameters passed in at
        # calculation initialization.
        keys = calculation_targets.keys()
        crosswalks = [self._get_dimension_crosswalk_(calculation_targets[k]) for k in keys]
        variable_shapes = [calculation_targets[k].shape for k in keys]
        arrs = [self.get_variable_value(calculation_targets[k]) for k in keys]
        archetype = calculation_targets[keys[0]]
        fill = self.get_fill_variable(archetype, self.alias, archetype.dimensions, self.file_only,
                                      add_repeat_record_archetype_name=False)
        fill.units = self.get_output_units()
        arr_fill = self.get_variable_value(fill)

        itrs = [self._iter_conformed_arrays_(crosswalks[idx], variable_shapes[idx], arrs[idx], arr_fill, None)
                for idx in range(len(crosswalks))]

        for yld in itertools.izip(*itrs):
            parms = {}
            for idx in range(len(keys)):
                parms[keys[idx]] = yld[idx][0]
            for k, v in self.parms.iteritems():
                if k not in self.required_variables:
                    parms.update({k: v})
            res = self.calculate(**parms)
            carr_fill = yld[0][1]
            carr_fill.data[:] = res.data
            carr_fill.mask[:] = res.mask

        if not self.file_only:
            fill.set_value(arr_fill)

        if self.tgd is not None:
            fill = self._get_temporal_agg_fill_(fill, fill.name, self.file_only, f=self.aggregate_temporal, parms={},
                                                add_repeat_record_archetype_name=False)
        else:
            fill = {'fill': fill}

        self._add_to_collection_(fill)

    @classmethod
    def validate(cls, ops):
        if ops.calc_sample_size:
            from ocgis.ops.parms.definition import CalcSampleSize

            exc = DefinitionValidationError(CalcSampleSize,
                                            'Multivariate functions do not calculate sample size at this time.')
            ocgis_lh(exc=exc, logger='calc.base')

        # ensure the required variables are present
        should_raise = False
        for c in ops.calc:
            if c['func'] == cls.key:
                kwds = c['kwds']

                # Check the required variables are keyword arguments.
                if not len(set(kwds.keys()).intersection(set(cls.required_variables))) >= 2:
                    should_raise = True
                    break

                # Ensure the mapped aliases exist.
                fnames = []
                for d in ops.dataset:
                    try:
                        for r in get_iter(d.rename_variable):
                            fnames.append(r)
                    except AttributeError:
                        # Fields do not have a rename variable attribute.
                        fnames += d.keys()
                for xx in cls.required_variables:
                    to_check = kwds[xx]
                    if to_check not in fnames:
                        should_raise = True
                break
        if should_raise:
            from ocgis.ops.parms.definition import Calc

            msg = 'These field names are missing for multivariate function "{0}": {1}.'
            exc = DefinitionValidationError(Calc, msg.format(cls.__name__, cls.required_variables))
            ocgis_lh(exc=exc, logger='calc.base')

    def validate_units(self):
        if self.required_units is not None:
            for required_variable in self.required_variables:
                alias_variable = self.parms[required_variable]
                variable = self.field[alias_variable]
                source = variable.units
                target = self.required_units[required_variable]
                match = get_are_units_equal_by_string_or_cfunits(source, target, try_cfunits=True)
                if match == False:
                    raise UnitsValidationError(variable, target, self.key)

    def _set_derived_variable_alias_(self, dv, parent_variables):
        pass


class AbstractKeyedOutputFunction(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def structure_dtype(self):
        dict
