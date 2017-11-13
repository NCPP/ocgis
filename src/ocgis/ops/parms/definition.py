import datetime
import itertools
import logging
import os
from collections import OrderedDict
from copy import deepcopy, copy
from os.path import exists
from types import FunctionType

import numpy as np
import six
from shapely.geometry.base import BaseGeometry
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon

import ocgis
from ocgis import RequestDataset
from ocgis import constants
from ocgis import env
from ocgis.calc.eval_function import EvalFunction, MultivariateEvalFunction
from ocgis.calc.library import register
from ocgis.collection.field import Field
from ocgis.constants import WrapAction, DimensionMapKey, KeywordArgument
from ocgis.conv.base import get_converter, get_converter_map
from ocgis.driver.request.base import AbstractRequestObject
from ocgis.exc import DefinitionValidationError
from ocgis.ops.parms import base
from ocgis.ops.parms.definition_helpers import MetadataAttributes
from ocgis.spatial.geom_cabinet import GeomCabinetIterator
from ocgis.spatial.grid import Grid
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.util.units import get_units_class, get_units_object
from ocgis.variable.crs import CoordinateReferenceSystem
from ocgis.variable.geom import GeometryVariable


class Abstraction(base.StringOptionParameter):
    name = 'abstraction'
    default = 'auto'
    valid = ('point', 'polygon', 'auto')
    nullable = True

    def _get_meta_(self):
        if self.value is None:
            msg = 'Highest order geometry available used for spatial output.'
        else:
            msg = 'Spatial dimension abstracted to {0}.'.format(self.value)
        return (msg)


class AddAuxiliaryFiles(base.BooleanParameter):
    name = 'add_auxiliary_files'
    default = True
    meta_true = 'Auxiliary metadata files added. A new directory was created.'
    meta_false = 'Auxiliary metadata files not added. Output file created in the output directory.'


class AllowEmpty(base.BooleanParameter):
    name = 'allow_empty'
    default = False
    meta_true = 'Empty returns are allowed. Selection geometries not overlapping with dataset geometries are excluded from a return. Empty output data may results for absolutely no overlap.'
    meta_false = 'Empty returns NOT allowed. If a selection geometry has no intersecting geometries from the target dataset, an exception is raised.'


class Aggregate(base.BooleanParameter):
    name = 'aggregate'
    default = False
    meta_true = ('Selected geometries are aggregated (unioned), and associated '
                 'data values are area-weighted based on final area following the '
                 'spatial operation. Weights are normalized using the maximum area '
                 'of the geometry set.')
    meta_false = 'Selected geometries are not aggregated (unioned).'


class AggregateSelection(base.BooleanParameter):
    name = 'agg_selection'
    default = False
    meta_true = 'Selection geometries were aggregated (unioned).'
    meta_false = 'Selection geometries left as is.'

    def __init__(self, init_value=None, output_format=None):
        self.output_format = output_format

        # We always aggregate the selection geometries if this is netCDF output.
        if self.output_format == constants.OutputFormatName.NETCDF:
            init_value = True

        super(AggregateSelection, self).__init__(init_value=init_value)


class Backend(base.StringOptionParameter):
    name = 'backend'
    default = 'ocg'
    valid = ('ocg',)

    def _get_meta_(self):
        if self.value == 'ocg':
            ret = 'OpenClimateGIS backend used for processing.'
        else:
            raise NotImplementedError
        return ret


class Callback(base.AbstractParameter):
    input_types = [FunctionType]
    name = 'callback'
    nullable = True
    default = None
    return_type = [FunctionType]

    def _get_meta_(self):
        if self.value is None:
            msg = 'No callback function provided.'
        else:
            msg = 'Callback enabled.'
        return (msg)


class Calc(base.IterableParameter, base.AbstractParameter):
    name = 'calc'
    default = None
    nullable = True
    input_types = [list, tuple]
    return_type = [list]
    element_type = [dict, str]
    unique = False
    _possible = ['es=tas+4', ['es=tas+4'], [{'func': 'mean', 'name': 'mean'}]]
    _required_keys_final = {'ref', 'meta_attrs', 'name', 'func', 'kwds'}
    _required_keys_initial = ('name', 'func')

    def __init__(self, *args, **kwargs):
        # this flag is used by the parser to determine if an eval function has been passed. very simple test for this...
        # if there is an equals sign in the string then it is considered an eval function
        self._is_eval_function = False
        base.AbstractParameter.__init__(self, *args, **kwargs)

    def __str__(self):
        if self.value is None:
            ret = base.AbstractParameter.__str__(self)
        else:
            cb = deepcopy(self.value)
            for ii in cb:
                ii.pop('ref')
                for k, v in ii['kwds'].items():
                    if type(v) not in [str, str, float, int, str]:
                        ii['kwds'][k] = type(v)
            ret = '{0}={1}'.format(self.name, cb)
        return ret

    def _get_meta_(self):
        if self.value is None:
            ret = 'No computations applied.'
        else:
            if self._is_eval_function:
                ret = 'An string function representation was used for calculation: "{0}"'.format(self.value[0])
            else:
                ret = ['The following computations were applied:']
                for ii in self.value:
                    ret.append('{0}: {1}'.format(ii['name'], ii['ref'].description))
        return ret

    def _parse_(self, value):
        # test if the value is an eval function and set internal flag
        if '=' in value:
            self._is_eval_function = True
        elif isinstance(value, dict) and value.get('func') is not None and '=' in value['func']:
            self._is_eval_function = True
        else:
            self._is_eval_function = False

        # format the output dictionary
        if self._is_eval_function:
            # select the function reference...
            try:
                eval_string = value['func']
            except TypeError:
                eval_string = value

            # determine if the eval string is multivariate
            if EvalFunction.is_multivariate(eval_string):
                eval_klass = MultivariateEvalFunction
            else:
                eval_klass = EvalFunction

            # reset the output dictionary
            new_value = {'func': eval_string, 'ref': eval_klass, 'name': None, 'kwds': OrderedDict()}
            # attempt to update the meta_attrs if they are present
            try:
                new_value.update({'meta_attrs': value['meta_attrs']})
            # attempting to index a string incorrectly
            except TypeError:
                new_value.update({'meta_attrs': None})
            # adjust the reference
            value = new_value

        # if it is not an eval function, then do the standard argument parsing
        else:

            # check for required keys
            if isinstance(value, dict):
                for key in self._required_keys_initial:
                    if key not in value:
                        msg = 'The key "{0}" is required for calculation dictionaries.'.format(key)
                        raise DefinitionValidationError(self, msg)

            fr = register.FunctionRegistry()

            # get the function key string form the calculation definition dictionary
            function_key = value['func']
            # this is the message for the DefinitionValidationError if this key may not be found.
            dve_msg = 'The function key "{0}" is not available in the function registry.'.format(function_key)

            # retrieve the calculation class reference from the function registry
            try:
                value['ref'] = fr[function_key]
            # if the function cannot be found, it may be part of a contributed library of calculations not registered by
            # default as the external library is an optional dependency.
            except KeyError:
                # this will register the icclim indices.
                if function_key.startswith('{0}_'.format(constants.ICCLIM_PREFIX_FUNCTION_KEY)):
                    register.register_icclim(fr)
                else:
                    raise DefinitionValidationError(self, dve_msg)
            # make another attempt to register the function
            try:
                value['ref'] = fr[function_key]
            except KeyError:
                raise DefinitionValidationError(self, dve_msg)

            # parameters will be set to empty if none are present in the calculation dictionary.
            if 'kwds' not in value:
                value['kwds'] = OrderedDict()
            # make the keyword parameter definitions lowercase.
            else:
                value['kwds'] = OrderedDict(value['kwds'])
                for k, v in value['kwds'].items():
                    try:
                        value['kwds'][k] = v.lower()
                    except AttributeError:
                        pass

        # add placeholder for meta_attrs if it is not present
        if 'meta_attrs' not in value:
            value['meta_attrs'] = None
        else:
            # replace with the metadata attributes class if the attributes are not none
            ma = value['meta_attrs']
            if ma is not None:
                value['meta_attrs'] = MetadataAttributes(ma)

        return value

    def _parse_string_(self, value):
        try:
            key, uname = value.split('~', 1)
            try:
                uname, kwds_raw = uname.split('!', 1)
                kwds_raw = kwds_raw.split('!')
                kwds = OrderedDict()
                for kwd in kwds_raw:
                    kwd_name, kwd_value = kwd.split('~')
                    try:
                        kwds.update({kwd_name: float(kwd_value)})
                    except ValueError:
                        kwds.update({kwd_name: str(kwd_value)})
            except ValueError:
                kwds = OrderedDict()
            ret = {'func': key, 'name': uname, 'kwds': kwds}
        except ValueError:
            # likely a string to use for an eval function
            if '=' not in value:
                msg = 'String may not be parsed: "{0}".'.format(value)
                raise DefinitionValidationError(self, msg)
            else:
                self._is_eval_function = True
                ret = value

        return ret

    def _validate_(self, value):
        if not self._is_eval_function:
            # get the aliases of the calculations
            aliases = [ii['name'] for ii in value]

            if len(aliases) != len(set(aliases)):
                raise DefinitionValidationError(self, 'User-provided calculation aliases must be unique: {0}'.format(
                    aliases))

            for v in value:
                if set(v.keys()) != self._required_keys_final:
                    msg = 'Required keys are: {0}'.format(self._required_keys_final)
                    raise DefinitionValidationError(self, msg)
                # run class-level definition
                v[constants.CALC_KEY_CLASS_REFERENCE].validate_definition(v)


class CalcGrouping(base.IterableParameter, base.AbstractParameter):
    name = 'calc_grouping'
    nullable = True
    input_types = [list, tuple]
    return_type = tuple
    default = None
    element_type = [str, list]
    unique = True
    _flags = ('unique', 'year')
    _standard_groups = ('day', 'month', 'year')

    @classmethod
    def iter_possible(cls):
        standard_seasons = [[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 1, 2]]
        for r in [1, 2, 3]:
            for combo in itertools.combinations(cls._standard_groups, r):
                yield (combo)
        for one in ['all']:
            yield (one)
        flags = list(cls._flags) + [None]
        for flag in flags:
            if flag is not None:
                yld = deepcopy(standard_seasons)
                yld.insert(0, flag)
            else:
                yld = standard_seasons
            yield (yld)

    def parse(self, value):
        try:
            # Not interested in looking for unique letters in the "_flags"
            parse_value = list(deepcopy(value))
            # If we do remove a flag, be sure and append it back
            add_back = None
            for flag in self._flags:
                if flag in parse_value:
                    parse_value.remove(flag)
                    add_back = flag
            # Call superclass method to parse the value for iteration
            ret = base.IterableParameter.parse(self, parse_value, check_basestrings=False)
            # Add the value back if it has been set
            if add_back is not None:
                ret.append(add_back)
        # Value is likely none.
        except TypeError:
            if value is None:
                ret = None
            else:
                raise
        return ret

    def finalize(self):
        if self._value == ('all',):
            self._value = 'all'

    def _get_meta_(self):
        if self.value is None:
            msg = 'No temporal aggregation applied.'
        else:
            msg = 'Temporal aggregation determined by the following group(s): {0}'.format(self.value)
        return (msg)

    def _validate_(self, value):
        # The 'all' parameter will be reduced to a string eventually
        if len(value) == 1 and value[0] == 'all':
            pass
        else:
            try:
                for val in value:
                    if val not in self._standard_groups:
                        raise DefinitionValidationError(self,
                                                        'not in standard groups: {}'.format(self._standard_groups))
            # The grouping may not be a date part but a seasonal aggregation
            except DefinitionValidationError:
                months = list(range(1, 13))
                for element in value:
                    # The keyword year and unique are okay for seasonal aggregations
                    if element in self._flags:
                        continue
                    elif isinstance(element, six.string_types):
                        if element not in self._flags:
                            raise DefinitionValidationError(self, 'element not in flags: {}'.format(self._flags))
                    else:
                        for month in element:
                            if month not in months:
                                raise DefinitionValidationError(self, 'month not in months: {}'.format(months))


class CalcRaw(base.BooleanParameter):
    name = 'calc_raw'
    default = False
    meta_true = 'Raw values will be used for calculations. These are the original data values linked to a selection geometry.'
    meta_false = 'Aggregated values will be used during the calculation.'


class CalcSampleSize(base.BooleanParameter):
    name = 'calc_sample_size'
    default = False
    meta_true = 'Statistical sample size for calculations added to output files.'
    meta_false = 'Statistical sample size not calculated.'


class ConformUnitsTo(base.AbstractParameter):
    name = 'conform_units_to'
    nullable = True
    default = None
    _lower_string = False

    @property
    def input_types(self):
        # CF units conversion packages are optional.
        uc = get_units_class(should_raise=False)
        if uc is None:
            ret = []
        else:
            ret = [uc]
        return ret

    @property
    def return_type(self):
        ret = [six.string_types]
        # CF units conversion packages are optional.
        uc = get_units_class(should_raise=False)
        if uc is not None:
            ret.append(uc)
        return ret

    def validate(self, value):
        if value is not None:
            try:
                get_units_object(value)
            except ValueError:
                msg = 'Units not recognized by conversion backend: {}'.format(value)
                raise DefinitionValidationError(self.__class__, msg)

    def _get_meta_(self):
        if self.value is None:
            ret = 'Units were not conformed.'
        else:
            ret = 'Units of all requested datasets were conformed to: "{0}".'.format(self.value)
        return ret


class Dataset(base.AbstractParameter):
    name = 'dataset'
    nullable = False
    default = None
    input_types = None
    return_type = None
    _perform_deepcopy = False
    _check_input_type = False
    _check_output_type = False

    def __init__(self, init_value, esmf_field_dimensions=('time', 'y', 'x')):
        """

        :param init_value: See :class:`ocgis.base.AbstractParameter`.
        :param esmf_field_dimensions: Tuple of :class:`~ocgis.Dimensions` object corresponding to the names of the ESMF
         field dimensions.
        """

        if isinstance(init_value, self.__class__):
            # Allow the dataset object to be initialized by an instance of itself.
            self.__dict__ = init_value.__dict__
        else:
            self.esmf_field_dimensions = esmf_field_dimensions
            super(Dataset, self).__init__(init_value)

    def __iter__(self):
        non_iterables = [AbstractRequestObject, dict, Field]
        if env.USE_ESMF:
            import ESMF
            non_iterables.append(ESMF.Field)

        if isinstance(self._value, tuple(non_iterables)):
            to_itr = [self._value]
        else:
            to_itr = self._value
        for uid, element in enumerate(to_itr, start=1):
            if isinstance(element, dict):
                element = RequestDataset(**element)

            if env.USE_ESMF and isinstance(element, ESMF.Field):
                from ocgis.regrid.base import get_ocgis_field_from_esmf_field
                element = get_ocgis_field_from_esmf_field(element)

            try:
                element = element.copy()
            except AttributeError:
                element = copy(element)

            if element.uid is None:
                element.uid = uid
                # TODO: Remove me once the driver does not accept request datasets at initialization.
                # Try to change the driver UID.
                try:
                    element.driver.rd.uid = uid
                except AttributeError:
                    # The field driver does not keep a copy of the request dataset.
                    if hasattr(element.driver, 'rd'):
                        raise

            yield element

    @property
    def data_model(self):
        dms = []
        for element in self:
            try:
                dm = element.driver.data_model
            except AttributeError:
                # Assume the data model or driver is not exposed by the object.
                continue
            else:
                if dm is not None:
                    dms.append(dm)
        # Allow for no collected data models.
        if len(dms) == 0:
            dms = None
        return dms

    @classmethod
    def from_query(cls, qi):

        def _update_from_query_dict_(store, key):
            value = qi.query_dict.get(key)
            if value is not None:
                value = value[0].split('|')
                store[key] = value

        keys = ['uri', 'variable', 'field_name', 'rename_variable']
        store = {}
        for k in keys:
            _update_from_query_dict_(store, k)

        rds = []
        for idx in range(len(store['uri'])):
            kwds = {}
            for k in keys:
                if k in store:
                    kwds[k] = store[k][idx]
            rd = RequestDataset(**kwds)
            rds.append(rd)
        return cls(rds)

    def get_meta(self):
        ret = ['* {}='.format(self.name)]
        for element in self:
            try:
                ret += element._get_meta_rows_()
            except AttributeError:
                ret.append('Field object with name: "{0}"'.format(element.name))
            ret.append('')
        return ret

    def iter_by_type(self, dtype):
        for element in self:
            if isinstance(element, dtype):
                yield element

    def parse_string(self, value):
        lowered = value.strip()
        if lowered == 'none':
            ret = None
        else:
            ret = self._parse_string_(lowered)
        return ret

    def _get_meta_(self):
        pass

    def _get_value_(self):
        return self.__iter__()

    value = property(_get_value_, base.AbstractParameter._set_value_)

    def _parse_string_(self, lowered):
        raise NotImplementedError


class DirOutput(base.StringParameter):
    _lower_string = False
    name = 'dir_output'
    nullable = False
    default = ocgis.env.DIR_OUTPUT
    return_type = str
    input_types = []

    def _get_meta_(self):
        ret = 'At execution time, data was originally written to this processor-local location: {0}'.format(self.value)
        return ret

    def _validate_(self, value):
        if not exists(value):
            raise DefinitionValidationError(self, 'Path does not exist: {}'.format(value))


class FileOnly(base.BooleanParameter):
    meta_true = 'File written with empty data.'
    meta_false = 'Actual data written to file.'
    default = False
    name = 'file_only'


class FormatTime(base.BooleanParameter):
    name = 'format_time'
    default = True
    meta_true = 'Time values converted to datetime stamps.'
    meta_false = 'Time values left in original form.'


class Geom(base.AbstractParameter):
    """
    :keyword list select_ugid: (``=None``) A sequence of sorted (ascending) unique identifiers to use when selecting
     geometries. These values should be a members of the attribute named ``geom_uid``.

    >>> [1, 45, 66]

    :keyword str geom_select_sql_where: (``=None``) A string suitable for insertion into a SQL WHERE statement. See http://www.gdal.org/ogr_sql.html
     for documentation (section titled "WHERE").

    >>> select_sql_where = 'STATE_NAME = "Wisconsin"'

    :keyword str geom_uid: (``=None``) The unique identifier name in the target geometry object.
    """

    name = 'geom'
    nullable = True
    default = None
    input_types = [list, tuple, GeomCabinetIterator, BaseGeometry, Field, GeometryVariable]
    return_type = [GeomCabinetIterator, tuple]
    _shp_key = None
    _bounds = None
    _ugid_key = constants.OCGIS_UNIQUE_GEOMETRY_IDENTIFIER

    def __init__(self, *args, **kwargs):
        self.select_ugid = kwargs.pop('select_ugid', None)
        self.union = kwargs.pop(KeywordArgument.UNION, False)
        self.geom_select_sql_where = kwargs.pop(GeomSelectSqlWhere.name, None)
        self.geom_uid = kwargs.pop(GeomUid.name, None)
        self.output_format_options = kwargs.pop(OutputFormatOptions.name, None)
        self.dataset = kwargs.pop(Dataset.name, None)
        # just store the value if it is a parameter object
        if isinstance(self.select_ugid, GeomSelectUid):
            self.select_ugid = self.select_ugid.value
        if isinstance(self.geom_uid, GeomUid):
            self.geom_uid = self.geom_uid.value
        if isinstance(self.geom_select_sql_where, GeomSelectSqlWhere):
            self.geom_select_sql_where = self.geom_select_sql_where.value
        if isinstance(self.output_format_options, OutputFormatOptions):
            self.output_format_options = self.output_format_options.value

        if self.output_format_options is not None:
            self.data_model = self.output_format_options.get(KeywordArgument.DATA_MODEL)
        else:
            if self.dataset is not None:
                self.data_model = list(self.dataset.data_model)[0]
            else:
                self.data_model = None

        args = [self] + list(args)
        base.AbstractParameter.__init__(*args, **kwargs)

    def __str__(self):
        if self.value is None:
            value = None
        elif self._shp_key is not None:
            value = '"{0}"'.format(self._shp_key)
        elif self._bounds is not None:
            value = '|'.join(map(str, self._bounds))
        else:
            value = '<{0} geometry(s)>'.format(len(self.value))
        ret = '{0}={1}'.format(self.name, value)
        return ret

    def _get_value_(self):
        if isinstance(self._value, GeomCabinetIterator):
            self._value.select_uid = self.select_ugid
        return base.AbstractParameter._get_value_(self)

    value = property(_get_value_, base.AbstractParameter._set_value_)

    def parse(self, value):
        if type(value) in [list, tuple]:
            if all([isinstance(element, dict) for element in value]):
                for ii, element in enumerate(value, start=1):
                    if 'geom' not in element:
                        ocgis_lh(exc=DefinitionValidationError(self, 'Geometry dictionaries must have a "geom" key.'))
                    if 'properties' not in element:
                        element['properties'] = {self._ugid_key: ii}
                    crs = element.get('crs', constants.UNINITIALIZED)
                    if 'crs' not in element:
                        ocgis_lh(msg='No CRS in geometry dictionary - assuming WGS84.', level=logging.WARN)
                ret = Field.from_records(value, crs=crs, uid=self.geom_uid, union=self.union,
                                         data_model=self.data_model)
            else:
                if len(value) == 2:
                    geom = Point(value[0], value[1])
                elif len(value) == 4:
                    minx, miny, maxx, maxy = value
                    geom = Polygon(((minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)))
                if not geom.is_valid:
                    raise DefinitionValidationError(self, 'Parsed geometry is not valid.')
                ret = [{'geom': geom, 'properties': {self._ugid_key: 1}}]
                ret = Field.from_records(ret, uid=self.geom_uid, union=self.union, data_model=self.data_model)
                self._bounds = geom.bounds
        elif isinstance(value, GeomCabinetIterator):
            self._shp_key = value.key or value.path
            # Always yield fields.
            value.as_field = True
            ret = value
        elif isinstance(value, BaseGeometry):
            ret = [{'geom': value, 'properties': {self._ugid_key: 1}}]
            ret = Field.from_records(ret, uid=self.geom_uid, union=self.union, data_model=self.data_model)
        elif value is None:
            ret = value
        elif isinstance(value, Field):
            ret = value
        elif isinstance(value, GeometryVariable):
            if value.ugid is None:
                msg = 'Geometry variables must have an associated "UGID".'
                raise DefinitionValidationError(self, msg)
            ret = Field(geom=value, crs=value.crs)
        else:
            raise NotImplementedError(type(value))

        # Convert to singular field if this is a field object.
        if isinstance(ret, Field):
            ret = tuple(self._iter_singular_fields_(ret))

        return ret

    def parse_string(self, value):
        if isinstance(value, six.string_types):
            if os.path.isdir(value):
                exc = DefinitionValidationError(self, 'The provided path is a directory.')
                ocgis_lh(exc=exc, logger='definition')

        elements = value.split('|')
        try:
            elements = [float(e) for e in elements]
            # switch geometry creation based on length. length of 2 is a point otherwise a bounding box
            if len(elements) == 2:
                geom = Point(elements[0], elements[1])
            else:
                minx, miny, maxx, maxy = elements
                geom = Polygon(((minx, miny),
                                (minx, maxy),
                                (maxx, maxy),
                                (maxx, miny)))
            if not geom.is_valid:
                raise DefinitionValidationError(self, 'Parsed geometry is not valid.')
            ret = [{'geom': geom, 'properties': {'ugid': 1}}]
            self._bounds = elements
        # try the value as a key or path
        except ValueError:
            # if the path exists, then assume it is a path to a shapefile, otherwise assume it is a key
            kwds = {}
            if os.path.exists(value):
                kwds['path'] = value
            else:
                kwds['key'] = value
            # this is saved for later use by the openclimategis metadata output as the input value is inherently
            # transformed
            self._shp_key = value
            # get the select_ugid test value
            try:
                test_value = self.select_ugid.value
            # it may not have been passed as a parameter object
            except AttributeError:
                test_value = self.select_ugid
            if test_value is None:
                select_ugid = None
            else:
                select_ugid = test_value
            kwds['select_uid'] = select_ugid

            kwds['select_sql_where'] = self.geom_select_sql_where
            kwds['uid'] = self.geom_uid
            kwds[KeywordArgument.UNION] = self.union
            kwds[KeywordArgument.DATA_MODEL] = self.data_model
            ret = GeomCabinetIterator(**kwds)
        return ret

    def _get_meta_(self):
        if self.value is None:
            ret = 'No user-supplied geometry. All data returned.'
        elif self._shp_key is not None:
            ret = 'The selection geometry "{0}" was used for subsetting.'.format(self._shp_key)
        elif self._bounds is not None:
            ret = 'The bounding box coordinates used for subset are: {0}.'.format(self._bounds)
        else:
            ret = '{0} custom user geometries provided.'.format(len(self.value))
        return ret

    @staticmethod
    def _iter_singular_fields_(field):
        """
        :param field:
        :type field: :class:`ocgis.new_interface.field.Field`
        :rtype: tuple
        """

        itr = [list(range(ii)) for ii in field.geom.shape]
        for slc in itertools.product(*itr):
            yield field.geom.__getitem__(slc).parent


class GeomSelectSqlWhere(base.AbstractParameter):
    name = 'geom_select_sql_where'
    return_type = [str]
    nullable = True
    default = None
    input_types = []
    _lower_string = False

    def _get_meta_(self):
        if self.value is None:
            msg = "No SQL where statement provided."
        else:
            msg = "A SQL where statement was used to select geometries from input data source: {0}".format(self.value)
        return msg


class GeomSelectUid(base.IterableParameter, base.AbstractParameter):
    name = 'geom_select_uid'
    return_type = tuple
    nullable = True
    default = None
    input_types = [list, tuple]
    element_type = int
    unique = True

    def _get_meta_(self):
        if self.value is None:
            ret = 'No geometry selection by unique identifier.'
        else:
            ret = 'The following UGID values were used to select from the input geometries: {0}.'.format(self.value)
        return ret


class GeomUid(base.AbstractParameter):
    name = 'geom_uid'
    nullable = True
    default = None
    input_types = []
    return_type = str
    _lower_string = False

    def _get_meta_(self):
        if self.value is None:
            msg = 'No geometry unique identifier provided.'
        else:
            msg = 'The unique geometry identifier used: "{0}".'.format(self.value)
        return msg


class InterpolateSpatialBounds(base.BooleanParameter):
    name = 'interpolate_spatial_bounds'
    default = False
    meta_true = 'If no bounds are present on the coordinate variables, an attempt will be made to interpolate boundary polygons.'
    meta_false = 'If no bounds are present on the coordinate variables, no attempt will be made to interpolate boundary polygons.'


class LevelRange(base.IterableParameter, base.AbstractParameter):
    name = 'level_range'
    element_type = [int, float]
    nullable = True
    input_types = [list, tuple]
    return_type = tuple
    unique = False
    default = None

    def validate_all(self, value):
        if len(value) != 2:
            msg = 'There must be two elements in the sequence.'
            raise DefinitionValidationError(self, msg)
        if value[0] > value[1]:
            msg = 'The second element must be >= the first element.'
            raise DefinitionValidationError(self, msg)

    def _get_meta_(self):
        if self.value is None:
            msg = 'No level subset.'
        else:
            msg = 'The following level subset was applied to all request datasets: {0}'.format(self.value)
        return msg


class Melted(base.BooleanParameter):
    """
    .. note:: Accepts all parameters to :class:`ocgis.driver.parms.base.BooleanParameter`.

    :keyword dataset:
    :type dataset: :class:`ocgis.driver.parms.definition.Dataset`
    :keyword output_format:
    :type output_format: :class:`ocgis.driver.parms.definition.OutputFormatName`
    """

    name = 'melted'
    default = False
    meta_true = 'Melted tabular iteration requested.'
    meta_false = 'Flat tabular iteration requested.'


class Optimizations(base.AbstractParameter):
    name = 'optimizations'
    default = None
    input_types = [dict]
    nullable = True
    return_type = [dict]
    # : 'tgds' - dictionary mapping field aliases to TemporalGroupDimension objects
    _allowed_keys = ['tgds', 'fields']
    _perform_deepcopy = False

    def _get_meta_(self):
        if self.value is None:
            ret = 'No optimizations were used.'
        else:
            ret = 'The following optimizations were used: {0}.'.format(list(self.value.keys()))
        return ret

    def _validate_(self, value):
        if len(value) == 0:
            msg = 'Empty dictionaries are not allowed for optimizations. Use None instead.'
            raise DefinitionValidationError(self, msg)
        if set(value.keys()).issubset(set(self._allowed_keys)) == False:
            msg = 'Allowed optimization keys are "{0}".'.format(self._allowed_keys)
            raise DefinitionValidationError(self, msg)


class OptimizedBoundingBoxSubset(base.BooleanParameter):
    name = 'optimized_bbox_subset'
    default = False
    meta_true = 'Optimized bounding box subset used.'
    meta_false = 'Standard subsetting procedure used.'


class OutputCRS(base.AbstractParameter):
    input_types = [CoordinateReferenceSystem]
    name = 'output_crs'
    nullable = True
    return_type = [CoordinateReferenceSystem]
    default = None

    def _get_meta_(self):
        if self.value is None:
            ret = "No CRS associated with dataset. WGS84 Lat/Lon Geographic (EPSG:4326) assumed."
        else:
            ret = 'The PROJ.4 definition of the coordinate reference system is: "{0}"'.format(
                self.value.sr.ExportToProj4())
        return ret

    def _parse_string_(self, value):
        return CoordinateReferenceSystem(epsg=int(value))


class OutputFormat(base.StringOptionParameter):
    name = 'output_format'
    default = constants.OutputFormatName.OCGIS
    valid = list(get_converter_map().keys())

    def __init__(self, init_value=None):
        try:
            if isinstance(init_value, six.string_types):
                init_value = init_value.lower()
            # Maintain the old CSV-Shapefile output format key.
            if init_value == 'csv+':
                init_value = constants.OutputFormatName.CSV_SHAPEFILE
            # Maintain the old NumPy key.
            if init_value == 'numpy':
                init_value = constants.OutputFormatName.OCGIS
        except AttributeError:
            # Allow the object to initialized by itself.
            if not isinstance(init_value, self.__class__):
                raise
        super(OutputFormat, self).__init__(init_value=init_value)

    @classmethod
    def iter_possible(cls):
        for element in cls.valid:
            yield element

    def get_converter_class(self):
        return get_converter(self.value)

    def _get_meta_(self):
        ret = 'The output format is "{0}".'.format(self.value)
        return ret


class OutputFormatOptions(base.AbstractParameter):
    name = 'output_format_options'
    nullable = True
    default = None
    input_types = [dict]
    return_type = [dict]

    def _get_meta_(self):
        if self.value is None:
            msg = 'No output format options supplied.'
        else:
            msg = 'Output-specific options were supplied: {}'.format(self.value)
        return msg


class Prefix(base.StringParameter):
    name = 'prefix'
    nullable = False
    default = 'ocgis_output'
    input_types = [str]
    return_type = str
    _lower_string = False

    def _get_meta_(self):
        msg = 'Data output given the following prefix: {0}.'.format(self.value)
        return msg


class RegridDestination(base.AbstractParameter):
    name = 'regrid_destination'
    nullable = True
    default = None
    input_types = None
    return_type = [Field, Grid]
    _check_input_type = False

    def __init__(self, **kwargs):
        self.dataset = kwargs.pop('dataset')
        super(RegridDestination, self).__init__(**kwargs)

    def _get_meta_(self):
        if self.value is not None:
            msg = 'Input data was regridded.'
        else:
            msg = 'Input data was not regridded.'
        return msg

    def _parse_(self, value):
        # Get the request dataset from the collection if the value is a string.
        if isinstance(value, six.string_types):
            value_to_find = value
            value = None
            for d in list(self.dataset):
                if d.field_name == value_to_find:
                    value = d
            if value is None:
                msg = 'String regrid destination "{}" not found.'.format(value_to_find)
                raise DefinitionValidationError(self.__class__, msg)
        elif value is None:
            is_regrid_destination = [rd.regrid_destination for rd in list(self.dataset)]
            if sum(is_regrid_destination) > 1:
                msg = 'Only one request dataset may be the destination regrid dataset.'
                raise DefinitionValidationError(self, msg)
            elif any(is_regrid_destination):
                for rd in self.dataset:
                    if rd.regrid_destination:
                        value = rd
                        break

        # If there is a destination, ensure at least one dataset is a source.
        if value is not None:
            if sum([d.regrid_source for d in list(self.dataset)]) < 1:
                msg = 'There is a destination reqrid target, but no datasets are set as a source.'
                raise DefinitionValidationError(self, msg)

        # Return a field if the value is a request dataset.
        if isinstance(value, RequestDataset):
            value = value.get()

        return value


class RegridOptions(base.AbstractParameter):
    name = 'regrid_options'
    nullable = True
    default = {'regrid_method': 'auto', 'value_mask': None, 'split': True}
    input_types = [dict]
    return_type = [dict]
    _possible_value_mask_types = [type(None), np.ndarray]

    def _parse_(self, value):
        for key in list(value.keys()):
            if key not in list(self.default.keys()):
                msg = 'The option "{0}" is not allowed.'.format(key)
                raise DefinitionValidationError(self, msg)

        if 'regrid_method' not in value:
            value['regrid_method'] = 'auto'
        if 'value_mask' not in value:
            value['value_mask'] = None

        if not any(isinstance(value['value_mask'], t) for t in self._possible_value_mask_types):
            msg = 'Types for "value_mask" are: {0}'.format(self._possible_value_mask_types)
            raise DefinitionValidationError(self, msg)
        if isinstance(value['value_mask'], np.ndarray):
            if value['value_mask'].dtype != np.bool:
                msg = '"value_mask" must be a boolean array.'
                raise DefinitionValidationError(self, msg)

        return value

    def _get_meta_(self):
        ret = {}
        for k, v in self.value.items():
            if k == 'value_mask' and isinstance(v, np.ndarray):
                ret[k] = np.ndarray
            else:
                ret[k] = v
        return str(ret)


class SearchRadiusMultiplier(base.AbstractParameter):
    input_types = [float]
    name = 'search_radius_mult'
    nullable = True
    return_type = [float]
    default = None

    def _get_meta_(self):
        if self.value is not None:
            msg = 'If point geometries were used for selection, a modifier of {0} times the data resolution was used to spatially select data.'.format(
                self.value)
        else:
            msg = 'No buffering applied to point selection geometries.'
        return msg

    def _validate_(self, value):
        if value <= 0:
            raise DefinitionValidationError(self, msg='must be > 0')


class SelectNearest(base.BooleanParameter):
    name = 'select_nearest'
    default = False
    meta_true = 'The nearest geometry to the centroid of the selection geometry was returned.'
    meta_false = 'All geometries returned regardless of distance.'


class Slice(base.IterableParameter, base.AbstractParameter):
    name = 'slice'
    return_type = dict
    nullable = True
    default = None
    input_types = [list, tuple, dict, OrderedDict]
    element_type = [type(None), int, tuple, list, slice]
    unique = False

    def __init__(self, init_value=None):
        # If this is a dictionary slice, leave as is and let slice infrastructure handling formatting.
        if isinstance(init_value, dict):
            is_dict = True
            use_init_value = None
        else:
            is_dict = False
            use_init_value = init_value

        super(Slice, self).__init__(use_init_value)

        # Set the dictionary slice directly. Do not use the iterable formatting for the list/tuple.
        if is_dict:
            self._value = deepcopy(init_value)

    def parse_all(self, values):
        try:
            new_values = {}
            new_values[DimensionMapKey.REALIZATION] = values[0]
            new_values[DimensionMapKey.TIME] = values[1]
            new_values[DimensionMapKey.LEVEL] = values[2]
            new_values[DimensionMapKey.Y] = values[3]
            new_values[DimensionMapKey.X] = values[4]
        except IndexError:
            # This implies the length is not correct. Let the validater catch this.
            new_values = values
        return new_values

    def validate_all(self, values):
        if len(values) != 5:
            raise DefinitionValidationError(self, 'Slices must have 5 values.')

    def _parse_(self, value):
        if value is None:
            ret = slice(None)
        elif type(value) == int:
            if value < 0:
                ret = slice(value, None)
            else:
                ret = slice(value, value + 1)
        elif type(value) in [list, tuple]:
            ret = slice(*value)
        else:
            raise DefinitionValidationError(self, '"{0}" cannot be converted to a slice object'.format(value))
        return ret

    def _get_meta_(self):
        if self.value is None:
            ret = 'No slice passed.'
        else:
            ret = 'A slice was used.'
        return ret


class Snippet(base.BooleanParameter):
    name = 'snippet'
    default = False
    meta_true = 'First temporal slice or temporal group returned.'
    meta_false = 'All time points returned.'


class SpatialOperation(base.StringOptionParameter):
    name = 'spatial_operation'
    default = 'intersects'
    valid = ('clip', 'intersects')

    @classmethod
    def iter_possible(cls):
        for v in cls.valid:
            yield (v)

    def _get_meta_(self):
        if self.value == 'intersects':
            ret = 'Geometries touching AND overlapping returned.'
        else:
            ret = 'A full geometric intersection occurred. Where geometries overlapped, a new geometry was created.'
        return ret


class SpatialReorder(base.BooleanParameter):
    name = 'spatial_reorder'
    default = False
    meta_true = 'Reorder data and coordinate arrays to have ascending (0-index to max index) longitudinal coordinates.'
    meta_false = 'Do not reorder data and coordinate arrays to have ascending longitudinal coordinates.'


class SpatialWrapping(base.StringOptionParameter):
    name = 'spatial_wrapping'
    default = None
    nullable = True
    valid = ('wrap', 'unwrap', None)
    _enum_mapping = {'wrap': WrapAction.WRAP, 'unwrap': WrapAction.UNWRAP}

    @property
    def as_enum(self):
        if self.value is None:
            ret = None
        else:
            ret = self._enum_mapping[self.value]
        return ret

    @classmethod
    def iter_possible(cls):
        for v in cls.valid:
            yield v

    def _get_meta_(self):
        msgs = {'wrap': 'Wrap data to -180 to 180 degree spatial domain.',
                'unwrap': 'Unwrap data to 0 to 360 degree spatial domain.',
                None: 'No wrapping action applied.'}
        return msgs[self.value]


class TimeRange(base.IterableParameter, base.AbstractParameter):
    name = 'time_range'
    element_type = [datetime.datetime]
    nullable = True
    input_types = [list, tuple]
    return_type = tuple
    unique = False
    default = None

    def validate_all(self, value):
        if len(value) != 2:
            msg = 'There must be two elements in the sequence.'
            raise DefinitionValidationError(self, msg)
        if value[0] > value[1]:
            msg = 'The second element must be >= the first element.'
            raise DefinitionValidationError(self, msg)

    def _get_meta_(self):
        if self.value == None:
            msg = 'No time range subset.'
        else:
            msg = 'The following time range subset was applied to all request datasets: {0}'.format(self.value)
        return msg

    def _parse_string_(self, value):
        formats = ['%Y%m%d', '%Y%m%d-%H%M%S']
        ret = None
        for f in formats:
            try:
                ret = datetime.datetime.strptime(value, f)
            except ValueError:
                # Conversion may fail. Try the next date format.
                continue
            else:
                break
        assert ret is not None
        return ret


class TimeRegion(base.AbstractParameter):
    name = 'time_region'
    nullable = True
    default = None
    return_type = dict
    input_types = [dict, OrderedDict]

    def _get_meta_(self):
        if self.value is None:
            msg = 'No time region subset.'
        else:
            msg = 'The following time region subset was applied to all request datasets: {0}'.format(self.value)
        return msg

    def _parse_(self, value):
        if value is not None:
            # Add missing keys
            for add_key in ['month', 'year']:
                if add_key not in value:
                    value.update({add_key: None})
            # Confirm only month and year keys are present
            for key in list(value.keys()):
                if key not in ['month', 'year']:
                    raise DefinitionValidationError(self, 'Only "month" and "year" keys allowed.')
            if all([i is None for i in list(value.values())]):
                value = None
        return value

    def _parse_string_(self, value):
        ret = {}
        values = value.split(',')
        for value in values:
            key, key_value = value.split('~')
            key_value = key_value.split('|')
            key_value = [int(e) for e in key_value]
            ret[key] = key_value
        return ret


class TimeSubsetFunction(base.AbstractParameter):
    name = 'time_subset_func'
    default = None
    nullable = True
    return_type = FunctionType
    input_types = [FunctionType]

    def _get_meta_(self):
        if self.value is None:
            msg = "No time subset function provided."
        else:
            msg = "A time subset function was provided."
        return msg


class VectorWrap(base.BooleanParameter):
    name = 'vector_wrap'
    default = True
    meta_true = 'Geographic coordinates wrapped from -180 to 180 degrees longitude.'
    meta_false = 'Geographic coordinates match the target dataset coordinate wrapping and may be in the range 0 to 360.'
