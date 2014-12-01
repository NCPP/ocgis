from copy import deepcopy
import logging
import netCDF4 as nc
from warnings import warn

import numpy as np

from ocgis import constants
from ocgis.api.request.driver.base import AbstractDriver
from ocgis.exc import ProjectionDoesNotMatch, VariableNotFoundError, DimensionNotFound
from ocgis.interface.base.crs import CFWGS84, CFCoordinateReferenceSystem
from ocgis.interface.base.dimension.spatial import SpatialGridDimension, SpatialDimension
from ocgis.interface.base.variable import VariableCollection, Variable
from ocgis.interface.metadata import NcMetadata
from ocgis.interface.nc.dimension import NcVectorDimension
from ocgis.interface.nc.field import NcField
from ocgis.interface.nc.temporal import NcTemporalDimension
from ocgis.util.helpers import assert_raise, itersubclasses
from ocgis.util.logging_ocgis import ocgis_lh


class DriverNetcdf(AbstractDriver):
    key = 'netCDF'

    def __init__(self, *args, **kwargs):
        AbstractDriver.__init__(self, *args, **kwargs)
        self._raw_metadata = None

    @property
    def raw_metadata(self):
        if self._raw_metadata is None:
            ds = self.open()
            try:
                self._raw_metadata = NcMetadata(ds)
            finally:
                self.close(ds)
        return self._raw_metadata

    def close(self, obj):
        obj.close()

    def get_crs(self):
        crs = None
        for potential in itersubclasses(CFCoordinateReferenceSystem):
            try:
                crs = potential.load_from_metadata(self.rd._variable[0], self.rd.source_metadata)
                break
            except ProjectionDoesNotMatch:
                continue
        return crs

    def get_dimensioned_variables(self):
        metadata = self.raw_metadata
        variables = metadata['variables'].keys()
        ret = []

        ## check each variable for appropriate dimensions.
        for variable in variables:
            try:
                dim_map = get_dimension_map(variable, metadata)
            except DimensionNotFound:
                ## if a dimension is not located, then it is not an appropriate variable for subsetting.
                continue
            missing_dimensions = []

            ## these dimensions are required for subsetting.
            for required_dimension in ['X', 'Y', 'T']:
                if dim_map[required_dimension] is None:
                    missing_dimensions.append(required_dimension)

            if len(missing_dimensions) > 0:
                ## if any of the required dimensions are missing, the variable is not appropriate for subsetting.
                continue
            else:
                ret.append(variable)

        return ret

    def _get_field_(self, format_time=True, interpolate_spatial_bounds=False):
        """
        :param bool format_time:
        :param bool interpolate_spatial_bounds:
        :raises ValueError:
        """

        def _get_temporal_adds_(ref_attrs):
            ## calendar should default to standard if it is not present and the
            ## t_calendar overload is not used.
            calendar = self.rd.t_calendar or ref_attrs.get('calendar', None) or 'standard'

            return ({'units': self.rd.t_units or ref_attrs['units'],
                     'calendar': calendar,
                     'format_time': format_time})

        ## this dictionary contains additional keyword arguments for the row
        ## and column dimensions.
        adds_row_col = {'interpolate_bounds': interpolate_spatial_bounds}

        ## parameters for the loading loop
        to_load = {'temporal': {'cls': NcTemporalDimension, 'adds': _get_temporal_adds_, 'axis': 'T', 'name_uid': 'tid',
                                'name_value': 'time'},
                   'level': {'cls': NcVectorDimension, 'adds': None, 'axis': 'Z', 'name_uid': 'lid',
                             'name_value': 'level'},
                   'row': {'cls': NcVectorDimension, 'adds': adds_row_col, 'axis': 'Y', 'name_uid': 'row_id',
                           'name_value': 'row'},
                   'col': {'cls': NcVectorDimension, 'adds': adds_row_col, 'axis': 'X', 'name_uid': 'col_id',
                           'name_value': 'col'},
                   'realization': {'cls': NcVectorDimension, 'adds': None, 'axis': 'R', 'name_uid': 'rlz_id',
                                   'name_value': 'rlz'}}
        loaded = {}

        for k, v in to_load.iteritems():
            ## this is the string axis representation
            axis_value = v['axis'] or v['cls']._axis
            ## pull the axis information out of the dimension map
            ref_axis = self.rd.source_metadata['dim_map'].get(axis_value)
            ref_axis = self.rd.source_metadata['dim_map'].get(axis_value)
            ## if the axis is not represented, fill it with none. this happens
            ## when a dataset does not have a vertical level or projection axis
            ## for example.
            if ref_axis is None:
                fill = None
            else:
                ref_variable = self.rd.source_metadata['variables'].get(ref_axis['variable'])

                ## for data with a projection/realization axis there may be no
                ## associated variable.
                try:
                    ref_variable['axis'] = ref_axis
                except TypeError:
                    if axis_value == 'R' and ref_variable is None:
                        ref_variable = {'axis': ref_axis, 'name': ref_axis['dimension'], 'attrs': {}}

                ## extract the data length to use when creating the source index
                ## arrays.
                length = self.rd.source_metadata['dimensions'][ref_axis['dimension']]['len']
                src_idx = np.arange(0, length, dtype=constants.np_int)

                ## get the target data type for the dimension
                try:
                    dtype = np.dtype(ref_variable['dtype'])
                ## the realization dimension may not be a associated with a variable
                except KeyError:
                    if k == 'realization' and ref_variable['axis']['variable'] is None:
                        dtype = None
                    else:
                        raise

                ## assemble parameters for creating the dimension class then initialize
                ## the class.
                kwds = dict(name_uid=v['name_uid'], name_value=v['name_value'], src_idx=src_idx,
                            data=self.rd, meta=ref_variable, axis=axis_value, name=ref_variable.get('name'),
                            dtype=dtype)

                ## there may be additional parameters for each dimension.
                if v['adds'] is not None:
                    try:
                        kwds.update(v['adds'](ref_variable['attrs']))
                    ## adds may not be a callable object. assume they are a
                    ## dictionary.
                    except TypeError:
                        kwds.update(v['adds'])
                kwds.update({'name': ref_variable.get('name')})
                fill = v['cls'](**kwds)

            loaded[k] = fill

        assert_raise(set(('temporal', 'row', 'col')).issubset(set([k for k, v in loaded.iteritems() if v != None])),
                     logger='request',
                     exc=ValueError('Target variable must at least have temporal, row, and column dimensions.'))

        grid = SpatialGridDimension(row=loaded['row'], col=loaded['col'])

        # crs = None
        # if rd.crs is not None:
        #     crs = rd.crs
        # else:
        #     crs = rd._get_crs_(rd._variable[0])
        # if crs is None:
        #     ocgis_lh('No "grid_mapping" attribute available assuming WGS84: {0}'.format(rd.uri),
        #              'request', logging.WARN)
        #     crs = CFWGS84()

        spatial = SpatialDimension(name_uid='gid', grid=grid, crs=self.rd.crs, abstraction=self.rd.s_abstraction)

        vc = VariableCollection()
        for vdict in self.rd:
            variable_meta = deepcopy(self.rd._source_metadata['variables'][vdict['variable']])
            variable_units = vdict['units'] or variable_meta['attrs'].get('units')
            dtype = np.dtype(variable_meta['dtype'])
            fill_value = variable_meta['fill_value']
            variable = Variable(vdict['variable'], vdict['alias'], units=variable_units, meta=variable_meta,
                                data=self.rd, conform_units_to=vdict['conform_units_to'], dtype=dtype,
                                fill_value=fill_value)
            vc.add_variable(variable)

        ret = NcField(variables=vc, spatial=spatial, temporal=loaded['temporal'], level=loaded['level'],
                      realization=loaded['realization'], meta=deepcopy(self.rd._source_metadata), uid=self.rd.did,
                      name=self.rd.name)

        ## apply any subset parameters after the field is loaded
        if self.rd.time_range is not None:
            ret = ret.get_between('temporal', min(self.rd.time_range), max(self.rd.time_range))
        if self.rd.time_region is not None:
            ret = ret.get_time_region(self.rd.time_region)
        if self.rd.level_range is not None:
            try:
                ret = ret.get_between('level', min(self.rd.level_range), max(self.rd.level_range))
            except AttributeError:
                ## there may be no level dimension
                if ret.level == None:
                    msg = ("A level subset was requested but the target dataset does not have a level dimension. The "
                           "dataset's alias is: {0}".format(self.rd.alias))
                    raise (ValueError(msg))
                else:
                    raise

        return ret

    def get_source_metadata(self):
        metadata = self.raw_metadata

        try:
            var = metadata['variables'][self.rd._variable[0]]
        except KeyError:
            raise VariableNotFoundError(self.rd.uri, self.rd._variable[0])
        if self.rd.dimension_map is None:
            metadata['dim_map'] = get_dimension_map(var['name'], metadata)
        else:
            for k, v in self.rd.dimension_map.iteritems():
                try:
                    variable_name = metadata['variables'][v]['name']
                except KeyError:
                    variable_name = None
                self.rd.dimension_map[k] = {'variable': variable_name,
                                            'dimension': v,
                                            'pos': var['dimensions'].index(v)}
                metadata['dim_map'] = self.rd.dimension_map

        return metadata

    def open(self):
        try:
            ret = nc.Dataset(self.rd.uri, 'r')
        except TypeError:
            ret = nc.MFDataset(self.rd.uri)
        return ret


def get_axis(dimvar, dims, dim):
    try:
        axis = dimvar['attrs']['axis']
    except KeyError:
        ocgis_lh('Guessing dimension location with "axis" attribute missing for variable "{0}".'.format(dimvar['name']),
                 logger='nc.dataset',
                 level=logging.WARN,
                 check_duplicate=True)
        axis = guess_by_location(dims, dim)
    return axis


def get_dimension_map(variable, metadata):
    """
    :param str variable:
    :param dict metadata:
    """
    dims = metadata['variables'][variable]['dimensions']
    mp = dict.fromkeys(['T', 'Z', 'X', 'Y'])

    ## try to pull dimensions
    for dim in dims:
        dimvar = None
        try:
            dimvar = metadata['variables'][dim]
        except KeyError:
            ## search for variable with the matching dimension
            for key, value in metadata['variables'].iteritems():
                if len(value['dimensions']) == 1 and value['dimensions'][0] == dim:
                    dimvar = metadata['variables'][key]
                    break
        ## the dimension variable may not exist
        if dimvar is None:
            ocgis_lh(logger='request.nc', exc=DimensionNotFound(dim))
        axis = get_axis(dimvar, dims, dim)
        ## pull metadata information the variable and dimension names
        mp[axis] = {'variable': dimvar['name'], 'dimension': dim}
        try:
            mp[axis].update({'pos': dims.index(dimvar['name'])})
        except ValueError:
            ## variable name may differ from the dimension name
            mp[axis].update({'pos': dims.index(dim)})

    ## look for bounds variables
    # bounds_names = set(constants.name_bounds)
    for key, value in mp.iteritems():

        if value is None:
            # this occurs for such things as levels or realizations where the dimensions is not present. the value is
            # set to none and should not be processed.
            continue

        # if the dimension is found, search for the bounds by various approaches.

        # try to get the bounds attribute from the variable directly. if the attribute is not present in the metadata
        # dictionary, continue looking for other options.
        bounds_var = metadata['variables'][value['variable']]['attrs'].get('bounds')
        var = metadata['variables'][variable]

        if bounds_var is None:
            # if no attribute is found, try some other options...

            ## if no bounds variable is found for time, it may be a climatological.
            if key == 'T':
                try:
                    bounds_var = metadata['variables'][value['variable']]['attrs']['climatology']
                    ocgis_lh('Climatological bounds found for variable: {0}'.format(var['name']), logger='request.nc',
                             level=logging.INFO)
                ## climatology is not found on time axis
                except KeyError:
                    pass

        # the bounds variable was found, but the variable is not actually present in the output file
        if bounds_var not in metadata['variables']:
            msg = 'Bounds listed for variable "{0}" but the destination bounds variable "{1}" does not exist.'.\
                format(var['name'], bounds_var)
            ocgis_lh(msg, logger='nc.driver', level=logging.WARNING, check_duplicate=True)
            bounds_var = None

        # bounds variables sometime appear oddly, if it is not none and not a string, display what the value is, raise a
        # warning and continue setting the bounds variable to None.
        if not isinstance(bounds_var, basestring):
            if bounds_var is not None:
                msg = 'Bounds variable is not a string and is not None. The value is "{0}". Setting bounds to None.'.format(bounds_var)
                warn(msg)
                bounds_var = None

        value.update({'bounds': bounds_var})

    return mp


def guess_by_location(dims, target):
    mp = {3: {0: 'T', 1: 'Y', 2: 'X'},
          4: {0: 'T', 2: 'Y', 3: 'X', 1: 'Z'}}
    try:
        axis_map = mp[len(dims)]
    except KeyError:
        # if there an improper number of dimensions, then the variable does not have appropriate dimensions for
        # subsetting
        raise DimensionNotFound(target)
    return axis_map[dims.index(target)]
