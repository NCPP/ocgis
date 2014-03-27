from ocgis.exc import DefinitionValidationError, ProjectionDoesNotMatch,\
    DimensionNotFound, NoUnitsError
from copy import deepcopy, copy
import inspect
import os
from ocgis import env, constants
from ocgis.util.helpers import locate, validate_time_subset, itersubclasses,\
    assert_raise
from datetime import datetime
import netCDF4 as nc
from ocgis.interface.metadata import NcMetadata
from ocgis.util.logging_ocgis import ocgis_lh
import logging
from ocgis.interface.nc.temporal import NcTemporalDimension
import numpy as np
from ocgis.interface.base.dimension.spatial import SpatialGridDimension,\
    SpatialDimension
from ocgis.interface.base.crs import CFCoordinateReferenceSystem, CFWGS84,\
    CFRotatedPole
from ocgis.interface.nc.dimension import NcVectorDimension
from ocgis.interface.nc.field import NcField
from ocgis.interface.base.variable import Variable, VariableCollection
from ocgis.util.inspect import Inspect


class NcRequestDataset(object):
    
    def __init__(self,uri=None,variable=None,alias=None,units=None,time_range=None,
                 time_region=None,level_range=None,conform_units_to=None,s_crs=None,
                 t_units=None,t_calendar=None,did=None,meta=None,s_abstraction=None,
                 dimension_map=None):
        
        self._uri = self._get_uri_(uri)
        self.variable = variable
        assert(self.uri is not None)
        assert(self.variable is not None)
        
        self.alias = self._str_format_(alias) or variable
        self.time_range = deepcopy(time_range)
        self.time_region = deepcopy(time_region)
        self.level_range = deepcopy(level_range)
        self.s_crs = deepcopy(s_crs)
        self.t_units = self._str_format_(t_units)
        self.t_calendar = self._str_format_(t_calendar)
        self.dimension_map = deepcopy(dimension_map)
        self.did = did
        self.meta = meta or {}
        
        self.__source_metadata = None
        self.units = units
        self.conform_units_to = conform_units_to
        
        self.s_abstraction = s_abstraction
        try:
            self.s_abstraction = self.s_abstraction.lower()
            assert(self.s_abstraction in ('point','polygon'))
        except AttributeError:
            if s_abstraction is None:
                pass
            else:
                raise
        
        self._format_()
        
    def _open_(self):
        try:
            ret = nc.Dataset(self.uri,'r')
        except TypeError:
            ret = nc.MFDataset(self.uri)
        return(ret)
    
    @property
    def conform_units_to(self):
        return(self._conform_units_to)
    @conform_units_to.setter
    def conform_units_to(self,value):
        if value is not None:
            ## import the cfunits package and attempt to construct a units object.
            ## if this is okay, save the units string
            from cfunits import Units
            dest = Units(value)
            ## the units of the source data need to be specified or available
            ## in the metadata.
            try:
                src = self.units or self._get_units_from_metadata_()
            except KeyError:
                exc = NoUnitsError(message='Units could not be read from source metadata. The "units" keyword argument may be needed.')
                ocgis_lh(exc=exc)
            src = Units(src)
            ## units must be equivalent.
            try:
                assert(src.equivalent(dest))
            except AssertionError:
                ocgis_lh(exc=ValueError('The units specified in "conform_units_to" ("{0}") are not equivalent to the source units "{1}".'.format(dest.format(names=True),src.format(names=True))))
            self._conform_units_to = value
        else:
            self._conform_units_to = None
            
    def _get_units_from_metadata_(self):
        ret = self._source_metadata['variables'][self.variable]['attrs']['units']
        return(ret)
            
    @property
    def _source_metadata(self):
        if self.__source_metadata is None:
            ds = self._open_()
            try:
                self.__source_metadata = NcMetadata(ds)
                var = ds.variables[self.variable]
                if self.dimension_map is None:
                    self.__source_metadata['dim_map'] = get_dimension_map(ds,var,self._source_metadata)
                else:
                    for k,v in self.dimension_map.iteritems():
                        try:
                            variable_name = ds.variables.get(v)._name
                        except AttributeError:
                            variable_name = None
                        self.dimension_map[k] = {'variable':variable_name,
                                                 'dimension':v,
                                                 'pos':var.dimensions.index(v)}
                        self.__source_metadata['dim_map'] = self.dimension_map
            finally:
                ds.close()
        return(self.__source_metadata)
        
    def get(self,format_time=True,interpolate_spatial_bounds=False):
        '''
        :param bool format_time:
        :param bool interpolate_spatial_bounds:
        '''
        
        def _get_temporal_adds_(ref_attrs):
            ## calendar should default to standard if it is not present and the
            ## t_calendar overload is not used.
            calendar = self.t_calendar or ref_attrs.get('calendar',None) or 'standard'
            
            return({'units':self.t_units or ref_attrs['units'],
                    'calendar':calendar,
                    'format_time':format_time})
            
        ## this dictionary contains additional keyword arguments for the row
        ## and column dimensions.
        adds_row_col = {'interpolate_bounds':interpolate_spatial_bounds}
        
        ## parameters for the loading loop
        to_load = {'temporal':{'cls':NcTemporalDimension,'adds':_get_temporal_adds_,'axis':'T','name_uid':'tid','name_value':'time'},
                   'level':{'cls':NcVectorDimension,'adds':None,'axis':'Z','name_uid':'lid','name_value':'level'},
                   'row':{'cls':NcVectorDimension,'adds':adds_row_col,'axis':'Y','name_uid':'row_id','name_value':'row'},
                   'col':{'cls':NcVectorDimension,'adds':adds_row_col,'axis':'X','name_uid':'col_id','name_value':'col'},
                   'realization':{'cls':NcVectorDimension,'adds':None,'axis':'R','name_uid':'rlz_id','name_value':'rlz'}}
        loaded = {}
        
        for k,v in to_load.iteritems():
            ## this is the string axis representation
            axis_value = v['axis'] or v['cls']._axis
            ## pull the axis information out of the dimension map
            ref_axis = self._source_metadata['dim_map'].get(axis_value)
            ## if the axis is not represented, fill it with none. this happens
            ## when a dataset does not have a vertical level or projection axis
            ## for example.
            if ref_axis is None:
                fill = None
            else:
                ref_variable = self._source_metadata['variables'].get(ref_axis['variable'])
                
                ## for data with a projection/realization axis there may be no 
                ## associated variable.
                try:
                    ref_variable['axis'] = ref_axis
                except TypeError:
                    if axis_value == 'R' and ref_variable is None:
                        ref_variable = {'axis':ref_axis,'name':ref_axis['dimension'],'attrs':{}}
                
                ## extract the data length to use when creating the source index
                ## arrays.
                length = self._source_metadata['dimensions'][ref_axis['dimension']]['len']
                src_idx = np.arange(0,length,dtype=constants.np_int)
                
                ## get the target data type for the dimension
                try:
                    dtype = np.dtype(ref_variable['dtype'])
                ## the realization dimension may not be a associated with a variable
                except KeyError:
                    if k == 'realization' and ref_variable['axis']['variable'] == None:
                        dtype = None
                    else:
                        raise
                
                ## assemble parameters for creating the dimension class then initialize
                ## the class.
                kwds = dict(name_uid=v['name_uid'],name_value=v['name_value'],src_idx=src_idx,
                        data=self,meta=ref_variable,axis=axis_value,name=ref_variable.get('name'),
                        dtype=dtype)

                ## there may be additional parameters for each dimension.
                if v['adds'] is not None:
                    try:
                        kwds.update(v['adds'](ref_variable['attrs']))
                    ## adds may not be a callable object. assume they are a
                    ## dictionary.
                    except TypeError:
                        kwds.update(v['adds'])
                kwds.update({'name':ref_variable.get('name')})
                fill = v['cls'](**kwds)
                
            loaded[k] = fill
            
        assert_raise(set(('temporal','row','col')).issubset(set([k for k,v in loaded.iteritems() if v != None])),
                     logger='request',exc=ValueError('Target variable must at least have temporal, row, and column dimensions.'))
            
        grid = SpatialGridDimension(row=loaded['row'],col=loaded['col'])
        crs = None
        if self.s_crs is not None:
            crs = self.s_crs
        else:
            crs = self._get_crs_()
        if crs is None:
            ocgis_lh('No "grid_mapping" attribute available assuming WGS84: {0}'.format(self.uri),
                     'request',logging.WARN)
            crs = CFWGS84()
            
#        ## rotated pole coordinate systems require transforming the coordinates to
#        ## WGS84 before they may be loaded.
#        if isinstance(crs,CFRotatedPole):
#            msg = 'CFRotatedPole projection found. Transforming coordinates to WGS84 and replacing the CRS with CFWGS84'
#            ocgis_lh(msg=msg,logger='request.nc',level=logging.WARN)
#            grid = get_rotated_pole_spatial_grid_dimension(crs,grid)
#            crs = CFWGS84()
            
        spatial = SpatialDimension(name_uid='gid',grid=grid,crs=crs,abstraction=self.s_abstraction)
        
        variable_meta = deepcopy(self._source_metadata['variables'][self.variable])
        variable_units = self.units or variable_meta['attrs'].get('units')
        dtype = np.dtype(variable_meta['dtype'])
        fill_value = variable_meta['fill_value']
        variable = Variable(self.variable,self.alias,units=variable_units,meta=variable_meta,
                            data=self,conform_units_to=self.conform_units_to,dtype=dtype,
                            fill_value=fill_value)
        vc = VariableCollection(variables=[variable])
        
        ret = NcField(variables=vc,spatial=spatial,temporal=loaded['temporal'],level=loaded['level'],
                      realization=loaded['realization'],meta=deepcopy(self._source_metadata),uid=self.did)
        
        ## apply any subset parameters after the field is loaded
        if self.time_range is not None:
            ret = ret.get_between('temporal',min(self.time_range),max(self.time_range))
        if self.time_region is not None:
            ret = ret.get_time_region(self.time_region)
        if self.level_range is not None:
            ret = ret.get_between('level',min(self.level_range),max(self.level_range))
            
        return(ret)
    
    def inspect(self):
        '''Print inspection output using :class:`~ocgis.Inspect`. This is a 
        convenience method.'''
        
        ip = Inspect(request_dataset=self)
        return(ip)
    
    def inspect_as_dct(self):
        '''
        Return a dictionary representation of the target's metadata. If the variable
        is `None`. An attempt will be made to find the target dataset's time bounds
        raising a warning if none is found or the time variable is lacking units
        and/or calendar attributes.
        
        >>> rd = ocgis.RequestDataset('rhs_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc','rhs')
        >>> ret = rd.inspect_as_dct()
        >>> ret.keys()
        ['dataset', 'variables', 'dimensions', 'derived']
        >>> ret['derived']
        OrderedDict([('Start Date', '2011-01-01 12:00:00'), ('End Date', '2020-12-31 12:00:00'), ('Calendar', '365_day'), ('Units', 'days since 1850-1-1'), ('Resolution (Days)', '1'), ('Count', '8192'), ('Has Bounds', 'True'), ('Spatial Reference', 'WGS84'), ('Proj4 String', '+proj=longlat +datum=WGS84 +no_defs '), ('Extent', '(-1.40625, -90.0, 358.59375, 90.0)'), ('Interface Type', 'NcPolygonDimension'), ('Resolution', '2.80091351339')])        
        
        :rtype: :class:`collections.OrderedDict`
        '''
        ip = Inspect(request_dataset=self)
        ret = ip._as_dct_()
        return(ret)
    
    @property
    def interface(self):
        attrs = ['s_crs','t_units','t_calendar','s_abstraction']
        ret = {attr:getattr(self,attr) for attr in attrs}
        return(ret)
    
    @property
    def uri(self):
        if len(self._uri) == 1:
            ret = self._uri[0]
        else:
            ret = self._uri
        return(ret)
        
    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return(self.__dict__ == other.__dict__)
        else:
            return(False)
        
    def __str__(self):
        msg = '{0}({1})'
        argspec = inspect.getargspec(self.__class__.__init__)
        parms = []
        for name in argspec.args:
            if name == 'self':
                continue
            else:
                as_str = '{0}={1}'
                value = getattr(self,name)
                if isinstance(value,basestring):
                    fill = '"{0}"'.format(value)
                else:
                    fill = value
                as_str = as_str.format(name,fill)
            parms.append(as_str)
        msg = msg.format(self.__class__.__name__,','.join(parms))
        return(msg)
    
    def _get_crs_(self):
        crs = None
        for potential in itersubclasses(CFCoordinateReferenceSystem):
            try:
                crs = potential.load_from_metadata(self.variable,self._source_metadata)
                break
            except ProjectionDoesNotMatch:
                continue
        return(crs)
    
    def _get_uri_(self,uri,ignore_errors=False,followlinks=True):
        out_uris = []
        if isinstance(uri,basestring):
            uris = [uri]
        else:
            uris = uri
        assert(len(uri) >= 1)
        for uri in uris:
            ret = None
            ## check if the path exists locally
            if os.path.exists(uri) or '://' in uri:
                ret = uri
            ## if it does not exist, check the directory locations
            else:
                if env.DIR_DATA is not None:
                    if isinstance(env.DIR_DATA,basestring):
                        dirs = [env.DIR_DATA]
                    else:
                        dirs = env.DIR_DATA
                    for directory in dirs:
                        for filepath in locate(uri,directory,followlinks=followlinks):
                            ret = filepath
                            break
                if ret is None:
                    if not ignore_errors:
                        raise(ValueError('File not found: "{0}". Check env.DIR_DATA or ensure a fully qualified URI is used.'.format(uri)))
                else:
                    if not os.path.exists(ret) and not ignore_errors:
                        raise(ValueError('Path does not exist and is likely not a remote URI: "{0}". Set "ignore_errors" to True if this is not the case.'.format(ret)))
            out_uris.append(ret)
        return(out_uris)
    
    def _str_format_(self,value):
        ret = value
        if isinstance(value,basestring):
            value = value.lower()
            if value == 'none':
                ret = None
        else:
            ret = value
        return(ret)
    
    def _format_(self):
        if self.time_range is not None:
            self._format_time_range_()
        if self.time_region is not None:
            self._format_time_region_()
        if self.level_range is not None:
            self._format_level_range_()
        ## ensure the time range and region overlaps
        if not validate_time_subset(self.time_range,self.time_region):
            raise(DefinitionValidationError('dataset','time_range and time_region do not overlap'))
    
    def _format_time_range_(self):
        try:
            ret = [datetime.strptime(v,'%Y-%m-%d') for v in self.time_range.split('|')]
        except AttributeError:
            ret = self.time_range
        if ret[0] > ret[1]:
            raise(DefinitionValidationError('dataset','Time ordination incorrect.'))
        self.time_range = ret
        
    def _format_time_region_(self):
        if isinstance(self.time_region,basestring):
            raise(NotImplementedError)
            ret = {}
            parts = self.time_region.split('|')
            for part in parts:
                tpart,values = part.split('~')
                try:
                    values = map(int,values.split('-'))
                ## may be nonetype
                except ValueError:
                    if isinstance(values,basestring):
                        if values.lower() == 'none':
                            values = None
                    else:
                        raise
                if values is not None and len(values) > 1:
                    values = range(values[0],values[1]+1)
                ret.update({tpart:values})
        else:
            ret = self.time_region
        ## add missing keys
        for add_key in ['month','year']:
            if add_key not in ret:
                ret.update({add_key:None})
        ## confirm only month and year keys are present
        for key in ret.keys():
            if key not in ['month','year']:
                raise(DefinitionValidationError('dataset','time regions keys must be month and/or year'))
        if all([i is None for i in ret.values()]):
            ret = None
        self.time_region = ret
        
    def _format_level_range_(self):
        try:
            ret = [int(v) for v in self.level_range.split('|')]
        except AttributeError:
            ret = self.level_range
        if ret[0] > ret[1]:
            raise(DefinitionValidationError('dataset','Level ordination incorrect.'))
        self.level_range = ret
        
    def _get_meta_rows_(self):
        if self.time_range is None:
            tr = None
        else:
            tr = '{0} to {1} (inclusive)'.format(self.time_range[0],self.time_range[1])
        if self.level_range is None:
            lr = None
        else:
            lr = '{0} to {1} (inclusive)'.format(self.level_range[0],self.level_range[1])
        
        rows = ['    URI: {0}'.format(self.uri),
                '    Variable: {0}'.format(self.variable),
                '    Alias: {0}'.format(self.alias),
                '    Time Range: {0}'.format(tr),
                '    Time Region/Selection: {0}'.format(self.time_region),
                '    Level Range: {0}'.format(lr),
                '    Overloaded Parameters:',
                '      PROJ4 String: {0}'.format(self.s_crs),
                '      Time Units: {0}'.format(self.t_units),
                '      Time Calendar: {0}'.format(self.t_calendar)]
        return(rows)

    
def get_axis(dimvar,dims,dim):
    try:
        axis = getattr(dimvar,'axis')
    except AttributeError:
        ocgis_lh('Guessing dimension location with "axis" attribute missing for variable "{0}".'.format(dimvar._name),
                 logger='nc.dataset',
                 level=logging.WARN,
                 check_duplicate=True)
        axis = guess_by_location(dims,dim)
    return(axis)

def get_dimension_map(ds,var,metadata):
    dims = var.dimensions
    mp = dict.fromkeys(['T','Z','X','Y'])
    
    ## try to pull dimensions
    for dim in dims:
        dimvar = None
        try:
            dimvar = ds.variables[dim]
        except KeyError:
            ## search for variable with the matching dimension
            for key,value in metadata['variables'].iteritems():
                if len(value['dimensions']) == 1 and value['dimensions'][0] == dim:
                    dimvar = ds.variables[key]
                    break
        ## the dimension variable may not exist
        if dimvar is None:
            msg = 'Dimension variable not found for axis: "{0}". You may need to use the "dimension_map" parameter.'.format(dim)
            ocgis_lh(logger='request.nc',exc=DimensionNotFound(msg))
        axis = get_axis(dimvar,dims,dim)
        ## pull metadata information the variable and dimension names
        mp[axis] = {'variable':dimvar._name,'dimension':dim}
        try:
            mp[axis].update({'pos':var.dimensions.index(dimvar._name)})
        except ValueError:
            ## variable name may differ from the dimension name
            mp[axis].update({'pos':var.dimensions.index(dimvar.dimensions[0])})
        
    ## look for bounds variables
    bounds_names = set(constants.name_bounds)
    for key,value in mp.iteritems():
        if value is None:
            continue
        bounds_var = None
        var = ds.variables[value['variable']]
        intersection = list(bounds_names.intersection(set(var.ncattrs())))
        try:
            bounds_var = ds.variables[getattr(var,intersection[0])]._name
        except KeyError:
            ## the data has listed a bounds variable, but the variable is not
            ## actually present in the dataset.
            ocgis_lh('Bounds listed for variable "{0}" but the destination bounds variable "{1}" does not exist.'.format(var._name,getattr(var,intersection[0])),
                             logger='nc.dataset',
                             level=logging.WARNING,
                             check_duplicate=True)
            bounds_var = None
        except IndexError:
            ## if no bounds variable is found for time, it may be a climatological.
            if key == 'T':
                try:
                    bounds_var = getattr(ds.variables[value['variable']],'climatology')
                    ocgis_lh('Climatological bounds found for variable: {0}'.format(var._name),
                             logger='request.nc',
                             level=logging.INFO)
                ## climatology is not found on time axis
                except AttributeError:
                    pass
            ## bounds variable not found by other methods
            if bounds_var is None:
                ocgis_lh('No bounds attribute found for variable "{0}". Searching variable dimensions for bounds information.'.format(var._name),
                         logger='request.nc',
                         level=logging.WARN,
                         check_duplicate=True)
                bounds_names_copy = bounds_names.copy()
                bounds_names_copy.update([value['dimension']])
                for key2,value2 in metadata['variables'].iteritems():
                    intersection = bounds_names_copy.intersection(set(value2['dimensions']))
                    if len(intersection) == 2:
                        bounds_var = ds.variables[key2]._name
        value.update({'bounds':bounds_var})
    return(mp)

def guess_by_location(dims,target):
    mp = {3:{0:'T',1:'Y',2:'X'},
          4:{0:'T',2:'Y',3:'X',1:'Z'}}
    return(mp[len(dims)][dims.index(target)])
