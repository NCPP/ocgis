from datetime import datetime
from copy import deepcopy
import os
from ocgis import env, constants
from ocgis.util.helpers import locate
from ocgis.exc import DefinitionValidationError
from collections import OrderedDict
from ocgis.util.inspect import Inspect
from ocgis.interface.nc.dataset import NcDataset


class RequestDataset(object):
    '''A :class:`ocgis.RequestDataset` contains all the information necessary to find
    and subset a variable (by time and/or level) contained in a local or 
    OpenDAP-hosted CF dataset.
    
    >>> from ocgis import RequestDataset
    >>> uri = 'http://some.opendap.dataset
    >>> variable = 'tasmax'
    >>> rd = RequestDataset(uri,variable)
    
    :param uri: The absolute path (URLs included) to the dataset's location.
    :type uri: str
    :param variable: The target variable.
    :type variable: str
    :param alias: An alternative name to identify the returned variable's data. If `None`, this defaults to `variable`. If variables having the same name occur in a request, this value will be required.
    :type alias: str
    :param time_range: Upper and lower bounds for time dimension subsetting. If `None`, return all time points.
    :type time_range: [:class:`datetime.datetime`, :class:`datetime.datetime`]
    :param level_range: Upper and lower bounds for level dimension subsetting. If `None`, return all levels.
    :type level_range: [int, int]
    :param s_proj: A `PROJ4 string`_ describing the dataset's spatial reference.
    :type s_proj: str
    :param t_units: Overload the autodiscover `time units`_.
    :type t_units: str
    :param t_calendar: Overload the autodiscover `time calendar`_.
    :type t_calendar: str
    
    .. _PROJ4 string: http://trac.osgeo.org/proj/wiki/FAQ
    .. _time units: http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html#num2date
    .. _time calendar: http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html#num2date
    '''
    _Dataset = NcDataset
    
    def __init__(self,uri=None,variable=None,alias=None,time_range=None,level_range=None,
                 s_proj=None,t_units=None,t_calendar=None,did=None,meta=None):
        self._uri = self._get_uri_(uri)
        self.variable = variable
        self.alias = self._str_format_(alias) or variable
        self.time_range = deepcopy(time_range)
        self.level_range = deepcopy(level_range)
        self.s_proj = self._str_format_(s_proj)
        self.t_units = self._str_format_(t_units)
        self.t_calendar = self._str_format_(t_calendar)
        self.did = did
        self.meta = meta or {}
        self._ds = None
        
        self._format_()
    
    def inspect(self):
        '''Print inspection output using :class:`~ocgis.Inspect`. This is a 
        convenience method.'''
        
        ip = Inspect(self.uri,variable=self.variable,
                     interface_overload=self.interface)
        return(ip)
    
    @property
    def ds(self):
        if self._ds is None:
            iface = self.interface
            try:
                iface.update({'request_dataset':self,
                              'abstraction':env.ops.abstraction})
            ## env likely not present
            except AttributeError:
                iface.update({'request_dataset':self,
                              'abstraction':None})
            self._ds = self._Dataset(**iface)
        return(self._ds)
    
    @property
    def interface(self):
        attrs = ['s_proj','t_units','t_calendar']
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
        
    def __repr__(self):
        msg = '<{0} ({1})>'.format(self.__class__.__name__,self.alias)
        return(msg)
    
    def __getitem__(self,item):
        ret = getattr(self,item)
        return(ret)
    
    def copy(self):
        return(deepcopy(self))
    
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
        if self.level_range is not None:
            self._format_level_range_()
    
    def _format_time_range_(self):
        try:
            ret = [datetime.strptime(v,'%Y-%m-%d') for v in self.time_range.split('|')]
            ref = ret[1]
            ret[1] = datetime(ref.year,ref.month,ref.day,23,59,59)
        except AttributeError:
            ret = self.time_range
        if ret[0] > ret[1]:
            raise(DefinitionValidationError('dataset','Time ordination incorrect.'))
        self.time_range = ret
        
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
                '    Level Range: {0}'.format(lr),
                '    Overloaded Parameters:',
                '      PROJ4 String: {0}'.format(self.s_proj),
                '      Time Units: {0}'.format(self.t_units),
                '      Time Calendar: {0}'.format(self.t_calendar)]
        return(rows)
    
    
class RequestDatasetCollection(object):
    '''Contains business logic ensuring multi-:class:`ocgis.RequestDataset` objects are
    compatible.
    
    >>> from ocgis import RequestDatasetCollection, RequestDataset
    >>> uris = ['http://some.opendap.dataset1', 'http://some.opendap.dataset2']
    >>> variables = ['tasmax', 'tasmin']
    >>> request_datasets = [RequestDatset(uri,variable) for uri,variable in zip(uris,variables)]
    >>> rdc = RequestDatasetCollection(request_datasets)
    ...
    >>> # Update object in place.
    >>> rdc = RequestDatasetCollection()
    >>> for rd in request_datasets:
    ...     rdc.update(rd)
    
    :param request_datasets: A sequence of :class:`ocgis.RequestDataset` objects.
    :type request_datasets: sequence of :class:`ocgis.RequestDataset` objects
    '''
    
    def __init__(self,request_datasets=[]):
        self._s = OrderedDict()
        self._did = []
        for rd in request_datasets:
            self.update(rd)
            
    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return(self.__dict__ == other.__dict__)
        else:
            return(False)
        
    def __len__(self):
        return(len(self._s))
        
    def __repr__(self):
        msg = ['{0} RequestDataset(s) in collection:'.format(len(self))]
        for rd in self:
            msg.append('  {0}'.format(rd.__repr__()))
        return('\n'.join(msg))
        
    def __iter__(self):
        for value in self._s.itervalues():
            yield(value)
            
    def __getitem__(self,index):
        try:
            ret = self._s[index]
        except KeyError:
            key = self._s.keys()[index]
            ret = self._s[key]
        return(ret)
    
    def keys(self):
        return(self._s.keys())
    
    def update(self,request_dataset):
        """Add a :class:`ocgis.RequestDataset` to the collection.
        
        :param request_dataset: The :class:`ocgis.RequestDataset` to add.
        :type request_dataset: :class:`ocgis.RequestDataset`
        """
        try:
            alias = request_dataset.alias
        except AttributeError:
            request_dataset = RequestDataset(**request_dataset)
            alias = request_dataset.alias
            
        if request_dataset.did is None:
            if len(self._did) == 0:
                did = 1
            else:
                did = max(self._did) + 1
            self._did.append(did)
            request_dataset.did = did
        else:
            self._did.append(request_dataset.did)
            
        if alias in self._s:
            raise(KeyError('Alias "{0}" already in collection.'\
                           .format(request_dataset.alias)))
        else:
            self._s.update({request_dataset.alias:request_dataset})
            
    def validate(self):
        ## confirm projections are equivalent
        projections = [rd.ds.spatial.projection.sr.ExportToProj4() for rd in self]
        if len(set(projections)) == 2:
            if constants.reference_projection is None:
                raise(ValueError('projections for input datasets must be equivalent if constants.reference_projection is None'))
            
    def _get_meta_rows_(self):
        rows = ['dataset=']
        for value in self._s.itervalues():
            rows += value._get_meta_rows_()
            rows.append('')
        return(rows)
