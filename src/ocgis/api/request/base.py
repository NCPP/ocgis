from nc import NcRequestDataset
from collections import OrderedDict
from ocgis.util.helpers import get_iter


class RequestDataset(NcRequestDataset):
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
    :param time_region: A dictionary with keys of 'month' and/or 'year' and values as sequences corresponding to target month and/or year values. Empty region selection for a key may be set to `None`.
    :type time_region: dict
        
    >>> time_region = {'month':[6,7],'year':[2010,2011]}
    >>> time_region = {'year':[2010]}
    
    :param level_range: Upper and lower bounds for level dimension subsetting. If `None`, return all levels.
    :type level_range: [int, int]
    :param s_crs: An ~`ocgis.interface.base.crs.CoordinateReferenceSystem` object to overload the projection autodiscovery.
    :type s_proj: `ocgis.interface.base.crs.CoordinateReferenceSystem`
    :param t_units: Overload the autodiscover `time units`_.
    :type t_units: str
    :param t_calendar: Overload the autodiscover `time calendar`_.
    :type t_calendar: str
    :param s_abstraction: Abstract the geometry data to either 'point' or 'polygon'. If 'polygon' is not possible due to missing bounds, 'point' will be used instead.
    :type s_abstraction: str
    
    .. note:: The `abstraction` argument in the :class:`ocgis.OcgOperations` will overload this.
    
    :param dimension_map: Maps dimensions to axes in the case of a projection/realization axis or an uncommon axis ordering. All axes must be in the dictionary.
    :type dimension_map: dict
    
    >>> dimension_map = {'T':'time','X':'longitude','Y':'latitude','R':'projection'}
    
    .. _time units: http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html#num2date
    .. _time calendar: http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html#num2date
    '''
    pass


class RequestDatasetCollection(object):
    '''Contains business logic ensuring multiple :class:`ocgis.RequestDataset` objects are
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
        for rd in get_iter(request_datasets):
            self.update(rd)
            
    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return(self.__dict__ == other.__dict__)
        else:
            return(False)
        
    def __len__(self):
        return(len(self._s))
        
    def __str__(self):
        msg = '{0}([{1}])'
        fill = [str(rd) for rd in self]
        msg = msg.format(self.__class__.__name__,','.join(fill))
        return(msg)
        
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
            raise(KeyError('Alias "{0}" already in collection. Attempted to add dataset with URI "{1}".'\
                           .format(request_dataset.alias,request_dataset.uri)))
        else:
            self._s.update({request_dataset.alias:request_dataset})
            
    def _get_meta_rows_(self):
        rows = ['* dataset=']
        for value in self._s.itervalues():
            rows += value._get_meta_rows_()
            rows.append('')
        return(rows)
