from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from ocgis import constants, env
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from copy import deepcopy
from ocgis.calc.base import KeyedFunctionOutput


class AbstractCollection(object):
    '''Abstract base class for all collection types.'''
    __metaclass__ = ABCMeta
    
    @property
    def _archetype(self):
        return(self.variables[self.variables.keys()[0]])
    
    @property
    def projection(self):
        return(self._archetype.spatial.projection)
    
    @property
    def ugid(self):
        try:
            ret = self.ugeom.spatial.uid[0]
        ## the geometry may be empty
        except AttributeError:
            if self.ugeom is None:
                ret = 1
            else:
                raise
        return(ret)
    
    def get_headers(self,upper=False):
        ## headers may have been overloaded by operations.
        try:
            if env.ops.headers is None: #@UndefinedVariable
                headers = self._get_headers_()
            else:
                headers = env.ops.headers #@UndefinedVariable
        except AttributeError:
            ## env.ops likely not present
            headers = self._get_headers_()
        if upper:
            ret = [h.upper() for h in headers]
        else:
            ret = headers
        return(ret)
    
    def get_iter(self,with_geometry_ids=False):
        '''
        :param with_geometry_ids: If True, return a dictionary containing geometry identifiers.
        :type with_geometry_ids: bool
        '''
        headers = self.get_headers()
        for geom,attrs in self._get_iter_():
            row = [attrs[h] for h in headers]
            if with_geometry_ids:
                geom_ids = {'ugid':attrs['ugid'],'gid':attrs['gid'],'did':attrs['did']}
                yld = (geom,row,geom_ids)
            else:
                yld = (geom,row)
            yield(yld)
    
    @abstractmethod
    def _get_headers_(self): list
    
    @abstractmethod
    def _get_iter_(self): 'generator'
    
    
class RawCollection(AbstractCollection):
    
    def __init__(self,ugeom=None):
        self.ugeom = ugeom
        self.variables = OrderedDict()
    
    def _get_iter_(self):
        ## we want to break out the date parts if any date parts are present
        ## in the headers argument.
        headers = self.get_headers()
        intersection = set(headers).intersection(set(['year','month','day']))
        if len(intersection) > 0:
            add_date_parts = True
        else:
            add_date_parts = False

        vid = 1
        ugid = self.ugid
        for alias,ds in self.variables.iteritems():
            did = ds.request_dataset.did
            variable = ds.request_dataset.variable
            for geom,attrs in ds.get_iter_value():
                attrs['did'] = did
                attrs['alias'] = alias
                attrs['variable'] = variable
                attrs['vid'] = vid
                attrs['ugid'] = ugid
                if add_date_parts:
                    attrs['year'] = attrs['time'].year
                    attrs['month'] = attrs['time'].month
                    attrs['day'] = attrs['time'].day
                if type(geom) == Polygon:
                    geom = MultiPolygon([geom])
                yield(geom,attrs)
            vid += 1
            
    def _get_headers_(self):
        return(deepcopy(constants.raw_headers))
        
        
class CalcCollection(AbstractCollection):
    
    def __init__(self,raw_collection,funcs=None):
        self.ugeom = raw_collection.ugeom
        self.variables = raw_collection.variables
        self.calc = OrderedDict()
        self.funcs = funcs
    
    def _get_iter_(self):
#        headers = self.get_headers()
        vid = 1
        cid = 1
        ugid = self.ugid
        for alias,calc in self.calc.iteritems():
            ds = self.variables[alias]
            did = ds.request_dataset.did
            variable = ds.request_dataset.variable
            for calc_name,calc_value in calc.iteritems():
                for geom,attrs in ds.get_iter_value(value=calc_value,temporal_group=True):
                    attrs['did'] = did
                    attrs['variable'] = variable
                    attrs['alias'] = alias
                    attrs['calc_name'] = calc_name
                    attrs['vid'] = vid
                    attrs['cid'] = cid
                    attrs['ugid'] = ugid
#                    row = [attrs[key] for key in headers]
                    if type(geom) == Polygon:
                        geom = MultiPolygon([geom])
                    yield(geom,attrs)
                cid += 1
            vid += 1
            
    def _get_headers_(self):
        return(deepcopy(constants.calc_headers))
            
            
class MultivariateCalcCollection(CalcCollection):
    
    def _get_iter_(self):
        arch = self._archetype
        ## determine if there is a temporal grouping
        temporal_group = False if arch.temporal.group is None else True
#        headers = self.get_headers()
        cid = 1
        ugid = self.ugid
        for calc_name,calc_value in self.calc.iteritems():
            for geom,attrs in arch.get_iter_value(value=calc_value,temporal_group=temporal_group,
                                                  add_masked=True):
                attrs['calc_name'] = calc_name
                attrs['cid'] = cid
                attrs['ugid'] = ugid
#                row = [attrs[key] for key in headers]
                if type(geom) == Polygon:
                    geom = MultiPolygon([geom])
                yield(geom,attrs)
            cid += 1

    def _get_headers_(self):
        ## get the representative dataset
        arch = self._archetype
        ## determine if there is a temporal grouping
        temporal_group = False if arch.temporal.group is None else True
        if temporal_group:
            headers = super(MultivariateCalcCollection,self)._get_headers_()
#            headers = [h.upper() for h in deepcopy(super(MultivariateCalcCollection,self)._get_headers_())]
            ## the variable name is not relevant for multivariate calculations
            headers.remove('did')
            headers.remove('alias')
            headers.remove('variable')
            headers.remove('vid')
        else:
            headers = constants.multi_headers
        return(headers)


class KeyedOutputCalcCollection(CalcCollection):
    
    def _get_iter_(self):
#        headers = self._get_headers_()
        vid = 1
        cid = 1
        ugid = self.ugid
        output_keys = self._get_target_ref_().output_keys
        for alias,calc in self.calc.iteritems():
            ds = self.variables[alias]
            did = ds.request_dataset.did
            variable = ds.request_dataset.variable
            for calc_name,calc_value in calc.iteritems():
                is_sample_size = False
                for geom,base_attrs in ds.get_iter_value(value=calc_value,temporal_group=True):
                    ## try to get the shape of the structure arrays
                    try:
                        iter_shape = base_attrs['value'].shape[0]
                    except IndexError:
                        iter_shape = 1
                    for ii_value in range(iter_shape):
                        attrs = base_attrs.copy()
                        for key in output_keys:
                            try:
                                attrs[key] = attrs['value'][key][ii_value]
                            except IndexError:
                                ## likely sample size
                                is_sample_size = True
                                attrs[key] = None
                        if not is_sample_size:
                            attrs['value'] = None
                        attrs['did'] = did
                        attrs['variable'] = variable
                        attrs['alias'] = alias
                        attrs['calc_name'] = calc_name
                        attrs['vid'] = vid
                        attrs['cid'] = cid
                        attrs['ugid'] = ugid
                        if type(geom) == Polygon:
                            geom = MultiPolygon([geom])
                        yield(geom,attrs)
                cid += 1
            vid += 1
        
    def _get_headers_(self):
        headers = super(self.__class__,self)._get_headers_()
        ref = self._get_target_ref_()
        headers += ref.output_keys
        return(headers)
        
    def _get_target_ref_(self):
        for func in self.funcs:
            if issubclass(func['ref'],KeyedFunctionOutput):
                return(func['ref'])