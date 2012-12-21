from collections import OrderedDict
import numpy as np
from ocgis.api.dataset.collection.dimension import TemporalDimension,\
    SpatialDimension
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.point import Point
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import cascaded_union
from shapely import prepared
from ocgis.util.helpers import iter_array, keep
from shapely.geometry.base import BaseGeometry


class OcgVariable(object):
    
    def __init__(self,name,value,temporal,spatial,level=None,uri=None):
        assert(value.shape[0] == len(temporal.value))
        if len(value) > 0:
            if level is None:
                assert(value.shape[1] == 1)
            assert(np.all(value.shape[2:] == spatial.value.shape))
        
        self.name = name
        self.value = value
        self.temporal = temporal
        self.spatial = spatial
        self.level = level
        self.uri = uri
        self.temporal_group = None
        
        ## hold aggregated values separate from raw
        self.raw_value = None
        ## hold calculated values
        self.calc_value = OrderedDict()
        
        self._is_empty = False
        self._i = None
        
    @classmethod
    def get_empty(cls,name,uri=None):
        temporal = TemporalDimension(np.empty(0),np.empty(0))
        spatial = SpatialDimension(np.empty(0),np.empty(0),np.empty(0))
        value = np.empty(0)
        ret = cls(name,value,temporal,spatial,uri=uri)
        ret._is_empty = True
        return(ret)
    
#    def snippet(self):
#        self.temporal_group._value = self.temporal_group.value[0,:].reshape(1,-1)
#        self.temporal_group.bounds = self.temporal_group.bounds[0,:].reshape(1,-1)
#        self.temporal_group._uid = self.temporal_group.uid[0].reshape(-1)
#        self.temporal_group.dgroups = self.temporal_group.dgroups[0]
#        
#        idx1 = self.temporal.value >= self.temporal_group.bounds[0,0]
#        idx2 = self.temporal.value <= self.temporal_group.bounds[0,1]
#        self.temporal.storage = self.temporal.storage[np.all(idx1*idx2,axis=1)]
        
    def group(self,*args,**kwds):
        self.temporal_group = self.temporal.group(*args,**kwds)
        
    def aggregate(self,new_id=1):
        ## will hold the unioned geometry
        new_geometry = np.empty((1,1),dtype=object)
        ## get the masked geometries
        geoms = self.spatial.value.compressed()
        if self.spatial.geomtype == 'point':
            pts = MultiPoint([pt for pt in geoms.flat])
            new_geometry[0,0] = Point(pts.centroid.x,pts.centroid.y)
        else:
            ## break out the MultiPolygon objects. inextricable geometry errors
            ## sometimes occur otherwise
            ugeom = []
            for geom in geoms:
                if isinstance(geom,MultiPolygon):
                    for poly in geom:
                        ugeom.append(poly)
                else:
                    ugeom.append(geom)
            ## execute the union
            new_geometry[0,0] = cascaded_union(ugeom)
        ## overwrite the original geometry
        self.spatial._value = new_geometry
        self.spatial._value_mask = np.array([[False]])
        self.spatial._uid = np.ma.array([[new_id]],mask=False)
        ## aggregate the values
        self.raw_value = self.value.copy()
        self.value = self._union_sum_()
            
    def _union_sum_(self):
        value = self.raw_value
        weight = self.spatial.weights
        
        ## make the output array
        wshape = (value.shape[0],value.shape[1],1,1)
        weighted = np.ma.array(np.empty(wshape,dtype=float),
                                mask=np.zeros(wshape,dtype=bool))
        ## next, weight and sum the data accordingly
        for dim_time in range(value.shape[0]):
            for dim_level in range(value.shape[1]):
                weighted[dim_time,dim_level,0,0] = np.ma.average(value[dim_time,dim_level,:,:],weights=weight)
        return(weighted)
    
    def clip(self,igeom):
        ## logic for convenience. just return the provided collection if a NoneType
        ## is passed for the 'igeom' arugment
        if igeom is not None:
            ## take advange of shapely speedups
            prep_igeom = prepared.prep(igeom)
            ## the weight array
            weights = np.empty(self.spatial.shape,dtype=float)
            weights = np.ma.array(weights,mask=self.spatial._value_mask)
            ## do the spatial operation
            for idx,geom in iter_array(self.spatial.value,
                                       return_value=True):
                if keep(prep_igeom,igeom,geom):
                    new_geom = igeom.intersection(geom)
                    weights[idx] = new_geom.area
                    self.spatial._value[idx] = new_geom
            ## set maximum weight to one
            self.spatial.weights = weights/weights.max()

   
class OcgCollection(object):
    
    def __init__(self,ugeom=None,projection=None):
        if ugeom is None:
            ugeom = {'ugid':1,'geom':None}
        self.ugeom = ugeom
        self._mode = 'raw'
        self.projection = projection
        ## collection level identifiers
        self.cid = Identifier(dtype=object) ## calculations
        self.vid = Identifier(dtype=object) ## variables
        self.did = Identifier(dtype=object) ## dataset (uri)
        ## variable storage
        self.variables = {}
        
    @property
    def geomtype(self):
        types = np.array([var.spatial.geomtype for var in self.variables.itervalues()])
        if np.all(types == 'point'):
            ret = 'point'
        elif np.all(types == 'polygon'):
            ret = 'polygon'
        else:
            raise(ValueError)
        return(ret)
    @property
    def is_empty(self):
        es = [var._is_empty for var in self.variables.itervalues()]
        if np.all(es):
            ret = True
            if np.any(np.invert(es)):
                raise(ValueError)
        else:
            ret = False
        return(ret)
    @property
    def _arch(self):
        return(self.variables[self.variables.keys()[0]])
    
    def aggregate(self,*args,**kwds):
        for var in self.variables.itervalues():
            var.aggregate(*args,**kwds)
            
    def clip(self,*args,**kwds):
        for var in self.variables.itervalues():
            var.clip(*args,**kwds)
    
    def add_calculation(self,var):
        self._mode = 'calc'
        for key in var.calc_value.keys():
            self.cid.add(key)

    def add_variable(self,var):
        ## check if the variable is already present. it is possible to request
        ## variables from different datasets with the same name. do not want to
        ## overwrite.
        if var.name in self.variables:
            raise(KeyError('variable is already present in the collection.'))
        ## add the variable to storage
        self.variables.update({var.name:var})
        
        ## update collection identifiers
        self.vid.add(np.array([var.name]))
        self.did.add(np.array([var.uri]))



class Identifier(object):
    
    def __init__(self,dtype,ncol):
        self.dtype = dtype
        self.ncol = ncol
        self.storage = None
        
    def __len__(self):
        if self.storage is None:
            return(0)
        else:
            return(self.storage.shape[0])
        
    def add(self,values,uids=None):
        values = self._format_(values)
        uids = self._format_uids_(uids)
        if self.storage is None:
            self.storage = self._make_storage_(values,uids)
        else:
            adds = self._get_adds_(values)
            if adds.any():
                new_values = values[adds]
                if uids is None:
                    new_uids = None
                else:
                    new_uids = self._format_uids_(uids)[adds]
                new_storage = self._make_storage_(new_values,new_uids)
                self.storage = np.concatenate((self.storage,new_storage))
                
    def get(self,value):
        value = self._format_(value)
        idx = (self.storage['value'] == value).squeeze()
        if idx.ndim == 1:
            idx.resize(idx.shape[0],1)
        idx2 = idx.all(axis=1)
        idx2.resize(idx2.shape[0])
        uid = self.storage['uid'][idx2,:]
        assert(uid.shape[0] == 1)
        return(int(uid))
                
    def populate(self):
        emptys = self.storage['uid'] == -1
        filled = np.invert(emptys)
        start = 1
        if filled.any():
            ref = self.storage['uid'][filled]
            if len(ref) > len(np.unique(ref)):
                raise(ValueError)
            start = self.storage['uid'][filled].max()+1
        uids = np.arange(start,start+emptys.sum()+1)
        self.storage['uid'][emptys] = uids
        
    def _format_(self,arr):
        arr = np.array(arr).reshape(-1,self.ncol)
        return(arr)
    
    def _format_uids_(self,uids):
        if uids is None:
            return(None)
        else:
            return(np.array(uids).flatten())
    
    def _get_adds_(self,values):
#        print values
#        import ipdb;ipdb.set_trace()
        
        adds = np.zeros(values.shape[0],dtype=bool)
        for idx in range(values.shape[0]):
            cmp1 = (self.storage['value'] == values[idx]).squeeze()
#            try:
            try:
                cmp2 = cmp1.all(axis=1)
            except ValueError:
                if cmp1.ndim == 1:
                    cmp2 = cmp1
                else:
                    raise
#            except ValueError:
#                cmp2 = cmp1
            if not cmp2.any():
                adds[idx] = True
        return(adds)
            
    def _make_storage_(self,values,uids):
        ret = np.empty(values.shape[0],dtype=[('uid',int,1),
                                              ('value',self.dtype,(1,self.ncol))])
        ret['value'] = values.reshape(ret['value'].shape)
        if uids is None:
            ret['uid'] = -1
        else:
            ret['uid'] = uids
        return(ret)
        


class Old2Identifier(object):
    
    def __init__(self,init_vals=None,init_uids=None,dtype=None):
        self.dtype = dtype
        self._curr = 1
        if init_vals is None:
            self.storage = None
        else:
            self.storage = self._init_storage_(init_vals,init_uids)
            
    def __len__(self):
        if self.storage is None:
            ret = 0
        else:
            ret = self.storage.shape[0]
        return(ret)
    
    @property
    def uid(self):
        try:
            return(self.storage['uid'])
        except TypeError:
            if self.storage is None:
                return(np.empty(0,dtype=int))
            else:
                raise
            
    def add(self,values,uids=None):
        values = self._reshape_(values)
        adds = np.zeros(values.shape[0],dtype=bool)
        for idx in range(values.shape[0]):
            cmp1 = self.storage['value'] == values[idx,:]
            try:
                cmp2 = cmp1.all(axis=1)
            except ValueError:
                cmp2 = cmp1
            if not cmp2.any():
                adds[idx] = True
        if adds.any():
            if uids is not None:
                import ipdb;ipdb.set_trace()
            new_values = values[adds,:]
            shp = new_values.shape[0]
            new_uids = self._get_curr_(shp)
            self._resize_(shp)
            self.storage['value'][-shp:] = new_values
            self.storage['uid'][-shp:] = new_uids
    
    def get(self,value):
        value = self._reshape_(value)
        cmp = (self.storage['value'] == value).all(axis=1)
        cmp.resize(cmp.shape[0])
        if not cmp.any():
            raise(ValueError('the requested value was not found in storage.'))
        else:
            uid = self.storage['uid'][cmp]
            try:
                assert(uid.shape[0] == 1)
            except:
                import ipdb;ipdb.set_trace()
            ret = int(uid)
        return(ret)
    
    def _reshape_(self,arr):
        try:
            if arr.ndim <= 1:
                arr = arr.reshape(-1,1)
        except AttributeError:
            arr = np.array(arr).reshape(-1,1)
        return(arr)
    
    def _resize_(self,n):
        self.storage = np.resize(self.storage,self.storage.shape[0]+n)
    
    def _init_storage_(self,init_vals,init_uids):
        init_vals = self._reshape_(init_vals)
        if self.dtype is None:
            self.dtype = init_vals.dtype
        ret = np.empty(init_vals.shape[0],dtype=[('uid',int,1),('value',self.dtype,init_vals.shape[1])])
        if init_uids is None:
            init_uids = self._get_curr_(init_vals.shape[0])
        ret['uid'][:] = init_uids
        try:
            ret['value'][:] = init_vals
        except:
            ret['value'][:] = init_vals.reshape(-1)
        return(ret)
        
    def _get_curr_(self,n):
        ret = np.arange(self._curr,self._curr+n,dtype=int)
        self._curr = self._curr + n
        return(ret)
        

class OldIdentifier(object):
    
    def __init__(self,init_vals=None,dtype=None,init_uids=None):
        if init_uids is not None:
            self._check_unique_uids_(init_uids)
        self.dtype = dtype
        if init_vals is None:
            self.storage = None
        else:
            self._init_storage_(init_vals,init_uids=init_uids)
        
    def __len__(self):
        try:
            return(self.storage.shape[0])
        except AttributeError:
            if self.storage is None:
                return(0)
    
    @property
    def uid(self):
        try:
            return(self.storage[:,0].astype(int))
        except TypeError:
            if self.storage is None:
                return(np.empty(0,dtype=int))
            else:
                raise
    
    def add(self,value,uids=None):
        if uids is not None:
            self._check_unique_uids_(uids)
        try:
            if value.ndim <= 1:
                value = value.reshape(-1,1)
        except AttributeError:
            value = np.array([[value]])
        if self.storage is None:
            self._init_storage_(value,init_uids=uids)
        else:
            adds = np.zeros(value.shape[0],dtype=bool)
            for idx in range(adds.shape[0]):
                cmp = self.storage[:,1:] == value[idx,:]
                cmp2 = cmp.all(axis=1)
                if not np.any(cmp2):
                    adds[idx] = True
            if adds.any():
                new_values = value[adds,:]
                shape = self.storage.shape
                if uids is None:
                    new_uids = self._get_curr_(new_values.shape[0])
                else:
                    new_uids = uids[adds]
                    if len(set(new_uids).intersection(set(self.uid))) > 0:
                        raise(ValueError('uids already exist in storage.'))
                self.storage.resize(shape[0]+new_values.shape[0],shape[1])
                self.storage[-new_values.shape[0]:,0] = new_uids
                self.storage[-new_values.shape[0]:,1:] = new_values
                self._curr = self.uid.max()+1
            
    def get(self,value):
        try:
            cmp = (self.storage[:,1:] == value).all(axis=1)
        except AttributeError:
            value = np.array([[value]])
            cmp = (self.storage[:,1:] == value).all(axis=1)
        try:
            return(int(self.storage[cmp,[0]]))
        except Exception as e:
            import ipdb;ipdb.set_trace()
    
    def _check_unique_uids_(self,uids):
        if len(np.unique(uids)) != len(uids):
            raise(ValueError)
    
    def _get_curr_(self,n):
        ret = np.arange(self._curr,self._curr+n,dtype=self.storage.dtype)
        self._curr = self._curr + n
        return(ret)
    
    def _init_storage_(self,init_vals,init_uids=None):
        if len(init_vals.shape) == 1:
            init_vals = init_vals.reshape(-1,1)
        shape = list(init_vals.shape)
        shape[1] += 1
        if self.dtype is None:
            self.dtype = init_vals.dtype
        self.storage = np.empty(shape,dtype=self.dtype)
        self.storage[:,1:] = init_vals
        if init_uids is None:
            self._curr = self.storage.shape[0]+1
            self.storage[:,0] = np.arange(1,self._curr,dtype=self.dtype)
        else:
            self.storage[:,0] = init_uids
            self._curr = init_uids.max()+1
