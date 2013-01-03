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
from ocgis.exc import UniqueIdNotFound


class OcgVariable(object):
    
    def __init__(self,name,value,temporal,spatial,level=None,uri=None,alias=None):
        assert(value.shape[0] == len(temporal.value))
        if len(value) > 0:
            if level is None:
                assert(value.shape[1] == 1)
            assert(np.all(value.shape[2:] == spatial.value.shape))
        
        self._name = name
        self.alias = alias
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
        
    @property
    def name(self):
        if self.alias is None:
            ret = self._name
        else:
            ret = self.alias
        return(ret)
        
    @classmethod
    def get_empty(cls,name,uri=None):
        temporal = TemporalDimension(np.empty(0),np.empty(0))
        spatial = SpatialDimension(np.empty(0),np.empty(0),np.empty(0))
        value = np.empty(0)
        ret = cls(name,value,temporal,spatial,uri=uri)
        ret._is_empty = True
        return(ret)
        
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
        self.cid = StringIdentifier() ## calculations
        self.vid = StringIdentifier() ## variables
        self.did = StringIdentifier() ## dataset (uri)
        ## variable storage
        self.variables = OrderedDict()
        ## storage for multivariate calculations
        self.calc_multi = OrderedDict()
        
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
        self.vid.add(var.name)
        self.did.add(var.uri)


class StringIdentifier(object):
    
    def __init__(self):
        self._curr = 1
        self.uid = np.empty(0,dtype=int)
        self.value = np.empty(0,dtype=object)

    def __repr__(self):
        msg = ('  uid={0}\n'
               'value={1}').format(self.uid,self.value)
        return(msg)
        
    def __len__(self):
        return(self.uid.shape[0])
        
    def add(self,value):
        idx = self.value == [value]
        try:
            should_add = not idx.any()
        except AttributeError:
            should_add = not idx
        if should_add:
            self.uid = np.concatenate((self.uid,np.array(self._get_curr_(1))))
            self.value = np.concatenate((self.value,[value]))
    
    def get(self,value):
        idx = self.value == [value]
        try:
            return(int(self.uid[idx]))
        except TypeError:
            if not idx.any():
                raise(UniqueIdNotFound)
    
    def _get_curr_(self,n):
        ret = np.arange(self._curr,self._curr+n,dtype=int)
        self._curr = self._curr + n
        return(ret)
    
 
class ArrayIdentifier(StringIdentifier):
    
    def __init__(self,ncol,dtype=object):
        self._curr = 1
        self.uid = np.empty(0,dtype=int)
        self.value = np.empty((0,ncol),dtype=dtype)
        
    def __iter__(self):
        uid = self.uid
        value = self.value
        template = [None]*(value.shape[1]+1)
        for idx in range(uid.shape[0]):
            template[0] = uid[idx]
            template[1:] = value[idx,:]
            yield(template)
        
    def add(self,values,uids):
        values = np.array(values)
        uids = np.array(uids)
        adds = np.zeros(values.shape[0],dtype=bool)
        for idx in range(adds.shape[0]):
            eq = (self.value == values[idx,:]).all(axis=1)
            if not eq.any():
                adds[idx] = True
        if adds.any():
            new_values = values[adds,:]
            new_uids = uids[adds]
            shp = new_values.shape[0]
            self.uid = np.resize(self.uid,self.uid.shape[0]+shp)
            self.value = np.resize(self.value,(self.value.shape[0]+shp,self.value.shape[1]))
            self.uid[-shp:] = new_uids
            self.value[-shp:] = new_values
        self._update_()
                
    def get(self,value):
        value = np.array(value)
        idx = (self.value == value).all(axis=1)
        return(int(self.uid[idx]))
    
    def _update_(self):
        uid = self.uid
        if uid.shape[0] > np.unique(uid).shape[0]:
            idxs = np.arange(uid.shape[0])
            for ii in uid.flat:
                eq = uid == ii
                if eq.sum() > 1:
                    umax = uid.max()
                    new_uids = np.arange(umax+1,umax+eq.sum())
                    uid[idxs[eq][1:]] = new_uids
        
        
class GeometryIdentifier(ArrayIdentifier):
    
    def __init__(self):
        self.uid = None
        self.ugid = None
        self.value = None
        
    def add(self,values,uids,ugid):
        if self.uid is None:
            self.value = values
            self.uid = uids
            self.ugid = np.repeat(ugid,self.uid.shape[0])
        else:
            adds = np.zeros(values.shape[0],dtype=bool)
            equals = self._equals_
            for idx in range(adds.shape[0]):
                eq = equals(values[idx],ugid)
                if not eq.any():
                    adds[idx] = True
            if adds.any():
                uid_adds = uids[adds]
                self.value = np.concatenate((self.value,values[adds]))
                self.uid = np.concatenate((self.uid,uid_adds))
                self.ugid = np.concatenate((self.ugid,np.repeat(ugid,uid_adds.shape[0])))
        self._update_()

    def get(self,value,ugid):
        idx = self._equals_(value,ugid)
        return(int(self.uid[idx]))
            
    def _equals_(self,value,ugid):
        ## stores equal geometry logical
        eq = np.zeros(self.uid.shape[0],dtype=bool)
        ## index array used for iteration
        idxs = np.arange(0,eq.shape[0])
        ## reference to geometries
        geoms = self.value
        ## first see if the user geometry is stored
        eq_ugid = self.ugid == ugid
        if eq_ugid.any():
            for idx in idxs[eq_ugid]:
                if geoms[idx].equals(value):
                    eq[idx] = True
        return(eq)
