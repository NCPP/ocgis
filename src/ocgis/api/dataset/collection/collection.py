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
    #            import ipdb;ipdb.set_trace()
                if keep(prep_igeom,igeom,geom):
    #                import ipdb;ipdb.set_trace()
                    new_geom = igeom.intersection(geom)
                    weights[idx] = new_geom.area
                    self.spatial._value[idx] = new_geom
            ## set maximum weight to one
            self.spatial.weights = weights/weights.max()
#        return(coll)

   
class OcgCollection(object):
    
    def __init__(self,ugeom=None,projection=None):
        if ugeom is None:
            ugeom = {'ugid':1,'geom':None}
        self.ugeom = ugeom
#        self._tid_name = 'tid'
#        self._tbid_name = 'tbid'
#        self._tgid_name = 'tgid'
#        self._tgbid_name = 'tgbid'
        self._mode = 'raw'
        self.projection = projection
        
#        ## variable level identifiers
#        self.tid = Identifier()
#        self.tbid = Identifier()
#        self.tgid = Identifier()
#        self.tgbid = Identifier()
#        self.lid = Identifier(dtype=object)
#        self.lbid = Identifier(dtype=object) ## level bounds identifier
#        self.gid = Identifier(dtype=object)
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
#        ## update time group identifiers
#        self.tgid.add(var.temporal.tgdim.value)
#        self.tgbid.add(var.temporal.tgdim.bounds)

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
        
#        ## update variable identifiers #########################################
#        
#        ## time
#        self.tid.add(var.temporal.value)
#        if var.temporal.bounds is not None:
#            self.tbid.add(var.temporal.bounds)
#        else:
#            self.tbid.add(np.array([[None,None]]))
#        
#        ## level
#        add_lbid = True
#        if var.level is None:
#            self.lid.add(np.array([None]))
#            add_lbid = False
#        else:
#            self.lid.add(var.level.value)
#            if var.level.bounds is None:
#                add_lbid = False
#        if add_lbid:
#            self.lbid.add(var.level.bounds)
#        else:
#            self.lbid.add(np.array([[None,None]]))
            
#        ## geometry
#        masked = var.spatial.get_masked()
#        gid = self.gid
#        for geom in masked.compressed():
#            gid.add(np.array([[geom.wkb]]))


class Identifier(object):
    
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
        return(self.storage[:,0].astype(int))
    
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
#                self.storage.resize(shape[0]+new_values.shape[0],shape[1])
                if uids is None:
                    new_uids = self._get_curr_(new_values.shape[0])
#                    self.storage[-new_values.shape[0]:,0] = self._get_curr_(new_values.shape[0])
                else:
                    new_uids = uids[adds]
                    if len(set(new_uids).intersection(set(self.uid))) > 0:
                        raise(ValueError('uids already exist in storage.'))
#                    if np.all(uids == 100): import ipdb;ipdb.set_trace()
#                    fill = new_uids
#                    self.storage[-new_values.shape[0]:,0] = uids
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
        return(int(self.storage[cmp,[0]]))
    
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
