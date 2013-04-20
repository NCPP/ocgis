import base
import numpy as np
from ocgis.util.spatial import index as si
from itertools import product
from ocgis.util.helpers import make_poly, iter_array
from shapely import prepared
import netCDF4 as nc
from ocgis import constants
from warnings import warn
from abc import ABCMeta, abstractproperty


class NcDimension(object):
    __metaclass__ = ABCMeta
    
    @abstractproperty
    def axis(self): str
    
    @classmethod
    def _load_(cls,gi,subset_by=None):
        if subset_by is not None:
            raise(NotImplementedError)
        else:
            ret = gi._load_axis_(cls)
            import ipdb;ipdb.set_trace()

    
class NcLevelDimension(NcDimension,base.AbstractLevelDimension):
    axis = 'Z'


class NcRowDimension(NcDimension,base.AbstractRowDimension):
    _name_id = None
    _name_long = None
    axis = 'Y'
    
    @property
    def extent(self):
        if self.bounds is None:
            ret = (self.value.min(),self.value.max())
        else:
            ret = (self.bounds.min(),self.bounds.max())
        return(ret)
    
    
class NcColumnDimension(NcRowDimension,base.AbstractColumnDimension):
    axis = 'X'


class NcTemporalDimension(NcDimension,base.AbstractTemporalDimension):
    axis = 'T'


class NcSpatialDimension(base.AbstractSpatialDimension):
    _name_id = 'gid'
    _name_long = None
    
    def __init__(self,*args,**kwds):
        super(self.__class__,self).__init__(projection=kwds.get('projection'))
        self.grid = NcGridDimension(*args,**kwds)
        if self.grid.row.bounds is None:
            self.vector = NcPointDimension(*args,**kwds)
        else:
            self.vector = NcPolygonDimension(grid=self.grid)
    
    @property
    def weights(self):
        raise(NotImplementedError,'Use "grid" or "vector" weights.')
    
    @classmethod
    def _load_(cls,gi,subset_by=None):
        if subset_by is not None:
            raise(NotImplementedError)
        else:
            row = gi._load_axis_(NcRowDimension)
            column = gi._load_axis_(NcColumnDimension)
            ret = cls(row=row,column=column)
        return(ret)

class NcGridDimension(base.AbstractSpatialGrid):
    _name_id = None
    _name_long = None
    
    def __init__(self,row=None,column=None,uid=None):
        self.row = row
        self.column = column
        self._weights = None
        if uid is None:
            shp = (self.row.shape[0],self.column.shape[0])
            uid = np.arange(1,(shp[0]*shp[1])+1,dtype=int).reshape(*shp)
            uid = np.ma.array(data=uid,mask=False)
        self.uid = uid
        
    def __getitem__(self,idx):
        rs,cs = idx
        row = self.row[rs.start:rs.stop]
        column = self.column[cs.start:cs.stop]
        uid = self.uid[rs.start:rs.stop,cs.start:cs.stop]
        ret = self.__class__(row=row,column=column,uid=uid)
        return(ret)
        
    @property
    def extent(self):
        if self.row.bounds is None:
            attr = 'value'
        else:
            attr = 'bounds'
        row = getattr(self.row,attr)
        column = getattr(self.column,attr)
        return(column.min(),row.min(),column.max(),row.max())
    
    @property
    def resolution(self):
        return(np.mean([self.row.resolution,self.column.resolution]))
    
    @property
    def shape(self):
        return(self.row.shape[0],self.column.shape[0])
    
    def subset(self,polygon=None):
        if polygon is not None:
            minx,miny,maxx,maxy = polygon.bounds
            row = self.row.subset(miny,maxy)
            column = self.column.subset(minx,maxx)
            uid = self.uid[row.real_idx.min():row.real_idx.max()+1,
                           column.real_idx.min():column.real_idx.max()+1]
            ret = self.__class__(row=row,column=column,uid=uid)
        else:
            ret = self
        return(ret)


class NcPolygonDimension(base.AbstractPolygonDimension):
    
    def __init__(self,grid=None,geom=None,weights=None):
        assert(grid.row.bounds is not None)
        
        self._geom = geom
        self._weights = weights
        self.grid = grid
        self.uid = self.grid.uid
        
    @property
    def extent(self):
        raise(NotImplementedError)
    
    @property
    def weights(self):
        if self._weights is None:
            geom = self.geom
            weights = np.ones(geom.shape,dtype=float)
            weights = np.ma.array(weights,mask=geom.mask)
            for ii,jj in iter_array(geom):
                weights[ii,jj] = geom[ii,jj].area
            weights = weights/weights.max()
            self._weights = weights
        return(self._weights)
    
    def clip(self,polygon):
        ## perform an intersects operation first
        vd = self.intersects(polygon)
        ## prepare the geometry for intersection
        prep_igeom = prepared.prep(polygon)
        
        ## loop for the intersection
        geom = vd._geom
        for ii,jj in iter_array(geom):
            ref = geom[ii,jj]
            if not prep_igeom.contains(ref):
                new_geom = polygon.intersection(ref)
                geom[ii,jj] = new_geom
        
        ret = self.__class__(grid=vd.grid,geom=geom)
        return(ret)
    
    def intersects(self,polygon):
        ## do the initial grid subset
        grid = self.grid.subset(polygon=polygon)
        
        ## construct the spatial index
        index_grid = si.build_index_grid(30.0,polygon)
        index = si.build_index(polygon,index_grid)
        
        ## the fill arrays
        geom = np.ones(grid.shape,dtype=object)
        geom = np.ma.array(geom,mask=True)
        
        ## loop performing the spatial operation
        row = grid.row.bounds
        col = grid.column.bounds
        index_intersects = si.index_intersects
        geom_mask = geom.mask
        for ii,jj in product(range(row.shape[0]),range(col.shape[0])):
            rref = row[ii,:]
            cref = col[jj,:]
            test_geom = make_poly(rref,cref)
            if index_intersects(test_geom,index):
                geom[ii,jj] = test_geom
                geom_mask[ii,jj] = False
        
        ret = self.__class__(grid=grid,geom=geom)
        return(ret)
    
    def _get_all_geoms_(self):
        ## the fill arrays
        geom = np.ones(self.grid.shape,dtype=object)
        geom = np.ma.array(geom,mask=False)
        ## loop performing the spatial operation
        row = self.grid.row.bounds
        col = self.grid.column.bounds
        for ii,jj in product(range(row.shape[0]),range(col.shape[0])):
            rref = row[ii,:]
            cref = col[jj,:]
            geom[ii,jj] = make_poly(rref,cref)
        
        return(geom)


class NcGlobalInterface(base.AbstractGlobalInterface):
    _dtemporal = NcTemporalDimension
    _dlevel = NcLevelDimension
    _dspatial = NcSpatialDimension
#    _metdata_cls = NcMetadata

    def __init__(self,*args,**kwds):
        super(self.__class__,self).__init__(*args,**kwds)
        self.__ds = None
        self.__dim_map = None
        
    def __del__(self):
        try:
            self._ds.close()
        finally:
            pass
    
    @property
    def _dim_map(self):
        if self.__dim_map is None:
            self.__dim_map = self._get_dimension_map_()
        return(self.__dim_map)
    
    @property
    def _ds(self):
        if self.__ds is None:
            self.__ds = nc.Dataset(self.request_dataset.uri,'r')
        return(self.__ds)
    
    def _get_axis_(self,dimvar,dims,dim):
        try:
            axis = getattr(dimvar,'axis')
        except AttributeError:
            warn('guessing dimension location with "axis" attribute missing')
            axis = self._guess_by_location_(dims,dim)
        return(axis)
    
    def _get_dimension_map_(self):
        var = self._ds.variables[self.request_dataset.variable]
        dims = var.dimensions
        mp = dict.fromkeys(['T','Z','X','Y'])
        ds = self._ds
        
        ## try to pull dimensions
        for dim in dims:
            try:
                dimvar = ds.variables[dim]
            except KeyError:
                ## search for variable with the matching dimension
                for key,value in self._meta['variables'].iteritems():
                    if len(value['dimensions']) == 1 and value['dimensions'][0] == dim:
                        dimvar = ds.variables[key]
                        break
            axis = self._get_axis_(dimvar,dims,dim)
            mp[axis] = {'variable':dimvar,'dimension':dim}
            
        ## look for bounds variables
        bounds_names = set(constants.name_bounds)
        for key,value in mp.iteritems():
            if value is None:
                continue
            bounds_var = None
            var = value['variable']
            intersection = list(bounds_names.intersection(set(var.ncattrs())))
            try:
                bounds_var = ds.variables[getattr(var,intersection[0])]
            except IndexError:
                warn('no bounds attribute found. searching variable dimensions for bounds information.')
                bounds_names_copy = bounds_names.copy()
                bounds_names_copy.update([value['dimension']])
                for key2,value2 in self._meta['variables'].iteritems():
                    intersection = bounds_names_copy.intersection(set(value2['dimensions']))
                    if len(intersection) == 2:
                        bounds_var = ds.variables[key2]
            value.update({'bounds':bounds_var})
        return(mp)
            
    def _guess_by_location_(self,dims,target):
        mp = {3:{0:'T',1:'Y',2:'X'},
              4:{0:'T',2:'Y',3:'X',1:'Z'}}
        return(mp[len(dims)][dims.index(target)])
    
    def _load_axis_(self,kls):
        ref = self._dim_map[kls.axis]
        value = ref['variable'][:]
        name = ref['dimension']
        try:
            bounds = ref['bounds'][:]
        except Exception as e:
            raise(NotImplementedError)
        ret = kls(value=value,name=name,bounds=bounds)
        return(ret)
