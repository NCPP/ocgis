import numpy as np
import models
from element import ElementNotFound
from warnings import warn
from ocgis.util.helpers import bounding_coords, approx_resolution, contains,\
    iter_array, make_poly, keep
from shapely.geometry.polygon import Polygon
from shapely import prepared
from shapely.geometry.point import Point
from ocgis.meta.interface.projection import get_projection
from copy import copy


class InterfaceElement(object):
    
    def __init__(self,Model):
        self._Model = Model
        
    def set(self,target,dataset):
        setattr(target,self._Model._ocg_name,self._Model(dataset))
        
        
class Interface(object):
    _Models = []
    
    def __init__(self,dataset):
        for Model in self._Models:
            InterfaceElement(Model).set(self,dataset)


class SpatialSelection(object):
    
    def __init__(self):
        self.row = []
        self.col = []
        self.idx = []
        
    def clear(self):
        self.row = []
        self.col = []
        self.idx = []


class SpatialInterface(Interface):
    
    def __init__(self,*args,**kwds):
        self.selection = SpatialSelection()
        super(SpatialInterface,self).__init__(*args,**kwds)


class SpatialInterfacePolygon(SpatialInterface):
    _Models = [
               models.RowBounds,
               models.ColumnBounds,
               models.Row,
               models.Column
               ]
    
    def __init__(self,*args,**kwds):
        super(SpatialInterfacePolygon,self).__init__(*args,**kwds)
        
        self.min_col,self.min_row = self.get_min_bounds()
        self.max_col,self.max_row = self.get_max_bounds()
        self.real_col,self.real_row = np.meshgrid(
                                np.arange(0,len(self.longitude_bounds.value)),
                                np.arange(0,len(self.latitude_bounds.value))
                                                 )
        self.resolution = approx_resolution(self.min_col[0,:])
        self.shape = self.real_col.shape
        self.gid = np.arange(1,self.real_col.shape[0]*
                               self.real_col.shape[1]+1).reshape(self.shape)
    
    def calc_weights(self,npd,geom):
        weight = np.ma.array(np.zeros((npd.shape[2],npd.shape[3]),dtype=float),
                             mask=npd.mask[0,0,:,:])
        for ii,jj in iter_array(weight):
#            try:
            weight[ii,jj] = geom[ii,jj].area
#            except:
#                import ipdb;ipdb.set_trace()
#                tdk
        weight = weight/weight.max()
        return(weight)
    
    def select(self,polygon):
        if polygon is not None:
            prep_polygon = prepared.prep(polygon)
            emin_col,emin_row,emax_col,emax_row = polygon.envelope.bounds
            smin_col = contains(self.min_col,
                                emin_col,emax_col,
                                self.resolution)
            smax_col = contains(self.max_col,
                                emin_col,emax_col,
                                self.resolution)
            smin_row = contains(self.min_row,
                                emin_row,emax_row,
                                self.resolution)
            smax_row = contains(self.max_row,
                                emin_row,emax_row,
                                self.resolution)
            include = np.any((smin_col,smax_col),axis=0)*\
                      np.any((smin_row,smax_row),axis=0)
        else:
            include = np.ones(self.shape,dtype=bool)
        
        ## construct the reference matrices
        self.selection.geom = np.empty(self.gid.shape,dtype=object)
#        row = []
#        col = []
##        gids = []
#        idx = []
        
        ## fill the matrices if value is included
        def _append(ii,jj,geom):
            self.selection.geom[ii,jj] = geom
            self.selection.row.append(self.real_row[ii,jj])
            self.selection.col.append(self.real_col[ii,jj])
#            gids.append(self.gids[ii,jj])
            self.selection.idx.append([self.real_row[ii,jj],
                                       self.real_col[ii,jj]])
        
        for ii,jj in iter_array(include,use_mask=False):
            if include[ii,jj]:
                test_geom = make_poly((self.min_row[ii,jj],
                                       self.max_row[ii,jj]),
                                      (self.min_col[ii,jj],
                                       self.max_col[ii,jj]))
                if polygon is not None and keep(prep_polygon,polygon,test_geom):
                    _append(ii,jj,test_geom)
                elif polygon is None:
                    _append(ii,jj,test_geom)
        
#        if len(self.selection.row) == 8:
#            check = np.empty(self.selection.geom.shape,dtype=object)
#            for (ii,jj),val in iter_array(self.selection.geom,use_mask=False,return_value=True):
#                if val is None:
#                    check[ii,jj] = None
#                else:
#                    check[ii,jj] = 'P'
#            import ipdb;ipdb.set_trace()
             
    def extent(self):
        minx = self.min_col.min()
        maxx = self.max_col.max()
        miny = self.min_row.min()
        maxy = self.max_row.max()
        poly = Polygon(((minx,miny),(maxx,miny),(maxx,maxy),(minx,maxy)))
        return(poly)
    
    def get_bounds(self,colidx):
        col,row = np.meshgrid(self.longitude_bounds.value[:,colidx],
                              self.latitude_bounds.value[:,colidx])
        ## some data uses 360 dynamic range for longitude coordinates. compliance
        ## with WGS84 data requires data ranging from -180 to 180.
        if col.max() > 180:
            col = col - 180
            warn(('0 to 360 longitude variable encountered. simple '
                  'remapping to [-180,180] occurred.'))
        return(col,row)
    
    def get_min_bounds(self):
        return(self.get_bounds(0))
    
    def get_max_bounds(self):
        return(self.get_bounds(1))
    
    def subset_bounds(self,polygon):
        bounds = bounding_coords(polygon)
        xbnd = self.longitude_bounds.value
        ybnd = self.latitude_bounds.value
        xbnd_idx1 = self._subset_(xbnd[:,0],bounds.min_x,bounds.max_x,ret_idx=True,method='open')
        xbnd_idx2 = self._subset_(xbnd[:,1],bounds.min_x,bounds.max_x,ret_idx=True,method='open')
        ybnd_idx1 = self._subset_(ybnd[:,0],bounds.min_y,bounds.max_y,ret_idx=True,method='open')
        ybnd_idx2 = self._subset_(ybnd[:,1],bounds.min_y,bounds.max_y,ret_idx=True,method='open')
        return(xbnd[xbnd_idx1*xbnd_idx2,:],ybnd[ybnd_idx1*ybnd_idx2,:])
    
    def subset_centroids(self,polygon):
        bounds = bounding_coords(polygon)
        y = self._subset_(self.row.value[:],bounds.min_y,bounds.max_y)
        x = self._subset_(self.col.value[:],bounds.min_x,bounds.max_x)
        return(x,y)
    
    @staticmethod
    def _subset_(ary,lower,upper,ret_idx=False,method='open'):
        if method == 'open':
            idx1 = ary >= lower
            idx2 = ary <= upper
        if method == 'closed':
            idx1 = ary > lower
            idx2 = ary < upper
        if not ret_idx:
            return(ary[idx1*idx2])
        else:
            return(idx1*idx2)


class SpatialInterfacePoint(SpatialInterface):
    _Models = [models.Row,
               models.Column]
    
    def __init__(self,*args,**kwds):
        super(SpatialInterfacePoint,self).__init__(*args,**kwds)

        ## change how the row and column point variables are created based
        ## on the shape of the incoming coordinates.
        if all([ii > 1 for ii in self.longitude.value.shape]):
            self.col_pt,self.row_pt = self.longitude.value,self.latitude.value
            self.real_col,self.real_row = np.meshgrid(
                                    np.arange(0,self.longitude.value.shape[1]),
                                    np.arange(0,self.longitude.value.shape[0]))
        else:
            self.col_pt,self.row_pt = np.meshgrid(self.longitude.value,
                                                  self.latitude.value)
            self.real_col,self.real_row = np.meshgrid(
                                    np.arange(0,len(self.longitude.value)),
                                    np.arange(0,len(self.latitude.value)))
        self.resolution = approx_resolution(np.ravel(self.col_pt))
        self.shape = self.real_col.shape
        self.gid = np.arange(1,self.real_col.shape[0]*
                               self.real_col.shape[1]+1).reshape(self.shape)
    
    def calc_weights(self,npd,geom):
        weight = np.ma.array(np.ones((npd.shape[2],npd.shape[3]),dtype=float),
                             mask=npd.mask[0,0,:,:])
        return(weight)
    
    def fill_geom(self):
        for ii,jj in iter_array(self.col_pt,use_mask=False):
            self.selection.geom[ii,jj] = Point(self.col_pt[ii,jj],self.row_pt[ii,jj])
    
    def select(self,polygon):
        self.selection.geom = np.empty(self.shape,dtype=object)
        
        if polygon is not None:
#            include = np.zeros(self.shape,dtype=bool)
            prep_polygon = prepared.prep(polygon)
            for ii,jj in iter_array(self.col_pt,use_mask=False):
                pt = Point(self.col_pt[ii,jj],self.row_pt[ii,jj])
                if prep_polygon.intersects(pt):
                    self.selection.geom[ii,jj] = pt
                    self.selection.row.append(self.real_row[ii,jj])
                    self.selection.col.append(self.real_col[ii,jj])
                    self.selection.idx.append([self.real_row[ii,jj],
                                               self.real_col[ii,jj]])
                    
        else:
            self.fill_geom()
            self.selection.row = self.real_row.flatten()
            self.selection.col = self.real_col.flatten()
            for ii,jj in iter_array(self.real_row,use_mask=False):
                self.selection.idx.append([self.real_row[ii,jj],
                                           self.real_col[ii,jj]])
#            include = np.ones(self.shape,dtype=bool)
        
        ## construct the reference matrices
#        self.selection.geom = np.empty(self.shape,dtype=object)
#        row = []
#        col = []
##        gids = []
#        idx = []
        
#        ## fill the matrices if value is included
#        def _append(ii,jj,geom):
#            self.selection.geom[ii,jj] = geom
#            self.selection.row.append(self.real_row[ii,jj])
#            self.selection.col.append(self.real_col[ii,jj])
##            gids.append(self.gids[ii,jj])
#            self.selection.idx.append([self.real_row[ii,jj],
#                                       self.real_col[ii,jj]])
#        
#        for ii,jj in iter_array(include,use_mask=False):
#            if include[ii,jj]:
#                test_geom = make_poly((self.min_row[ii,jj],
#                                       self.max_row[ii,jj]),
#                                      (self.min_col[ii,jj],
#                                       self.max_col[ii,jj]))
#                if polygon is not None and keep(prep_polygon,polygon,test_geom):
#                    _append(ii,jj,test_geom)
#                elif polygon is None:
#                    _append(ii,jj,test_geom)
             
    def extent(self):
        minx = self.longitude.value.min()
        maxx = self.longitude.value.max()
        miny = self.latitude.value.min()
        maxy = self.latitude.value.max()
        poly = Polygon(((minx,miny),(maxx,miny),(maxx,maxy),(minx,maxy)))
        return(poly)


class TemporalInterface(Interface):
    _Models = [models.Time]
    
    def __init__(self,*args,**kwds):
        super(TemporalInterface,self).__init__(*args,**kwds)
        
        self.timeidx = np.arange(0,len(self.time.value))
        self.tid = np.arange(1,len(self.time.value)+1)
        
    def subset_timeidx(self,time_range):
        if time_range is None:
            ret = self.timeidx
        else:
            ret = self.timeidx[(self.time.value>=time_range[0])*
                               (self.time.value<=time_range[1])]
        return(ret)
    
    def get_approx_res_days(self):
        diffs = np.array([],dtype=float)
        for tidx,tval in iter_array(self.time.value,return_value=True):
            try:
                diffs = np.append(diffs,
                                np.abs((tval-self.time.value[tidx[0]+1]).days))
            except IndexError:
                break
        return(diffs.mean())
        

class LevelInterface(Interface):
    _Models = [models.Level]
    
    def __init__(self,*args,**kwds):
        super(LevelInterface,self).__init__(*args,**kwds)
        
        self.levelidx = np.arange(0,len(self.level.value))
        self.lid = np.arange(1,len(self.level.value)+1)
        
        
class DummyLevelInterface(object):
    pass


class DummyLevelVariable(object):
    pass


class GlobalInterface(object):
    
    def __init__(self,dataset):
        try:
            self.spatial = SpatialInterfacePolygon(dataset)
        except ElementNotFound:
            self.spatial = SpatialInterfacePoint(dataset)
        self._projection = get_projection(dataset)
        ## necessary to rebuild after pickling
        self._projection_class = copy(self.projection.__class__)
        self.temporal = TemporalInterface(dataset)
        try:
            self.level = LevelInterface(dataset)
            self._has_level = True
        except ElementNotFound:
            warn('no level variable found.')
            self.level = DummyLevelInterface()
            self.level.level = DummyLevelVariable()
            self.level.level.value = np.array([1])
            self.level.levelidx = np.array([0])
            self._has_level = False
            self.level.lid = np.array([1])
            
    @property
    def projection(self):
        if self._projection is None:
            self._projection = self._projection_class._build_()
        return(self._projection)
    
    def __getstate__(self):
        '''SpatialReference objects are Swig objects and must handled correctly
        for the pickler.'''
        
        self.__dict__['_projection'] = None
        return(self.__dict__)
    
    def __setstate__(self,dct):
        self.__dict__.update(dct)
        self.__dict__['_projection'] = self._projection_class._build_()
