from warnings import warn
from ocgis.interface.ncmeta import NcMetadata
import inspect
import netCDF4 as nc
import datetime
from ocgis.util.helpers import iter_array, approx_resolution, contains,\
    append
import numpy as np
from ocgis.interface.projection import get_projection
from shapely.geometry.polygon import Polygon
from shapely import prepared
from shapely.geometry.point import Point
from ocgis.exc import DummyLevelEncountered


class GlobalInterface(object):
    
    def __init__(self,rootgrp,target_var,overload={}):
        self.target_var = target_var
        self._meta = NcMetadata(rootgrp)
        self._dim_map = self._get_dimension_map_(rootgrp)

        interfaces = [LevelInterface,TemporalInterface,RowInterface,ColumnInterface]
        for interface in interfaces:
            try:
                argspec = inspect.getargspec(interface.__init__)
                overloads = argspec.args[-len(argspec.defaults):]
                kwds = dict(zip(overloads,[overload.get(o) for o in overloads]))
            except TypeError:
                kwds = {}
            try:
                setattr(self,interface._name,interface(self,**kwds))
            except DummyLevelEncountered:
                setattr(self,interface._name,None)
        
        ## check for proj4 string to initialize a projection
        s_proj = overload.get('s_proj')
        if s_proj is None:
            projection = get_projection(rootgrp)
#            projection = WGS84()
        else:
            raise(NotImplementedError)
        
        ## get the geometric abstraction
        s_abstraction = overload.get('s_abstraction')
        if s_abstraction is None:
            if self._row.bounds is None:
                warn('no bounds found for spatial dimensions. abstracting to point.')
                s_abstraction = 'point'
            else:
                s_abstraction = 'polygon'
        if s_abstraction == 'polygon':
            self.spatial = SpatialInterfacePolygon(self._row,self._col,projection)
        else:
            self.spatial = SpatialInterfacePoint(self._row,self._col,projection)
    
    def _get_axis_(self,dimvar,dims,dim):
        try:
            axis = getattr(dimvar,'axis')
        except AttributeError:
            warn('guessing dimension location with "axis" attribute missing')
            axis = self._guess_by_location_(dims,dim)
        return(axis)
    
    def _get_dimension_map_(self,rootgrp):
        var = rootgrp.variables[self.target_var]
        dims = var.dimensions
        mp = dict.fromkeys(['T','Z','X','Y'])
        
        ## try to pull dimensions
        for dim in dims:
            try:
                dimvar = rootgrp.variables[dim]
            except KeyError:
                ## search for variable with the matching dimension
                for key,value in self._meta['variables'].iteritems():
                    if len(value['dimensions']) == 1 and value['dimensions'][0] == dim:
                        dimvar = rootgrp.variables[key]
                        break
            axis = self._get_axis_(dimvar,dims,dim)
            mp[axis] = {'variable':dimvar,'dimension':dim}
            
        ## look for bounds variables
        bounds_names = set(['bounds','bnds','bound','bnd'])
        bounds_names.update(['d_'+b for b in bounds_names])
        for key,value in mp.iteritems():
            if value is None:
                continue
            bounds_var = None
            var = value['variable']
            intersection = list(bounds_names.intersection(set(var.ncattrs())))
            try:
                bounds_var = rootgrp.variables[getattr(var,intersection[0])]
            except IndexError:
                warn('no bounds attribute found. searching variable dimensions for bounds information.')
                bounds_names_copy = bounds_names.copy()
                bounds_names_copy.update([value['dimension']])
                for key2,value2 in self._meta['variables'].iteritems():
                    intersection = bounds_names_copy.intersection(set(value2['dimensions']))
                    if len(intersection) == 2:
                        bounds_var = rootgrp.variables[key2]
            value.update({'bounds':bounds_var})
        return(mp)
            
    def _guess_by_location_(self,dims,target):
        mp = {3:{0:'T',1:'Y',2:'X'},
              4:{0:'T',2:'Y',3:'X',1:'Z'}}
        return(mp[len(dims)][dims.index(target)])
        
        
class AbstractInterface(object):
    _axis = None
    _name = None
    
    def __init__(self,gi):
        self.gi = gi
        self._ref = gi._dim_map[self._axis]
        if self._ref is not None:
            self._ref_var = self._ref.get('variable')
            self._ref_bnds = self._ref.get('bounds')
        else:
            if type(self) == LevelInterface:
                raise(DummyLevelEncountered)
            else:
                self._ref_var = None
                self._ref_bnds = None
        
        try:
            self.name = self._ref_var._name
        ## variables without levels will have no attributes
        except AttributeError:
            self.name = None
        try:
            self.name_bounds = self._ref_bnds._name
        except AttributeError:
            self.name_bounds = None
        
    def format(self):
        if self._ref_var is None:
            self.value = None
        else:
            self.value = self._format_value_()
        if self._ref_bnds is None:
            self.bounds = None
        else:
            self.bounds = self._format_bounds_()
        
    def _get_attribute_(self,overloaded,default,target='variable'):
        if overloaded is not None:
            ret = overloaded
        else:
            ret = getattr(self._ref[target],default)
        return(ret)
        
    def _format_value_(self):
        ret = self._ref_var[:]
        return(ret)
    
    def _format_bounds_(self):
        ret = self._ref_bnds[:]
        return(ret)
        
        
class TemporalInterface(AbstractInterface):
    _axis = 'T'
    _name = 'temporal'
    
    def __init__(self,gi,t_calendar=None,t_units=None):
        super(TemporalInterface,self).__init__(gi)
        
        self.calendar = self._get_attribute_(t_calendar,'calendar')
        self.units = self._get_attribute_(t_units,'units')
        
        self.format()
        
        self.timeidx = np.arange(0,len(self.value))
        self.tid = np.arange(1,len(self.value)+1)
        self.resolution = self.get_approx_res_days()
                
    def _format_value_(self):
        ret = nc.num2date(self._ref_var[:],self.units,self.calendar)
        self._to_datetime_(ret)
        return(ret)
    
    def _format_bounds_(self):
        ret = nc.num2date(self._ref_bnds[:],self.units,self.calendar)
        self._to_datetime_(ret)
        return(ret)
        
    def _to_datetime_(self,arr):
        for idx,t in iter_array(arr,return_value=True):
            arr[idx] = datetime.datetime(t.year,t.month,t.day,
                                         t.hour,t.minute,t.second)
            
    def subset_timeidx(self,time_range):
        if time_range is None:
            ret = self.timeidx
        else:
            if self.bounds is None:
                ret = self.timeidx[(self.value>=time_range[0])*
                                   (self.value<=time_range[1])]
            else:
                select = np.empty(self.value.shape,dtype=bool)
                for idx in np.arange(self.bounds.shape[0]):
                    bnds = self.bounds[idx,:]
                    idx1 = (time_range[0]>bnds[0])*(time_range[0]<bnds[1])
                    idx2 = (time_range[0]<=bnds[0])*(time_range[1]>=bnds[1])
                    idx3 = (time_range[1]>bnds[0])*(time_range[1]<bnds[1])
                    select[idx] = np.logical_or(np.logical_or(idx1,idx2),idx3)
                ret = self.timeidx[select]
        return(ret)
    
    def get_approx_res_days(self):
        diffs = np.array([],dtype=float)
        for tidx,tval in iter_array(self.value,return_value=True):
            try:
                diffs = np.append(diffs,
                                np.abs((tval-self.value[tidx[0]+1]).days))
            except IndexError:
                break
        return(diffs.mean())
    
    def calculate(self,values):
        ret = nc.date2num(values,self.units,calendar=self.calendar)
        return(ret)
            
            
class LevelInterface(AbstractInterface):
    _axis = 'Z'
    _name = 'level'
    
    def __init__(self,gi):
        super(LevelInterface,self).__init__(gi)
        self.format()
        
        if self.value is None:
            self.is_dummy = True
            self.value = np.array([1])
            self.levelidx = np.array([0])
            self.lid = np.array([1])
        else:
            self.is_dummy = False
            self.levelidx = np.arange(len(self.value))
            self.lid = self.levelidx + 1
        
        
class RowInterface(AbstractInterface):
    _axis = 'Y'
    _name = '_row'
    
    def __init__(self,gi):
        super(RowInterface,self).__init__(gi)
        self.format()


class ColumnInterface(AbstractInterface):
    _axis = 'X'
    _name = '_col'
    
    def __init__(self,gi):
        super(ColumnInterface,self).__init__(gi)
        self.format()
        

class AbstractSpatialInterface(object):
    
    def __init__(self,row,col,projection):
        self.row = row
        self.col = col
        self.projection = projection
        
        self.is_360,self.pm = self._get_wrapping_()
        self.resolution = self._get_resolution_()
        
        self._count = None
        
    @property
    def count(self):
        if self._count is None:
            self._count = self.gid.shape[0]*self.gid.shape[1]
        return(self._count)
        
    def select(self,polygon=None):
        if polygon is None:
            return(self._get_all_geoms_())
        else:
            return(self._select_(polygon))
        
    def _get_resolution_(self):
        return(approx_resolution(self.row.value))
        
    def _select_(self,polygon):
        raise(NotImplementedError)
        
    def _get_all_geoms_(self):
        raise(NotImplementedError)
        
    def _get_wrapping_(self):
        raise(NotImplementedError)


class SpatialInterfacePolygon(AbstractSpatialInterface):
    
    def __init__(self,*args,**kwds):
        super(self.__class__,self).__init__(*args,**kwds)
        
        self.min_col,self.min_row = self.get_min_bounds()
        self.max_col,self.max_row = self.get_max_bounds()
        
        self.real_col,self.real_row = np.meshgrid(
                                np.arange(0,len(self.col.bounds)),
                                np.arange(0,len(self.row.bounds)))

        self.shape = self.real_col.shape
        self.gid = np.ma.array(np.arange(1,self.real_col.shape[0]*
                                           self.real_col.shape[1]+1)\
                               .reshape(self.shape),
                               mask=False)
        
    def get_bounds(self,colidx):
        col,row = np.meshgrid(self.col.bounds[:,colidx],
                              self.row.bounds[:,colidx])
        return(col,row)
    
    def get_min_bounds(self):
        return(self.get_bounds(0))
    
    def get_max_bounds(self):
        return(self.get_bounds(1))
    
    def extent(self):
        minx = self.min_col.min()
        maxx = self.max_col.max()
        miny = self.min_row.min()
        maxy = self.max_row.max()
        poly = Polygon(((minx,miny),(maxx,miny),(maxx,maxy),(minx,maxy)))
        return(poly)
    
    def calc_weights(self,npd,geom):
        weight = np.ma.array(np.zeros((npd.shape[2],npd.shape[3]),dtype=float),
                             mask=npd.mask[0,0,:,:])
        for ii,jj in iter_array(weight):
            weight[ii,jj] = geom[ii,jj].area
        weight = weight/weight.max()
        return(weight)
    
    def _get_wrapping_(self):
        ## check for values over 180 in the bounds variables. if higher values
        ## exists, user geometries will need to be wrapped and data may be 
        ## wrapped later in the conversion process.
        pm = 0.0
        if np.any(self.col.bounds > 180):
            is_360 = True
            ## iterate bounds coordinates to identify upper bound for left
            ## clip polygon for geometry wrapping.
            ref = self.col.bounds
            for idx in range(ref.shape[0]):
                if ref[idx,0] < 0 and ref[idx,1] > 0:
                    pm = ref[idx,0]
                    break
        else:
            is_360 = False
            
        return(is_360,pm)
    
    def _get_all_geoms_(self):
        geom = np.empty(self.shape,dtype=object)
        min_col,max_col,min_row,max_row = self.min_col,self.max_col,self.min_row,self.max_row
        
        for ii,jj in iter_array(geom,use_mask=False):
            geom[ii,jj] = Polygon(((min_col[ii,jj],min_row[ii,jj]),
                                   (max_col[ii,jj],min_row[ii,jj]),
                                   (max_col[ii,jj],max_row[ii,jj]),
                                   (min_col[ii,jj],max_row[ii,jj])))

        row = self.real_row.reshape(-1)
        col = self.real_col.reshape(-1)
        
        return(geom,row,col)
    
    def _select_(self,polygon):
#        prep_polygon = prepared.prep(polygon)
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
        
        from ocgis.util.spatial import index as si
        grid = si.build_index_grid(30.0,polygon)
        index = si.build_index(polygon,grid)
        index_intersects = si.index_intersects
        
        ## construct the reference matrices
        geom = np.empty(self.shape,dtype=object)
        row = np.array([],dtype=int)
        col = np.array([],dtype=int)
        
        real_row = self.real_row
        real_col = self.real_col
        min_row = self.min_row
        min_col = self.min_col
        max_row = self.max_row
        max_col = self.max_col
#        append = append
        
        for ii,jj in iter_array(include,use_mask=False):
            if include[ii,jj]:
                test_geom = Polygon(((min_col[ii,jj],min_row[ii,jj]),
                                     (max_col[ii,jj],min_row[ii,jj]),
                                     (max_col[ii,jj],max_row[ii,jj]),
                                     (min_col[ii,jj],max_row[ii,jj])))
                geom[ii,jj] = test_geom
                if index_intersects(test_geom,index):
                    append(row,real_row[ii,jj])
                    append(col,real_col[ii,jj])
        
        return(geom,row,col)  


class SpatialInterfacePoint(AbstractSpatialInterface):
    
    def __init__(self,*args,**kwds):
        super(self.__class__,self).__init__(*args,**kwds)
        
        ## change how the row and column point variables are created based
        ## on the shape of the incoming coordinates.
        try:
            self.col_pt,self.row_pt = self.col.value,self.row.value
            self.real_col,self.real_row = np.meshgrid(
                                    np.arange(0,self.col.value.shape[1]),
                                    np.arange(0,self.col.value.shape[0]))
        except IndexError:
            self.col_pt,self.row_pt = np.meshgrid(self.col.value,
                                                  self.row.value)
            self.real_col,self.real_row = np.meshgrid(
                                    np.arange(0,len(self.col.value)),
                                    np.arange(0,len(self.row.value)))
        self.resolution = approx_resolution(np.ravel(self.col_pt))
        self.shape = self.real_col.shape
        self.gid = np.ma.array(np.arange(1,self.real_col.shape[0]*
                               self.real_col.shape[1]+1).reshape(self.shape),
                               mask=False)
        
    def calc_weights(self,npd,geom):
        weight = np.ma.array(np.ones((npd.shape[2],npd.shape[3]),dtype=float),
                             mask=npd.mask[0,0,:,:])
        return(weight)

    def _select_(self,polygon):
        geom = np.empty(self.shape,dtype=object)
        row = np.array([],dtype=int)
        col = np.array([],dtype=int)
#        append = append
        
        prep_polygon = prepared.prep(polygon)
        for ii,jj in iter_array(self.col_pt,use_mask=False):
            pt = Point(self.col_pt[ii,jj],self.row_pt[ii,jj])
            geom[ii,jj] = pt
            if prep_polygon.intersects(pt):
                append(row,self.real_row[ii,jj])
                append(col,self.real_col[ii,jj])
        
        return(geom,row,col)
        
    def _get_all_geoms_(self):
        geom = np.empty(self.col_pt.shape,dtype=object)
        for ii,jj in iter_array(self.col_pt,use_mask=False):
            geom[ii,jj] = Point(self.col_pt[ii,jj],self.row_pt[ii,jj])
            
        row = self.real_row.reshape(-1)
        col = self.real_col.reshape(-1)
        
        return(geom,row,col)
        
    def _get_wrapping_(self):
        if self.col.value.max() > 180:
            is_360 = True
        else:
            is_360 = False
        return(is_360,0.0)
    
    def extent(self):
        minx = self.col.value.min()
        maxx = self.col.value.max()
        miny = self.row.value.min()
        maxy = self.row.value.max()
        poly = Polygon(((minx,miny),(maxx,miny),(maxx,maxy),(minx,maxy)))
        return(poly)