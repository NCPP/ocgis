from ocgis.interface import base
import numpy as np
from ocgis.util.spatial import index as si
from itertools import product
from ocgis.util.helpers import make_poly, iter_array
from shapely import prepared
import netCDF4 as nc
from ocgis.exc import DummyDimensionEncountered, EmptyData,\
    TemporalResolutionError
import datetime
from ocgis.interface.projection import get_projection, RotatedPole
from shapely.geometry.point import Point
from ocgis.util.spatial.wrap import Wrapper
from copy import copy
from ocgis import constants
from ocgis.util.logging_ocgis import ocgis_lh
import logging
from ocgis import env


class NcDimension(object):
    axis = None
    
    @classmethod
    def _load_(cls,gi,subset_by=None):
        if subset_by is not None:
            raise(NotImplementedError)
        else:
            try:
                ret = gi._load_axis_(cls)
            except DummyDimensionEncountered:
                ret = None
        if ret is not None:
            cls._set_after_load_(ret,gi)
        return(ret)
    
    @classmethod
    def _set_after_load_(cls,state,dataset):
        pass


class NcLevelDimension(NcDimension,base.AbstractLevelDimension):
    axis = 'Z'
    _name_id = 'lid'
    _name_long = 'level'


class NcRowDimension(NcDimension,base.AbstractRowDimension):
    _name_id = None
    _name_long = None
    axis = 'Y'
    
    
class NcColumnDimension(NcRowDimension,base.AbstractColumnDimension):
    axis = 'X'


class NcTemporalGroupDimension(base.AbstractTemporalGroupDimension):
    _name_id = 'tgid'
    _name_long = 'time_group'


class NcTemporalDimension(NcDimension,base.AbstractTemporalDimension):
    axis = 'T'
    _name_id = 'tid'
    _name_long = 'time'
    _dtemporal_group_dimension = NcTemporalGroupDimension
    
    def __init__(self,*args,**kwargs):
        super(self.__class__,self).__init__(*args,**kwargs)
        assert(self.dataset is not None)
        self._value_datetime = None
        self._bounds_datetime = None
    
    @property
    def extent(self):
        if self.bounds is None:
            ret = (self.value_datetime.min(),self.value_datetime.max())
        else:
            ret = (self.bounds_datetime.min(),self.bounds_datetime.max())
        return(ret)
        
    @property
    def value_datetime(self):
        if self._value_datetime is None:
            if self._get_optimized_('_value_datetime') is False:
                ocgis_lh('getting value_datetime','nc.dimension',logging.DEBUG)
                self._value_datetime = np.atleast_1d(self.get_datetime(self.value))
        return(self._value_datetime)
    
    @property
    def bounds_datetime(self):
        if self.bounds is None:
            pass
        else:
            if self._bounds_datetime is None:
                if self._get_optimized_('_bounds_datetime') is False:
                    self._bounds_datetime = np.atleast_2d(self.get_datetime(self.bounds))
        return(self._bounds_datetime)
    
    def _get_optimized_(self,attr):
        if env.OPTIMIZE_FOR_CALC:
            setattr(self,attr,env._optimize_store[self.dataset.request_dataset.alias][attr])
            ret = True
        else:
            ret = False
        return(ret)
    
    @property
    def resolution(self):
        ## resolution cannot be calculated from a single value
        if self.value.shape[0] == 1:
            raise(TemporalResolutionError)
        datetimes = []
        for ii in range(constants.resolution_limit):
            try:
                datetimes.append(self.get_datetime(self.value[ii])[0])
            except IndexError:
                if ii == self.value.shape[0]:
                    break
                else:
                    raise
        ret = np.mean([dt.days for dt in np.diff(datetimes).flat])
        return(ret)
    
    def __getitem__(self,*args,**kwds):
        ret = super(self.__class__,self).__getitem__(*args,**kwds)
        ret.units = self.units
        ret.calendar = self.calendar
        ret.name_bounds = self.name_bounds
        return(ret)
    
    def get_datetime(self,arr):
        arr = np.atleast_1d(nc.num2date(arr,self.units,calendar=self.calendar))
        dt = datetime.datetime
        for idx,t in iter_array(arr,return_value=True):
            arr[idx] = dt(t.year,t.month,t.day,
                          t.hour,t.minute,t.second)
        return(arr)
    
    def get_nc_time(self,values):
        ret = np.atleast_1d(nc.date2num(values,self.units,calendar=self.calendar))
        return(ret)
    
    def subset(self,*args,**kwds):
        try:
            if isinstance(args[0][0],datetime.datetime):
                as_nc_time = self.get_nc_time(args[0])
            else:
                ## data may already by float, so try to subset with the float date
                ## representations.
                as_nc_time = args[0]
            ret = super(self.__class__,self).subset(*as_nc_time,**kwds)
            ret.units = self.units
            ret.calendar = self.calendar
            ret.name_bounds = self.name_bounds
        ## likely a region subset
        except:
            regions = args[0]
            assert(isinstance(regions,dict))
            value = self.value_datetime
            bounds = self.bounds_datetime
            
            if regions['month'] is None and regions['year'] is None:
                ret = self
            else:
                if bounds is None:
                    ## get years and months from dates
                    parts = np.array([[dt.year,dt.month] for dt in value],dtype=int)
                    ## get matching months
                    if regions['month'] is not None:
                        idx_months = np.zeros(parts.shape[0],dtype=bool)
                        for month in regions['month']:
                            idx_months = np.logical_or(idx_months,parts[:,1] == month)
                    ## potentially return all months if none are in the region
                    ## dictionary.
                    else:
                        idx_months = np.ones(parts.shape[0],dtype=bool)
                    ## get matching years
                    if regions['year'] is not None:
                        idx_years = np.zeros(parts.shape[0],dtype=bool)
                        for year in regions['year']:
                            idx_years = np.logical_or(idx_years,parts[:,0] == year)
                    ## potentially return all years.
                    else:
                        idx_years = np.ones(parts.shape[0],dtype=bool)
                    ## combine the index arrays
                    idx_dates = np.logical_and(idx_months,idx_years)
                    ret = self[idx_dates]
                else:
                    
                    def _get_parts_(start,end,day_step=29.5):
                        parts_months = set()
                        parts_years = set()
                        delta = datetime.timedelta(days=day_step)
                        while start < end:
                            parts_months.update([start.month])
                            parts_years.update([start.year])
                            start += delta
                        return(parts_months,parts_years)
                    
                    ## get the temporal resolution
                    try:
                        res = self.resolution
                        if res > 28 and res < 31:
                            res = 'month'
                        else:
                            res = 'day'
                    except TemporalResolutionError:
                        res = 'day'
                    
                    ## assemble ranges from the bounds
                    select_years = np.zeros(bounds.shape[0],dtype=bool)
                    select_months = np.zeros(bounds.shape[0],dtype=bool)
                    if res == 'day':
                        for ii in range(bounds.shape[0]):
                            row = bounds[ii]
                            p_months,p_years = _get_parts_(row[0],row[1])
                            if regions['month'] is not None:
                                if any([month in p_months for month in regions['month']]):
                                    select_months[ii] = True
                            else:
                                select_months[:] = True
                            if regions['year'] is not None:
                                if any([year in p_years for year in regions['year']]):
                                    select_years[ii] = True
                            else:
                                select_years[:] = True
                    elif res == 'month':
                        if regions['month'] is None:
                            select_months[:] = True
                        else:
                            for ii in range(value.shape[0]):
                                if value[ii].month in regions['month']:
                                    select_months[ii] = True
                        if regions['year'] is None:
                            select_years[:] = True
                        else:
                            for ii in range(self.bounds.shape[0]):
                                row = bounds[ii]
                                p_months,p_years = _get_parts_(row[0],row[1])
                                if any([year in p_years for year in regions['year']]):
                                    select_years[ii] = True
                    select = np.logical_and(select_years,select_months)
                    ret = self[select]
        return(ret)
    
    @classmethod
    def _set_after_load_(cls,state,dataset):
        attrs = dataset.metadata['variables'][state.name]['attrs']
        state.units = dataset._t_units or attrs['units']
        state.calendar = dataset._t_calendar or attrs['calendar']


class NcSpatialDimension(base.AbstractSpatialDimension):
    _name_id = 'gid'
    _name_long = None
    
    def __init__(self,grid=None,vector=None,projection=None,abstraction='polygon',
                 row=None,column=None):
        self.abstraction = abstraction
        super(self.__class__,self).__init__(projection=projection)
        if grid is None:
            self.grid = NcGridDimension(row=row,column=column)
        else:
            self.grid = grid
        if vector is None:
            if self.abstraction == 'point' or self.grid.row.bounds is None:
                self.vector = NcPointDimension(grid=self.grid,uid=self.grid.uid)
            else:
                self.vector = NcPolygonDimension(grid=self.grid,uid=self.grid.uid)
        else:
            self.vector = vector
            
    def __getitem__(self,slc):
        grid = self.grid[slc]
        ret = self.__class__(grid=grid,projection=self.projection,
                             abstraction=self.abstraction)
        return(ret)
    
    @property
    def is_360(self):
        if self.grid.column.bounds is not None:
            check = self.grid.column.bounds
        else:
            check = self.grid.column.value
        if np.any(check > 180.):
            ret = True
        else:
            ret = False
        return(ret)
    
    @property
    def pm(self):
        
        if self.grid.column.bounds is None:
            pm = 0.0
        else:
            pm = 0.0
            ref = self.grid.column.bounds
            for idx in range(ref.shape[0]):
                if ref[idx,0] < 0 and ref[idx,1] > 0:
                    pm = ref[idx,0]
                    break
        return(pm)
    
    @property
    def weights(self):
        raise(NotImplementedError,'Use "grid" or "vector" weights.')
    
    def get_iter(self):
        geoms = self.vector.geom
        name_id = self._name_id
        uid = self.vector.uid

        ret = {}
        for ii,jj in iter_array(geoms):
            ret[name_id] = uid[ii,jj]
            yield(((ii,jj),geoms[ii,jj],ret))
    
    @classmethod
    def _load_(cls,gi,subset_by=None):
        if subset_by is not None:
            raise(NotImplementedError)
        else:
            row = gi._load_axis_(NcRowDimension)
            column = gi._load_axis_(NcColumnDimension)
            
            ## check for overloaded projections
            if gi._s_proj is None:
                projection = get_projection(gi._ds)
            else:
                projection = gi._s_proj
            
            if isinstance(projection,RotatedPole):
                import csv
                import itertools
                import subprocess
                import tempfile
                class ProjDialect(csv.excel):
                    lineterminator = '\n'
                    delimiter = '\t'
                f = tempfile.NamedTemporaryFile()
                writer = csv.writer(f,dialect=ProjDialect)
                _row = row.value
                _col = column.value
                real_idx = []
                shp = (row.shape[0],column.shape[0])
                uid = np.arange(1,(shp[0]*shp[1])+1,dtype=int).reshape(*shp)
                uid = np.ma.array(data=uid,mask=False)
                for row_idx,col_idx in itertools.product(range(row.value.shape[0]),range(column.value.shape[0])):
                    real_idx.append([col_idx,row_idx])
                    writer.writerow([_col[col_idx],_row[row_idx]])
                f.flush()
                cmd = projection._trans_proj.split(' ')
                cmd.append(f.name)
                cmd = ['proj','-f','"%.6f"','-m','57.2957795130823'] + cmd
                capture = subprocess.check_output(cmd)
                f.close()
                coords = capture.split('\n')
                new_coords = []
                for ii,coord in enumerate(coords):
                    coord = coord.replace('"','')
                    coord = coord.split('\t')
                    try:
                        coord = map(float,coord)
                    ## likely empty string
                    except ValueError:
                        if coord[0] == '':
                            continue
                        else:
                            raise
                    new_coords.append(coord)
                new_coords = np.array(new_coords)
                real_idx = np.array(real_idx)
                new_row = new_coords[:,1].reshape(*shp)
                new_col = new_coords[:,0].reshape(*shp)
                new_real_row_idx = real_idx[:,1].reshape(*shp)
                new_real_column_idx = real_idx[:,0].reshape(*shp)
                
                grid = NcGridMatrixDimension(new_row,new_col,new_real_row_idx,new_real_column_idx,uid)
                ret = cls(grid=grid,projection=projection,abstraction='point')
            else:
                ret = cls(row=row,column=column,projection=projection,
                          abstraction=gi._s_abstraction)
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
        
    def __getitem__(self,slc):
        rs,cs = slc
        row = self.row[rs]
        column = self.column[cs]
        uid = self.uid[rs,cs]
        ## we want a two-dimension uid
        if len(uid.shape) == 1:
            uid = uid.reshape(-1,1)
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
    def is_bounded(self):
        check = map(lambda x: x is not None,[self.row.bounds,self.column.bounds])
        if sum(check) == 1:
            raise(ValueError('Only one axis is bounded. This should not be possible.'))
        elif all(check):
            ret = True
        else:
            ret = False
        return(ret)
    
    @property
    def resolution(self):
        return(np.mean([self.row.resolution,self.column.resolution]))
    
    @property
    def shape(self):
        return(self.row.shape[0],self.column.shape[0])
    
    def get_iter(self):
        raise(NotImplementedError)
    
    def subset(self,polygon=None):
        if polygon is not None:
            minx,miny,maxx,maxy = polygon.bounds
            row = self.row.subset(miny,maxy)
            column = self.column.subset(minx,maxx)
            try:
                uid = self.uid[row.real_idx.min():row.real_idx.max()+1,
                               column.real_idx.min():column.real_idx.max()+1]
            ## likely empty row or column data
            except ValueError:
                if len(row.value) == 0 or len(column.value) == 0:
                    raise(EmptyData)
                else:
                    raise
            ret = self.__class__(row=row,column=column,uid=uid)
        else:
            ret = self
        return(ret)
    
    
class NcGridMatrixDimension(base.AbstractSpatialGrid):
    _name_id = None
    _name_long = None
    
    def __init__(self,row,column,real_idx_row,real_idx_column,uid):
        self.row = row
        self.column = column
        self.real_idx_row = real_idx_row
        self.real_idx_column = real_idx_column
        self.uid = uid
        self._weights = None
        
    def __getitem__(self,slc):
        idx_row,idx_col = slc
        
        sub_idx_row = self.real_idx_row[idx_row]
        slc_row = slice(sub_idx_row.min(),sub_idx_row.max()+1)
        sub_idx_col = self.real_idx_column[idx_col]
        slc_col = slice(sub_idx_col.min(),sub_idx_col.max()+1)
        
        new_row = self.row[slc_row,slc_col]
        new_column = self.column[slc_row,slc_col]
        new_real_row = self.real_idx_row[slc_row,slc_col]
        new_real_column = self.real_idx_column[slc_row,slc_col]
        new_uid = self.uid[slc_row,slc_col]

        ret = self.__class__(new_row,new_column,new_real_row,new_real_column,new_uid)
        
        return(ret)
        
    @property
    def extent(self):
        raise(NotImplementedError)
    
    @property
    def resolution(self):
        raise(NotImplementedError)
    
    @property
    def shape(self):
        return(self.row.shape)
    
    def get_iter(self):
        raise(NotImplementedError)
    
    def subset(self,polygon=None):
        if polygon is None:
            ret = copy(self)
        else:
            minx,miny,maxx,maxy = polygon.bounds
            
            lidx = self.row >= miny
            uidx = self.row <= maxy
            idx_row = np.logical_and(lidx,uidx)
            lidx = self.column >= minx
            uidx = self.column <= maxx
            idx_col = np.logical_and(lidx,uidx)
            
            ret = self[idx_row,idx_col]
        
        return(ret)


class NcPolygonDimension(base.AbstractPolygonDimension):
    
    def __init__(self,grid=None,geom=None,uid=None):
        self._geom = geom
        self._weights = None
        self.grid = grid
        self.uid = uid
        
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
        
        ret = self.__class__(grid=vd.grid,geom=geom,uid=vd.uid)
        return(ret)
    
    def get_iter(self):
        raise(NotImplementedError)
    
    def intersects(self,polygon):
        ## reset the values to ensure mask is properly applied
        self._value = None
        ## reset the weights
        self._weights = None
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
            geom[ii,jj] = test_geom
            if index_intersects(test_geom,index):
                geom_mask[ii,jj] = False
            else:
                geom_mask[ii,jj] = True

        ret = self.__class__(grid=grid,geom=geom,uid=grid.uid)
        return(ret)
    
    def unwrap(self):
        raise(NotImplementedError)
    
    def wrap(self):
        wrap = Wrapper().wrap
        geom = self.geom
        for (ii,jj),to_wrap in iter_array(geom,return_value=True):
            geom[ii,jj] = wrap(to_wrap)
    
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
    
    
class NcPointDimension(NcPolygonDimension):

    @property
    def extent(self):
        raise(NotImplementedError)

    @property
    def weights(self):
        return(np.ma.array(self.grid.weights,mask=self.geom.mask))

    def clip(self,polygon):
        return(self.intersects(polygon))
    
    def intersects(self,polygon):
        ## do the initial grid subset
        grid = self.grid.subset(polygon=polygon)
        ## a prepared polygon
        prep_polygon = prepared.prep(polygon)
        ## the fill arrays
        geom = np.ones(grid.shape,dtype=object)
        geom = np.ma.array(geom,mask=True)
        geom_mask = geom.mask
        
        try:
            row = grid.row.value
            col = grid.column.value
            for ii,jj in product(range(row.shape[0]),range(col.shape[0])):
                pt = Point(col[jj],row[ii])
                geom[ii,jj] = pt
                if prep_polygon.intersects(pt):
                    geom_mask[ii,jj] = False
                else:
                    geom_mask[ii,jj] = True
        ## NcGridMatrixDimension correction
        except AttributeError:
            _row = grid.row
            _col = grid.column
            for ii,jj in iter_array(_row):
                pt = Point(_col[ii,jj],_row[ii,jj])
                geom[ii,jj] = pt
                if prep_polygon.intersects(pt):
                    geom_mask[ii,jj] = False
                else:
                    geom_mask[ii,jj] = True
            
        ret = self.__class__(grid=grid,geom=geom,uid=grid.uid)

        return(ret)

    def _get_all_geoms_(self):
        ## the fill arrays
        geom = np.ones(self.grid.shape,dtype=object)
        geom = np.ma.array(geom,mask=False)
        ## loop performing the spatial operation
        try:
            row = self.grid.row.value
            col = self.grid.column.value
            for ii,jj in product(range(row.shape[0]),range(col.shape[0])):
                geom[ii,jj] = Point(col[jj],row[ii])
        ## NcGridMatrixDimension correction
        except AttributeError:
            _row = self.grid.row
            _col = self.grid.column
            for ii,jj in iter_array(_row):
                geom[ii,jj] = Point(_col[ii,jj],_row[ii,jj])
        return(geom)