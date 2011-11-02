import netCDF4 as nc
import numpy as np
from shapely import prepared
from helpers import *
import ipdb


class OcgDataset(object):
    """
    Wraps and netCDF4-python Dataset object providing extraction methods by 
    spatial and temporal queries.
    
    dataset -- netCDF4-python Dataset object
    **kwds -- arguments for the names of multiple configuration parameters:
        rowbnds_name
        colbnds_name
        time_name
        time_units
        level_name
        calendar
        verbose
    """
    
    def __init__(self,dataset_or_uri,**kwds):
        ## the dataset can be passed as an open object or a uri.
        if isinstance(dataset_or_uri,nc.Dataset):
            self.uri = None
            self.dataset = dataset_or_uri
        else:
            self.uri = dataset_or_uri
            self.dataset = nc.Dataset(self.uri,'r')
            
        ## extract other keyword arguments -------------------------------------
        
        self.verbose = kwds.get('verbose')
        self.rowbnds_name = kwds.get('rowbnds_name') or 'bounds_latitude'
        self.colbnds_name = kwds.get('colbnds_name') or 'bounds_longitude'
        self.time_name = kwds.get('time_name') or 'time'
        self.time_units = kwds.get('time_units') or 'days since 1950-01-01 00:00:00'
        self.calendar = kwds.get('calendar') or 'proleptic_gregorian'
        self.level_name = kwds.get('level_name') or 'levels'

        ## extract the row and column bounds from the dataset
        self.row_bnds = self.dataset.variables[self.rowbnds_name][:]
        self.col_bnds = self.dataset.variables[self.colbnds_name][:]
        
        ## convert the time vector to datetime objects
        self.timevec = nc.netcdftime.num2date(self.dataset.variables[self.time_name][:],
                                              self.time_units,
                                              self.calendar)
        self.timeidx = np.arange(0,len(self.timevec))
        
        ## pull levels if possible
        try:
            self.levelvec = self.dataset.variables[self.level_name][:]
            self.levelidx = np.arange(0,len(self.levelvec))
        except:
            pass
        
        ## these are base numpy arrays used by spatial operations. -------------

        ## four numpy arrays one for each bounding coordinate of a polygon
        self.min_col,self.min_row = np.meshgrid(self.col_bnds[:,0],self.row_bnds[:,0])
        self.max_col,self.max_row = np.meshgrid(self.col_bnds[:,1],self.row_bnds[:,1])
        ## these are the original indices of the row and columns. they are
        ## referenced after the spatial subset to retrieve data from the dataset
        self.real_col,self.real_row = np.meshgrid(np.arange(0,len(self.col_bnds)),
                                                  np.arange(0,len(self.row_bnds)))
        ## calculate approximate data resolution
        self.res = approx_resolution(self.min_col[0,:])
        ## generate unique id for each grid cell
        self.cell_ids = np.empty(self.real_col.shape,dtype=int)
        curr_id = 1
        for i,j in itr_array(self.cell_ids):
            self.cell_ids[i,j] = curr_id
            curr_id += 1
        ## set the array shape.
        self.shape = self.real_col.shape
            
    def broadcast_geom(self):
        """
        returns -- matrix of same geographic dimension with shapely Polygon
            objects as value
        """
        pass
        
    def subset(self,var_name,polygon=None,time_range=None,level_range=None): ## intersects + touches
        """
        polygon -- shapely Polygon object
        return -- SubOcgDataset
        """

        ## the base cell selection. does basic find operation to identify
        ## cells to keep.
        if polygon is not None:
            prep_polygon = prepared.prep(polygon)
            emin_col,emin_row,emax_col,emax_row = polygon.envelope.bounds
            smin_col = contains(self.min_col,emin_col,emax_col,self.res)
            smax_col = contains(self.max_col,emin_col,emax_col,self.res)
            smin_row = contains(self.min_row,emin_row,emax_row,self.res)
            smax_row = contains(self.max_row,emin_row,emax_row,self.res)
            include = np.any((smin_col,smax_col),axis=0)*np.any((smin_row,smax_row),axis=0)
        else:
            include = np.empty(self.min_row.shape,dtype=bool)
            include[:,:] = True
        
        ## construct the reference matrices
        geometry = []
        row = []
        col = []
        cell_ids = []
        idx = []
        
        ## fill the matrices if value is included
        def _append(ii,jj,geom):
            geometry.append(geom)
            row.append(self.real_row[ii,jj])
            col.append(self.real_col[ii,jj])
            cell_ids.append(self.cell_ids[ii,jj])
            idx.append([self.real_row[ii,jj],self.real_col[ii,jj]])
        
        for ii,jj in itr_array(include):
            if include[ii,jj]:
                test_geom = make_poly((self.min_row[ii,jj],self.max_row[ii,jj]),
                                      (self.min_col[ii,jj],self.max_col[ii,jj]))
                if polygon is not None and keep(prep_polygon,polygon,test_geom):
                    _append(ii,jj,test_geom)
                elif polygon is None:
                    _append(ii,jj,test_geom)
        
        
        ## get the number of dimensions of the target variable
        ndim = len(self.dataset.variables[var_name].dimensions)
                
        ## get the time indices
        if time_range is not None:
            timeidx = self.timeidx[(self.timevec>=time_range[0])*
                                   (self.timevec<=time_range[1])]
        else:
            timeidx = self.timeidx
        
        ## convert the level indices
        if ndim == 4:
            if level_range is not None:
                level_range = np.array([ii-1 for ii in level_range])
                levelidx = sub_range(level_range)
            else:
                levelidx = np.array([0])
        else:
            if level_range is not None:
                raise ValueError('Target variable has no levels.')
            
        ## extract the data
        var = self.dataset.variables[var_name]
        rowidx = sub_range(row)
        colidx = sub_range(col)
        if ndim == 3:
            npd = var[timeidx,rowidx,colidx]
        if ndim == 4:
            npd = var[timeidx,levelidx,rowidx,colidx]
        
        ## ensure we have four-dimensional data.
        len_sh = len(npd.shape)
        if ndim == 3:
            raise NotImplementedError
        if ndim == 4:
            if len_sh == 3:
                npd = npd.reshape(1,1,npd.shape[1],npd.shape[2])
        
        ## we need to remove the unwanted data and reshape in the process. first,
        ## construct the relative indices.
        if ndim == 3:
            sh = (npd.shape[1],npd.shape[2])
        if ndim == 4:
            sh = (npd.shape[2],npd.shape[3])
        rel_mask = np.repeat(False,sh[0]*sh[1]).reshape(sh)
        ## now iterate and remove the data
        min_row = min(row)
        min_col = min(col)
        for ii in idx:
            rel_mask[ii[0]-min_row,ii[1]-min_col] = True
        ipdb.set_trace()
        ## reshape the data
        if ndim == 3:
            raise NotImplementedError
            npd = npd.reshape(npd.shape[0],1,-1)
        if ndim == 4:
            npd = npd.reshape(npd.shape[0],npd.shape[1],-1)
        
#        import matplotlib.pyplot as plt
#        from descartes.patch import PolygonPatch
#        ax = plt.axes()
#        for geom in geometry:
#                ax.add_patch(PolygonPatch(geom,alpha=0.5))
#        patch1 = PolygonPatch(polygon,alpha=0.5,fc='#999999')
#        ax.add_patch(patch1)
#        ax.scatter(self.min_col,self.min_row)
#        ax.scatter(self.max_col,self.max_row)
#        plt.show()
        
#        ipdb.set_trace()

        return(SubOcgDataset(geometry,npd,cell_ids))
    
    def mapped_subset(self,geom,cell_dimension=None,max_proc=None):
        """
        returns -- dictionary containing ids and SubOcgDataset objects for
            sending to parallel
        """
        pass
        

class SubOcgDataset(object):
    
    def __init__(self,geometry,value,cell_id,id=None):
        """
        geometry -- numpy array with dimension (n) of shapely Polygon 
            objects
        value -- numpy array with dimension (level,datetime,n)
        cell_id -- numpy array containing integer unique ids for the grid cells.
            has dimension (n)
        """
        
        self.id = id
        self.geometry = np.array(geometry)
        self.value = np.array(value)
        self.cell_id = np.array(cell_id)
        ## calculate nominal weights
        self.weight = np.zeros(self.geometry.shape,dtype=float)
        
    def clip(self,igeom):
        prep_igeom = prepared.prep(igeom)
        for i,j in itr_array(self.geometry):
            geom = self.geometry[i,j]
            if geom is not None:
                if self.keep(prep_igeom,igeom,geom):
                    area_prev = geom.area()
                    geom = igeom.intersection(geom)
                    area_new = geom.area()
                    weight = area_new/area_prev
                    assert(weight != 0.0) #tdk
                    self.weight[i,j] = weight
            self.geometry[i,j] = geom