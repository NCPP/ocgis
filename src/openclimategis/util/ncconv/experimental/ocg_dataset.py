import netCDF4 as nc
import numpy as np
from shapely import prepared


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
        
        ## these are base numpy arrays used by spatial operations. -------------

        ## four numpy arrays one for each bounding coordinate of a polygon
        self.min_col,self.min_row = np.meshgrid(self.col_bnds[:,0],self.row_bnds[:,0])
        self.max_col,self.max_row = np.meshgrid(self.col_bnds[:,1],self.row_bnds[:,1])
        ## these are the original indices of the row and columns. they are
        ## referenced after the spatial subset to retrieve data from the dataset
        self.real_col,self.real_row = np.meshgrid(np.arange(0,len(self.col_bnds)),
                                                  np.arange(0,len(self.row_bnds)))
        
    def broadcast_geom(self):
        """
        returns -- matrix of same geographic dimension with shapely Polygon
            objects as value
        """
        pass
        
    def subset(self,geom): ## intersects + touches
        """
        geom -- shapely Polygon object
        return -- SubOcgDataset
        """
        pass
    
    def mapped_subset(self,geom,cell_dimension=None,max_proc=None):
        """
        returns -- dictionary containing ids and SubOcgDataset objects for
            sending to parallel
        """
        pass
        

class SubOcgDataset(object):
    
    def __init__(self,geometry,id=None):
        """
        geometry -- numpy array with dimension(ncol,nrow) of shapely Polygon 
            objects (optionally None at is_masked locations)
        """
        
        self.id = id
        self.geometry = geometry
        ## calculate nominal weights
        self.weight = np.zeros(self.geometry.shape,dtype=float)
        
    def clip(self,igeom):
        prep_igeom = prepared.prep(igeom)
        for i in range(0,self.geometry.shape[0]):
            for j in range(0,self.geometry.shape[1]):
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
    
    def keep(self,prep_igeom,igeom,target):
        if prep_igeom.intersects(target) and not target.touches(igeom):
            ret = True
        else:
            ret = False
        return(ret)