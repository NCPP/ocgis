import netCDF4 as nc
from shapely import prepared
from ocgis.interface.interface import GlobalInterface
from ocgis.util.helpers import keep, sub_range, iter_array
import numpy as np
import ocgis.exc as exc
from ocgis.api.dataset.collection.collection import OcgVariable
from ocgis.api.dataset.collection.dimension import TemporalDimension,\
    LevelDimension, SpatialDimension


class OcgDataset(object):
    """
    Wraps and netCDF4-python Dataset object providing extraction methods by 
    spatial and temporal queries.
    
    interface_overload -- dictionary containing overloaded parameters for interface
        objects
    """
    
    def __init__(self,dataset,interface_overload={}):
        self.uri = dataset['uri']
        self.variable = dataset['variable']
        self._interface_overload = interface_overload

        ## construct interface
        self.dataset = self.connect(self.uri)
        try:
            self.i = GlobalInterface(self.dataset,self.variable,overload=interface_overload)
        finally:
            self.dataset.close()
            
    def __getstate__(self):
        self.i = None
        self.dataset = None
        return(self.__dict__)
    
    def __setstate__(self,state):
        self.__init__({'uri':state['uri'],'variable':state['variable']},
                      interface_overload=state['_interface_overload'])
    
    def connect(self,uri):
        try:
            ret = nc.Dataset(uri,'r')
        except TypeError:
            ret = nc.MFDataset(uri)
        return(ret)
    
    def check_extent(self,target):
        extent = self.i.spatial.extent()
        return(keep(prepared.prep(extent),extent,target))
    
    def check_masked(self,var_name,polygon):
        try:
            self.subset(var_name,
                        polygon=polygon,
                        time_range=[self.i.temporal.time.value[0],
                                    self.i.temporal.time.value[0]])
            ret = True
        except exc.MaskedDataError:
            ret = False
        return(ret)
        
#    def display(self,show=True,overlays=None):
#        import matplotlib.pyplot as plt
#        from descartes.patch import PolygonPatch
#        
#        ax = plt.axes()
#        if overlays is not None:
#            for geom in overlays:
#                ax.add_patch(PolygonPatch(geom,alpha=0.5,fc='#999999'))
#        ax.scatter(self.min_col,self.min_row)
#        ax.scatter(self.max_col,self.max_row)
#        if show: plt.show()
    
    def get_numpy_data(self,var,args):
        if len(args) == 3:
            npd = var[args[0],args[1],args[2]]
        if len(args) == 4:
            npd = var[args[0],args[1],args[2],args[3]]
        
        ## resize the data to match returned counts
        new_shape = [len(a) for a in args]
        try:
            npd.resize(new_shape)
        except ValueError:
            npd = np.ma.resize(npd,new_shape)
        ## ensure we have four-dimensional data.
        if len(npd.shape) == 3:
            new_shape = (npd.shape[0],1,npd.shape[1],npd.shape[2])
            try:
                npd.resize(new_shape)
            except ValueError:
                npd = np.ma.resize(npd,new_shape)
        
        return(npd)
    
    def _subset_(self,polygon=None,time_range=None,level_range=None,
                 allow_empty=False,slice_row=None,slice_column=None): ## intersects + touches
        """
        polygon -- shapely Polygon object
        return -- SubOcgDataset
        """
        try:
            ## do a quick extent check if a polygon is passed
            if polygon is not None:
                if not self.check_extent(polygon):
                    raise(exc.ExtentError)
                
            ## the initial selection
            geom,row,col = self.i.spatial.select(polygon)
            if len(row) == 0 and len(col) == 0:
                raise(exc.ExtentError)
        except exc.ExtentError:
            if allow_empty:
                return(OcgVariable.get_empty(self.variable,self.uri))
            else:
                raise
        
        ## get the number of dimensions of the target variable
        ndim = len(self.dataset.variables[self.variable].dimensions)
                
        ## get the time indices
        timeidx = self.i.temporal.subset_timeidx(time_range)
        if len(timeidx) == 0:
            raise(IndexError('time range returned no data.'))

        ## convert the level indices
        try:
            levelidx = self.i.level.levelidx
            if ndim == 4:
                if level_range is not None:
                    level_range = np.array([ii-1 for ii in level_range])
                    levelidx = sub_range(level_range)
            else:
                if level_range is not None:
                    if len(set(level_range)) == 1 and level_range[0] == 1:
                        level_range = None
                    else:
                        raise IndexError('target variable has no levels.')
        except AttributeError:
            if self.i.level is None:
                pass
            else:
                raise

            
        ## extract the data ####################################################
        
        ## get the variable from the netcdf dataset
        var = self.dataset.variables[self.variable]
        ## restructure arrays for fancy indexing in the dataset
        is_slice = False #: use for additional subsetting due to a slice being used.
        if slice_row is None:
            rowidx = sub_range(row)
            colidx = sub_range(col)
        else:
            is_slice = True
            rowidx = np.arange(*slice_row)
            colidx = np.arange(*slice_column)

        ## make the subset arguments depending on the number of dimensions
        if ndim == 3:
            args = [timeidx,rowidx,colidx]
        elif ndim == 4:
            args = [timeidx,levelidx,rowidx,colidx]
        else:
            raise(NotImplementedError('cannot hold dimension count of "{0}"'.format(ndim)))
        ## actually get the data
        npd = self.get_numpy_data(var,args)

        ## we need to remove the unwanted data and reshape in the process. first,
        ## construct the relative indices, but not if slices are used...
        if not is_slice:
            rel_mask = np.ones(npd.shape,dtype=bool)
            min_row = row.min()
            min_col = col.min()
            ## now iterate and remove the data
            for ii in iter_array(row,use_mask=False):
                rel_mask[:,:,row[ii]-min_row,col[ii]-min_col] = False
        
        ## test for masked data differently depending on slice operation
        if is_slice:
            if not hasattr(npd,'mask'):
                npd = np.ma.array(npd,mask=False)
        else:
            if hasattr(npd,'mask'):
                ## if all the data values are masked, raise an error
                if npd.mask.all():
                    if allow_empty:
                        return(OcgVariable.get_empty(self.variable,self.uri))
                    else:
                        raise(exc.MaskedDataError)
                else:
                    npd.mask = np.logical_or(npd.mask,rel_mask)
            else:
                npd = np.ma.array(npd,mask=rel_mask)
        
        ## create geometry identifier
        gid = self.i.spatial.gid[rowidx][:,colidx].\
              reshape((npd.shape[2],npd.shape[3]))
        gid = np.ma.array(gid,mask=npd.mask[0,0,:,:])
        
        ## keeping the geometry mask separate is necessary related to this error:
        ## http://projects.scipy.org/numpy/ticket/897
        geom = geom[rowidx][:,colidx].reshape((npd.shape[2],npd.shape[3]))
        geom_mask = npd.mask[0,0,:,:]
        
        ## make dimensions #####################################################
        
        if self.i.temporal.bounds is None:
            timevec_bounds = None
        else:
            timevec_bounds = self.i.temporal.bounds[timeidx,:]
        d_temporal = TemporalDimension(self.i.temporal.tid[timeidx],
                                       self.i.temporal.value[timeidx],
                                       timevec_bounds)
        
        if self.i.level is None:
            d_level = None
#            d_level = LevelDimension(is_dummy=True)
        else:
            if self.i.level.bounds is None:
                levelvec_bounds = None
            else:
                levelvec_bounds = self.i.level.bounds[levelidx]
            d_level = LevelDimension(self.i.level.lid[levelidx],
                                     self.i.level.value[levelidx],
                                                levelvec_bounds)
        
        d_spatial = SpatialDimension(gid,geom,geom_mask,weights=None)
        
        ########################################################################
        
        ocg_variable = OcgVariable(self.variable,npd,d_temporal,
                                        d_spatial,level=d_level,uri=self.uri)
        
        return(ocg_variable)
    
    def subset(self,*args,**kwds):
        try:
            self.dataset = self.connect(self.uri)
            return(self._subset_(*args,**kwds))
        finally:
            self.dataset.close()
