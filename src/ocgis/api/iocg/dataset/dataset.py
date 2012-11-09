import netCDF4 as nc
from shapely import prepared
from ocgis.meta.interface.interface import GlobalInterface
from ocgis.util.helpers import keep, sub_range, iter_array
import numpy as np
import ocgis.exc as exc
from ocgis.api.iocg.dataset import collection


class OcgDataset(object):
    """
    Wraps and netCDF4-python Dataset object providing extraction methods by 
    spatial and temporal queries.
    
    uri -- location of the dataset object.
    interface_overload -- dictionary containing overloaded parameters for interface
        objects
    """
    
    def __init__(self,uri,interface_overload={}):
        self.uri = uri

        ## construct interface
        self.dataset = self.connect(uri)
        try:
            self.i = GlobalInterface(self.dataset,overload=interface_overload)
        finally:
            self.dataset.close()
    
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
            try:
                npd.resize(npd.shape[0],1,npd.shape[1],npd.shape[2])
            except ValueError:
                npd = np.ma.resize(npd,(npd.shape[0],1,npd.shape[1],npd.shape[2]))
            
        return(npd)
    
    def _subset_(self,var_name,polygon=None,time_range=None,level_range=None,
                 return_collection=True): ## intersects + touches
        """
        polygon -- shapely Polygon object
        return -- SubOcgDataset
        """
        ## do a quick extent check if a polygon is passed
        if polygon is not None:
            if not self.check_extent(polygon):
                raise(exc.ExtentError)
            
        ## the initial selection
#        self.i.spatial.selection.clear()
        geom,row,col = self.i.spatial.select(polygon)
        if len(row) == 0 and len(col) == 0:
            raise(exc.ExtentError)
        
        ## get the number of dimensions of the target variable
        ndim = len(self.dataset.variables[var_name].dimensions)
                
        ## get the time indices
        timeidx = self.i.temporal.subset_timeidx(time_range)

        ## convert the level indices
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
                    raise ValueError('Target variable has no levels.')
            
        ## extract the data ####################################################
        
        ## get the variable from the netcdf dataset
        var = self.dataset.variables[var_name]
        ## restructure arrays for fancy indexing in the dataset
        rowidx = sub_range(row)
        colidx = sub_range(col)
#        rowidx = sub_range(self.i.spatial.selection.row)
#        colidx = sub_range(self.i.spatial.selection.col)

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
        ## construct the relative indices.
        rel_mask = np.ones(npd.shape,dtype=bool)
        min_row = row.min()
        min_col = col.min()
#        min_row = min(self.i.spatial.selection.row)
#        min_col = min(self.i.spatial.selection.col)
        ## now iterate and remove the data
#        for ii in self.i.spatial.selection.idx:
#            rel_mask[:,:,ii[0]-min_row,ii[1]-min_col] = False
        for ii in iter_array(row,use_mask=False):
            rel_mask[:,:,row[ii]-min_row,col[ii]-min_col] = False
#        import ipdb;ipdb.set_trace()
        
        ## test for masked data
        if hasattr(npd,'mask'):
            ## if all the data values are masked, raise an error.
            if npd.mask.all():
                raise(exc.MaskedDataError)
            else:
                npd.mask = np.logical_or(npd.mask,rel_mask)
        else:
            npd = np.ma.array(npd,mask=rel_mask)

        ## create masked arrays for other relevant variables
        gid = self.i.spatial.gid[rowidx][:,colidx].\
              reshape((npd.shape[2],npd.shape[3]))
        gid = np.ma.array(gid,mask=npd.mask[0,0,:,:])
        
        ## keeping the geometry mask separate is necessary related to this error:
        ## http://projects.scipy.org/numpy/ticket/897
        geom = geom[rowidx][:,colidx].reshape((npd.shape[2],npd.shape[3]))
        geom_mask = npd.mask[0,0,:,:]
        
        ocg_variable = collection.OcgVariable(
          var_name,
          self.i.level.lid[levelidx],
          self.i.level.level.value[levelidx],
          npd,
          self
                                   )
        
        if return_collection:
            coll = collection.OcgCollection(
              self.i.temporal.tid[timeidx],
              gid,
              geom,
              geom_mask,
              self.i.temporal.time.value[timeidx],
              self.i.spatial.calc_weights(npd,geom),
                                         )
            return(coll,ocg_variable)
        else:
            return(ocg_variable)
    
    def subset(self,*args,**kwds):
        try:
            self.dataset = self.connect(self.uri)
            return(self._subset_(*args,**kwds))
        finally:
            self.dataset.close()
