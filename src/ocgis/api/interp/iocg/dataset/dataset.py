import netCDF4 as nc
from shapely import prepared
from ocgis.meta.interface.interface import GlobalInterface
from ocgis.api.interp.iocg.dataset import collection
from ocgis.util.helpers import keep, sub_range
import numpy as np
import ocgis.exc as exc


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
        
#    def __del__(self):
#        try:
#            self.dataset.close()
#        except:
#            pass
#        finally:
#            pass
    
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
        self.i.spatial.selection.clear()
        self.i.spatial.select(polygon)
        if self.i.spatial.selection.is_empty:
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
                if len(level_range) == 1 and level_range[0] == 1:
                    level_range = None
                else:
                    raise ValueError('Target variable has no levels.')
            
        ## extract the data
        var = self.dataset.variables[var_name]
        rowidx = sub_range(self.i.spatial.selection.row)
        colidx = sub_range(self.i.spatial.selection.col)
        
        if ndim == 3:
            args = [timeidx,rowidx,colidx]
        elif ndim == 4:
            args = [timeidx,levelidx,rowidx,colidx]
        else:
            raise(NotImplementedError('cannot hold dimension count of "{0}"'.format(ndim)))
        npd = self.get_numpy_data(var,args)
        
        ## ensure we have four-dimensional data.
        len_sh = len(npd.shape)

        if ndim == 3:
            if len_sh == 3 and len(timeidx) == 1 and level_range is None:
                npd = npd.reshape(1,1,npd.shape[1],npd.shape[2])
            elif len_sh == 3 and len(timeidx) > 1 and level_range is None:
                npd = npd.reshape(npd.shape[0],1,npd.shape[1],npd.shape[2])
            elif len_sh == 2 and len(timeidx) > 1 and level_range is None:
                npd = npd.reshape(npd.shape[0],1,npd.shape[1],npd.shape[1])
            else:
                raise(NotImplementedError)
        if ndim == 4:
            if len_sh == 3:
                npd = npd.reshape(1,1,npd.shape[1],npd.shape[2])

        ## we need to remove the unwanted data and reshape in the process. first,
        ## construct the relative indices.
        rel_mask = np.ones(npd.shape,dtype=bool)
        ## now iterate and remove the data
        min_row = min(self.i.spatial.selection.row)
        min_col = min(self.i.spatial.selection.col)
        for ii in self.i.spatial.selection.idx:
            rel_mask[:,:,ii[0]-min_row,ii[1]-min_col] = False
        
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
        geom = self.i.spatial.selection.geom[rowidx][:,colidx].reshape((npd.shape[2],npd.shape[3]))
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
