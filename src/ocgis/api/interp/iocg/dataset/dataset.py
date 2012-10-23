import netCDF4 as nc
from shapely import prepared
from ocgis.meta.interface.interface import GlobalInterface
from ocgis.api.interp.iocg.dataset import collection
from ocgis.util.helpers import keep, sub_range
import numpy as np


class MaskedDataError(Exception):
    def __str__(self):
        return('Geometric intersection returns all masked values.')
    
    
class ExtentError(Exception):
    def __str__(self):
        return('Geometric intersection is empty.')
    
    
class EmptyDataNotAllowed(Exception):
    def __str__(self):
        return('Interesection returned empty, but empty data not allowed.')
    
    
class EmptyData(Exception):
    def __str__(self):
        return('Empty data returned.')


class OcgDataset(object):
    """
    Wraps and netCDF4-python Dataset object providing extraction methods by 
    spatial and temporal queries.
    
    uri -- location of the dataset object.
    **kwds -- arguments for the names of multiple configuration parameters:
        rowbnds_name
        colbnds_name
        time_name
        time_units
        level_name
        calendar
        verbose
    """
    
    def __init__(self,uri,**kwds):
        self.uri = uri
        self.dataset = self.connect(uri)

        ## construct interface
        self.i = GlobalInterface(self.dataset)
#        self.spatial = SpatialInterface(self.dataset,**kwds)
#        self.temporal = TemporalInterface(self.dataset,**kwds)
#        try:
#            self.level = LevelInterface(self.dataset,**kwds)
#        except PolyElementNotFound:
#            warn('No "level" variable found. Assuming NoneType.')
#            self.level = None
        
        ## extract other keyword arguments -------------------------------------
#        self.verbose = kwds.get('verbose')
#        self.time_units = kwds.get('time_units')# or 'days since 1950-01-01 00:00:00'
#        self.time_units = self.temporal.units.value
#        self.calendar = kwds.get('calendar')# or 'proleptic_gregorian'
#        self.calendar = self.temporal.calendar.value
#        self.level_name = kwds.get('level_name') or 'levels'

        ## extract the row and column bounds from the dataset
#        self.row_bnds = self.spatial.rowbnds.value[:]
#        self.col_bnds = self.spatial.colbnds.value[:]
        
        ## convert the time vector to datetime objects
#        self.timevec = nc.netcdftime.num2date(self.temporal.time.value[:],
#                                              self.time_units,
#                                              self.calendar)
#        self.timevec = self.temporal.get_timevec()
#        self.timeidx = np.arange(0,len(self.timevec))
#        self.tids = np.arange(1,len(self.timevec)+1)
        
        ## pull levels if possible
#        if self.level is not None:
##            self.levelvec = np.arange(1,len(self.level.level.value[:])+1)
#            self.levelvec = self.level.level.value[:]
#            self.levelidx = np.arange(0,len(self.levelvec))
#            self.lids = np.arange(1,len(self.levelvec)+1)
#        else:
#            self.levelvec = np.array([1])
#            self.levelidx = np.array([0])
#            self.lids = np.array([1])
        
        ## these are base numpy arrays used by spatial operations. -------------

        ## four numpy arrays one for each bounding coordinate of a polygon
#        self.min_col,self.min_row = self.spatial.get_min_bounds()
#        self.max_col,self.max_row = self.spatial.get_max_bounds()
#        ## these are the original indices of the row and columns. they are
#        ## referenced after the spatial subset to retrieve data from the dataset
#        self.real_col,self.real_row = np.meshgrid(np.arange(0,len(self.col_bnds)),
#                                                  np.arange(0,len(self.row_bnds)))
#        ## calculate approximate data resolution
#        self.res = approx_resolution(self.min_col[0,:])
#        ## generate unique id for each grid cell
#        self.gids = np.arange(1,self.real_col.shape[0]*self.real_col.shape[1]+1)
#        self.gids = self.gids.reshape(self.real_col.shape)
#        self.gids = np.empty(self.real_col.shape,dtype=int)
#        curr_id = 1
#        for i,j in itr_array(self.gids):
#            self.gids[i,j] = curr_id
#            curr_id += 1
        ## set the array shape.
#        self.shape = self.real_col.shape

        self.dataset.close()
        
#    def __del__(self):
#        try:
#            self.dataset.close()
#        finally:
#            pass
    
    def connect(self,uri):
        try:
            ret = nc.Dataset(uri,'r')
        except TypeError:
            ret = nc.MFDataset(uri)
        return(ret)
        
#    def extent(self):
#        minx = self.i.spatial.min_col.min()
#        maxx = self.i.spatial.max_col.max()
#        miny = self.i.spatial.min_row.min()
#        maxy = self.i.spatial.max_row.max()
#        poly = Polygon(((minx,miny),(maxx,miny),(maxx,maxy),(minx,maxy)))
#        return(poly)
    
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
        except MaskedDataError:
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
                raise(ExtentError)
            
        ## the initial selection
        self.i.spatial.selection.clear()
        self.i.spatial.select(polygon)

#        ## the base cell selection. does basic find operation to identify
#        ## cells to keep.
#        if polygon is not None:
#            prep_polygon = prepared.prep(polygon)
#            emin_col,emin_row,emax_col,emax_row = polygon.envelope.bounds
#            smin_col = contains(self.i.spatial.min_col,
#                                emin_col,emax_col,
#                                self.i.spatial.resolution)
#            smax_col = contains(self.i.spatial.max_col,
#                                emin_col,emax_col,
#                                self.i.spatial.resolution)
#            smin_row = contains(self.i.spatial.min_row,
#                                emin_row,emax_row,
#                                self.i.spatial.resolution)
#            smax_row = contains(self.i.spatial.max_row,
#                                emin_row,emax_row,
#                                self.i.spatial.resolution)
#            include = np.any((smin_col,smax_col),axis=0)*\
#                      np.any((smin_row,smax_row),axis=0)
#        else:
#            include = np.empty(self.i.spatial.min_row.shape,dtype=bool)
#            include[:,:] = True
#        
#        ## construct the reference matrices
#        geometry = np.empty(self.i.spatial.gid.shape,dtype=object)
#        row = []
#        col = []
##        gids = []
#        idx = []
#        
#        ## fill the matrices if value is included
#        def _append(ii,jj,geom):
#            geometry[ii,jj] = geom
#            row.append(self.i.spatial.real_row[ii,jj])
#            col.append(self.i.spatial.real_col[ii,jj])
##            gids.append(self.gids[ii,jj])
#            idx.append([self.i.spatial.real_row[ii,jj],
#                        self.i.spatial.real_col[ii,jj]])
#        
#        for ii,jj in itr_array(include):
#            if include[ii,jj]:
#                test_geom = make_poly((self.i.spatial.min_row[ii,jj],
#                                       self.i.spatial.max_row[ii,jj]),
#                                      (self.i.spatial.min_col[ii,jj],
#                                       self.i.spatial.max_col[ii,jj]))
#                if polygon is not None and keep(prep_polygon,polygon,test_geom):
#                    _append(ii,jj,test_geom)
#                elif polygon is None:
#                    _append(ii,jj,test_geom)
        
        
        ## get the number of dimensions of the target variable
        ndim = len(self.dataset.variables[var_name].dimensions)
                
        ## get the time indices
        timeidx = self.i.temporal.subset_timeidx(time_range)
#        if time_range is not None:
#            timeidx = self.i.temporal.timeidx\
#                      [(self.timevec>=time_range[0])*
#                       (self.timevec<=time_range[1])]
#        else:
#            timeidx = self.i.temporal.timeidx
        
        #tdk
#        timeidx = np.array([0,1])

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
        
#        ## extract the global gids
#        gids = self.gids[min(rowidx):max(rowidx)+1,min(colidx):max(colidx)+1]
        if ndim == 3:
            args = [timeidx,rowidx,colidx]
        elif ndim == 4:
            args = [timeidx,levelidx,rowidx,colidx]
        else:
            raise(NotImplementedError('cannot hold dimension count of "{0}"'.format(ndim)))
        npd = self.get_numpy_data(var,args)
        
#        if npd.shape[3] == 4:
#            import ipdb;ipdb.set_trace()
        
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
#        rel_mask = np.repeat(False,npd.shape[2]*npd.shape[3]).reshape((npd.shape[2],npd.shape[3]))
        rel_mask = np.ones(npd.shape,dtype=bool)
        ## now iterate and remove the data
        min_row = min(self.i.spatial.selection.row)
        min_col = min(self.i.spatial.selection.col)
        for ii in self.i.spatial.selection.idx:
            rel_mask[:,:,ii[0]-min_row,ii[1]-min_col] = False
        
#        ## reshape the data
#        npd = npd[:,:,rel_mask]
        
        ## test for masked data
        if hasattr(npd,'mask'):
#            mask = npd.mask
            ## if all the data values are masked, raise an error.
            if npd.mask.all():
                raise(MaskedDataError)
            else:
                npd.mask = np.logical_or(npd.mask,rel_mask)
        else:
            npd = np.ma.array(npd,mask=rel_mask)
                ## remove masked data
#                invmask = np.invert(mask)
#                npd = npd[invmask].reshape(invmask.shape[0],invmask.shape[1],-1)
#                geometry = np.array(geometry)[invmask[0,0,:]]
#                gids = np.array(gids)[invmask[0,0,:]]

#        import datetime
#        shapely_to_shp(geometry,str(datetime.datetime.now()))
#        shapely_to_shp(polygon,str(datetime.datetime.now()))

        ## create masked arrays for other relevant variables
#        gid = np.array(gids).reshape((npd.shape[2],npd.shape[3]))
        gid = self.i.spatial.gid[rowidx][:,colidx].\
              reshape((npd.shape[2],npd.shape[3]))
        gid = np.ma.array(gid,mask=npd.mask[0,0,:,:])
        
        ## keeping the geometry mask separate is necessary related to this error:
        ## http://projects.scipy.org/numpy/ticket/897
        geom = self.i.spatial.selection.geom[rowidx][:,colidx].reshape((npd.shape[2],npd.shape[3]))
        geom_mask = npd.mask[0,0,:,:]
        
#        weight = np.ma.array(np.zeros((npd.shape[2],npd.shape[3]),dtype=float),
#                             mask=npd.mask[0,0,:,:])
#        for ii,jj in iter_array(weight):
#            weight[ii,jj] = geom[ii,jj].area
#        weight = weight/weight.max()
#        if gid.shape[1] == 4:
#            g = []
#            for ii in geom.flat:
#                if ii is not None:
#                    g.append(ii)
#            shapely_to_shp(polygon,'pp_polygon')
#            shapely_to_shp(g,'pp_geom')
#            import ipdb;ipdb.set_trace()
#            tdk
        
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
        
#        ret = dict(
#          gid=gid,
#          tid=self.i.temporal.tid[timeidx],
#          lid=self.i.level.lid[levelidx],
#          geom=geom,
#          geom_mask=geom_mask,
#          weights=self.i.spatial.calc_weights(npd,geom),
#          timevec=self.i.temporal.time.value[timeidx],
#          levelvec=self.i.level.level.value[levelidx],
#          value=npd,
##          var_name=var_name
#                   )

#        return(ret)
    
    def subset(self,*args,**kwds):
        try:
            self.dataset = self.connect(self.uri)
            return(self._subset_(*args,**kwds))
        finally:
            self.dataset.close()
        
        
#        return(SubOcgDataset(geometry,
#                             npd,
#                             self.timevec[timeidx],
#                             gid=gids,
#                             levelvec=self.levelvec[levelidx],
#                             mask=mask,
#                             tid=self.tids[timeidx]))
    
#    def split_subset(self,var_name,
#                           max_proc=1,
#                           subset_opts={}):
#        """
#        returns -- list of SubOcgDatasets
#        """
#        ## the initial subset
#        ref = self.subset(var_name,**subset_opts)
#        ## make base process map
#        ref_idx_array = np.arange(0,len(ref.geometry))
#        splits = np.array_split(ref_idx_array,max_proc)
#        ## for the case of a single value, truncate the last split if it is
#        ## empty
#        if len(splits[-1]) == 0:
#            splits = splits[0:-1]
#        ## will hold the subsets
#        subs = []
#        ## create the subsets
#        for ii,split in enumerate(splits):
#            geometry = ref.geometry[split]
#            value = ref.value[:,:,split]
#            gid = ref.gid[split]
#            sub = SubOcgDataset(geometry,
#                                value,
#                                ref.timevec,
#                                gid=gid,
#                                levelvec=ref.levelvec,
#                                id=ii,
#                                tid=ref.tid)
#            subs.append(sub)
#        return(subs)
#    
#    def parallel_process_subsets(self,subs,polygon=None,clip=False,union=False,debug=False):
#        
#        def f(out,sub,polygon,clip,union):
#            if clip:
#                sub.clip(polygon)
#            if union:
#                sub.union_nosum()
#                if not clip:
#                    sub.select_values(clip=True,igeom=polygon)
#                else:
#                    sub.select_values(clip=False)
#            out.append(sub)
#        
#        if not debug:
#            import multiprocessing as mp
#            
#            out = mp.Manager().list()
#            pps = [mp.Process(target=f,args=(out,sub,polygon,clip,union)) for sub in subs]
#            for pp in pps: pp.start()
#            for pp in pps: pp.join()
#        else:
#            out = []
#            for sub in subs:
#                f(out,sub,polygon,clip,union)
#        return(list(out))
#                
#    def combine_subsets(self,subs,union=False):
#        ## collect data from subsets
#        for ii,sub in enumerate(subs): 
#            if ii == 0:
#                base = sub
#            else:
#                base = base.merge(sub,union=union)
#        
#        ## if union is true, sum the values, add new gid, and union the 
#        ## geometries.
#        if union:
#            base.geometry = np.array([cascaded_union(base.geometry)],dtype=object)
#            base.value = union_sum(base.weight,base.value,normalize=True)
#            base.gid = np.array([1])
#        return(base)