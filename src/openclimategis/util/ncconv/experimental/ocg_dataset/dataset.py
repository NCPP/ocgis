import netCDF4 as nc
import numpy as np
from shapely import prepared
from shapely.ops import cascaded_union
from util.ncconv.experimental.ocg_meta.interface import SpatialInterface,\
    TemporalInterface, LevelInterface
from util.ncconv.experimental.ocg_meta.element import PolyElementNotFound
from warnings import warn
from util.ncconv.experimental.helpers import *
from shapely.geometry.polygon import Polygon
from util.ncconv.experimental.ocg_dataset.sub import SubOcgDataset


class MaskedDataError(Exception):
    def __str__(self):
        return('Geometric intersection returns all masked values.')
    
    
class ExtentError(Exception):
    def __str__(self):
        return('Geometric intersection is empty.')


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
    
    @timing
    def __init__(self,uri,**kwds):
        self.uri = uri
        self.dataset = self.connect(uri)

        ## construct interfaces
        self.spatial = SpatialInterface(self.dataset,**kwds)
        self.temporal = TemporalInterface(self.dataset,**kwds)
        try:
            self.level = LevelInterface(self.dataset,**kwds)
        except PolyElementNotFound:
            warn('No "level" variable found. Assuming NoneType.')
            self.level = None
        
        ## extract other keyword arguments -------------------------------------
        self.verbose = kwds.get('verbose')
        self.time_units = kwds.get('time_units') or 'days since 1950-01-01 00:00:00'
        self.calendar = kwds.get('calendar') or 'proleptic_gregorian'
        self.level_name = kwds.get('level_name') or 'levels'

        ## extract the row and column bounds from the dataset
        self.row_bnds = self.spatial.rowbnds.value[:]
        self.col_bnds = self.spatial.colbnds.value[:]
        
        ## convert the time vector to datetime objects
        self.timevec = nc.netcdftime.num2date(self.temporal.time.value[:],
                                              self.time_units,
                                              self.calendar)
        self.timeidx = np.arange(0,len(self.timevec))
        self.tids = np.arange(1,len(self.timevec)+1)
        
        ## pull levels if possible
        if self.level is not None:
            self.levelvec = np.arange(1,len(self.level.level.value[:])+1)
            self.levelidx = np.arange(0,len(self.levelvec))
        else:
            self.levelvec = np.array([1])
            self.levelidx = np.array([0])
        
        ## these are base numpy arrays used by spatial operations. -------------

        ## four numpy arrays one for each bounding coordinate of a polygon
        self.min_col,self.min_row = self.spatial.get_min_bounds()
        self.max_col,self.max_row = self.spatial.get_max_bounds()
        ## these are the original indices of the row and columns. they are
        ## referenced after the spatial subset to retrieve data from the dataset
        self.real_col,self.real_row = np.meshgrid(np.arange(0,len(self.col_bnds)),
                                                  np.arange(0,len(self.row_bnds)))
        ## calculate approximate data resolution
        self.res = approx_resolution(self.min_col[0,:])
        ## generate unique id for each grid cell
        self.gids = np.arange(1,self.real_col.shape[0]*self.real_col.shape[1]+1)
        self.gids = self.gids.reshape(self.real_col.shape)
#        self.gids = np.empty(self.real_col.shape,dtype=int)
#        curr_id = 1
#        for i,j in itr_array(self.gids):
#            self.gids[i,j] = curr_id
#            curr_id += 1
        ## set the array shape.
        self.shape = self.real_col.shape
        
    def __del__(self):
        try:
            self.dataset.close()
        finally:
            pass
    
    @timing
    def connect(self,uri):
        return(nc.Dataset(uri,'r'))
        
    def extent(self):
        minx = self.min_col.min()
        maxx = self.max_col.max()
        miny = self.min_row.min()
        maxy = self.max_row.max()
        poly = Polygon(((minx,miny),(maxx,miny),(maxx,maxy),(minx,maxy)))
        return(poly)
    
    def check_extent(self,target):
        extent = self.extent()
        return(keep(prepared.prep(extent),extent,target))
    
    def check_masked(self,var_name,polygon):
        try:
            self.subset(var_name,
                        polygon=polygon,
                        time_range=[self.timevec[0],self.timevec[1]])
            ret = True
        except MaskedDataError:
            ret = False
        return(ret)
        
    def display(self,show=True,overlays=None):
        import matplotlib.pyplot as plt
        from descartes.patch import PolygonPatch
        
        ax = plt.axes()
        if overlays is not None:
            for geom in overlays:
                ax.add_patch(PolygonPatch(geom,alpha=0.5,fc='#999999'))
        ax.scatter(self.min_col,self.min_row)
        ax.scatter(self.max_col,self.max_row)
        if show: plt.show()
    
    @timing
    def get_numpy_data(self,var,args):
        if len(args) == 3:
            npd = var[args[0],args[1],args[2]]
        if len(args) == 4:
            npd = var[args[0],args[1],args[2],args[3]]
        return(npd)
    
    @timing
    def subset(self,var_name,polygon=None,time_range=None,level_range=None): ## intersects + touches
        """
        polygon -- shapely Polygon object
        return -- SubOcgDataset
        """

        ## do a quick extent check if a polygon is passed
        if polygon is not None:
            if not self.check_extent(polygon):
                raise(ExtentError)

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
        gids = []
        idx = []
        
        ## fill the matrices if value is included
        def _append(ii,jj,geom):
            geometry.append(geom)
            row.append(self.real_row[ii,jj])
            col.append(self.real_col[ii,jj])
            gids.append(self.gids[ii,jj])
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
        levelidx = self.levelidx
        if ndim == 4:
            if level_range is not None:
                level_range = np.array([ii-1 for ii in level_range])
                levelidx = sub_range(level_range)
        else:
            if level_range is not None:
                raise ValueError('Target variable has no levels.')
            
        ## extract the data
        var = self.dataset.variables[var_name]
        rowidx = sub_range(row)
        colidx = sub_range(col)
        
#        ## extract the global gids
#        gids = self.gids[min(rowidx):max(rowidx)+1,min(colidx):max(colidx)+1]

        if ndim == 3:
            args = [timeidx,rowidx,colidx]
        if ndim == 4:
            args = [timeidx,levelidx,rowidx,colidx]
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
        rel_mask = np.repeat(False,npd.shape[2]*npd.shape[3]).reshape((npd.shape[2],npd.shape[3]))
        ## now iterate and remove the data
        min_row = min(row)
        min_col = min(col)
        for ii in idx:
            rel_mask[ii[0]-min_row,ii[1]-min_col] = True

        ## reshape the data
        npd = npd[:,:,rel_mask]
        
        ## test for masked data
        if hasattr(npd,'mask'):
            mask = npd.mask
            ## if all the data values are masked, raise an error.
            if mask.all():
                raise(MaskedDataError)
        else:
            mask = None

        return(SubOcgDataset(geometry,
                             npd,
                             self.timevec[timeidx],
                             gid=gids,
                             levelvec=self.levelvec[levelidx],
                             mask=mask,
                             tid=self.tids[timeidx]))
    
    def split_subset(self,var_name,
                           max_proc=1,
                           subset_opts={}):
        """
        returns -- list of SubOcgDatasets
        """
        ## the initial subset
        ref = self.subset(var_name,**subset_opts)
        ## make base process map
        ref_idx_array = np.arange(0,len(ref.geometry))
        splits = np.array_split(ref_idx_array,max_proc)
        ## for the case of a single value, truncate the last split if it is
        ## empty
        if len(splits[-1]) == 0:
            splits = splits[0:-1]
        ## will hold the subsets
        subs = []
        ## create the subsets
        for ii,split in enumerate(splits):
            geometry = ref.geometry[split]
            value = ref.value[:,:,split]
            gid = ref.gid[split]
            sub = SubOcgDataset(geometry,
                                value,
                                ref.timevec,
                                gid=gid,
                                levelvec=ref.levelvec,
                                id=ii,
                                tid=ref.tid)
            subs.append(sub)
        return(subs)
    
    def parallel_process_subsets(self,subs,polygon=None,clip=False,union=False,debug=False):
        
        def f(out,sub,polygon,clip,union):
            if clip:
                sub.clip(polygon)
            if union:
                sub.union_nosum()
                if not clip:
                    sub.select_values(clip=True,igeom=polygon)
                else:
                    sub.select_values(clip=False)
            out.append(sub)
        
        if not debug:
            import multiprocessing as mp
            
            out = mp.Manager().list()
            pps = [mp.Process(target=f,args=(out,sub,polygon,clip,union)) for sub in subs]
            for pp in pps: pp.start()
            for pp in pps: pp.join()
        else:
            out = []
            for sub in subs:
                f(out,sub,polygon,clip,union)
        return(list(out))
                
    def combine_subsets(self,subs,union=False):
        ## collect data from subsets
        for ii,sub in enumerate(subs): 
            if ii == 0:
                base = sub
            else:
                base = base.merge(sub,union=union)
        
        ## if union is true, sum the values, add new gid, and union the 
        ## geometries.
        if union:
            base.geometry = np.array([cascaded_union(base.geometry)],dtype=object)
            base.value = union_sum(base.weight,base.value,normalize=True)
            base.gid = np.array([1])
        return(base)
