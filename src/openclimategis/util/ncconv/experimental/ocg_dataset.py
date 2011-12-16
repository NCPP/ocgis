import netCDF4 as nc
import numpy as np
from shapely import prepared
from helpers import *
from shapely.ops import cascaded_union
import copy
from util.helpers import get_temp_path
from sqlalchemy.pool import NullPool
import ploader as pl
from util.ncconv.experimental.ocg_meta.interface import SpatialInterface,\
    TemporalInterface, LevelInterface
from util.ncconv.experimental.ocg_meta.element import PolyElementNotFound
from warnings import warn


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
        self.gids = np.empty(self.real_col.shape,dtype=int)
        curr_id = 1
        for i,j in itr_array(self.gids):
            self.gids[i,j] = curr_id
            curr_id += 1
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
            else:
                raise NotImplementedError
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
                             mask=mask))
    
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
                                id=ii)
            subs.append(sub)
                
        return(subs)
    
    def parallel_process_subsets(self,subs,polygon=None,clip=False,union=False):
        
        def f(out,sub,polygon,clip,union):
            if clip:
                sub.clip(polygon)
            if union:
                sub.union_nosum()
            out.append(sub)
        
        parallel = True
        if parallel:
            import multiprocessing as mp
            
            out = mp.Manager().list()
            pps = [mp.Process(target=f,args=(out,sub,polygon,clip,union)) for sub in subs]
            for pp in pps: pp.start()
            for pp in pps: pp.join()
        else:
            out = []
            for sub in subs:
                f(out,sub,polygon,clip,union)
        
        return([sub for sub in out])
                
    def combine_subsets(self,subs,union=False):
        ## collect data from subsets
        for ii,sub in enumerate(subs): 
            if ii == 0:
                all_geometry = sub.geometry
                all_weights = sub.weight
                all_value = sub.value
                all_gid = sub.gid
            else:
                all_gid = np.hstack((all_gid,sub.gid))
                all_geometry = np.hstack((all_geometry,sub.geometry))
                all_weights = np.hstack((all_weights,sub.weight))
                all_value = np.dstack((all_value,sub.value))
        
        ## if union is true, sum the values, add new gid, and union the 
        ## geometries.
        if union:
            all_geometry = np.array([cascaded_union(all_geometry)],dtype=object)
            all_value = union_sum(all_weights,all_value,normalize=True)
        
        return(SubOcgDataset(all_geometry,
                             all_value,
                             sub.timevec,
                             gid=all_gid,
                             levelvec=sub.levelvec))
        

class SubOcgDataset(object):
    __attrs__ = ['geometry','value','gid','weight','timevec','levelvec']
    
    def __init__(self,geometry,value,timevec,gid=None,levelvec=None,mask=None,id=None):
        """
        geometry -- numpy array with dimension (n) of shapely Polygon 
            objects
        value -- numpy array with dimension (time,level,n)
        gid -- numpy array containing integer unique ids for the grid cells.
            has dimension (n)
        timevec -- numpy array with indices corresponding to time dimension of
            value
        mask -- boolean mask array with same dimension as value. will subset other
            inputs if passed. a value to be masked is indicated by True.
        """
        
        self.id = id
        self.geometry = np.array(geometry)
        self.value = np.array(value)
        self.timevec = np.array(timevec)
        
        if gid is not None:
            self.gid = np.array(gid)
        else:
            self.gid = np.arange(1,len(self.geometry) + 1)
        if levelvec is not None:
            self.levelvec = np.array(levelvec)
        else:
            if len(self.value) == 0:
                self.levelvec = np.array()
            else:
                self.levelvec = np.arange(1,self.value.shape[1]+1)
        
        ## if the mask is passed, subset the data
        if mask is not None:
            mask = np.invert(mask)[0,0,:]
            self.geometry = self.geometry[mask]
            self.gid = self.gid[mask]
            self.value = self.value[:,:,mask]
        
        ## calculate nominal weights
        self.weight = np.ones(self.geometry.shape,dtype=float)
        
    def to_grid_dict(self,ocg_dataset):
        """assumes an intersects-like operation with no union"""
        ## make the bounding polygon
        envelope = MultiPolygon(self.geometry.tolist()).envelope
        ## get the x,y vectors
        x,y = ocg_dataset.spatial.subset_centroids(envelope)
        ## make the grids
        gx,gy = np.meshgrid(x,y)
        ## make the empty boolean array
        mask = np.empty((self.value.shape[0],
                        self.value.shape[1],
                        gx.shape[0],
                        gx.shape[1]),dtype=bool)
        mask[:,:,:,:] = True
        ## make the empty geometry id
        gidx = np.empty(gx.shape,dtype=int)
        gidx = np.ma.array(gidx,mask=mask[0,0,:,:])
        ## make the empty value array
#        value = np.empty(mask.shape,dtype=float)
        ## loop for each centroid
        for ii,geom in enumerate(self.geometry):
            diffx = abs(gx - geom.centroid.x)
            diffy = abs(gy - geom.centroid.y)
            diff = diffx + diffy
            idx = diff == diff.min()
            mask[:,:,idx] = False
            gidx[idx] = ii
#            for dt in self.dim_time:
#                for dl in self.dim_level:
#                    value[dt,dl,idx] = self.value[dt,dl,ii]
        # construct the masked array
#        value = np.ma.array(value,mask=mask,fill_value=fill_value)
        ## if level is not included, squeeze out the dimension
#        if not include_level:
#            value = value.squeeze()
#        ## construct row and column bounds
#        xbnds = np.empty((len(self.geometry),2),dtype=float)
#        ybnds = xbnds.copy()
        ## subset the bounds
        xbnds,ybnds = ocg_dataset.spatial.subset_bounds(envelope)
        ## put the data together
        ret = dict(xbnds=xbnds,ybnds=ybnds,x=x,y=y,mask=mask,gidx=gidx)
        return(ret)
        
    def copy(self,**kwds):
        new_ds = copy.copy(self)
        def _find_set(kwd):
            val = kwds.get(kwd)
            if val is not None:
                setattr(new_ds,kwd,val) 
        for attr in self.__attrs__:  _find_set(attr)  
        return(new_ds)
    
    def merge(self,sub,id=None):
        """Assumes same time and level vectors."""
        geometry = np.hstack((self.geometry,sub.geometry))
        value = np.dstack((self.value,sub.value))
        gid = np.hstack((self.gid,sub.gid))
        ## if there are non-unique cell ids (which may happen with union
        ## operations, regenerate the unique values.
        if len(gid) > len(np.unique(gid)):
            gid = np.arange(1,len(gid)+1)
        return(self.copy(geometry=geometry,
                         value=value,
                         gid=gid,
                         id=id))
    
    @timing 
    def purge(self):
        """looks for duplicate geometries"""
        unique,uidx = np.unique([geom.wkb for geom in self.geometry],return_index=True)
        self.geometry = self.geometry[uidx]
        self.gid = self.gid[uidx]
        self.value = self.value[:,:,uidx]
        
        
    def __iter__(self):
        for dt,dl,dd in itertools.product(self.dim_time,self.dim_level,self.dim_data):
            d = dict(geometry=self.geometry[dd],
                     value=float(self.value[dt,dl,dd]),
                     time=self.timevec[dt],
                     level=int(self.levelvec[dl]),
                     gid=int(self.gid[dd]))
            yield(d)
            
    def iter_with_area(self,area_srid=3005):
        sr_orig = get_sr(4326)
        sr_dest = get_sr(area_srid)
        for attrs in self:
            attrs.update(dict(area_m2=get_area(attrs['geometry'],sr_orig,sr_dest)))
            yield(attrs)
    
    def _range_(self,idx):
        try:
            return(range(self.value.shape[idx]))
        except IndexError:
            return([])
    
    @property
    def dim_time(self):
        return(self._range_(0))

    @property
    def dim_level(self):
        return(self._range_(1))

    @property
    def dim_data(self):
        return(self._range_(2))
                     
    @property
    def area(self):
        area = 0.0
        for geom in self.geometry:
            area += geom.area
        return(area)
        
    def clip(self,igeom):
        prep_igeom = prepared.prep(igeom)
        for ii,geom in enumerate(self.geometry):
            if keep(prep_igeom,igeom,geom):
                new_geom = igeom.intersection(geom)
                weight = new_geom.area/geom.area
                assert(weight != 0.0) #tdk
                self.weight[ii] = weight
                self.geometry[ii] = new_geom
        
    def report_shape(self):
        for attr in self.__attrs__:
            rattr = getattr(self,attr)
            msg = '{0}={1}'.format(attr,getattr(rattr,'shape'))
            print(msg)
        
    def union(self):
        self._union_geom_()
        self._union_sum_()
        
    def union_nosum(self):
        self._union_geom_()
        
    def _union_geom_(self):
        ## union the geometry. just using np.array() on a multipolgon object
        ## results in a (1,n) array of polygons.
        new_geometry = np.array([None],dtype=object)
        new_geometry[0] = cascaded_union(self.geometry)
        self.geometry = new_geometry
        
    def _union_sum_(self):
        self.value = union_sum(self.weight,self.value,normalize=True)
        self.gid = np.array([1])
    
    @timing
    def as_sqlite(self,add_area=True,
                       area_srid=3005,
                       wkt=True,
                       wkb=False,
                       as_multi=True,
                       to_disk=False,
                       procs=1):
        from sqlalchemy import create_engine
        from sqlalchemy.orm.session import sessionmaker
        import db
        
        path = 'sqlite://'
        if to_disk or procs > 1:
            path = path + '/' + get_temp_path('.sqlite',nest=True)
            db.engine = create_engine(path,
                                      poolclass=NullPool)
        else:
            db.engine = create_engine(path,
#                                      connect_args={'check_same_thread':False},
#                                      poolclass=StaticPool
                                      )
        db.metadata.bind = db.engine
        db.Session = sessionmaker(bind=db.engine)
        db.metadata.create_all()

        print('  loading geometry...')
        ## spatial reference for area calculation
        sr = get_sr(4326)
        sr2 = get_sr(area_srid)

#        data = dict([[key,list()] for key in ['gid','wkt','wkb','area_m2']])
#        for dd in self.dim_data:
#            data['gid'].append(int(self.gid[dd]))
#            geom = self.geometry[dd]
#            if isinstance(geom,Polygon):
#                geom = MultiPolygon([geom])
#            if wkt:
#                wkt = str(geom.wkt)
#            else:
#                wkt = None
#            data['wkt'].append(wkt)
#            if wkb:
#                wkb = str(geom.wkb)
#            else:
#                wkb = None
#            data['wkb'].append(wkb)
#            data['area_m2'].append(get_area(geom,sr,sr2))
#        self.load_parallel(db.Geometry,data,procs)

        def f(idx,geometry=self.geometry,gid=self.gid,wkt=wkt,wkb=wkb,sr=sr,sr2=sr2,get_area=get_area):
            geom = geometry[idx]
            if isinstance(geom,Polygon):
                geom = MultiPolygon([geom])
            if wkt:
                wkt = str(geom.wkt)
            else:
                wkt = None
            if wkb:
                wkb = str(geom.wkb)
            else:
                wkb = None
            return(dict(gid=int(gid[idx]),
                        wkt=wkt,
                        wkb=wkb,
                        area_m2=get_area(geom,sr,sr2)))
        fkwds = dict(geometry=self.geometry,gid=self.gid,wkt=wkt,wkb=wkb,sr=sr,sr2=sr2,get_area=get_area)
        gen = pl.ParallelGenerator(db.Geometry,self.dim_data,f,fkwds=fkwds,procs=procs)
        gen.load()

        print('  loading time...')
        ## load the time data
        data = dict([[key,list()] for key in ['tid','time','day','month','year']])
        for ii,dt in enumerate(self.dim_time,start=1):
            data['tid'].append(ii)
            data['time'].append(self.timevec[dt])
            data['day'].append(self.timevec[dt].day)
            data['month'].append(self.timevec[dt].month)
            data['year'].append(self.timevec[dt].year)
        self.load_parallel(db.Time,data,procs)
            
        print('  loading value...')
        ## set up parallel loading data
        data = dict([key,list()] for key in ['gid','level','tid','value'])
        for ii,dt in enumerate(self.dim_time,start=1):
            for dl in self.dim_level:
                for dd in self.dim_data:
                    data['gid'].append(int(self.gid[dd]))
                    data['level'].append(int(self.levelvec[dl]))
                    data['tid'].append(ii)
                    data['value'].append(float(self.value[dt,dl,dd]))
        self.load_parallel(db.Value,data,procs)

        return(db)
    
    def load_parallel(self,Model,data,procs):
        pmodel = pl.ParallelModel(Model,data)
        ploader = pl.ParallelLoader(procs=procs)
        ploader.load_model(pmodel)
    
    def display(self,show=True,overlays=None):
        import matplotlib.pyplot as plt
        from descartes.patch import PolygonPatch
        
        ax = plt.axes()
        x = []
        y = []
        for geom in self.geometry:
            if isinstance(geom,MultiPolygon):
                for geom2 in geom:
                    try:
                        ax.add_patch(PolygonPatch(geom2,alpha=0.5))
                    except:
                        geom2 = wkt.loads(geom2.wkt)
                        ax.add_patch(PolygonPatch(geom2,alpha=0.5))
                    ct = geom2.centroid
                    x.append(ct.x)
                    y.append(ct.y)
            else:
                ax.add_patch(PolygonPatch(geom,alpha=0.5))
                ct = geom.centroid
                x.append(ct.x)
                y.append(ct.y)
        if overlays is not None:
            for geom in overlays:
                ax.add_patch(PolygonPatch(geom,alpha=0.5,fc='#999999'))
        ax.scatter(x,y,alpha=1.0)
        if show: plt.show()