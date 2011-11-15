import netCDF4 as nc
import numpy as np
from shapely import prepared, wkt
from helpers import *
from shapely.ops import cascaded_union
from shapely.geometry.multipolygon import MultiPolygon
import time
import copy


class MaskedDataError(Exception):
    
    def __str__(self):
        return('Geometric intersection returns all masked values.')


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
            self.levelvec = np.arange(1,len(self.dataset.variables[self.level_name][:])+1)
            self.levelidx = np.arange(0,len(self.levelvec))
        except:
            self.levelvec = np.array([1])
        
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
        
    def __del__(self):
        try:
            self.dataset.close()
        finally:
            pass
        
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
        
    def get_numpy_data(self,var,args):
        if len(args) == 3:
            npd = var[args[0],args[1],args[2]]
        if len(args) == 4:
            npd = var[args[0],args[1],args[2],args[3]]
        return(npd)
        
    def subset(self,var_name,polygon=None,time_range=None,level_range=None): ## intersects + touches
        """
        polygon -- shapely Polygon object
        return -- SubOcgDataset
        """

        ## do a quick extent check if a polygon is passed
        if polygon is not None:
            if not self.check_extent(polygon):
                raise(ValueError("Polygon boundary & domain extent return an empty intersection."))

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
                             cell_ids,
                             self.timevec[timeidx],
                             levelvec=self.levelvec,
                             mask=mask))
    
    def mapped_subset(self,var_name,
                           max_proc=1,
                           subset_opts={}):
        """
        returns -- list of SubOcgDatasets
        """

        ## the initial subset
        ref = self.subset(var_name,**subset_opts)
        ## make base process map
        ref_idx_array = np.arange(0,len(ref.geometry))
        if max_proc == 1:
            splits = [ref_idx_array]
        elif max_proc > 1:
            splits = np.array_split(ref_idx_array,max_proc)
        else:
            raise ValueError("max_proc must be one or greater.")
        ## will hold the subsets
        subs = []
        ## create the subsets
        for ii,split in enumerate(splits):
            geometry = ref.geometry[split]
            value = ref.value[:,:,split]
            cell_id = ref.cell_id[split]
            sub = SubOcgDataset(geometry,
                                value,
                                cell_id,
                                timevec=ref.timevec,
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
            while True:
                alive = [pp.is_alive() for pp in pps]
                if any(alive):
                    time.sleep(0.1)
                else:
                    break
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
                all_cell_id = sub.cell_id
            else:
                all_cell_id = np.hstack((all_cell_id,sub.cell_id))
                all_geometry = np.hstack((all_geometry,sub.geometry))
                all_weights = np.hstack((all_weights,sub.weight))
                all_value = np.dstack((all_value,sub.value))
        
        ## if union is true, sum the values, add new cell_id, and union the 
        ## geometries.
        if union:
            all_geometry = np.array([cascaded_union(all_geometry)],dtype=object)
            all_value = union_sum(all_weights,all_value,normalize=True)
            all_cell_id = np.array([1])
        
        return(SubOcgDataset(all_geometry,
                             all_value,
                             all_cell_id,
                             sub.timevec))
        

class SubOcgDataset(object):
    __attrs__ = ['geometry','value','cell_id','weight','timevec','levelvec']
    
    def __init__(self,geometry,value,cell_id,timevec,levelvec=None,mask=None,id=None):
        """
        geometry -- numpy array with dimension (n) of shapely Polygon 
            objects
        value -- numpy array with dimension (time,level,n)
        cell_id -- numpy array containing integer unique ids for the grid cells.
            has dimension (n)
        timevec -- numpy array with indices corresponding to time dimension of
            value
        mask -- boolean mask array with same dimension as value. will subset other
            inputs if passed. a value to be masked is indicated by True.
        """
        
        self.id = id
        self.geometry = np.array(geometry)
        self.value = np.array(value)
        self.cell_id = np.array(cell_id)
        self.timevec = timevec
        if levelvec is not None:
            self.levelvec = levelvec
        else:
            self.levelvec = np.array([1])
        
        ## if the mask is passed, subset the data
        if mask is not None:
            mask = np.invert(mask)[0,0,:]
            self.geometry = self.geometry[mask]
            self.cell_id = self.cell_id[mask]
            self.value = self.value[:,:,mask]
        
        ## calculate nominal weights
        self.weight = np.ones(self.geometry.shape,dtype=float)
        
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
        cell_id = np.hstack((self.cell_id,sub.cell_id))
        ## if there are non-unique cell ids (which may happen with union
        ## operations, regenerate the unique values.
        if len(cell_id) > np.unique(cell_id):
            cell_id = np.arange(1,len(cell_id)+1)
        return(SubOcgDataset(geometry,
                             value,
                             cell_id,
                             self.timevec,
                             self.levelvec,
                             id=id))
        
    def __iter__(self):
        for dt,dl,dd in itertools.product(self.dim_time,self.dim_level,self.dim_data):
            atime = self.timevec[dt]
            geometry = self.geometry[dd]
            d = dict(geometry=geometry,
                     value=float(self.value[dt,dl,dd]),
                     time=atime,
                     level=int(self.levelvec[dl]))
            yield(d)
        
    @property
    def dim_time(self):
        return(range(self.value.shape[0]))
    
    @property
    def dim_level(self):
        return(range(self.value.shape[1]))
    
    @property
    def dim_data(self):
        return(range(self.value.shape[2]))
                     
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
        self.cell_id = np.array([1])
    
    def as_sqlite(self,add_area=True,area_srid=3005):
        from sqlalchemy import create_engine
        from sqlalchemy.orm.session import sessionmaker
        import db
        
        engine = create_engine('sqlite://')
        db.metadata.bind = engine
        db.Session = sessionmaker(bind=engine)
        
        ## spatial reference for area calculation
        sr = get_sr(4326)
        sr2 = get_sr(area_srid)
        
        db.metadata.create_all()
        s = db.Session()
        try:
            ## create the geometries
            for dd in self.dim_data:
                s.add(db.Geometry(gid=int(self.cell_id[dd]),
                                  wkt=str(self.geometry[dd].wkt),
                                  area_m2=get_area(self.geometry[dd],sr,sr2)))
                s.commit()
            ## fill in the rest of the data
            for dt in self.dim_time:
                dtime = db.Time(time=self.timevec[dt])
                for dl in self.dim_level:
                    for dd in self.dim_data:
                        geometry = s.query(db.Geometry).filter(db.Geometry.gid == int(self.cell_id[dd])).one()
                        val = db.Value(geometry=geometry,
                                       level=int(self.levelvec[dl]),
                                       time=dtime,
                                       value=float(self.value[dt,dl,dd]))
                        s.add(val)
            s.commit()
        finally:
            s.close()
        return(db)
    
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