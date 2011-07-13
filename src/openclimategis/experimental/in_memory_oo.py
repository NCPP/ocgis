import numpy as np
from shapely.geometry.polygon import Polygon
import datetime
import netCDF4 as nc
import itertools
import geojson
from shapely.ops import cascaded_union
from ipdb import set_trace as tr
from openclimategis.util.helpers import get_temp_path
from openclimategis.util.toshp import OpenClimateShp


class OcgDataset(object):
    """
    Wraps and netCDF4-python Dataset object providing extraction methods by 
    spatial and temporal queries.
    
    dataset -- netCDF4-python Dataset object
    **kwds -- arguments for the names of spatial and temporal dimensions.
        rowbnds_name
        colbnds_name
        time_name
        time_units
        calendar
    """
    
    def __init__(self,dataset,**kwds):
        self.dataset = dataset
        
#        self.polygon = kwds.get('polygon')
#        self.temporal = kwds.get('temporal')
#        self.row_name = kwds.get('row_name') or 'latitude'
#        self.col_name = kwds.get('col_name') or 'longitude'
        
        ## extract the names of the spatiotemporal variables/dimensions from
        ## the keyword arguments.
        self.rowbnds_name = kwds.get('rowbnds_name') or 'bounds_latitude'
        self.colbnds_name = kwds.get('colbnds_name') or 'bounds_longitude'
        self.time_name = kwds.get('time_name') or 'time'
        self.time_units = kwds.get('time_units') or 'days since 1950-01-01 00:00:00'
        self.calendar = kwds.get('calendar') or 'proleptic_gregorian'
#        self.clip = kwds.get('clip') or False
#        self.dissolve = kwds.get('dissolve') or False
        
#        self.row = self.dataset.variables[self.row_name][:]
#        self.col = self.dataset.variables[self.col_name][:]
        ## extract the row and column bounds from the dataset
        self.row_bnds = self.dataset.variables[self.rowbnds_name][:]
        self.col_bnds = self.dataset.variables[self.colbnds_name][:]
        ## convert the time vector to datetime objects
        self.timevec = nc.netcdftime.num2date(self.dataset.variables[self.time_name][:],
                                              self.time_units,
                                              self.calendar)
        
        ## these are base numpy arrays used by spatial operations.
        
        ## four numpy arrays one for each bounding coordinate of a polygon
        self.min_col,self.min_row = np.meshgrid(self.col_bnds[:,0],self.row_bnds[:,0])
        self.max_col,self.max_row = np.meshgrid(self.col_bnds[:,1],self.row_bnds[:,1])
        ## these are the original indices of the row and columns. they are
        ## referenced after the spatial subset to retrieve data from the dataset
        self.real_col,self.real_row = np.meshgrid(np.arange(0,len(self.col_bnds)),
                                                  np.arange(0,len(self.row_bnds)))

    def _itr_array_(self,a):
        "a -- 2-d ndarray"
        ix = a.shape[0]
        jx = a.shape[1]
        for ii,jj in itertools.product(xrange(ix),xrange(jx)):
            yield ii,jj
            
    def _contains_(self,grid,lower,upper):
        s1 = grid >= lower
        s2 = grid <= upper
        return(s1*s2)
            
    def _set_overlay_(self,polygon=None,clip=False):
        """
        Perform spatial operations.
        
        polygon=None -- shapely polygon object
        clip=False -- set to True to perform an intersection
        """
        
        print('overlay...')
        
        ## holds polygon objects
        self._igrid = np.empty(self.min_row.shape,dtype=object)
        ## holds weights for area weighting in the case of a dissolve
        self._weights = np.empty(self.min_row.shape)
        
        ## initial subsetting to avoid iterating over all polygons unless abso-
        ## lutely necessary
        if polygon is not None:
            emin_col,emin_row,emax_col,emax_row = polygon.envelope.bounds
            smin_col = self._contains_(self.min_col,emin_col,emax_col)
            smax_col = self._contains_(self.max_col,emin_col,emax_col)
            smin_row = self._contains_(self.min_row,emin_row,emax_row)
            smax_row = self._contains_(self.max_row,emin_row,emax_row)
            include = np.any((smin_col,smax_col,smin_row,smax_row),axis=0)
        else:
            include = np.empty(self.min_row.shape,dtype=bool)
            include[:,:] = True
        
        ## loop for each spatial grid element
        for ii,jj in self._itr_array_(include):
            if not include[ii,jj]: continue
            ## create the polygon
            g = self._make_poly_((self.min_row[ii,jj],self.max_row[ii,jj]),
                                 (self.min_col[ii,jj],self.max_col[ii,jj]))
            ## add the polygon if it intersects the aoi of if all data is being
            ## returned.
            if g.intersects(polygon) or polygon is None:
                ## get the area before the intersection
                prearea = g.area
                ## full intersection in the case of a clip and an aoi is passed
                if clip is True and polygon is not None:
                    ng = g.intersection(polygon)
                ## otherwise, just keep the geometry
                else:
                    ng = g
                ## calculate the weight
                w = ng.area/prearea
                ## a polygon can have a true intersects but actually not overlap
                ## i.e. shares a border.
                if w > 0:
                    self._igrid[ii,jj] = ng
                    self._weights[ii,jj] = w
        ## the mask is used as a subset
        self._mask = self._weights > 0
#        self._weights = self._weights/self._weights.sum()
    
    def _make_poly_(self,rtup,ctup):
        """
        rtup = (row min, row max)
        ctup = (col min, col max)
        """
        return Polygon(((ctup[0],rtup[0]),
                        (ctup[0],rtup[1]),
                        (ctup[1],rtup[1]),
                        (ctup[1],rtup[0])))
        
    def _get_numpy_data_(self,var_name,polygon=None,time_range=None,clip=False):
        """
        var_name -- NC variable to extract from
        polygon -- shapely polygon object
        time_range -- [lower datetime, upper datetime]
        """
        print('getting numpy data...')
        variable = self.dataset.variables[var_name]
        self._set_overlay_(polygon=polygon,clip=clip)
        
        def _u(arg):
            un = np.unique(arg)
            return(np.arange(un.min(),un.max()+1))
        
        def _sub(arg):
            return arg[self._idxrow.min():self._idxrow.max()+1,
                       self._idxcol.min():self._idxcol.max()+1]
        
        if time_range is not None:
            self._idxtime = np.arange(
             0,
             len(self.timevec))[(self.timevec>=time_range[0])*
                                (self.timevec<=time_range[1])]
        else:
            self._idxtime = np.arange(0,len(self.timevec))
        
        self._idxrow = _u(self.real_row[self._mask])
        self._idxcol = _u(self.real_col[self._mask])
         
        self._mask = _sub(self._mask)
        self._weights = _sub(self._weights)
        self._igrid = _sub(self._igrid)
        
        npd = variable[self._idxtime,self._idxrow,self._idxcol]
        
        return(npd)
    
    def _is_masked_(self,arg):
        if isinstance(arg,np.ma.core.MaskedConstant):
            return None
        else:
            return arg
    
    def extract_elements(self,*args,**kwds):
        print('extracting elements...')
        if 'dissolve' in kwds:
            dissolve = kwds.pop('dissolve')
        else:
            dissolve = False

        npd = self._get_numpy_data_(*args,**kwds)
        features = []
        ids = self._itr_id_()
        if dissolve:
            for kk in range(len(self._idxtime)):
                if kk == 0:
                    ## need to remove geometries that have masked data
                    lyr = npd[kk,:,:]
                    select = self._mask*np.invert(lyr.mask)
                    geoms = self._igrid[select]
                    sub_weights = self._weights*select
                    self._weights = sub_weights/sub_weights.sum()
                    weighted = npd*self._weights
                    unioned = cascaded_union([p for p in geoms])
                ## generate the feature
                feature = dict(
                    id=ids.next(),
                    geometry=unioned,
                    properties=dict({VAR:float(weighted[kk,:,:].sum()),
                                     'timestamp':str(self.timevec[self._idxtime[kk]])}))
                features.append(feature)
        else:
            for ii,jj in self._itr_array_(self._mask):
                if self._mask[ii,jj] == True:
                    if len(self._idxtime) > 1:
                        data = npd[:,ii,jj]
                    else:
                        data = npd[ii,jj]
                    data = [self._is_masked_(da) for da in data]
                    for kk in range(len(data)):
                        if data[kk] == None: continue
                        feature = dict(
                            id=ids.next(),
                            geometry=self._igrid[ii,jj],
                            properties=dict({VAR:float(data[kk]),
                                             'timestamp':str(self.timevec[self._idxtime[kk]])}))
                        features.append(feature)
        return(features)
        
        
    def _itr_id_(self,start=1):
        while True:
            try:
                yield start
            finally:
                start += 1
                
    def as_geojson(self,elements):
        features = [geojson.Feature(**e) for e in elements]
        fc = geojson.FeatureCollection(features)
        return(geojson.dumps(fc))
    
    def as_shp(self,elements,path=None):
        if path is None:
            path = get_temp_path(suffix='.shp')
        ocs = OpenClimateShp(path,elements)
        ocs.write()
        return(path)
        
        
if __name__ == '__main__':

    #NC = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data/wcrp_cmip3/pcmdi.ipcc4.bccr_bcm2_0.1pctto2x.run1.monthly.cl_A1_1.nc'
    NC = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data/maurer/bccr_bcm2_0.1.sresa1b.monthly.Prcp.1950.nc'
    ## all
    #POLYINT = Polygon(((-99,39),(-94,38),(-94,40),(-100,39)))
    ## great lakes
    POLYINT = Polygon(((-90.35,40.55),(-83,43),(-80.80,49.87),(-90.35,49.87)))
    ## two areas
    #POLYINT = [wkt.loads('POLYGON ((-85.324076923076916 44.028020242914977,-84.280765182186229 44.16008502024291,-84.003429149797569 43.301663967611333,-83.607234817813762 42.91867611336032,-84.227939271255053 42.060255060728736,-84.941089068825903 41.307485829959511,-85.931574898785414 41.624441295546553,-85.588206477732783 43.011121457489871,-85.324076923076916 44.028020242914977))'),
    #           wkt.loads('POLYGON ((-89.24640080971659 46.061817813765174,-88.942651821862341 46.378773279352224,-88.454012145748976 46.431599190283393,-87.952165991902831 46.11464372469635,-88.163469635627521 45.190190283400803,-88.889825910931165 44.503453441295541,-88.770967611336033 43.552587044534405,-88.942651821862341 42.786611336032379,-89.774659919028338 42.760198380566798,-90.038789473684204 43.777097165991897,-89.735040485829956 45.097744939271251,-89.24640080971659 46.061817813765174))')]
    TEMPORAL = [datetime.datetime(1950,2,1),datetime.datetime(1950,4,30)]
    DISSOLVE = True
    CLIP = True
    VAR = 'Prcp'

    dataset = nc.Dataset(NC,'r')
    ncp = OcgDataset(dataset)
#    ncp._set_overlay_(POLYINT)
#    npd = ncp._get_numpy_data_(VAR,POLYINT,TEMPORAL)
    elements = ncp.extract_elements(VAR,polygon=POLYINT,time_range=TEMPORAL,clip=CLIP,dissolve=DISSOLVE)
#    gj = ncp.as_geojson(elements)
    path = ncp.as_shp(elements)
    print(path)