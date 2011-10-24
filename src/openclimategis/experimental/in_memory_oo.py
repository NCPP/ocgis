import numpy as np
from shapely.geometry.polygon import Polygon
import datetime
import netCDF4 as nc
import itertools
import geojson
from shapely.ops import cascaded_union
from openclimategis.util.helpers import get_temp_path
from openclimategis.util.toshp import OpenClimateShp
from shapely.geometry.multipolygon import MultiPolygon, MultiPolygonAdapter
from shapely import prepared, wkt
from shapely.geometry.geo import asShape


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
        s1 = grid > lower
        s2 = grid < upper
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
        self._weights = np.zeros(self.min_row.shape)
        
        ## initial subsetting to avoid iterating over all polygons unless abso-
        ## lutely necessary
        if polygon is not None:
            emin_col,emin_row,emax_col,emax_row = polygon.envelope.bounds
            smin_col = self._contains_(self.min_col,emin_col,emax_col)
            smax_col = self._contains_(self.max_col,emin_col,emax_col)
            smin_row = self._contains_(self.min_row,emin_row,emax_row)
            smax_row = self._contains_(self.max_row,emin_row,emax_row)
            include = np.any((smin_col,smax_col),axis=0)*np.any((smin_row,smax_row),axis=0)
        else:
            include = np.empty(self.min_row.shape,dtype=bool)
            include[:,:] = True
        
#        print('constructing grid...')
#        ## construct the subset of polygon geometries
#        vfunc = np.vectorize(self._make_poly_array_)
#        self._igrid = vfunc(include,
#                            self.min_row,
#                            self.min_col,
#                            self.max_row,
#                            self.max_col,
#                            polygon)
#        
#        ## calculate the areas for potential weighting
#        print('calculating area...')
#        def _area(x):
#            if x != None:
#                return(x.area)
#            else:
#                return(0.0)
#        vfunc_area = np.vectorize(_area,otypes=[np.float])
#        preareas = vfunc_area(self._igrid)
#        
#        ## if we are clipping the data, modify the geometries and record the weights
#        if clip and polygon:
#            print('clipping...')
##            polys = []
##            for p in self._igrid.reshape(-1):
##                polys.append(self._intersection_(polygon,p))
#            vfunc = np.vectorize(self._intersection_)
#            self._igrid = vfunc(polygon,self._igrid)
#            
#        ## calculate weights following intersection
#        areas = vfunc_area(self._igrid)
#        def _weight(x,y):
#            if y == 0:
#                return(0.0)
#            else:
#                return(x/y)
#        self._weights=np.vectorize(_weight)(areas,preareas)
#        
#        ## set the mask
#        self._mask = self._weights > 0
#        
#        print('overlay done.')
        
        ## loop for each spatial grid element
        if polygon:
#            prepared_polygon = polygon
            prepared_polygon = prepared.prep(polygon)
        for ii,jj in self._itr_array_(include):
            if not include[ii,jj]: continue
            ## create the polygon
            g = self._make_poly_((self.min_row[ii,jj],self.max_row[ii,jj]),
                                 (self.min_col[ii,jj],self.max_col[ii,jj]))
            ## add the polygon if it intersects the aoi of if all data is being
            ## returned.
            if polygon:
                if not prepared_polygon.intersects(g): continue
#            if g.intersects(polygon) or polygon is None:
                ## get the area before the intersection
            prearea = g.area
            ## full intersection in the case of a clip and an aoi is passed
#                if g.overlaps(polygon) and clip is True and polygon is not None:
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
    
    @staticmethod    
    def _make_poly_array_(include,min_row,min_col,max_row,max_col,polygon=None):
        ret = None
        if include:
            poly = Polygon(((min_col,min_row),
                         (max_col,min_row),
                         (max_col,max_row),
                         (min_col,max_row),
                         (min_col,min_row)))
            if polygon != None:
                if polygon.intersects(poly):
                    ret = poly
            else:
                ret = poly
        return(ret)
        
    @staticmethod
    def _intersection_(polygon,target):
        ret = None
        if target != None:
            ppp = target.intersection(polygon)
            if not ppp.is_empty:
                ret = ppp
        return(ret)
        
        
    def _get_numpy_data_(self,var_name,polygon=None,time_range=None,clip=False):
        """
        var_name -- NC variable to extract from
        polygon=None -- shapely polygon object
        time_range=None -- [lower datetime, upper datetime]
        clip=False -- set to True to perform a full intersection
        """
        print('getting numpy data...')
        
        ## perform the spatial operations
        self._set_overlay_(polygon=polygon,clip=clip)

        def _u(arg):
            "Pulls unique values and generates an evenly spaced array."
            un = np.unique(arg)
            return(np.arange(un.min(),un.max()+1))
        
        def _sub(arg):
            "Subset an array."
            return arg[self._idxrow.min():self._idxrow.max()+1,
                       self._idxcol.min():self._idxcol.max()+1]
        
        ## get the time indices
        if time_range is not None:
            self._idxtime = np.arange(
             0,
             len(self.timevec))[(self.timevec>=time_range[0])*
                                (self.timevec<=time_range[1])]
        else:
            self._idxtime = np.arange(0,len(self.timevec))
        
        ## reference the original (world) coordinates of the netCDF when selecting
        ## the spatial subset.
        self._idxrow = _u(self.real_row[self._mask])
        self._idxcol = _u(self.real_col[self._mask])
         
        ## subset our reference arrays in a similar manner
        self._mask = _sub(self._mask)
        self._weights = _sub(self._weights)
        self._igrid = _sub(self._igrid)
        
        ## hit the dataset and extract the block
        npd = self.dataset.variables[var_name][self._idxtime,self._idxrow,self._idxcol]
        
        ## add in an extra dummy dimension in the case of one time layer
        if len(npd.shape) == 2:
            npd = npd.reshape(1,npd.shape[0],npd.shape[1])
        
        print('numpy extraction done.')
        
        return(npd)
    
    def _is_masked_(self,arg):
        "Ensures proper formating of masked data."
        if isinstance(arg,np.ma.core.MaskedConstant):
            return None
        else:
            return arg
    
    def extract_elements(self,*args,**kwds):
        """
        Merges the geometries and extracted attributes into a GeoJson-like dictionary
        list.
        
        var_name -- NC variable to extract from
        dissolve=False -- set to True to merge geometries and calculate an 
            area-weighted average
        polygon=None -- shapely polygon object
        time_range=None -- [lower datetime, upper datetime]
        clip=False -- set to True to perform a full intersection
        """
        print('extracting elements...')
        
        ## dissolve argument is unique to extract_elements
        if 'dissolve' in kwds:
            dissolve = kwds.pop('dissolve')
        else:
            dissolve = False
        ## pull the variable name from the arguments
        var_name = args[0]
        
        ## extract numpy data from the nc file
        npd = self._get_numpy_data_(*args,**kwds)
        ## will hold feature dictionaries
        features = []
        ## the unique identified iterator
        ids = self._itr_id_()

        if dissolve:
            ## one feature is created for each unique time
            for kk in range(len(self._idxtime)):
                ## check if this is the first iteration. approach assumes that
                ## masked values are homogenous through the time layers. this
                ## avoids multiple union operations on the geometries. i.e.
                ##    time 1 = masked, time 2 = masked, time 3 = masked
                ##        vs.
                ##    time 1 = 0.5, time 2 = masked, time 3 = 0.46
                if kk == 0:
                    ## on the first iteration:
                    ##    1. make the unioned geometry
                    ##    2. weight the data according to area
                    
                    ## reference layer for the masked data
                    lyr = npd[kk,:,:]
                    ## select values with spatial overlap and not masked
                    if hasattr(lyr,'mask'):
                        select = self._mask*np.invert(lyr.mask)
                    else:
                        select = self._mask
                    ## select those geometries
                    geoms = self._igrid[select]
                    ## union the geometries
                    unioned = cascaded_union([p for p in geoms])
                    ## select the weight subset and normalize to unity
                    sub_weights = self._weights*select
                    self._weights = sub_weights/sub_weights.sum()
                    ## apply the weighting
                    weighted = npd*self._weights
                ## generate the feature
                feature = dict(
                    id=ids.next(),
                    geometry=unioned,
                    properties=dict({var_name:float(weighted[kk,:,:].sum()),
                                     'timestamp':self.timevec[self._idxtime[kk]]}))
                features.append(feature)
        else:
            ## loop for each feature. no dissolving.
            for ii,jj in self._itr_array_(self._mask):
                ## if the data is included, add the feature
                if self._mask[ii,jj] == True:
                    ## extract the data and convert any mask values
                    data = [self._is_masked_(da) for da in npd[:,ii,jj]]
                    for kk in range(len(data)):
                        ## do not add the feature if the value is a NoneType
                        if data[kk] == None: continue
                        feature = dict(
                            id=ids.next(),
                            geometry=self._igrid[ii,jj],
                            properties=dict({var_name:float(data[kk]),
                                             'timestamp':self.timevec[self._idxtime[kk]]}))
                        features.append(feature)
        
        print('extraction complete.')

        return(features)
    
    def _itr_id_(self,start=1):
        while True:
            try:
                yield start
            finally:
                start += 1
                
def as_geojson(elements):
    features = []
    for e in elements:
        e['properties']['timestamp'] = str(e['properties']['timestamp'])
        features.append(geojson.Feature(**e))
    fc = geojson.FeatureCollection(features)
    return(geojson.dumps(fc))
    
def as_shp(elements,path=None):
    if path is None:
        path = get_temp_path(suffix='.shp')
    ocs = OpenClimateShp(path,elements)
    ocs.write()
    return(path)

def multipolygon_operation(dataset,var,polygons,time_range=None,clip=None,dissolve=None,ocg_kwds={}):
    elements = []
    ncp = OcgDataset(dataset,**ocg_kwds)
    for polygon in polygons:
#        if ii != 2: continue
#        print(ii)
#        ncp = OcgDataset(dataset)
        elements += ncp.extract_elements(var,
                                         polygon=polygon,
                                         time_range=time_range,
                                         clip=clip,
                                         dissolve=dissolve)
    return(elements)
        
        
if __name__ == '__main__':

    #NC = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data/wcrp_cmip3/pcmdi.ipcc4.bccr_bcm2_0.1pctto2x.run1.monthly.cl_A1_1.nc'
    NC = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data/maurer/bccr_bcm2_0.1.sresa1b.monthly.Prcp.1950.nc'
    ## all
#    POLYINT = Polygon(((-99,39),(-94,38),(-94,40),(-100,39)))
    ## great lakes
#    POLYINT = Polygon(((-90.35,40.55),(-83,43),(-80.80,49.87),(-90.35,49.87)))
    ## return all data
#    POLYINT = None
    ## two areas
#    POLYINT = [wkt.loads('POLYGON ((-85.324076923076916 44.028020242914977,-84.280765182186229 44.16008502024291,-84.003429149797569 43.301663967611333,-83.607234817813762 42.91867611336032,-84.227939271255053 42.060255060728736,-84.941089068825903 41.307485829959511,-85.931574898785414 41.624441295546553,-85.588206477732783 43.011121457489871,-85.324076923076916 44.028020242914977))'),
#               wkt.loads('POLYGON ((-89.24640080971659 46.061817813765174,-88.942651821862341 46.378773279352224,-88.454012145748976 46.431599190283393,-87.952165991902831 46.11464372469635,-88.163469635627521 45.190190283400803,-88.889825910931165 44.503453441295541,-88.770967611336033 43.552587044534405,-88.942651821862341 42.786611336032379,-89.774659919028338 42.760198380566798,-90.038789473684204 43.777097165991897,-89.735040485829956 45.097744939271251,-89.24640080971659 46.061817813765174))')]
    ## watersheds
    path = '/home/bkoziol/git/OpenClimateGIS/bin/geojson/watersheds_4326.geojson'
#    select = ['HURON']
    select = []
    with open(path,'r') as f:
        data = ''.join(f.readlines())
#        data2 = f.read()
    gj = geojson.loads(data)
    POLYINT = []
    for feature in gj['features']:
        if select:
            prop = feature['properties']
            if prop['HUCNAME'] in select:
                pass
            else:
                continue
        geom = asShape(feature['geometry'])
        if not isinstance(geom,MultiPolygonAdapter):
            geom = [geom]
        for polygon in geom:
            POLYINT.append(polygon)
    
    TEMPORAL = [datetime.datetime(1950,2,1),datetime.datetime(1950,4,30)]
#    TEMPORAL = [datetime.datetime(1950,2,1),datetime.datetime(1950,3,1)]
    DISSOLVE = True
    CLIP = True
    VAR = 'Prcp'

    dataset = nc.Dataset(NC,'r')
    
    if type(POLYINT) not in (list,tuple): POLYINT = [POLYINT]
        
    elements = multipolygon_operation(dataset,
                                      VAR,
                                      POLYINT,
                                      time_range=TEMPORAL,
                                      clip=CLIP,
                                      dissolve=DISSOLVE)
    
#    ncp = OcgDataset(dataset)
#    ncp._set_overlay_(POLYINT)
#    npd = ncp._get_numpy_data_(VAR,POLYINT,TEMPORAL)
#    elements = ncp.extract_elements(VAR,polygon=POLYINT,time_range=TEMPORAL,clip=CLIP,dissolve=DISSOLVE)
#    gj = ncp.as_geojson(elements)
#    out = as_shp(elements)
    out = as_geojson(elements)
#    print(out)
    with open('/tmp/out','w') as f:
        f.write(out)