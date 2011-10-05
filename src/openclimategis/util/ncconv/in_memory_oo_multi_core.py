# -*- coding: utf-8 -*-
import numpy as np
from shapely.geometry.polygon import Polygon
import datetime
import netCDF4 as nc
import itertools
import geojson
from shapely.ops import cascaded_union
#from openclimategis.util.helpers import get_temp_path
#from openclimategis.util.toshp import OpenClimateShp
from shapely.geometry.multipolygon import MultiPolygon, MultiPolygonAdapter
from shapely import prepared, wkt
from shapely.geometry.geo import asShape
import time, sys
from multiprocessing import Process, Queue, Lock
from math import sqrt
import ipdb
import os
from osgeo import osr, ogr
from util.helpers import get_temp_path
from util.toshp import OpenClimateShp

dtime = 0

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
        self.url = dataset

        self.dataset = nc.Dataset(dataset,'r')
        self.multiReset = kwds.get('multiReset') or False
        self.verbose = kwds.get('verbose')
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
        self.level_name = kwds.get('level_name') or 'levels'
#        self.clip = kwds.get('clip') or False
#        self.dissolve = kwds.get('dissolve') or False

        #print self.dataset.variables[self.time_name].units
        #sys.exit()
        
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

        #data file must be closed and reopened to work properly with multiple threads
        if self.multiReset:
            if self.verbose>1: print 'closed'
            self.dataset.close()

    def _itr_array_(self,a):
        "a -- 2-d ndarray"
        ix = a.shape[0]
        jx = a.shape[1]
        for ii,jj in itertools.product(xrange(ix),xrange(jx)):
            yield ii,jj
            
    def _contains_(self,grid,lower,upper):
        
        ## small ranges on coordinates requires snapping to closest coordinate
        ## to ensure values are selected through logical comparison.
        ugrid = np.unique(grid)
        lower = ugrid[np.argmin(np.abs(ugrid-lower))]
        upper = ugrid[np.argmin(np.abs(ugrid-upper))]
        
        s1 = grid >= lower
        s2 = grid <= upper
        ret = s1*s2

        return(ret)
            
    def _set_overlay_(self,polygon=None,clip=False):
        """
        Perform spatial operations.
        
        polygon=None -- shapely polygon object
        clip=False -- set to True to perform an intersection
        """
        
        if self.verbose>1: print('overlay...')
        
        ## holds polygon objects
        self._igrid = np.empty(self.min_row.shape,dtype=object)
        ## hold point objects
        self._jgrid = np.empty(self.min_row.shape,dtype=object)
        ##holds locations that would be partial if the data were clipped for use in dissolve
        self._pgrid = np.zeros(self.min_row.shape,dtype=bool)
        ## holds weights for area weighting in the case of a dissolve
        self._weights = np.zeros(self.min_row.shape)
        ## initial subsetting to avoid iterating over all polygons unless abso-
        ## lutely necessary
        if polygon is not None:
            emin_col,emin_row,emax_col,emax_row = polygon.envelope.bounds
            #print emin_col,emin_row,emax_col,emax_row
            #print self.min_col
            #print self.max_col
            #print self.min_row
            #print self.max_row
#            ipdb.set_trace()
            smin_col = self._contains_(self.min_col,emin_col,emax_col)
            smax_col = self._contains_(self.max_col,emin_col,emax_col)
            smin_row = self._contains_(self.min_row,emin_row,emax_row)
            smax_row = self._contains_(self.max_row,emin_row,emax_row)
            #print smin_col
            #print smax_col
            #print smin_row
            #print smax_row
            #include = smin_col*smax_col*smin_row*smax_row
            include = np.any((smin_col,smax_col),axis=0)*np.any((smin_row,smax_row),axis=0)
            #print include
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
#        tr()
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


            if polygon.intersection(g).area==0:
                continue
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
                #check if the geometry partially intersects the AoI
                #without this multiple features covering the same location will 
                #occur when threading is enabled
                if g.intersection(polygon).area<g.area and g.intersection(polygon).area>0:
                    self._pgrid[ii,jj]=True
            ## calculate the weight
            w = ng.area/prearea

            ## a polygon can have a true intersects but actually not overlap
            ## i.e. shares a border.
            if w > 0:
                self._igrid[ii,jj] = ng
                self._weights[ii,jj] = w
                self._jgrid[ii,jj] = (g.centroid.x,g.centroid.y)
        ## the mask is used as a subset
        self._mask = self._weights > 0
        #print self._mask
        #print self._pgrid
#        self._weights = self._weights/self._weights.sum()
        #print self._weights
        #print self._mask.shape
    
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
        
        
    def _get_numpy_data_(self,var_name,polygon=None,time_range=None,clip=False,levels = [0],lock=Lock()):
        """
        var_name -- NC variable to extract from
        polygon=None -- shapely polygon object
        time_range=None -- [lower datetime, upper datetime]
        clip=False -- set to True to perform a full intersection
        """
        if self.verbose>1: print('getting numpy data...')

        ## perform the spatial operations
        self._set_overlay_(polygon=polygon,clip=clip)

        def _u(arg):
            "Pulls unique values and generates an evenly spaced array."
            un = np.unique(arg)
            ret = np.arange(un.min(),un.max()+1)
            return(ret)
        
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
        self._jgrid = _sub(self._jgrid)
        self._pgrid = _sub(self._pgrid)
        
        ## hit the dataset and extract the block
        npd = None

        #print an error message and return if the selection doesn't include any data
        if len(self._idxrow)==0:
            if self.verbose>0: print "Invalid Selection, unable to select row"
            return
        if len(self._idxcol)==0:
            if self.verbose>0: print "Invalid Selection, unable to select column"
            return
        if len(self._idxtime)==0:
            if self.verbose>0: print "Invalid Selection, unable to select time range"
            return
        
        narg = time.clock()

        #attempt to aquire the file lock
        while not(lock.acquire(False)):
            time.sleep(.1)

        #reopen the data file
        if self.multiReset:
            self.dataset = nc.Dataset(self.url,'r')

        ##check if data is 3 or 4 dimensions
        dimShape = len(self.dataset.variables[var_name].dimensions)

        #grab the data
        if dimShape == 3:
            npd = self.dataset.variables[var_name][self._idxtime,self._idxrow,self._idxcol]
            # reshape the data if the selection causes a loss of dimension(s)
            if len(npd.shape) <= 2:
                npd = npd.reshape(len(self._idxtime),len(self._idxrow),len(self._idxcol))
        elif dimShape == 4:
            #check if 1 or more levels have been selected
            if len(levels)==0:
                if self.verbose>0: print "Invalid Selection, unable to select levels"
                return

            #grab level values
            self.levels = self.dataset.variables[self.level_name][:]

            npd = self.dataset.variables[var_name][self._idxtime,levels,self._idxrow,self._idxcol]

            # reshape the data if the selection causes a loss of dimension(s)
            if len(npd.shape)<=3:
                npd = npd.reshape(len(self._idxtime),len(levels),len(self._idxrow),len(self._idxcol))

            #print self._weights

        #close the dataset
        if self.multiReset:
            self.dataset.close()
    
        #release the file lock
        lock.release()

        if self.verbose>1: print "dtime: ", time.clock()-narg
 
        
        if self.verbose>1: print('numpy extraction done.')
        
        return(npd)
    
    def _is_masked_(self,arg):
        "Ensures proper formating of masked data for single-layer data."
        if isinstance(arg,np.ma.MaskedArray):
            return None
        else:
            return arg

    def _is_masked2_(self,arg):
        "Ensures proper formating of masked data for multi-layer data."
        #print arg
        if isinstance(arg[0],np.ma.MaskedArray):
            return None
        else:
            return np.array(arg)
    
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
        if self.verbose>1: print('extracting elements...')
        ## dissolve argument is unique to extract_elements
        if 'dissolve' in kwds:
            dissolve = kwds.pop('dissolve')
        else:
            dissolve = False

        if 'levels' in kwds:
            levels = kwds.get('levels')

        #get the parent polygon ID so geometry/features can be recombined later
        if 'parentPoly' in kwds:
            parent = kwds.pop('parentPoly')
        else:
            parent = None

        clip = kwds.get('clip')
            
        ## extract numpy data from the nc file
        q=args[0]
        var = args[1]
        npd = self._get_numpy_data_(*args[1:],**kwds)


        #if hasattr(npd,'mask'):
            #print self.url," has a mask layer"
        #else:
            #print self.url," does not have a mask layer"

        #cancel if there is no data
        if npd is None:
            return
        ##check which flavor of climate data we are dealing with
        ocgShape = len(npd.shape)
        ## will hold feature dictionaries
        features = []
        ## partial pixels
        recombine = {}
        ## the unique identified iterator
        ids = self._itr_id_()
        gpass = True
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
                    lyr = None

                    if ocgShape==3:
                        lyr = npd[kk,:,:]
                    elif ocgShape==4:
                        lyr = npd[kk,0,:,:]
                    ## select values with spatial overlap and not masked
                    if hasattr(lyr,'mask'):
                        select = self._mask*np.invert(lyr.mask)
                    else:
                        select = self._mask


                    #cut out partial values
                    if not clip:
                        pselect = select*self._pgrid
                        select *= np.invert(self._pgrid)

                    ## select those geometries
                    geoms = self._igrid[select]
                    #print geoms
                    if len(geoms)>0:
                        ## union the geometries
                        unioned = cascaded_union([p for p in geoms])
                        ## select the weight subset and normalize to unity
                        sub_weights = self._weights*select
                        #print sub_weights
                        #print unioned.area
                        self._weights = sub_weights/sub_weights.sum()
                        ## apply the weighting
                        weighted = npd*self._weights
                        #print self._weights
                        #print weighted
                        #print (npd*sub_weights).sum()
                        #print select.sum()
                        #weighted = npd/sub_weights.sum()*sub_weights
                    else:
                        gpass = False
                ## generate the feature

                #only bother with dissolve if there are one or more features that
                #fully intersect the AoI
                if gpass:
                    if ocgShape==3:
                        feature = dict(
                            id=ids.next(),
                            geometry=unioned,
                            properties=dict({var:float(weighted[kk,:,:].sum()),
                                            'timestamp':self.timevec[self._idxtime[kk]]}))
                    elif ocgShape==4:
                        feature = dict(
                            id=ids.next(),
                            geometry=unioned,
                            properties=dict({var:float(list(weighted[kk,x,:,:].sum() for x in xrange(len(levels)))),
                                            'timestamp':self.timevec[self._idxtime[kk]],
                                            'levels':list(x for x in self.levels[levels])}))
                    
                    #record the weight used so the geometry can be
                    #properly recombined later
                    if not(parent == None) and dissolve:
                        feature['weight']=sub_weights.sum()
                    features.append(feature)

                #Record pieces that partially cover geometry so duplicates
                #can later be filtered out the the unique values recombined
                if not clip:
                    for ii,jj in self._itr_array_(pselect):
                        if self._pgrid[ii,jj]:
                            ctr = self._jgrid[ii,jj]
                            if kk==0:
                                recombine[ctr] = []
                            if ocgShape==3:
                                feature = dict(
                                    id=ids.next(),
                                    geometry=self._igrid[ii,jj],
                                    weight=1.0,
                                    properties=dict({var:float(npd[kk,ii,jj]),
                                                    'timestamp':self.timevec[self._idxtime[kk]]}))
                                #print npd[kk,ii,jj]
                            if ocgShape==4:
                                feature = dict(
                                    id=ids.next(),
                                    geometry=self._igrid[ii,jj],
                                    weight=1.0,
                                    properties=dict({var:float(list(npd[kk,x,ii,jj] for x in xrange(len(levels)))),
                                                    'timestamp':self.timevec[self._idxtime[kk]],
                                                    'level':list(x for x in self.levels[levels])}))
                            recombine[ctr].append(feature)
                            
        else:
            #print self._mask
            ctr = None
            ## loop for each feature. no dissolving.
            for ii,jj in self._itr_array_(self._mask):
                ## if the data is included, add the feature
                if self._mask[ii,jj] == True:
                    #if the geometry has a fraction of a pixel, the other factions could be handled by a different thread
                    #these must be recombined later, or if it's not clipped there will be duplicates to filter out
                    if self._weights[ii,jj] < 1 or not clip:
                        #tag the location this data value is at so it can be compared later
                        ctr = self._jgrid[ii,jj]
                        recombine[ctr] = []
                    ## extract the data and convert any mask values
                    #print ocgShape
                    if ocgShape == 3:
                        #if hasattr(npd,'mask'):
                            #print np.invert(npd.mask)
                        #else:
                            #print 'no mask found'
                        data = [self._is_masked_(da) for da in npd[:,ii,jj]]
                        #print data
                        for kk in range(len(data)):
                            ## do not add the feature if the value is a NoneType
                            if data[kk] == None: continue
                            feature = dict(
                                id=ids.next(),
                                geometry=self._igrid[ii,jj],
                                properties=dict({var:float(data[kk]),
                                                'timestamp':self.timevec[self._idxtime[kk]]}))
                            #if the data point covers a partial pixel or isn't clipped add it to the recombine set, otherwise leave it alone
                            if self._weights[ii,jj] < 1 or (self._pgrid[ii,jj] and not clip):
                                recombine[ctr].append(feature)
                            else:
                                features.append(feature)
                            

                    elif ocgShape == 4:
                        #if hasattr(npd,'mask'):
                            #print np.invert(npd.mask)
                        #else:
                            #print 'no mask found'
                        if self._weights[ii,jj] < 1 or not clip:
                            ctr = self._jgrid[ii,jj]
                            recombine[ctr] = []

                        data = [self._is_masked2_(da) for da in npd[:,:,ii,jj]]
                        #print data
                        for kk in range(len(data)):
                            ## do not add the feature if the value is a NoneType
                            if data[kk] == None: continue
                            feature = dict(
                                id=ids.next(),
                                geometry=self._igrid[ii,jj],
                                properties=dict({var:list(float(data[kk][x]) for x in xrange(len(levels))),
                                                'timestamp':self.timevec[self._idxtime[kk]],
                                                'level':list(x for x in self.levels[levels])}))
                            #q.put(feature)
                            if self._weights[ii,jj] < 1 or (self._pgrid[ii,jj] and not clip):
                                recombine[ctr].append(feature)
                            else:
                                features.append(feature)
        if self.verbose>1: print('extraction complete.')

        if not(parent == None) and dissolve:
            q.put((parent,features,recombine))
        else:
            q.put((features,recombine))
        return
        #sys.exit(0)
        #return(features)
    
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

def as_tabular(elements,var,wkt=False,wkb=False,path = None):
    '''writes output in a tabular, CSV format
geometry output is optional'''
    import osgeo.ogr as ogr

    if path is None:
        path = get_temp_path(suffix='.txt')

    #define spatial references for the projection
    sr = ogr.osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    sr2 = ogr.osr.SpatialReference()
    sr2.ImportFromEPSG(3005) #Albers Equal Area is used to ensure legitimate area values

    with open(path,'w') as f:

        for ii,element in enumerate(elements):

            #convert area from degrees to m^2
            geo = ogr.CreateGeometryFromWkb(element['geometry'].wkb)
            geo.AssignSpatialReference(sr)
            geo.TransformTo(sr2)
            area = geo.GetArea()

            #write id, timestamp, variable
            f.write(','.join([repr(ii+1),element['properties']['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),repr(element['properties'][var])]))

            #write level if the dataset has levels
            if 'level' in element['properties'].keys():
                f.write(','+repr(element['properties']['level']))

            #write the area
            f.write(','+repr(area))

            #write wkb if requested
            if wkb:
                f.write(','+repr(element['geometry'].wkb))

            #write wkt if requested
            if wkt:
                f.write(','+repr(element['geometry'].wkt))

            f.write('\n')
        f.close()

    return path

def as_keyTabular(elements,var,wkt=False,wkb=False,path = None):
    '''writes output as tabular csv files, but uses foreign keys
on time and geometry to reduce file size'''
    import osgeo.ogr as ogr

    if path is None:
        path = get_temp_path(suffix='')

    if len(path)>4 and path[-4] == '.':
        path = path[:-4]

    patht = path+"_time.txt"
    pathg = path+"_geometry.txt"
    pathd = path+"_data.txt"

    #define spatial references for the projection
    sr = ogr.osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    sr2 = ogr.osr.SpatialReference()
    sr2.ImportFromEPSG(3005)
    data = {}

    #sort the data into dictionaries so common times and geometries can be identified
    for ii,element in enumerate(elements):

        #record new element ids (otherwise threads will produce copies of ids)
        element['id']=ii

        #get the time and geometry
        time = element['properties']['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        ewkt = element['geometry'].wkt

        if not (time in data):
            data[time] = {}

        #put the data into the dictionary
        if not (ewkt in data[time]):
            data[time][ewkt] = [element]
        else:
            data[time][ewkt].append(element)


    #get a unique set of geometry keys
    locs = []

    for key in data:
        locs.extend(data[key].keys())

    locs = set(locs)

    ft = open(patht,'w')
    fg = open(pathg,'w')
    fd = open(pathd,'w')

    #write the features to file
    for ii,time in enumerate(data.keys()):

        #write out id's and time values to the time file
        tdat = data[time]
        ft.write(repr(ii+1)+','+time+'\n')

        for jj,loc in enumerate(locs):
            if ii==0:

                #find the geometry area
                geo = ogr.CreateGeometryFromWkt(loc)
                geo.AssignSpatialReference(sr)
                geo.TransformTo(sr2)

                #write the id and area
                fg.write(repr(jj+1))
                fg.write(','+repr(geo.GetArea()))

                #write out optional geometry
                if wkt:
                    fg.write(','+loc)
                if wkb:
                    fg.write(','+repr(ogr.CreateGeometryFromWkt(loc).ExportToWkb()))
                fg.write('\n')

            if loc in tdat:
                for element in tdat[loc]:
                    #write out id, foreign keys (time then geometry) and the variable value
                    fd.write(','.join([repr(element['id']),repr(ii+1),repr(jj+1),repr(element['properties'][var])]))
                    #write out level if appropriate
                    if 'level' in element['properties']:
                        fd.write(','+repr(element['properties']['level']))
                    fd.write('\n')

    ft.close()
    fg.close()
    fd.close()

            

#['AddGeometry', 'AddGeometryDirectly', 'AddPoint', 'AddPoint_2D', 'AssignSpatialReference', 'Buffer', 
#'Centroid', 'Clone', 'CloseRings', 'Contains', 'ConvexHull', 'Crosses', 'Destroy', 'Difference', 
#'Disjoint', 'Distance', 'Empty', 'Equal', 'ExportToGML', 'ExportToJson', 'ExportToKML', 'ExportToWkb', 
#'ExportToWkt', 'FlattenTo2D', 'GetArea', 'GetBoundary', 'GetCoordinateDimension', 'GetDimension', 
#'GetEnvelope', 'GetGeometryCount', 'GetGeometryName', 'GetGeometryRef', 'GetGeometryType', 'GetPoint', 
#'GetPointCount', 'GetPoint_2D', 'GetSpatialReference', 'GetX', 'GetY', 'GetZ', 'Intersect', 'Intersection', 
#'IsEmpty', 'IsRing', 'IsSimple', 'IsValid', 'Overlaps', 'Segmentize', 'SetCoordinateDimension', 'SetPoint', 
#'SetPoint_2D', 'SymmetricDifference', 'Touches', 'Transform', 'TransformTo', 'Union', 'Within', 'WkbSize', 
#'__class__', '__del__', '__delattr__', '__dict__', '__doc__', '__format__', '__getattr__', '__getattribute__', 
#'__hash__', '__init__', '__iter__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', 
#'__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', '__swig_destroy__', 
#'__swig_getmethods__', '__swig_setmethods__', '__weakref__', 'next', 'this']


def multipolygon_multicore_operation(dataset,var,polygons,time_range=None,clip=None,dissolve=None,levels = None,ocgOpts=None,subdivide=False,subres='detect',verbose=1):

    elements = []
    ret = []
    q = Queue()
    l = Lock()
    pl = []

    #set the file reset option if the file is local
    if not('http:' in dataset or 'www.' in dataset):
        if ocgOpts == None:
            ocgOpts = {}
        ocgOpts['multiReset'] = True
    ocgOpts['verbose'] = verbose
    ncp = OcgDataset(dataset,**ocgOpts)

    #if no polygon was specified
    #create a polygon covering the whole area so that the job can be split
    if polygons == [None]:
        polygons = [Polygon(((ncp.col_bnds.min(),ncp.row_bnds.min()),(ncp.col_bnds.max(),ncp.row_bnds.min()),(ncp.col_bnds.max(),ncp.row_bnds.max()),(ncp.col_bnds.min(),ncp.row_bnds.max())))]

    for ii,polygon in enumerate(polygons):
        if verbose>1: print(ii)

        #skip invalid polygons
        if not polygon.is_valid:
            if verbose>0: print "Polygon "+repr(ii+1)+" is not valid. "+polygon.wkt
            continue

        #if polygons have been specified and subdivide is True, each polygon will be subdivided
        #into a grid with resolution of subres. If subres is undefined the resolution is half the square root of the area of the polygons envelope, or approximately 4 subpolygons
        if subdivide and not(polygons == None):

            #figure out the resolution and subdivide
            #default value uses sqrt(polygon envelop area)
            #generally resulting in 4-6 threads per polygon
            if subres == 'detect':
                subpolys = make_shapely_grid(polygon,sqrt(polygon.envelope.area)/2.0,clip=True)
            else:
                subpolys = make_shapely_grid(polygon,subres,clip=True)
            #generate threads for each subpolygon

#            subpolys = [subpolys[1]] ## tdk
            for poly in subpolys:
                ## during regular gridding used to create sub-polygons, a polygon
                ## may not intersect the actual extraction extent returning None
                ## in the process as opposed to a Polygon. skip the Nones.
                if poly is None: continue
                ## continue generating threads
                if clip is False:
                    poly2 = poly.intersection(polygon.envelope)
                    if poly2 is None: continue
                
                ## tdk #####################
#                ipdb.set_trace()
#                ncp.extract_elements(q,var,lock=l,polygon=poly,time_range=time_range,clip=clip,dissolve=dissolve,levels=levels,parentPoly=11)
                ############################
                
                p = Process(target = ncp.extract_elements,
                                args =       (q,
                                                var,),
                                kwargs= {
                                                'lock':l,
                                                'polygon':poly,
                                                'time_range':time_range,
                                                'clip':clip,
                                                'dissolve':dissolve,
                                                'levels' : levels,
                                                'parentPoly':ii})
                p.start()
                pl.append(p)

        #if no polygons are specified only 1 thread will be created per polygon
        else:
            p = Process(target = ncp.extract_elements,
                            args =       (
                                            q,
                                            var,),
                            kwargs= {
                                            'lock':l,
                                            'polygon':polygon,
                                            'time_range':time_range,
                                            'clip':clip,
                                            'dissolve':dissolve,
                                            'levels' : levels,
                                            'parentPoly':ii})
            p.start()
            pl.append(p)

    #for p in pl:
        #p.join()

    #consumer loop, the main process will grab any feature lists added by the
    #processing threads and continues until those threads have terminated.
    #without this the processing threads will NOT terminate
    a=True
    while a:
        a=False
        #check if any threads are still active
        for p in pl:
            a = a or p.is_alive()

        #remove anything from the queue if present
        while not q.empty():
            ret.append(q.get())

        #give the threads some time to process more stuff
        time.sleep(.1)

    #The subdivided geometry must be recombined into the original polygons
    if dissolve:
        groups = {}

        #form groups of elements based on which polygon they belong to
        for x in ret:
            if not x[0] in groups:
                groups[x[0]] = []

            groups[x[0]].append((x[1],x[2]))
        #print '>',groups.keys()
        #print groups
        
        #for each group, recombine the geometry and average the data points
        for x in groups.keys():

            #for y in groups[x]:
                #print len(y[0])
            group = [y[0] for y in groups[x] if len(y[0])>0]

            #print groups[x][0][1]
            recombine ={}
            for y in groups[x]:
                recombine.update(y[1])

            for key in recombine.keys():
                group.append(recombine[key])

            #recombine the geometry using the first time period
            total = cascaded_union([y[0]['geometry'] for y in group])

            #form subgroups consisting of subpolygons that cover the same time period
            subgroups = [[g[t] for g in group] for t in xrange(len(group[0]))]

            ta = sum([y['weight'] for y in subgroups[0]])
            #print 't',ta

            #average the data values and form new features
            for subgroup in subgroups:
                if not(levels == None):
                    avg = [sum([y['properties'][var][z]*(y['weight']/ta) for y in subgroup]) for z in xrange(len(levels))]
                    elements.append(    dict(
                                        id=subgroup[0]['id'],
                                        geometry=total,
                                        properties=dict({VAR: avg,
                                                        'timestamp':subgroup[0]['properties']['timestamp'],
                                                        'level': subgroup[0]['properties']['levels']})))
                    #print total.area
                    #print avg
                else:
                    #print (y['weight']/ta)
                    avg = sum([y['properties'][var]*(y['weight']/ta) for y in subgroup])
                    elements.append(    dict(
                                        id=subgroup[0]['id'],
                                        geometry=total,
                                        properties=dict({var:float(avg),
                                                        'timestamp':subgroup[0]['properties']['timestamp']})))
    #handle recombining undissolved features
    else:
        recombine = []
        #pull out unique elements and potentially repeated elements
        for x in ret:
            elements.extend(x[0])
            recombine.append(x[1])

        #get a list of all unique locations
        keylist = []
        for x in recombine:
            keylist.extend(x.keys())
        keylist = set(keylist)

        #find all the locations that have duplicated features
        for key in keylist:
            cur = []
            for x in recombine:
                if key in x:
                    cur.append(x[key])
            #print cur

            #if there is only 1 feature, it is unique so toss it into the element list
            if len(cur)==1:
                elements.extend(cur[0])
            
            else:
                #if clip=False then all the features are identical, pick one and discard the rest
                if not clip:
                    elements.extend(cur[0])
                #if clip=True then the features have the same values but the geometry is fragmented
                else:
                    #recombine the geometry
                    geo = cascaded_union([x[0]['geometry'] for x in cur])
                    #pick a feature, update the geometry, and discard the rest
                    for x in cur[0]:
                        x['geometry'] = geo
                        elements.append(x)
                        
    elements2 = []

    #expand elements in the case of multi-level data
    dtime = time.time()
    if not (levels == None):
        for x in elements:
            #create a new feature for each data level
            for i in xrange(len(levels)):
                e = x.copy()
                e['properties'] = x['properties'].copy()
                e['properties'][var] = e['properties'][var][i]
                e['properties']['level'] = e['properties']['level'][i]
                elements2.append(e)
    else:
        elements2 = elements
    if verbose>1: print "expansion time: ",time.time()-dtime
    if verbose>1: print "points: ",repr(len(elements2))
    return(elements2)


def make_shapely_grid(poly,res,as_numpy=False,clip=True):
    """
    Return a list or NumPy matrix of shapely Polygon objects.
    
    poly -- shapely Polygon to discretize
    res -- target grid resolution in the same units as |poly|
    """
    
    ## ensure we have a floating point resolution
    res = float(res)
    ## check that the target polygon is a valid geometry
    assert(poly.is_valid)
    ## vectorize the polygon creation
    vfunc_poly = np.vectorize(make_poly_array)#,otypes=[np.object])
    ## prepare the geometry for faster spatial relationship checking. throws a
    ## a warning so leaving out for now.
#    prepped = prep(poly)
    
    ## extract bounding coordinates of the polygon
    min_x,min_y,max_x,max_y = poly.envelope.bounds
    ## convert to matrices
    X,Y = np.meshgrid(np.arange(min_x,max_x,res),
                      np.arange(min_y,max_y,res))
    #print X,Y

    ## shift by the resolution
    pmin_x = X
    pmax_x = X + res
    pmin_y = Y
    pmax_y = Y + res
    ## make the 2-d array
#    print pmin_y,pmin_x,pmax_y,pmax_x,poly.wkt
    if clip:
        poly_array = vfunc_poly(pmin_y,pmin_x,pmax_y,pmax_x,poly)
    else:
        poly_array = vfunc_poly(pmin_y,pmin_x,pmax_y,pmax_x)
    #print poly_array
    #sys.exit()
    ## format according to configuration arguments
    if as_numpy:
        ret = poly_array
    else:
        ret = list(poly_array.reshape(-1))
    
    return(ret)

    
def make_poly_array(min_row,min_col,max_row,max_col,polyint=None):
    ret = Polygon(((min_col,min_row),
                    (max_col,min_row),
                    (max_col,max_row),
                    (min_col,max_row),
                    (min_col,min_row)))
    if polyint is not None:
        if polyint.intersects(ret) == False:
            ret = None
        else:
            ret = polyint.intersection(ret)
    return(ret)
        
def shapely_to_shp(obj,outname):
    path = os.path.join('/tmp',outname+'.shp')
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ogr_geom = 3
    
    dr = ogr.GetDriverByName('ESRI Shapefile')
    ds = dr.CreateDataSource(path)
    try:
        if ds is None:
            raise IOError('Could not create file on disk. Does it already exist?')
            
        layer = ds.CreateLayer('lyr',srs=srs,geom_type=ogr_geom)
        feature_def = layer.GetLayerDefn()
        feat = ogr.Feature(feature_def)
        feat.SetGeometry(ogr.CreateGeometryFromWkt(obj.wkt))
        layer.CreateFeature(feat)
    finally:
        ds.Destroy()

       
if __name__ == '__main__':
    narg = time.time()
    ## all
#    POLYINT = Polygon(((-99,39),(-94,38),(-94,40),(-100,39)))
    ## great lakes
    #POLYINT = Polygon(((-90.35,40.55),(-83,43),(-80.80,49.87),(-90.35,49.87)))
    #POLYINT = Polygon(((-90,30),(-70,30),(-70,50),(-90,50)))
    #POLYINT = Polygon(((-90,40),(-80,40),(-80,50),(-90,50)))
    #POLYINT = Polygon(((-130,18),(-60,18),(-60,98),(-130,98)))
    #POLYINT = Polygon(((0,0),(0,10),(10,10),(10,0)))
    ## return all data
    POLYINT = None
    ## two areas
    #POLYINT = [wkt.loads('POLYGON ((-85.324076923076916 44.028020242914977,-84.280765182186229 44.16008502024291,-84.003429149797569 43.301663967611333,-83.607234817813762 42.91867611336032,-84.227939271255053 42.060255060728736,-84.941089068825903 41.307485829959511,-85.931574898785414 41.624441295546553,-85.588206477732783 43.011121457489871,-85.324076923076916 44.028020242914977))'),
              #wkt.loads('POLYGON ((-89.24640080971659 46.061817813765174,-88.942651821862341 46.378773279352224,-88.454012145748976 46.431599190283393,-87.952165991902831 46.11464372469635,-88.163469635627521 45.190190283400803,-88.889825910931165 44.503453441295541,-88.770967611336033 43.552587044534405,-88.942651821862341 42.786611336032379,-89.774659919028338 42.760198380566798,-90.038789473684204 43.777097165991897,-89.735040485829956 45.097744939271251,-89.24640080971659 46.061817813765174))')]
    ## watersheds
#    path = '/home/bkoziol/git/OpenClimateGIS/bin/geojson/watersheds_4326.geojson'
##    select = ['HURON']
#    select = []
#    with open(path,'r') as f:
#        data = ''.join(f.readlines())
##        data2 = f.read()
##        tr()
##    tr()
#    gj = geojson.loads(data)
#    POLYINT = []
#    for feature in gj['features']:
#        if select:
#            prop = feature['properties']
#            if prop['HUCNAME'] in select:
#                pass
#            else:
#                continue
#        geom = asShape(feature['geometry'])
#        if not isinstance(geom,MultiPolygonAdapter):
#            geom = [geom]
#        for polygon in geom:
#            POLYINT.append(polygon)
    
    #NC = '/home/reid/Desktop/ncconv/pcmdi.ipcc4.bccr_bcm2_0.1pctto2x.run1.monthly.cl_A1_1.nc'
    #NC = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data/maurer/bccr_bcm2_0.1.sresa1b.monthly.Prcp.1950.nc'
    #NC = '/home/reid/Desktop/ncconv/bccr_bcm2_0.1.sresa1b.monthly.Prcp.1950.nc'
    #NC = 'http://hydra.fsl.noaa.gov/thredds/dodsC/oc_gis_downscaling.bccr_bcm2.sresa1b.Prcp.Prcp.1.aggregation.1'
    NC = 'test.nc'

#    TEMPORAL = [datetime.datetime(1950,2,1),datetime.datetime(1950,4,30)]
    TEMPORAL = [datetime.datetime(1950,2,1),datetime.datetime(1950,5,1)]
    #TEMPORAL = [datetime.datetime(1960,3,16),datetime.datetime(1961,3,16)] #time range for multi-level file
    DISSOLVE = False
    CLIP = False
    #VAR = 'cl'
    VAR = 'Prcp'
    #kwds={}
    kwds = {
        #'rowbnds_name': 'lat_bnds', 
        #'colbnds_name': 'lon_bnds',
        #'time_units': 'days since 1800-1-1 00:00:0.0',
        'time_units': 'days since 1950-1-1 0:0:0.0',
        #'level_name': 'lev'
    }
    LEVELS = None
    #LEVELS = [x for x in range(0,1)]
    #LEVELS = [x for x in range(0,10)]
    ## open the dataset for reading
    dataset = NC#nc.Dataset(NC,'r')
    ## make iterable if only a single polygon requested
    if type(POLYINT) not in (list,tuple): POLYINT = [POLYINT]
    ## convenience function for multiple polygons
    elements = multipolygon_multicore_operation(dataset,
                                      VAR,
                                      POLYINT,
                                      time_range=TEMPORAL,
                                      clip=CLIP,
                                      dissolve=DISSOLVE,
                                      levels = LEVELS,
                                      ocgOpts=kwds,
                                      subdivide=True,
                                      #subres = 90
                                      )


#    out = as_shp(elements)
    dtime = time.time()
    #out = as_geojson(elements)
    #with open('./out_M3.json','w') as f:
        #f.write(out)
    as_keyTabular(elements,VAR,path='./out_tabular.txt',wkt=True)
    dtime = time.time()-dtime

    blarg = time.time()
    print blarg-narg,dtime,blarg-narg-dtime
