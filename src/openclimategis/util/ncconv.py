from django.contrib.gis.db.models.manager import GeoManager
from django.contrib.gis.db import models
import numpy as np
from types import ClassType
import netCDF4 as n
from django.contrib.gis.db.models.query import GeoQuerySet
import itertools


class GeoQuerySetFactory(object):
    """
    Create GeoQuerySets from netCDF4 Dataset objects.
    
    rootgrp -- base netCDF4.Dataset object.
    var -- name of the variable to extract. ie. 'Tavg'
    **kwds -- customizable NC attribute names. see object's __init__ method for
        explanation.
    """
    
    _dtype_mapping = {np.dtype('float32'):[float,models.FloatField]}
    
    def __init__(self,rootgrp,var,**kwds):
        self.rootgrp = rootgrp
        self.var = var

        ## KEYWORD EXTRACTION --------------------------------------------------

        self.time = kwds.get('time') or 'time'
        self.srid = kwds.get('srid') or 4326
        self.app_label = kwds.get('app_label') or 'climatedata'
        self.objects = kwds.get('objects') or GeoManager
        self.nodata_var = kwds.get('nodata') or 'missing_value'
        
        ## ---------------------------------------------------------------------
        
        self._pyfrmt = None ## used to format values from NumPy data types
        
        ## pull the nodata value
        self.nodata = getattr(rootgrp.variables[self.var],self.nodata_var)
        ## construct the time vector
        times = self.rootgrp.variables[self.time]
        self._timevec = n.num2date(times[:],units=times.units,calendar=times.calendar)
        ## attributes of the Django Model
        self._base_attrs = {'id':models.IntegerField(primary_key=True),
                            'geom':models.MultiPolygonField(srid=self.srid),
                            'timestamp':models.DateTimeField(),
                            self.var:self._dtype_mapping[rootgrp.variables[self.var].dtype][1](),
                            'Meta':ClassType('Meta',(),{'app_label':self.app_label}),
                            'objects':self.objects(),}
        ## initialize the class
        self._klass = ClassType('NcModel',(models.Model,),self._base_attrs)
                        
    def get_pyfrmt(self,val):
        """
        Pulls the correct Python format from the type mapping.
        """  
        
        if self._pyfrmt == None:
            try:
                self._pyfrmt = self._dtype_mapping[val][0]
            ## numpy 'masked' values throws exception
            except KeyError:
                ## map this to a NoneType returning function
                if isinstance(val,np.ma.core.MaskedConstant):
                    self._pyfrmt = self._masked_
                else:
                    raise
        return self._pyfrmt
        
    def get_numpy_data(self,time_indices=[],x_indices=[],y_indices=[]):
        """
        Returns multi-dimensional NumPy array extracted from a NC.
        """
        
        sh = self.rootgrp.variables[self.var].shape
        
        ## If indices are not passed, this method defaults to returning all
        ##  data.
        
        if not time_indices:
            time_indices = range(0,sh[0])
        
        if not x_indices:
            x_indices = range(0,sh[2])
        else:
            x_indices = range(min(x_indices),max(x_indices)+1)
        
        if not y_indices:
            y_indices = range(0,sh[1])
        else:
            y_indices = range(min(y_indices),max(y_indices)+1)
        
        data = self.rootgrp.variables[self.var][time_indices,y_indices,x_indices]
        
        return data
    
    def get_queryset(self,geom_list,mask=None,aggregate=False,time_indices=[],x_indices=[],y_indices=[]):
        """
        Returns a populated GeoQuerySet object.
        
        geom_list -- list of GEOSGeometry objects to associate with each
            requested NC index location.
        mask -- a NumPy array with dimension equal to the extracted NC block.
        aggregate -- set to True and the mean of the NC values will be
            associated with a single geometry.
        time_indices -- list of index locations for the time slices of interest.
        x & y_indices -- must have same dimension. coordinate pairs to
            pull from the NC.
        """
        
        ## code expects the geometry to be included in a list, but for convenience
        ##  in the aggregation case a single geometry may be passed
        if type(geom_list) not in [list,tuple]:
            geom_list = [geom_list]
        
        ## if indices are passed for the x&y dimensions, they must be equal to
        ##  return data correctly from a netcdf.
        if len(x_indices) != len(y_indices):
            raise ValueError('Row and column index counts must be equal.')
        
        ## do some checking for geometry counts depending on the request type
        if aggregate is True:
            if len(geom_list) > 1:
                raise ValueError('When aggregating, only a single geometry is permitted.')
        elif aggregate is False:
            if len(x_indices) > 0 and len(geom_list) != len(x_indices):
                raise ValueError('The number of geometries and the number of requested indices must be equal.')

        ## return the netcdf data as a multi-dimensional numpy array
        data = self.get_numpy_data(time_indices,x_indices,y_indices)
        
        ## once more check in the case of all data being returned unaggregated
        if (aggregate is False) and not x_indices and (len(geom_list) < data.shape[1]*data.shape[2]):
            msg = ('The number of geometries and the number of requested indices must be equal. '
                   '{0} geometry(s) passed with {1} geometry(s) required.'.format(len(geom_list),data.shape[1]*data.shape[2]))
            raise ValueError(msg)
        
        ## if a weighting mask is not passed, use the identity masking function
        if mask == None: mask = np.ones((data.shape[1],data.shape[2]))
        
        ## initialize a custom GeoQuerySet 
        qs = GeoQuerySet(self._klass)
        qs._result_cache = []
        
        ## POPULATE QUERYSET ---------------------------------------------------
        
        attrs = [] ## will hold "rows" in the queryset
        ## loop for each time slice
        for ii in xrange(data.shape[0]):
            ## retrieve the corresponding time stamp
            timestamp = self._timevec[ii]
            ## apply the mask and remove the singleton (time) dimension
            slice = np.squeeze(data[ii,:,:])*mask
            
            ## INDEX INTO NUMPY ARRAY ------------------------------------------
            
            ## in the aggretation case, we summarize a time layer and link it with
            ##  the lone geometry
            if aggregate:
                attrs.append({'timestamp':timestamp,'geom':geom_list[0],self.var:slice.mean()})
            ## otherwise, we create the rows differently if a subset or the entire
            ##  dataset was requested.
            else:
                ## the case of requesting specific indices
                if x_indices:
                    for jj in xrange(len(x_indices)):
                        ## offsets are required as the indices were shifted due
                        ##  to subsetting when querying the netcdf
                        x_offset = min(x_indices)
                        y_offset = min(y_indices)
                        val = self._value_(slice,y_indices[jj]-y_offset,x_indices[x_offset]-x_offset)
                        attrs.append({'timestamp':timestamp,'geom':geom_list[jj],self.var:val})
                ## case that all data is being returned
                else:
                    ctr = 0
                    for row,col in itertools.product(xrange(slice.shape[0]),xrange(slice.shape[1])):
                        attrs.append({'timestamp':timestamp,'geom':geom_list[ctr],self.var:slice[row,col]})
                        ctr += 1
            
        ## add data to queryset
        for ii,attr in enumerate(attrs,start=1):
            attr.update({'id':ii}) ## need to add the unique identifier
            qs._result_cache.append(self._klass(**attr))
        
        return qs
    
    def _value_(self,array,row,col):
        """
        Correctly formats a value returned from a NumPy array.
        
        array -- NumPy array.
        row & col -- index locations.
        """
        
        val = array[row,col]
        fmt = self.get_pyfrmt(val)
        val = fmt(val)
        if val == self.nodata:
            return None
        else:
            return val
        
    def _masked_(self,val):
        return None