import numpy as np
import netCDF4 as n
import itertools
from django.contrib.gis.db.models.query import GeoValuesListQuerySet


class NetCdfAccessor(object):
    """
    Access netCDF4 Dataset objects for profiling and conversion.
    
    rootgrp -- base netCDF4.Dataset object.
    var -- name of the variable to extract. ie. 'Tavg'
    **kwds -- customizable NC attribute names. see object's __init__ method for
        explanation.
    """

    
    def __init__(self,rootgrp,var,**kwds):
        self.rootgrp = rootgrp
        self.var = var

        ## KEYWORD EXTRACTION --------------------------------------------------

        self.time = kwds.get('time') or 'time'
        self.srid = kwds.get('srid') or 4326
        self.nodata_var = kwds.get('nodata') or 'missing_value'
        
        ## ---------------------------------------------------------------------
        
        ## pull the nodata value
        try:
            self.nodata = getattr(rootgrp.variables[self.var],self.nodata_var)
        except AttributeError:
            self.nodata = None
        ## construct the time vector
        times = self.rootgrp.variables[self.time]
        self._timevec = n.num2date(times[:],units=times.units,calendar=times.calendar)
                        
    def get_pyfrmt(self,val):
        """
        Pulls the correct Python format from the type mapping.
        """  
        
        if isinstance(val,np.ma.core.MaskedConstant):
            ret = self._masked_
        else:
            ret = float
        return ret
        
    def get_numpy_data(self,time_indices=[],x_indices=[],y_indices=[]):
        """
        Returns multi-dimensional NumPy array extracted from a NC.
        """

        sh = self.rootgrp.variables[self.var].shape
        
        ## If indices are not passed, this method defaults to returning all
        ##  data.
        
        if not time_indices:
            time_indices = range(0,sh[0])
        elif len(time_indices) == 1:
            time_indices = time_indices[0]
        else:
            time_indices = range(min(time_indices,max(time_indices)+1))
#            time_indices = range(min(time_indices))
        
        if not x_indices:
            x_indices = range(0,sh[2])
        else:
            x_indices = range(min(x_indices),max(x_indices)+1)
        
        if not y_indices:
            y_indices = range(0,sh[1])
        else:
            y_indices = range(min(y_indices),max(y_indices)+1)
        
#        import ipdb;ipdb.set_trace()       
        data = self.rootgrp.variables[self.var][time_indices,y_indices,x_indices]        
        return data
    
    def get_dict(self,geom_list,weights=None,aggregate=False,time_indices=[],col=[],row=[]):
        """
        Returns a dict list containing target attributes.
        
        geom_list -- list of GEOSGeometry objects to associate with each
            requested NC index location. a single geometry is acceptable in the
            case of an aggregate query.
        weights -- list with dimension equal to col & row index lists. the weight
            sums should sum to one.
        aggregate -- set to True and the mean of the NC values will be
            associated with a single geometry.
        time_indices -- list of index locations for the time slices of interest.
        row & col -- must have same dimension. coordinate pairs to
            pull from the NC.
        """
        
        ## code expects at least one geometry
        if len(geom_list) == 0:
            raise ValueError('At least one geometry is expected.')
        
        ## code expects the geometry to be included in a list, but for convenience
        ##  in the aggregation case a single geometry may be passed
        if type(geom_list) not in [list,tuple,GeoValuesListQuerySet]:
            geom_list = [geom_list]
        
        ## if indices are passed for the x&y dimensions, they must be equal to
        ##  return data correctly from a netcdf.
        if len(col) != len(row):
            raise ValueError('Row and column index counts must be equal.')
        
        ## do some checking for geometry counts depending on the request type
        if aggregate is True:
            if len(geom_list) > 1:
                raise ValueError('When aggregating, only a single geometry is permitted.')
        elif aggregate is False:
            if len(col) > 0 and len(geom_list) != len(col):
                raise ValueError('The number of geometries and the number of requested indices must be equal.')

        ## return the netcdf data as a multi-dimensional numpy array
#        import ipdb;ipdb.set_trace()
        data = self.get_numpy_data(time_indices,col,row)
        
        ## once more check in the case of all data being returned unaggregated
        if (aggregate is False) and not col and (len(geom_list) < data.shape[1]*data.shape[2]):
            msg = ('The number of geometries and the number of requested indices must be equal. '
                   '{0} geometry(s) passed with {1} geometry(s) required.'.format(len(geom_list),data.shape[1]*data.shape[2]))
            raise ValueError(msg)
        
        ## extract array shape depending on the number of time indices passed. 
        ## if only one time slice is requested, the array is two-dimensional. the
        ## variable is used to properly size the weighting mask.
        if len(time_indices) == 1:
            sh = (data.shape[0],data.shape[1])
        else:
            sh = (data.shape[1],data.shape[2])
            
        ## if a weighting weights is not passed, use the identity masking
        ## function
        if weights == None:
            mask = np.ones(sh)
        else:
            mask = np.array(weights).reshape(*sh)
#        import ipdb;ipdb.set_trace()
        
        ## POPULATE QUERYSET ---------------------------------------------------
        
        attrs = [] ## will hold "rows" in the queryset
        ids = self._gen_id_(start=1)
        ## loop for each time slice, checking that there is more than one
        if len(data.shape) == 2:
            itrval = 1
        else:
            itrval = data.shape[0]
        for ii in xrange(itrval):
            ## retrieve the corresponding time stamp
            timestamp = self._timevec[ii]
            ## apply the weights and remove the singleton (time) dimension
            if len(data.shape) == 2:
                slice = data[:,:]*mask
            else:
                slice = np.squeeze(data[ii,:,:])*mask
            
            ## INDEX INTO NUMPY ARRAY ------------------------------------------
            
            ## in the aggretation case, we summarize a time layer and link it with
            ##  the lone geometry
            if aggregate:
#                import ipdb;ipdb.set_trace()
                attrs.append({'id':ids.next(),'timestamp':timestamp,'geom':geom_list[0],self.var:float(slice.sum())})
            ## otherwise, we create the rows differently if a subset or the entire
            ##  dataset was requested.
            else:
                ## the case of requesting specific indices
                if len(col) > 0:
                    for jj in xrange(len(col)):
                        ## offsets are required as the indices were shifted due
                        ##  to subsetting when querying the netcdf
                        x_offset = min(col)
                        y_offset = min(row)
                        val = self._value_(slice,row[jj]-y_offset,col[jj]-x_offset)
                        attrs.append({'id':ids.next(),'timestamp':timestamp,'geom':geom_list[jj],self.var:val})
                ## case that all data is being returned
                else:
                    ctr = 0
                    for r,c in itertools.product(xrange(slice.shape[0]),xrange(slice.shape[1])):
                        attrs.append({'id':ids.next(),'timestamp':timestamp,'geom':geom_list[ctr],self.var:self._value_(slice,r,c)})
                        ctr += 1
        
        return attrs
    
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
    
    def _gen_id_(self,start=1):
        adj = 0
        while True:
            try:
                yield start + adj
            finally:
                adj += 1


#class GeoQuerySetFactory(object):
#    """
#    Create GeoQuerySets from netCDF4 Dataset objects.
#    
#    rootgrp -- base netCDF4.Dataset object.
#    var -- name of the variable to extract. ie. 'Tavg'
#    **kwds -- customizable NC attribute names. see object's __init__ method for
#        explanation.
#    """
#    
##    _dtype_mapping = {type(np.dtype('float32')):[float,models.FloatField],
##                      type(np.dtype('float64')):[float,models.FloatField]}
#    
#    def __init__(self,rootgrp,var,**kwds):
#        self.rootgrp = rootgrp
#        self.var = var
#
#        ## KEYWORD EXTRACTION --------------------------------------------------
#
#        self.time = kwds.get('time') or 'time'
#        self.srid = kwds.get('srid') or 4326
#        self.app_label = kwds.get('app_label') or 'climatedata'
#        self.objects = kwds.get('objects') or GeoManager
#        self.nodata_var = kwds.get('nodata') or 'missing_value'
#        
#        ## ---------------------------------------------------------------------
#        
#        self._pyfrmt = None ## used to format values from NumPy data types
#        
#        ## pull the nodata value
#        try:
#            self.nodata = getattr(rootgrp.variables[self.var],self.nodata_var)
#        except AttributeError:
#            self.nodata = None
#        ## construct the time vector
#        times = self.rootgrp.variables[self.time]
#        self._timevec = n.num2date(times[:],units=times.units,calendar=times.calendar)
#        ## attributes of the Django Model
#        self._base_attrs = {'id':models.IntegerField(primary_key=True),
#                            'geom':models.MultiPolygonField(srid=self.srid),
#                            'timestamp':models.DateTimeField(),
##                            self.var:self._dtype_mapping[rootgrp.variables[self.var].dtype][1](),
#                            self.var:models.FloatField(),
#                            'Meta':ClassType('Meta',(),{'app_label':self.app_label}),
#                            'objects':self.objects(),}
#        ## initialize the class
#        self._klass = ClassType('NcModel',(models.Model,),self._base_attrs)
#                        
#    def get_pyfrmt(self,val):
#        """
#        Pulls the correct Python format from the type mapping.
#        """  
#        
#        if self._pyfrmt == None:
##            self._pyfrmt = float
##            try:
##                self._pyfrmt = self._dtype_mapping[type(val)][0]
##            ## numpy 'masked' values throws exception
##            except KeyError:
##                import ipdb;ipdb.set_trace()
#                ## map this to a NoneType returning function
#            if isinstance(val,np.ma.core.MaskedConstant):
#                self._pyfrmt = self._masked_
#            else:
#                self._pyfrmt = float
##                else:
##                    raise
#        return self._pyfrmt
#        
#    def get_numpy_data(self,time_indices=[],x_indices=[],y_indices=[]):
#        """
#        Returns multi-dimensional NumPy array extracted from a NC.
#        """
#        
#        sh = self.rootgrp.variables[self.var].shape
#        
#        ## If indices are not passed, this method defaults to returning all
#        ##  data.
#        
#        if not time_indices:
#            time_indices = range(0,sh[0])
#        
#        if not x_indices:
#            x_indices = range(0,sh[2])
#        else:
#            x_indices = range(min(x_indices),max(x_indices)+1)
#        
#        if not y_indices:
#            y_indices = range(0,sh[1])
#        else:
#            y_indices = range(min(y_indices),max(y_indices)+1)
#        
#        data = self.rootgrp.variables[self.var][time_indices,y_indices,x_indices]
#        
#        return data
#    
#    def get_queryset(self,geom_list,mask=None,aggregate=False,time_indices=[],x_indices=[],y_indices=[]):
#        """
#        Returns a populated GeoQuerySet object.
#        
#        geom_list -- list of GEOSGeometry objects to associate with each
#            requested NC index location.
#        mask -- a NumPy array with dimension equal to the extracted NC block.
#        aggregate -- set to True and the mean of the NC values will be
#            associated with a single geometry.
#        time_indices -- list of index locations for the time slices of interest.
#        x & y_indices -- must have same dimension. coordinate pairs to
#            pull from the NC.
#        """
#        
#        ## code expects the geometry to be included in a list, but for convenience
#        ##  in the aggregation case a single geometry may be passed
#        if type(geom_list) not in [list,tuple]:
#            geom_list = [geom_list]
#        
#        ## if indices are passed for the x&y dimensions, they must be equal to
#        ##  return data correctly from a netcdf.
#        if len(x_indices) != len(y_indices):
#            raise ValueError('Row and column index counts must be equal.')
#        
#        ## do some checking for geometry counts depending on the request type
#        if aggregate is True:
#            if len(geom_list) > 1:
#                raise ValueError('When aggregating, only a single geometry is permitted.')
#        elif aggregate is False:
#            if len(x_indices) > 0 and len(geom_list) != len(x_indices):
#                raise ValueError('The number of geometries and the number of requested indices must be equal.')
#
#        ## return the netcdf data as a multi-dimensional numpy array
#        data = self.get_numpy_data(time_indices,x_indices,y_indices)
#        
#        ## once more check in the case of all data being returned unaggregated
#        if (aggregate is False) and not x_indices and (len(geom_list) < data.shape[1]*data.shape[2]):
#            msg = ('The number of geometries and the number of requested indices must be equal. '
#                   '{0} geometry(s) passed with {1} geometry(s) required.'.format(len(geom_list),data.shape[1]*data.shape[2]))
#            raise ValueError(msg)
#        
#        ## if a weighting mask is not passed, use the identity masking function
#        if mask == None: mask = np.ones((data.shape[1],data.shape[2]))
#        
#        ## initialize a custom GeoQuerySet 
#        qs = GeoQuerySet(self._klass)
#        qs._result_cache = []
#        
#        ## POPULATE QUERYSET ---------------------------------------------------
#        
#        attrs = [] ## will hold "rows" in the queryset
#        ## loop for each time slice
#        for ii in xrange(data.shape[0]):
#            ## retrieve the corresponding time stamp
#            timestamp = self._timevec[ii]
#            ## apply the mask and remove the singleton (time) dimension
#            slice = np.squeeze(data[ii,:,:])*mask
#            
#            ## INDEX INTO NUMPY ARRAY ------------------------------------------
#            
#            ## in the aggretation case, we summarize a time layer and link it with
#            ##  the lone geometry
#            if aggregate:
#                attrs.append({'timestamp':timestamp,'geom':geom_list[0],self.var:slice.mean()})
#            ## otherwise, we create the rows differently if a subset or the entire
#            ##  dataset was requested.
#            else:
#                ## the case of requesting specific indices
#                if x_indices:
#                    for jj in xrange(len(x_indices)):
#                        ## offsets are required as the indices were shifted due
#                        ##  to subsetting when querying the netcdf
#                        x_offset = min(x_indices)
#                        y_offset = min(y_indices)
#                        val = self._value_(slice,y_indices[jj]-y_offset,x_indices[x_offset]-x_offset)
#                        attrs.append({'timestamp':timestamp,'geom':geom_list[jj],self.var:val})
#                ## case that all data is being returned
#                else:
#                    ctr = 0
#                    for row,col in itertools.product(xrange(slice.shape[0]),xrange(slice.shape[1])):
#                        attrs.append({'timestamp':timestamp,'geom':geom_list[ctr],self.var:slice[row,col]})
#                        ctr += 1
#            
#        ## add data to queryset
#        for ii,attr in enumerate(attrs,start=1):
#            attr.update({'id':ii}) ## need to add the unique identifier
#            qs._result_cache.append(self._klass(**attr))
#        
#        return qs
#    
#    def _value_(self,array,row,col):
#        """
#        Correctly formats a value returned from a NumPy array.
#        
#        array -- NumPy array.
#        row & col -- index locations.
#        """
#        
#        val = array[row,col]
#        fmt = self.get_pyfrmt(val)
#        val = fmt(val)
#        if val == self.nodata:
#            return None
#        else:
#            return val
#        
#    def _masked_(self,val):
#        return None