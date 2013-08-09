import groups
from base import OcgFunction, OcgCvArgFunction, OcgArgFunction
import numpy as np
from ocgis.util.helpers import iter_array
from ocgis.calc.base import KeyedFunctionOutput
from ocgis.constants import np_int
from ocgis.exc import DefinitionValidationError
import datetime
import os
import csv


class FrequencyPercentile(OcgArgFunction):
    name = 'freq_perc'
    nargs = 2
    Group = groups.Percentiles
    dtype = np.float32
    description = 'The percentile value along the time axis. See: http://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.percentile.html.'
    
    def _calculate_(self,values,percentile=None):
        '''
        :param percentile: Percentile to compute.
        :type percentile: float on the interval [0,100]
        '''
        ret = np.percentile(values,percentile,axis=0)
        return(ret)


class SampleSize(OcgFunction):
    '''
    .. note:: Automatically added by OpenClimateGIS. This should generally not be invoked manually.
    '''
    name = 'n'
    description = 'Statistical sample size.'
    Group = groups.BasicStatistics
    dtype = np.int32
    
    def _calculate_(self,values):
        ret = np.empty(values.shape[-2:],dtype=int)
        ret[:] = values.shape[0]
        ret = np.ma.array(ret,mask=values.mask[0,0,:])
        return(ret)
    
    def _aggregate_spatial_(self,values,weights):
        return(np.ma.sum(values))


class Median(OcgFunction):
    description = 'Median value for the series.'
    Group = groups.BasicStatistics
    dtype = np.float32
    
    def _calculate_(self,values):
        return(np.ma.median(values,axis=0))
    
    
class Mean(OcgFunction):
    description = 'Mean value for the series.'
    Group = groups.BasicStatistics
    dtype = np.float32
    
    def _calculate_(self,values):
        return(np.ma.mean(values,axis=0))
    
    
class Max(OcgFunction):
    description = 'Max value for the series.'
    Group = groups.BasicStatistics
    dtype = np.float32
    
    def _calculate_(self,values):
        return(np.ma.max(values,axis=0))
    
    
class Min(OcgFunction):
    description = 'Min value for the series.'
    Group = groups.BasicStatistics
    dtype = np.float32
    
    def _calculate_(self,values):
        return(np.ma.min(values,axis=0))
    
    
class StandardDeviation(OcgFunction):
    description = 'Standard deviation for the series.'
    Group = groups.BasicStatistics
    dtype = np.float32
    name = 'std'
    
    def _calculate_(self,values):
        return(np.ma.std(values,axis=0))


class Duration(OcgArgFunction):
    name = 'duration'
    nargs = 3
    Group = groups.Thresholds
    dtype = np.float32
    description = 'Summarizes consecutive occurrences in a sequence where the logical operation returns TRUE. The summary operation is applied to the sequences within a temporal aggregation.'
    
    def _calculate_(self,values,threshold=None,operation=None,summary='mean'):
        '''
        :param threshold: The threshold value to use for the logical operation.
        :type threshold: float
        :param operation: The logical operation. One of 'gt','gte','lt', or 'lte'.
        :type operation: str
        :param summary: The summary operation to apply the durations. One of 'mean','median','std','max', or 'min'.
        :type summary: str
        '''
        ## storage array for counts
        shp_out = list(values.shape)
        shp_out[0] = 1
        store = np.zeros(shp_out,dtype=self.dtype).flatten()
        ## get the summary operation from the numpy library
        summary_operation = getattr(np,summary)

        ## find longest sequence for each geometry across time dimension
        for ii,fill in enumerate(self._iter_consecutive_(values,threshold,operation)):
            ## case of only a singular occurrence
            if len(fill) > 1:
                fill = summary_operation(fill)
            else:
                try:
                    fill = fill[0]
                ## value is likely masked
                except IndexError:
                    fill = 0
            store[ii] = fill
        
        store.resize(shp_out)
        return(store)
    
    def _iter_consecutive_(self,values,threshold,operation):
        ## time index reference
        ref = np.arange(0,values.shape[0])
        ## perform requested logical operation
        if operation == 'gt':
            arr = values > threshold
        elif operation == 'lt':
            arr = values < threshold
        elif operation == 'gte':
            arr = values >= threshold
        elif operation == 'lte':
            arr = values <= threshold

        ## find longest sequence for each geometry across time dimension
        for zidx,rowidx,colidx in iter_array(values[0,:,:,:],use_mask=False):
            vec = arr[:,zidx,rowidx,colidx]
            ## check first if there is a longer series than 1
            if np.any(np.diff(ref[vec]) == 1):
                split_idx = ref[np.diff(vec)] + 1
                splits = np.array_split(vec,split_idx)
                fill = [a.sum() for a in splits if np.all(a)]
            ## case of only a singular occurrence
            elif np.any(vec):
                fill = [1]
            ## case for no occurrence
            else:
                fill = [0]
            
            yield(fill)
    
    @classmethod 
    def validate(cls,ops):
        if 'year' not in ops.calc_grouping:
            msg = 'Calculation grouping must include "year" for duration calculations.'
            raise(DefinitionValidationError('calc',msg))
    
    
class FrequencyDuration(KeyedFunctionOutput,Duration):
    name = 'freq_duration'
    description = 'Count the frequency of spell durations within the temporal aggregation.'
    nargs = 2
    dtype = object
    output_keys = ['duration','count']
    
    def _calculate_(self,values,threshold=None,operation=None):
        '''
        :param threshold: The threshold value to use for the logical operation.
        :type threshold: float
        :param operation: The logical operation. One of 'gt','gte','lt', or 'lte'.
        :type operation: str
        '''
        shp_out = list(values.shape)
        shp_out[0] = 1
        store = np.zeros(shp_out,dtype=self.dtype).flatten()
        for ii,duration in enumerate(self._iter_consecutive_(values,threshold,operation)):
            summary = self._get_summary_(duration)
            store[ii] = summary
        store.resize(shp_out)
        return(store)
        
    def _get_summary_(self,duration):
        set_duration = set(duration)
        ret = np.empty(len(set_duration),dtype=[('duration',np_int),('count',np_int)])
        for ii,sd in enumerate(set_duration):
            idx = np.array(duration) == sd
            count = idx.sum()
            ret[ii]['duration'] = sd
            ret[ii]['count'] = count
        return(ret)
    
    @classmethod
    def validate(cls,ops):
        KeyedFunctionOutput.validate(ops)
        Duration.validate(ops)
        

class QEDDynamicPercentileThreshold(OcgArgFunction):
    name = 'qed_dynamic_percentile_threshold'
    nargs = 3
    Group = groups.Thresholds
    dtype = np.int32
    description = 'Compares to a dynamic base dataset of daily thresholds. Only relevant for daily Maurer spatially coincident with QED City Centroids or North Carolina. Only works for "standard" calendars.'
    
    def __init__(self,*args,**kwds):
        super(self.__class__,self).__init__(*args,**kwds)
        self.__map_day_index = None
    
    def _calculate_(self,values,percentile=None,operation=None):
        ## first map the dates to dynamic percentiles index days
        from ocgis import env
        day_idx = self._get_day_index_(self.dataset.temporal.value[self._curr_group])
        dy_day_idx = map(self._get_dynamic_index_,day_idx.flat)
        gp = self._get_geometries_with_percentiles_(env.ops.dataset[0].variable,env.ops.geom.key,env.DIR_BIN,percentile)
        ## get threshold for each geometry
        ## special case for north carolina counties
        if env.ops.geom.key == 'us_counties':
            select_ugid = 39
        else:
            select_ugid = self.dataset.spatial._ugid
        ugid_ref = gp[select_ugid]
        compare = np.empty_like(values,dtype=float)
        for ii,jj,kk,ll in iter_array(values):
            ## get the geometry id
            gid = self.dataset.spatial.vector.uid[kk,ll]
            gid_ref = ugid_ref[gid]
            percentile_static = gid_ref[dy_day_idx[ii]]
            compare[ii,jj,kk,ll] = percentile_static
            
        ## perform requested logical operation
        if operation == 'gt':
            idx = values > compare
        elif operation == 'lt':
            idx = values < compare
        elif operation == 'gte':
            idx = values >= compare
        elif operation == 'lte':
            idx = values <= compare
        else:
            raise(NotImplementedError('The operation "{0}" was not recognized.'.format(operation)))
        
        ret = np.ma.sum(idx,axis=0)
        return(ret)
        
    @property
    def _map_day_index(self):
        if self.__map_day_index is None:
            self.__map_day_index = {key:self._get_day_index_reference_(year) for key,year in zip(['leap','noleap'],[1996,1995])}
        return(self.__map_day_index)
    
    def _get_day_index_reference_(self,year):
        store = []
        delta = datetime.timedelta(days=1)
        start = datetime.datetime(year,1,1)
        ctr = 1
        while start <= datetime.datetime(year,12,31):
            store.append([start.month,start.day,ctr])
            ctr += 1
            start += delta
        fill = np.empty(len(store),dtype=[('month',int),('day',int),('index',int)])
        for ii,row in enumerate(store):
            fill[ii] = tuple(row)
        return(fill)
    
    def _get_day_index_(self,dates):
        ## make date index
        fill_day_idx = np.empty(len(dates),dtype=[('index',int),('is_leap',bool)])
        ## make leap year designation
        is_leap_year = np.array([self._get_is_leap_year_(dt.year) for dt in dates])
        ## reference the day index
        map_day_index = self._map_day_index
        ## calculate index for each date
        for ii,date in enumerate(dates.flat):
            if is_leap_year[ii]:
                key = 'leap'
            else:
                key = 'noleap'
            ref_map_day_index = map_day_index[key]
            idx1 = ref_map_day_index['month'] == date.month
            idx2 = ref_map_day_index['day'] == date.day
            idx = np.logical_and(idx1,idx2)
            fill_day_idx[ii] = (ref_map_day_index[idx]['index'][0],is_leap_year[ii])
        return(fill_day_idx)
    
    def _get_dynamic_index_(self,k):
        ref_idx = k['index']
        if k['is_leap']:
            if ref_idx in (1,2):
                ret = 1
            elif ref_idx >= 3 and ref_idx <= 59:
                ret = ref_idx - 2
            elif ref_idx >= 60 and ref_idx <= 364:
                ret = ref_idx - 3
            elif ref_idx in (365,366):
                ret = 361
            else:
                raise(NotImplementedError)
        else:
            if ref_idx in (1,2):
                ret = 1
            elif ref_idx >= 364:
                ret = 361
            else:
                ret = ref_idx - 2
        return(ret)
        
    def _get_is_leap_year_(self,year):
        try:
            datetime.datetime(year,2,29)
            is_leap = True
        except ValueError:
            is_leap = False
        return(is_leap)
    
    def _get_geometries_with_percentiles_(self,variable,shp_key,bin_directory,percentile):
        '''
        :rtype: dict
        :returns: {ugid(int):gid(int),...:window(int),...:dynamic_percent(float)}
        '''
        def _get_or_create_dict_(key,dct):
            try:
                ret = dct[key]
            except KeyError:
                dct.update({key:{}})
                ret = dct[key]
            return(ret)
        
        variable_map = {'tasmax':'tmax',
                        'tasmin':'tmin'}
        variable = variable_map[variable]
        
        if shp_key in ('state_boundaries','us_counties'):
            shp_key = 'north_carolina'
        select_key = variable + '_' + shp_key
        data_map = {'tmax_qed_city_centroids':'DynPercTmax_19712000_maurer_citycentroids.csv',
                    'tmin_qed_city_centroids':'DynPercTmin_19712000_maurer_citycentroids.csv',
                    'tmax_north_carolina':'DynPercTmax_19712000_maurer_NCarolina.csv',
                    'tmin_north_carolina':'DynPercTmin_19712000_maurer_NCarolina.csv'}
        csv_path = os.path.join(bin_directory,data_map[select_key])
        percentile_key = 'q{0}{1}'.format(percentile,variable)
        
        store = {}
        with open(csv_path,'r') as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [field.strip().lower() for field in reader.fieldnames]
            for row in reader:
                ugid = int(row['ugid'].strip())
                gid = int(row['gid'].strip())
                window = int(row['5-day window number'].strip())
                dypercentile = float(row[percentile_key].strip())
                
                ref_ugid = _get_or_create_dict_(ugid,store)
                ref_gid = _get_or_create_dict_(gid,ref_ugid)
                ref_gid[window] = dypercentile
        
        return(store)


class Between(OcgArgFunction):
    nargs = 2
    description = 'Count of values falling within the limits lower and upper (inclusive).'
    Group = groups.Thresholds
    dtype = np.int32
    
    def _calculate_(self,values,lower=None,upper=None):
        '''
        :param lower: The lower value of the range.
        :type lower: float
        :param upper: The upper value of the range.
        :type upper: float
        '''
        idx = (values >= float(lower))*(values <= float(upper))
        return(np.ma.sum(idx,axis=0))
    
    
class Threshold(OcgArgFunction):
    nargs = 2
    description = 'Count of values where the logical operation returns TRUE.'
    Group = groups.Thresholds
    dtype = np.int32
    
    def _calculate_(self,values,threshold=None,operation=None):
        '''
        :param threshold: The threshold value to use for the logical operation.
        :type threshold: float
        :param operation: The logical operation. One of 'gt','gte','lt', or 'lte'.
        :type operation: str
        '''
        threshold = float(threshold)
        
        ## perform requested logical operation
        if operation == 'gt':
            idx = values > threshold
        elif operation == 'lt':
            idx = values < threshold
        elif operation == 'gte':
            idx = values >= threshold
        elif operation == 'lte':
            idx = values <= threshold
        else:
            raise(NotImplementedError('The operation "{0}" was not recognized.'.format(operation)))
        
        ret = np.ma.sum(idx,axis=0)
        return(ret)
        
    def _aggregate_spatial_(self,values,weights):
        return(np.ma.sum(values))
    

class HeatIndex(OcgCvArgFunction):
    description = 'Heat Index following: http://en.wikipedia.org/wiki/Heat_index. If temperature is < 80F or relative humidity is < 40%, the value is masked during calculation. Output units are Fahrenheit.'
    Group = groups.MultivariateStatistics
    dtype = np.float32
    nargs = 2
    keys = ['tas','rhs']
    name = 'heat_index'
    
    def _calculate_(self,tas=None,rhs=None,units=None):
        if units == 'k':
            tas = 1.8*(tas - 273.15) + 32
        else:
            raise(NotImplementedError)
        
        c1 = -42.379
        c2 = 2.04901523
        c3 = 10.14333127
        c4 = -0.22475541
        c5 = -6.83783e-3
        c6 = -5.481717e-2
        c7 = 1.22874e-3
        c8 = 8.5282e-4
        c9 = -1.99e-6
        
        idx = tas < 80
        tas.mask = np.logical_or(idx,tas.mask)
        idx = rhs < 40
        rhs.mask = np.logical_or(idx,rhs.mask)
        
        tas_sq = np.square(tas)
        rhs_sq = np.square(rhs)
        
        hi = c1 + c2*tas + c3*rhs + c4*tas*rhs + c5*tas_sq + c6*rhs_sq + \
             c7*tas_sq*rhs + c8*tas*rhs_sq + c9*tas_sq*rhs_sq
        
        return(hi)
