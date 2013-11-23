from abc import ABCMeta, abstractproperty
from ocgis.test.base import TestBase
from ocgis.interface.shp import ShpDataset
import datetime
import ocgis
import os


class AbstractParameter(object):
    __metaclass__ = ABCMeta
    
    def __iter__(self):
        name = self.name
        for value in self.values:
            yield({name:value})
    
    @abstractproperty
    def name(self): str
    
    @abstractproperty
    def values(self): ['<varying>']
    
    
class AbstractBooleanParameter(AbstractParameter):
    __metaclass__ = ABCMeta
    values = [True,False]
    
    
class Abstraction(AbstractParameter):
    name = 'abstraction'
    values = ['point','polygon']


class Aggregate(AbstractBooleanParameter):
    name = 'aggregate'
    
    
class AggregateSelection(AbstractBooleanParameter):
    name = 'agg_selection'
    
    
class Calc(AbstractParameter):
    name = 'calc'
    values = [
              None,
              [{'func':'mean','name':'mean'}]
              ]
    
class CalcGrouping(AbstractParameter):
    name = 'calc_grouping'
    
    @property
    def values(self):
        ret = [None,
               ['month'],
               ['year'],
               ['month','year']
               ]
        return(ret)
    
class CalcRaw(AbstractBooleanParameter):
    name = 'calc_raw'

        
class Dataset(AbstractParameter):
    name = 'dataset'
    
    @property
    def values(self):
        
        path = '/usr/local/climate_data/maurer/2010-concatenated'
        filenames = ['Maurer02new_OBS_pr_daily.1971-2000.nc',
                     'Maurer02new_OBS_tasmax_daily.1971-2000.nc',
                     'Maurer02new_OBS_tasmin_daily.1971-2000.nc',
                     'Maurer02new_OBS_tas_daily.1971-2000.nc']
        time_range = [None,[datetime.datetime(2001,3,1),datetime.datetime(2001,3,31,23)]]
        time_region = [None,{'month':[6,7],'year':[2006,2007]}]
        
        for filename in filenames:
            variable = filename.split('_')[2]
            for time in [None,time_range,time_region]:
                if time is None:
                    trange = None
                    tregion = None
                elif isinstance(time,list):
                    trange = time
                    tregion = None
                else:
                    trange = None
                    tregion = time
                rd = ocgis.RequestDataset(os.path.join('path'),variable,time_range=trange,time_region=tregion)
                yield(rd)
            
#        values = []
#        
#        tdata = TestBase.get_tdata()
#        
#        rd = tdata.get_rd('cancm4_tasmax_2001')
#        
#        rd_time_range = tdata.get_rd('cancm4_tasmax_2001')
#        rd_time_range.time_range = [datetime.datetime(2001,3,1),datetime.datetime(2001,3,31,23)]
#        
#        rd_time_region = tdata.get_rd('cancm4_tasmax_2001')
#        rd_time_region.time_region = {'month':[6,7],'year':[2006,2007]}
#        
#        return([rd,rd_time_range,rd_time_region])
    
    
class Geometry(AbstractParameter):
    name = 'geom'
    
    def __iter__(self):
        for value in self.values:
            yld = {self.name:value[0],'select_ugid':value[1]}
            yield(yld)
    
    @property
    def values(self):
        california = ['state_boundaries',[25]]
        states = ['state_boundaries',[14,16,32]]
        city_centroids = ['gg_city_centroids',[622,3051]]
        return([california,states,city_centroids])


class OutputFormat(AbstractParameter):
    name = 'output_format'
    values = [
#              'numpy',
#              'csv',
              'shp',
              'csv+',
              'nc'
              ]
    
    
#class Snippet(AbstractBooleanParameter):
#    name = 'snippet'

    
class SpatialOperation(AbstractParameter):
    name = 'spatial_operation'
    values = [
              'clip',
              'intersects',
#              None
              ]
    
## environmental parameters ####################################################

#class AbstractEnvironmentalParameter(AbstractParameter):
#    __metaclass__ = ABCMeta
    
    
#class Verbose(AbstractEnvironmentalParameter):
#    name = 'VERBOSE'
#    values = [True,False]
#    
#    
#class FileLogging(AbstractEnvironmentalParameter):
#    name = 'ENABLE_FILE_LOGGING'
#    values = [True,False]
    
    
#class ReferenceProjection(AbstractEnvironmentalParameter):
#    name = 'WRITE_TO_REFERENCE_PROJECTION'
#    values = [True,False]