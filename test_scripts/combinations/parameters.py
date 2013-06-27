from abc import ABCMeta, abstractproperty
from ocgis.test.base import TestBase
from ocgis.interface.shp import ShpDataset
import datetime


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
        tdata = TestBase.get_tdata()
        
        rd = tdata.get_rd('cancm4_tasmax_2001')
        
        rd_time_range = tdata.get_rd('cancm4_tasmax_2001')
        rd_time_range.time_range = [datetime.datetime(2001,3,1),datetime.datetime(2001,3,31,23)]
        
        rd_time_region = tdata.get_rd('cancm4_tasmax_2001')
        rd_time_region.time_region = {'month':[6,7],'year':[2006,2007]}
        
        return([rd,rd_time_range,rd_time_region])
    
    
class Geometry(AbstractParameter):
    name = 'geom'
    
    @property
    def values(self):
        california = ShpDataset('state_boundaries',select_ugid=[25])
        ne_ia_co = ShpDataset('state_boundaries',select_ugid=[14,16,32])
        split_counties = ShpDataset('us_counties',select_ugid=[622,3051])
        return([california,ne_ia_co,split_counties,None])


class OutputFormat(AbstractParameter):
    name = 'output_format'
    values = ['numpy','csv','shp','csv+','nc']
    
    
class Snippet(AbstractBooleanParameter):
    name = 'snippet'

    
class SpatialOperation(AbstractParameter):
    name = 'spatial_operation'
    values = ['clip','intersects',None]
    
## environmental parameters ####################################################

class AbstractEnvironmentalParameter(AbstractParameter):
    __metaclass__ = ABCMeta
    
    
#class Verbose(AbstractEnvironmentalParameter):
#    name = 'VERBOSE'
#    values = [True,False]
#    
#    
#class FileLogging(AbstractEnvironmentalParameter):
#    name = 'ENABLE_FILE_LOGGING'
#    values = [True,False]
    
    
class ReferenceProjection(AbstractEnvironmentalParameter):
    name = 'WRITE_TO_REFERENCE_PROJECTION'
    values = [True,False]