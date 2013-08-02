from abc import ABCMeta, abstractproperty
import datetime
import ocgis
import os


class CounterLimit(Exception):
    pass


class ConditionalNotMet(Exception):
    pass


class Counter(object):
    
    def __init__(self,predicate,limit=1):
        self.predicate = predicate
        self.counter = 0
        self.limit = limit
        
    def check(self,key,kwargs):
        value = kwargs[key]
        if self.predicate(value):
            self.counter += 1
            if self.counter > self.limit:
                raise(CounterLimit(key,value,self.limit))


class AbstractParameter(object):
    __metaclass__ = ABCMeta
    counters = []
    
    def __iter__(self):
        name = self.name
        for value in self.values:
            yield({name:value})
    
    @abstractproperty
    def name(self): str
    
    @abstractproperty
    def values(self): ['<varying>']
    
    @classmethod
    def get_conditional(cls,kwargs,name):
        func = getattr(cls,name)
        ret = func(kwargs)
        if ret is None:
            raise(ConditionalNotMet)
        return(ret)
    
    
class AbstractBooleanParameter(AbstractParameter):
    __metaclass__ = ABCMeta
    values = [True,False]


################################################################################


class Aggregate(AbstractBooleanParameter):
    name = 'aggregate'

    
class Calc(AbstractParameter):
    name = 'calc'
    values = [
              None,
              [{'func':'mean','name':'mean'}],
              '_conditional_tasmax_duration_',
              '_conditional_tasmax_frequency_duration_',
              ]
    
    @classmethod
    def _conditional_tasmax_duration_(cls,kwargs):
        ret = None
        rd = kwargs['dataset']
        if rd.variable == 'tasmax':
            ret = [{'func':'duration','name':'duration','kwds':{'threshold':21.0,'operation':'gte','summary':'mean'}}]
        return(ret)
    
    @classmethod
    def _conditional_tasmax_frequency_duration_(cls,kwargs):
        ret = None
        rd = kwargs['dataset']
        if rd.variable == 'tasmax':
            ret = [{'func':'freq_duration','name':'freq_duration','kwds':{'threshold':21.0,'operation':'gte'}}]
        return(ret)
    
class CalcGrouping(AbstractParameter):
    name = 'calc_grouping'
    
    @property
    def values(self):
        ret = [
               ['month'],
               ['year'],
               ['month','year']
               ]
        return(ret)
    
class CalcRaw(AbstractBooleanParameter):
    name = 'calc_raw'


def _counter_dataset_(x):
    if x.variable == 'tasmax':
        ret = False
    else:
        ret = True
    return(ret)


class Dataset(AbstractParameter):
    name = 'dataset'
    counters = [Counter(_counter_dataset_,limit=3)]
    
    @property
    def values(self):
        
        path = '/usr/local/climate_data/maurer/2010-concatenated'
        filenames = ['Maurer02new_OBS_pr_daily.1971-2000.nc',
                     'Maurer02new_OBS_tasmax_daily.1971-2000.nc',
                     'Maurer02new_OBS_tasmin_daily.1971-2000.nc',
                     'Maurer02new_OBS_tas_daily.1971-2000.nc']
        time_range = [datetime.datetime(1990,1,1),datetime.datetime(1990,12,31)]
        time_region = {'month':[6,7],'year':[1989,1990]}
        
        for filename in filenames:
            variable = filename.split('_')[2]
            for time in [time_range,time_region]:
                if time is None:
                    trange = None
                    tregion = None
                elif isinstance(time,list):
                    trange = time
                    tregion = None
                else:
                    trange = None
                    tregion = time
                rd = ocgis.RequestDataset(os.path.join(path,filename),variable,time_range=trange,time_region=tregion)
                yield(rd)
    

def _counter_geometry_(x):
    if x == 'state_boundaries':
        ret = False
    else:
        ret = True
    return(ret)


class Geometry(AbstractParameter):
    name = 'geom'
    counters = [Counter(_counter_geometry_,limit=3)]
    
    def __iter__(self):
        for value in self.values:
            yld = {self.name:value[0],'select_ugid':value[1]}
            yield(yld)
    
    @property
    def values(self):
        states = ['state_boundaries',[14,16,32]]
        city_centroids = ['gg_city_centroids',None]
        us_counties = ['us_counties',[1337,1340]]
        huc8 = ['WBDHU8_June2013',[472]]
        return([states,city_centroids,us_counties,huc8])


class OutputFormat(AbstractParameter):
    name = 'output_format'
    values = ['csv+','nc','shp','csv']
    counters = [Counter(lambda x: x in ('shp','csv'),limit=1)]

    
class SpatialOperation(AbstractParameter):
    name = 'spatial_operation'
    values = ['clip','intersects']