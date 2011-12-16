from element import *
import datetime
from netCDF4 import date2num


class Register(object):
    pass


class Conditional(object):
    pass


class RowBounds(Register,VariablePolyElement,SpatialTranslationalElement):
    _names = ['bounds_latitude',
              'bnds_latitude',
              'latitude_bounds',
              'lat_bnds']
    _dtype = 'f4'
    
    def calculate(self,grid):
        return(grid['ybnds'])
    
    
class ColumnBounds(Register,VariablePolyElement,SpatialTranslationalElement):
    _names = ['bounds_longitude',
              'bnds_longitude',
              'longitude_bounds',
              'lon_bnds']
    _dtype = 'f4'
    
    def calculate(self,grid):
        return(grid['xbnds'])
    
    
class Row(Register,VariablePolyElement,SpatialTranslationalElement):
    _names = ['latitude','lat']
    _dtype = 'f4'
    
    def calculate(self,grid):
        return(grid['y'])
    
    
class Column(Register,VariablePolyElement,SpatialTranslationalElement):
    _names = ['longitude','lon']
    _dtype = 'f4'
    
    def calculate(self,grid):
        return(grid['x'])
    
    
class GeospatialLonMax(Register,DatasetPolyElement,SpatialTranslationalElement):
    _names = ['geospatial_lon_max']
    
    def calculate(self,grid):
        return(float(max(grid['x'])))
    
    
class GeospatialLonMin(Register,DatasetPolyElement,SpatialTranslationalElement):
    _names = ['geospatial_lon_min']
    
    def calculate(self,grid):
        return(float(min(grid['x'])))
    
    
class GeospatialLatMax(Register,DatasetPolyElement,SpatialTranslationalElement):
    _names = ['geospatial_lat_max']
    
    def calculate(self,grid):
        return(float(max(grid['y'])))
    
    
class GeospatialLatMin(Register,DatasetPolyElement,SpatialTranslationalElement):
    _names = ['geospatial_lat_min']
    
    def calculate(self,grid):
        return(float(min(grid['y'])))
    
    
class TimeCoverageStart(Register,DatasetPolyElement,TemporalTranslationalElement):
    _names = ['time_coverage_start']
    
    def calculate(self,timevec):
        return(str(min(timevec)))
    
    
class TimeCoverageEnd(Register,DatasetPolyElement,TemporalTranslationalElement):
    _names = ['time_coverage_end']
    
    def calculate(self,timevec):
        return(str(max(timevec)))
    
    
class DateModified(Register,DatasetPolyElement,SimpleTranslationalElement):
    _names = ['date_modified']
    
    def calculate(self):
        return(str(datetime.datetime.now().date()))
    
    
class Time(Register,VariablePolyElement,TemporalTranslationalElement):
    _names = ['time']
    _dtype = 'i4'
    
    def calculate(self,timevec):
        time_units = TimeUnits(self,self.dataset)
        calendar = Calendar(self,self.dataset)
        return(date2num(timevec,units=time_units.value,calendar=calendar.value))
    
class Calendar(VariableAttrPolyElement):
    _names = ['calendar']
    
    
class TimeUnits(VariableAttrPolyElement):
    _names = ['units']
    
    
class Level(Register,VariablePolyElement):
    _names = ['level','levels','lvl','lvls']
    _dtype = 'i4'
    
    
class MissingValue(VariableAttrPolyElement):
    _names = ['missing_value']
    
    
class FileName(DatasetPolyElement,Register,TranslationalElement):
    _names = ['file_name']
    
    
class TimeDimension(TemporalDimensionElement,Register):
    _names = ['time']
    
class LatitudeDimension(SpatialDimensionElement,Register):
    _names = ['latitude','lat']
    
    def calculate(self,grid):
        return(len(grid['y']))
    
class LongitudeDimension(SpatialDimensionElement,Register):
    _names = ['longitude','lon']
    
    def calculate(self,grid):
        return(len(grid['x']))
    
class BoundsDimension(SpatialDimensionElement,Register):
    _names = ['bound','bounds','bnd','bnds']
    
    def calculate(self,grid):
        return(2)
    
class LevelDimension(LevelDimensionElement,Register):
    _names = ['level','levels','lvl','lvls']