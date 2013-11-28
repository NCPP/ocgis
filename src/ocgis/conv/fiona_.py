from ocgis.conv.base import OcgConverter
import datetime
import numpy as np
from types import NoneType
import fiona
from collections import OrderedDict
from shapely.geometry.geo import mapping
from fiona.rfc3339 import FionaTimeType, FionaDateType
import abc
from ocgis.util.logging_ocgis import ocgis_lh

    
class FionaConverter(OcgConverter):
    __metaclass__ = abc.ABCMeta
    
    _add_ugeom = True
    _add_ugeom_nest = False
    _fiona_conversion = {np.int32:int,
                         np.int16:int,
                         np.int64:int,
                         np.float64:float,
                         np.float32:float,
                         np.float16:float,
                         datetime.datetime:FionaTimeType,
                         datetime.date:FionaDateType}
    _fiona_type_mapping = {datetime.date:'date',
                           datetime.datetime:'datetime',
                           np.int64:'int',
                           NoneType:None,
                           np.int32:'int',
                           np.float64:'float',
                           np.float32:'float',
                           np.float16:'float',
                           np.int16:'int',
                           str:'str'}
    
    def _finalize_(self,f):
        f['fiona_object'].close()
    
    def _build_(self,coll):
        fiona_conversion = {}
        
        def _get_field_type_(key,the_type):
            ret = None
            for k,v in fiona.FIELD_TYPES_MAP.iteritems():
                if the_type == v:
                    ret = k
                    break
            if ret is None:
                ret = self._fiona_type_mapping[the_type]
            if the_type in self._fiona_conversion:
                fiona_conversion.update({key.lower():self._fiona_conversion[the_type]})
            return(ret)
        
        ## pull the fiona schema properties together by mapping fiona types to
        ## the data types of the first row of the output data file
        archetype_field = coll._archetype_field
        fiona_crs = archetype_field.spatial.crs.value
        geom,arch_row = coll.get_iter_dict().next()
        fiona_properties = OrderedDict()
        for header in coll.headers:
            fiona_field_type = _get_field_type_(header,type(arch_row[header]))
            fiona_properties.update({header.upper():fiona_field_type})
            
        ## we always want to convert the value. if the data is masked, it comes
        ## through as a float when unmasked data is in fact a numpy data type.
        ## however, this should only occur if 'value' is in the output headers!
        if 'value' in coll.headers and 'value' not in fiona_conversion:
            value_dtype = archetype_field.variables.values()[0].value.dtype
            try:
                to_update = self._fiona_conversion[value_dtype]
            ## may have to do type comparisons
            except KeyError as e:
                to_update = None
                for k,v in self._fiona_conversion.iteritems():
                    if value_dtype == k:
                        to_update = v
                        break
                if to_update is None:
                    ocgis_lh(exc=e,logger='fiona_')
            fiona_conversion.update({'value':to_update})
        
        ## polygon geometry types are always converted to multipolygons to avoid
        ## later collections having multipolygon geometries.
        geometry_type = archetype_field.spatial.abstraction_geometry._geom_type
        if geometry_type == 'Polygon':
            geometry_type = 'MultiPolygon'
        
        fiona_schema = {'geometry':geometry_type,
                        'properties':fiona_properties}
        
        ## if there is no data for a header, it may be empty. in this case, the
        ## value comes through as none and it should be replaced with bool.
        for k,v in fiona_schema['properties'].iteritems():
            if v is None:
                fiona_schema['properties'][k] = 'str:1'

        fiona_object = fiona.open(self.path,'w',driver=self._driver,crs=fiona_crs,schema=fiona_schema)
        
        ret = {'fiona_object':fiona_object,'fiona_conversion':fiona_conversion}
        
        return(ret)
    
    def _write_coll_(self,f,coll):
        fiona_object = f['fiona_object']
        for geom,properties in coll.get_iter_dict(use_upper_keys=True,conversion_map=f['fiona_conversion']):
            to_write = {'geometry':mapping(geom),'properties':properties}
            fiona_object.write(to_write)


class ShpConverter(FionaConverter):
    _ext = 'shp'
    _driver = 'ESRI Shapefile'


class GeoJsonConverter(FionaConverter):
    _ext = 'json'
    _driver = 'GeoJSON'
