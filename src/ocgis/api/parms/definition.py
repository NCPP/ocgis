from ocgis.api.parms import base


class Snippet(base.BooleanParameter):
    name = 'snippet'
    default = False
    
    
class SpatialOperation(base.StringOptionParameter):
    name = 'spatial_operation'
    default = 'intersects'
    valid = ('clip','intersects')
    
    
class OutputFormat(base.StringOptionParameter):
    name = 'output_format'
    default = 'numpy'
    valid = ('numpy','shp','csv','keyed','meta','nc')
    
    
class SelectUgid(base.OcgParameter):
    name = 'select_ugid'
    return_type = tuple
    nullable = True
    default = None
    input_types = [list,tuple]
