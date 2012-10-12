import os
from ocgis import env
from ConfigParser import ConfigParser
from ocgis.util.helpers import get_shp_as_multi


class ShpCabinet(object):
    '''
    >>> sc = ShpCabinet()
    >>> path = sc.get_shp_path('mi_watersheds')
    >>> assert(path.endswith('mi_watersheds.shp'))
    >>> geom_dict = sc.get_geom_dict('mi_watersheds')
    >>> len(geom_dict)
    60
    '''
    
    def __init__(self,path=None):
        self.path = path or env.SHP_DIR
        
    def get_shp_path(self,key):
        return(os.path.join(self.path,key,'{0}.shp'.format(key)))
    
    def get_cfg_path(self,key):
        return(os.path.join(self.path,key,'{0}.cfg'.format(key)))
    
    def get_geom_dict(self,key):
        shp_path = self.get_shp_path(key)
        cfg_path = self.get_cfg_path(key)
        config = ConfigParser()
        config.read(cfg_path)
        id_attr = config.get('mapping','id')
        other_attrs = config.get('mapping','attributes').split(',')
        geom_dict = get_shp_as_multi(shp_path,
                                     uid_field=id_attr,
                                     attr_fields=other_attrs)
        return(geom_dict)
        
        
if __name__ == '__main__':
    import doctest
    doctest.testmod()