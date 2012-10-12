from ocgis.conv.converter import OcgConverter

    
class NumpyConverter(OcgConverter):
    
    def __init__(self,*args,**kwds):
        super(NumpyConverter,self).__init__(*args,**kwds)
    
    def write(self):
        ret = {}
        for coll,geom_dict in self:
            ret.update({geom_dict['id']:{'geom':geom_dict['geom'],
                                         'coll':coll}})
        return(ret)