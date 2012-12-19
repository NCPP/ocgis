from ocgis.conv.converter import OcgConverter

    
class NumpyConverter(OcgConverter):
    
    def __init__(self,*args,**kwds):
        super(NumpyConverter,self).__init__(*args,**kwds)
    
    def write(self):
        ret = {}
        for coll in self:
            ret.update({coll.ugeom['ugid']:coll})
        return(ret)