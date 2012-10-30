from ocgis.util.shp_cabinet import ShpCabinet
from shapely.geometry.polygon import Polygon
import datetime
from cdata.models import Address
from ConfigParser import ConfigParser
import ocgis
import os.path
import exc


_cp = ConfigParser()
with open(os.path.join(os.path.split(
                       os.path.split(
                       os.path.split(ocgis.__file__)[0])[0])[0],
                       'ocgis.cfg'),'r') as f:
    _cp.readfp(f)
    PARMS = dict(_cp.items('parms'))


class QueryParm(object):
    
    def __init__(self,query,key,default=None,scalar=False,nullable=True,
                 split_char='|',name_map=None,dtype=str):
        self.query = query
        self.key = key
        self.default = default
        self.scalar = scalar
        self.nullable = nullable
        self.split_char = split_char
        self.name_map = name_map or {}
        self.dtype = dtype
        self.value = self._get_()
        
    def __repr__(self):
        msg = '{0}={1}'.format(self.__class__.__name__,self.value)
        return(msg)
        
    def __iter__(self):
        for ii in self.value:
            yield(ii)
            
    def _get_(self):
        value = self.query.get(self.key) or self.query.get(self.name_map.get(self.key))
        try:
            value = value.lower()
            if value == 'none':
                value = None
        except AttributeError:
            pass
        if value is None and self.nullable:
            value = self.default
        elif value is None and not self.nullable:
            raise(exc.NotNullableError(self.key))
        else:
            value = self.format(value)
        return(value)
        
    def format(self,value):
        value = self.split(value)
        value = [self._format_element_(v) for v in value]
        value = self._format_all_elements_(value)
        if self.scalar:
            if len(value) > 1:
                raise(exc.ScalarError(self.key))
            value = value[0]
        return(value)
    
    def split(self,value):
        return(value.split(self.split_char))
    
    def _format_element_(self,value):
        return(self.dtype(value))
    
    def _format_all_elements_(self,value):
        return(value)
    
    
class OcgQueryParm(QueryParm):
    
    def __init__(self,*args,**kwds):
        kwds.update({'name_map':PARMS})
        super(OcgQueryParm,self).__init__(*args,**kwds)
    
    
class UidParm(OcgQueryParm):
    
    def __init__(self,*args,**kwds):
        kwds.update({'nullable':False})
        super(UidParm,self).__init__(*args,**kwds)
    
    def _format_element_(self,value):
        return(Address.objects.get(pk=int(value)).uri)
    
    
class UriParm(UidParm):
    
    def _format_element_(self,value):
        return(value)

  
class LevelParm(OcgQueryParm):
    
    def __init__(self,*args,**kwds):
        kwds.update({'dtype':int})
        super(LevelParm,self).__init__(*args,**kwds)
    
    
class TimeParm(OcgQueryParm):
    
    def _format_element_(self,value):
        return(datetime.datetime.strptime(value,'%Y-%m-%d'))
    
    
class SpaceParm(OcgQueryParm):
    
    def _format_element_(self,value):
        try:
            ret = float(value)
        except ValueError:
            ret = value
        return(ret)
    
    def _format_all_elements_(self,value):
        try:
            minx,miny,maxx,maxy = value
            geom = Polygon(((minx,miny),
                            (minx,maxy),
                            (maxx,maxy),
                            (maxx,miny)))
            assert(geom.is_valid)
            ret = [{'id':1,'geom':geom}]
        except ValueError:
            sc = ShpCabinet()
            ret = sc.get_geom_dict(value[0])
        return(ret)
    
    
class BooleanParm(OcgQueryParm):
    
    def _format_element_(self,value):
        if value in ['false','f']:
            ret = False
        elif value in ['true','t']:
            ret = True
        return(ret)
    
    
#class QueryParm(Slug):
#    
#    def __init__(self,query,key,default=None,split_char='|',scalar=False):
#        self.query = query
#        self.key = key
#        self.default = default
#        self.split_char = split_char
#        self.scalar = scalar
#        try:
#            self.value = self.format(str(self.query[key][0]))
#        except KeyError:
#            self.value = default
#            
#    def __repr__(self):
#        msg = '{0}[{1}]={2}'.format(self.__class__.__name__,self.key,self.value)
#        return(msg)
#    
#    
#class BoolQueryParm(QueryParm):
#    
#    def _format_element_(self,value):
#        if value in ['false','f']:
#            ret = False
#        elif value in ['true','t']:
#            ret = True
#        return(ret)
#    
#    
class CalcParm(OcgQueryParm):
    
    def _format_element_(self,value):
        func,name = value.split('~',1)
        try:
            name,kwds_raw = name.split('!',1)
            kwds_raw = kwds_raw.split('!')
            kwds = {}
            for kwd in kwds_raw:
                kwd_name,kwd_value = kwd.split('~')
                try:
                    kwds.update({kwd_name:float(kwd_value)})
                except ValueError:
                    kwds.update({kwd_name:str(kwd_value)})
        except ValueError:
            kwds = {}
        ret = {'func':func,'name':name,'kwds':kwds}
        return(ret)
