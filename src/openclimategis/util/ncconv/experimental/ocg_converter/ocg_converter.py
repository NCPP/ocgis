from util.ncconv.experimental.helpers import timing
import sys
import math


class OcgConverter(object):
    
    def __init__(self,db,base_name,use_stat=False,meta=None):
        self.db = db
        self.base_name = base_name
        self.use_stat = use_stat
        self.meta = meta
        
        if self.use_stat:
            self.value_table = self.db.Stat
        else:
            self.value_table = self.db.Value
    
    def get_iter(self,table,headers=None):
        if headers is None: headers = self.get_headers(table)
        s = self.db.Session()
        try:
            for obj in s.query(table).all():
                yield(self._todict_(obj,headers))
        finally:
            s.close()
            
    def get_headers(self,table,adds=[]):
        keys = table.__mapper__.columns.keys()
        keys += adds
        headers = [h.upper() for h in keys]
        return(headers)
    
    def get_tablename(self,table):
        return(table.__tablename__)
            
    @staticmethod
    def _todict_(obj,headers):
        return(dict(zip(headers,
                        [getattr(obj,h.lower()) for h in headers])))
    
    @timing
    def convert(self,*args,**kwds):
        return(self._convert_(*args,**kwds))
    
    def _convert_(self,*args,**kwds):
        raise(NotImplementedError)
    
    def response(self,*args,**kwds):
        payload = self.convert(*args,**kwds)
        try:
            return(self._response_(payload))
        finally:
            self.cleanup()
    
    def _response_(self,payload):
        return(payload)
    
    def cleanup(self):
        pass
    
    def write(self):
        raise(NotImplementedError)

    def write_meta(self,zip):
        if self.meta is not None:
            zip.writestr('meta.txt',self.meta.response())