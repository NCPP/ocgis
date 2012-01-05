from util.ncconv.experimental.helpers import timing


class OcgConverter(object):
    
    def __init__(self,db,base_name,use_stat=False):
        self.db = db
        self.base_name = base_name
        self.use_stat = use_stat
        
        if self.use_stat:
            self.value_table = self.db.Stat
        else:
            self.value_table = self.db.Value
    
#    @property
#    def mapper(self):
#        return(self.get_mapper(self.value_table))
    
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
    
#    @staticmethod
#    def get_mapper(table):
#        try:
#            table_mapper = table.__mapper__
#        except AttributeError:
#            table_mapper = class_mapper(table)
#        return(table_mapper)
    
    def get_tablename(self,table):
        return(table.__tablename__)
#        m = self.get_mapper(table)
#        return(m.mapped_table.name)
            
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
