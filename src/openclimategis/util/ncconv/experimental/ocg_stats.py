import os
from sqlalchemy.sql.expression import func, cast
from sqlalchemy.types import INTEGER, Integer, Float
import copy
from sqlalchemy.schema import Table, Column
from sqlalchemy.orm import mapper


class OcgStat(object):
    __types = {int:Integer,
               float:Float}
    __reserved = ['gid','day','month','year','level']
    
    def __init__(self,db,grouping):
        self.db = db
        self.grouping = list(grouping) + ['level','gid']
        self._groups = None
        
    @property
    def groups(self):
        if self._groups is None:
            self._groups = [attrs for attrs in self.iter_grouping()]
        return(self._groups)
        
    def get_date_query(self,session):
        q = session.query(cast(func.strftime('%m',self.db.Time.time),INTEGER).label('month'),
                        cast(func.strftime('%d',self.db.Time.time),INTEGER).label('day'),
                        cast(func.strftime('%Y',self.db.Time.time),INTEGER).label('year'),
                        self.db.Value)
        q = q.filter(self.db.Time.tid == self.db.Value.tid)
        return(q.subquery())
            
    def iter_grouping(self):
        s = self.db.Session()
        try:
            ## return the date subquery
            sq = self.get_date_query(s)
            ## retrieve the unique groups over which to iterate
            columns = [getattr(sq.c,grp) for grp in self.grouping]
            q = s.query(*columns).distinct()
            ## iterate over the grouping returning a list of values for that
            ## group.
            for obj in q.all():
                filters = [getattr(sq.c,grp) == getattr(obj,grp) for grp in self.grouping]
                data = s.query(sq.c.value)
                for filter in filters:
                    data = data.filter(filter)
                attrs = obj.__dict__;attrs.pop('_labels')
                attrs['value'] = [d[0] for d in data]
                yield(attrs)
        finally:
            s.close()
            
    def calculate(self,funcs):
        """
        funcs -- dict[] {'function':sum,'name':'foo','kwds':{}} - kwds optional
        """
        for group in self.groups:
            grpcpy = group.copy()
            value = grpcpy.pop('value')
            for f in funcs:
                kwds = f.get('kwds',{})
                grpcpy[f['name']] = float(f['function'](value,**kwds))
            yield(grpcpy)
            
    def calculate_load(self,funcs):
#        s = self.db.Session()
#        try:
        coll = []
        for ii,attrs in enumerate(self.calculate(funcs)):
            if ii == 0:
                table = self.get_table(attrs)
                i = table.insert()
            coll.append(attrs)
            i.execute(*coll)
#            s.commit()
#        finally:
#            s.close()    
            
    def get_table(self,arch):
        args = ['stats',self.db.metadata,Column('ocgid',Integer,primary_key=True)]
        for key,value in arch.iteritems():
            args.append(Column(key,self.__types[type(value)]))
        table = Table(*args)
        mapper(self.db.Stat,table)
        table.create()
        return(table)
