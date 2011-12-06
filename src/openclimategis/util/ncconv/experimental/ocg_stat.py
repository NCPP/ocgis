import os
from sqlalchemy.sql.expression import func, cast
from sqlalchemy.types import INTEGER, Integer, Float
import copy
from sqlalchemy.schema import Table, Column, ForeignKey
from sqlalchemy.orm import mapper, relationship
import numpy as np
from util.ncconv.experimental.ordered_dict import OrderedDict
import re
import inspect
from sqlalchemy.exc import InvalidRequestError, ArgumentError, OperationalError
from util.ncconv.experimental.helpers import timing
import types


class OcgStat(object):
    __types = {int:Integer,
               float:Float}
    
    def __init__(self,db,grouping):
        self.db = db
        self.grouping = ['gid','level'] + list(grouping)
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
                attrs = OrderedDict(zip(obj.keys(),[getattr(obj,key) for key in obj.keys()]))
                attrs['value'] = [d[0] for d in data]
                yield(attrs)
        finally:
            s.close()
            
    def calculate(self,funcs):
        """
        funcs -- dict[] {'function':sum,'name':'foo','kwds':{}} - kwds optional
        """
        funcs = [{'function':len,'name':'count'}] + funcs
        for group in self.groups:
            grpcpy = group.copy()
            value = grpcpy.pop('value')
            for f in funcs:
                kwds = f.get('kwds',{})
                args = [value] + f.get('args',[])
                name = f.get('name',f['function'].__name__)
                ivalue = f['function'](*args,**kwds)
                if type(ivalue) == np.float_:
                    ivalue = float(ivalue)
                elif type(ivalue) == np.int_:
                    ivalue = int(ivalue)
                grpcpy[name] = ivalue
            yield(grpcpy)
    
    @timing   
    def calculate_load(self,funcs):
        coll = []
        for ii,attrs in enumerate(self.calculate(funcs)):
            if ii == 0:
                self.set_table(attrs)
                i = self.db.Stat.__table__.insert()
            coll.append(attrs)
        i.execute(*coll)
            
    def set_table(self,arch):
        attrs = OrderedDict({'__tablename__':'stats',
                             'ocgid':Column(Integer,primary_key=True),
                             'gid':Column(Integer,ForeignKey(self.db.Geometry.gid)),
                             'level':Column(Integer,nullable=False)})
        for key,value in arch.iteritems():
            if key in ['gid','level']: continue
            attrs.update({key:Column(self.__types[type(value)],nullable=False)})
        try:
            self.db.Stat = type('Stat',
                                (self.db.AbstractValue,self.db.Base),
                                attrs)
        except InvalidRequestError:
            self.db.metadata.remove(self.db.Stat.__table__)
            self.db.Stat = type('Stat',
                                (self.db.AbstractValue,self.db.Base),
                                attrs)
        self.db.Stat.__table__.create()


class OcgStatFunction(object):
    """
    >>> functions = ['mean','median','max','min','gt2','between1,2']
    >>> stat = OcgStatFunction()
    >>> function_list = stat.get_function_list(functions)
    >>> potentials = stat.get_potentials()
    """
    
    __descs = {'mean':'Calculate mean for standard normal distribution.'}
    
    def get_function_list(self,functions):
        funcs = []
        for f in functions:
            fname = re.search('([A-Za-z]+)',f).group(1)
            try:
                args = re.search('([\d,]+)',f).group(1)
            except AttributeError:
                args = None
            attrs = {'function':getattr(self,fname)}
            if args is not None:
                args = [float(a) for a in args.split(',')]
                attrs.update({'args':args})
            funcs.append(attrs)
        return(funcs)
    
    @classmethod
    def get_potentials(cls):
        filters = ['_','get_']
        ret = []
        for member in inspect.getmembers(cls):
            if inspect.isfunction(member[1]):
                test = [member[0].startswith(filter) for filter in filters]
                if not any(test):
                    ret.append([member[0],member[0]+' ('+cls.__descs.get(member[0],member[0])+')'])
        return(ret)
    
    @staticmethod
    def mean(values):
        return(np.mean(values))
    
    @staticmethod
    def median(values):
        return(np.median(values))
    
    @staticmethod
    def std(values):
        return(np.std(values))
    
    @staticmethod
    def max(values):
        return(max(values))
    
    @staticmethod
    def min(values):
        return(min(values))
    
#    @staticmethod
#    def gt(values,threshold=None):
#        if threshold is None:
#            raise(ValueError('a threshold must be passed'))
#        days = filter(lambda x: x > threshold, values)
#        return(len(days))
#    
#    @staticmethod
#    def between(values,lower=None,upper=None):
#        if lower is None or upper is None:
#            raise(ValueError('a lower and upper limit are required'))
#        days = filter(lambda x: x >= lower and x <= upper, values)
#        return(len(days))
    
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()