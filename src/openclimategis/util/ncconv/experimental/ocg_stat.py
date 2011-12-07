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
from util.ncconv.experimental.helpers import timing, chunks, array_split
import types
from multiprocessing import Pool, Manager
from multiprocessing.process import Process


class OcgStat(object):
    __types = {int:Integer,
               float:Float}
    
    def __init__(self,db,grouping,procs=1):
        self.db = db
        self.grouping = ['gid','level'] + list(grouping)
        self.procs = procs
        self._groups = None
        
    @property
    def groups(self):
        if self.procs == 1:
            self._groups = self.iter_grouping()
        elif self.procs > 1:
            self._groups = self.get_groups()
        return(self._groups)
    
    def get_date_query(self,session):
        qdate = session.query(self.db.Value,
                              self.db.Time.day,
                              self.db.Time.month,
                              self.db.Time.year)
        qdate = qdate.join(self.db.Time)
        return(qdate.subquery())
    
    @staticmethod
    def get_group(all_attrs,db,group_objs,grouping):
        s = db.Session()
        try:
            for group_obj in group_objs:
                qtids = s.query(db.Time.tid)
                for grp in grouping:
                    if grp in ['year','day','month']:
                        qtids = qtids.filter(getattr(db.Time,grp) == getattr(group_obj,grp))
#                tids = [dd[0] for dd in qtids.all()]
                data = s.query(db.Value.value).filter(db.Value.tid.in_(qtids))
                for grp in grouping:
                    if grp in ['gid','level']:
                        data = data.filter(getattr(db.Value,grp) == getattr(group_obj,grp))
                print(data);import ipdb;ipdb.set_trace()
                attrs = OrderedDict(zip(group_obj.keys(),[getattr(group_obj,key) for key in group_obj.keys()]))
                attrs['value'] = [d[0] for d in data.all()]
                all_attrs.append(attrs)
        finally:
            s.close()
    
    @timing     
    def get_groups(self):
        s = self.db.Session()
        try:
            ## return the date subquery
            sq = self.get_date_query(s)
            ## retrieve the unique groups over which to iterate
            columns = [getattr(sq.c,grp) for grp in self.grouping]
            qdistinct = s.query(*columns).distinct()
            distinct_groups = array_split(qdistinct.all(),self.procs)
            ## process in parallel
            all_attrs = Manager().list()
            processes = [Process(target=self.get_group,
                                 args=(all_attrs,self.db,group_objs,self.grouping))
                         for group_objs in distinct_groups]
            for process in processes:
                process.start()
            for p in processes:
                p.join()
            return(all_attrs)
        finally:
            s.close()
            
    def iter_grouping(self):
        s = self.db.Session()
        try:
            ## return the date subquery
            sq = self.get_date_query(s)
            ## retrieve the unique groups over which to iterate
            columns = [getattr(sq.c,grp) for grp in self.grouping]
            qdistinct = s.query(*columns).distinct()
            ## iterate over the grouping returning a list of values for that
            ## group.
            for obj in qdistinct.all():
                data = s.query(self.db.Value.value).join(self.db.Time)
                for grp in self.grouping:
                    if grp in ['gid','level']:
                        data = data.filter(getattr(self.db.Value,grp) == getattr(obj,grp))
                    else:
                        data = data.filter(getattr(self.db.Time,grp) == getattr(obj,grp))
                attrs = OrderedDict(zip(obj.keys(),[getattr(obj,key) for key in obj.keys()]))
                attrs['value'] = [d[0] for d in data.all()]
                yield(attrs)
        finally:
            s.close()
            
    def calculate(self,funcs):
        """
        funcs -- dict[] {'function':sum,'name':'foo','kwds':{}} - kwds optional
        """
        funcs = [{'function':len,'name':'count'}] + funcs
        for ii,group in enumerate(self.groups):
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
                             'level':Column(Integer,nullable=False,index=True)})
        for key,value in arch.iteritems():
            if key in ['gid','level']: continue
            if key in ['day','month','year']:
                index = True
            else:
                index = False
            attrs.update({key:Column(self.__types[type(value)],
                                     nullable=False,
                                     index=index)})
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