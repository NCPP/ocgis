from sqlalchemy.types import Integer, Float
from sqlalchemy.schema import Column, ForeignKey
import numpy as np
from util.ncconv.experimental.ordered_dict import OrderedDict
from sqlalchemy.exc import InvalidRequestError, OperationalError
from util.ncconv.experimental.helpers import timing, array_split,\
    check_function_dictionary
from multiprocessing import Manager
from multiprocessing.process import Process
from base import OcgFunctionTree
import groups
from util.ncconv.experimental import ploader as pl
from collections import namedtuple
import itertools


class OcgStat(object):
    __types = {int:Integer,
               float:Float}
    
    def __init__(self,db,sub,grouping,procs=1):
        self.db = db
        self.sub = sub
        self.grouping = ['gid','level'] + list(grouping)
        self.time_grouping = grouping
        self.procs = procs
        self._groups = None
        
    @property
    def groups(self):
        if self._groups is None:
            self._groups = array_split(self.get_distinct_groups(),self.procs)
        return(self._groups)
    
    def run_parallel(self,processes):
        for process in processes:
            process.start()
        for p in processes:
            p.join()
    
#    def get_date_query(self,session):
#        qdate = session.query(self.db.Value.gid,
#                              self.db.Value.level,
#                              self.db.Time.day,
#                              self.db.Time.month,
#                              self.db.Time.year)
##        qdate = session.query(self.db.Time.day,
##                              self.db.Time.month,
##                              self.db.Time.year)
#        qdate = qdate.join(self.db.Time)
#        return(qdate.subquery())
    
    def get_distinct(self,Model,attr):
        s = self.db.Session()
        try:
            qq = s.query(getattr(Model,attr)).distinct()
            return([obj[0] for obj in qq.all()])
        finally:
            s.close()
    
    @timing
    def get_distinct_groups(self):
        gids = self.sub.gid.tolist()
        levels = self.get_distinct(self.db.Value,'level')
        years = [None]
        months = [None]
        days = [None]
        if 'year' in self.grouping:
            years = self.get_distinct(self.db.Time,'year')
        if 'month' in self.grouping:
            months = self.get_distinct(self.db.Time,'month')
        if 'day' in self.grouping:
            days = self.get_distinct(self.db.Time,'day')
        Group = namedtuple('Group',['gid','level','day','month','year'])
        groups = []
        for gid,level,year,month,day in itertools.product(gids,levels,years,months,days):
            groups.append(Group(gid=gid,level=level,year=year,day=day,month=month))
        return(groups)
#        s = self.db.Session()
#        try:
#            ## return the date subquery
#            sq = self.get_date_query(s)
#            ## retrieve the unique groups over which to iterate
#            columns = [getattr(sq.c,grp) for grp in self.grouping]
#            qdistinct = s.query(*columns).distinct()
##            return(qdistinct)
#            return(qdistinct.all())
#        finally:
#            s.close()
    
    @staticmethod
    def f_calculate(all_attrs,sub,groups,funcs,time_conv,time_grouping,grouping):
        """
        funcs -- dict[] {'function':sum,'name':'foo','kwds':{}} - kwds optional
        """ 
        ## construct the base variable dictionary
        keys = list(grouping) + [f.get('name') for f in funcs]
        pvars = OrderedDict([[key,list()] for key in keys])
        ## loop through the groups adding the data
        for group in groups:
            ## append the grouping information
            for grp in grouping:
                pvars[grp].append(getattr(group,grp))
            ## extract the time indices of the values
            cmp = [getattr(group,grp) for grp in time_grouping]
            ## loop through time vector selecting the time indices
            tids = [ii for ii,time in enumerate(time_conv) if cmp == time]
            ## get the index for the particular geometry
            gid_idx = sub.gid == group.gid
            ## extract the values
            values = sub.value[tids,group.level-1,gid_idx].tolist()
            ## calculate function value and update variables
            for f in funcs:
                kwds = f.get('kwds',{})
                args = [values] + f.get('args',[])
                ivalue = f['function'](*args,**kwds)
                if type(ivalue) == np.float_:
                    ivalue = float(ivalue)
                elif type(ivalue) == np.int_:
                    ivalue = int(ivalue)
                pvars[f.get('name')].append(ivalue)

        all_attrs.append(pvars)
            
    @timing
    def calculate(self,funcs):
        ## always count the data
        funcs = [{'function':len,'name':'count'}] + funcs
        ## check the function definition dictionary for common problems
        check_function_dictionary(funcs)
        ## convert the time vector for faster referencing
        time_conv = [[getattr(time,grp) for grp in self.time_grouping] 
                     for time in self.sub.timevec]
        ## the shared list
        all_attrs = Manager().list()
        if self.procs > 1:
            processes = [Process(target=self.f_calculate,
                                 args=(all_attrs,
                                       self.sub,
                                       groups,
                                       funcs,
                                       time_conv,
                                       self.time_grouping,
                                       self.grouping))
                         for groups in self.groups]
            self.run_parallel(processes)
        else:
            self.f_calculate(all_attrs,self.sub,self.groups[0],funcs,time_conv,self.time_grouping,self.grouping)

        return(all_attrs)
    
    @timing   
    def calculate_load(self,funcs):
        all_attrs = self.calculate(funcs)
        self.load(all_attrs)
    
    @timing
    def load(self,all_attrs):
        self.set_table(all_attrs[0])
        pmodels = [pl.ParallelModel(self.db.Stat,data) for data in all_attrs]
        ploader = pl.ParallelLoader(procs=self.procs)
        ploader.load_models(pmodels)
            
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
            attrs.update({key:Column(self.__types[type(value[0])],
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
        try:
            self.db.Stat.__table__.create()
        except OperationalError:
            self.db.Stat.__table__.drop()
            self.db.Stat.__table__.create()


class OcgStatFunction(OcgFunctionTree):
    """
    >>> functions = ['mean','median','max','min','gt(2)','between(1,2)']
    >>> stat = OcgStatFunction()
    >>> function_list = stat.get_function_list(functions)
    """
    
    Groups = [groups.BasicStatistics,groups.Thresholds]
    
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()