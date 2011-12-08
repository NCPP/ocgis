from sqlalchemy.types import Integer, Float
from sqlalchemy.schema import Column, ForeignKey
import numpy as np
from util.ncconv.experimental.ordered_dict import OrderedDict
import re
import inspect
from sqlalchemy.exc import InvalidRequestError
from util.ncconv.experimental.helpers import timing, array_split
from multiprocessing import Manager
from multiprocessing.process import Process
import ploader as pl


class OcgStat(object):
    __types = {int:Integer,
               float:Float}
    
    def __init__(self,db,grouping,procs=1):
        self.db = db
        self.grouping = ['gid','level'] + list(grouping)
        self.time_grouping = grouping
        self.procs = procs
        self._groups = None
        
    @property
    def groups(self):
        if self._groups is None:
            self._groups = self.get_groups()
        return(self._groups)
    
    def run_parallel(self,processes):
        for process in processes:
            process.start()
        for p in processes:
            p.join()
    
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
            ## create the query statements with bind parameters
            qtids = s.query(db.Time.tid)
            qdata = s.query(db.Value.value)#.filter(db.Value.tid.in_(qtids))
            for grp in grouping:
                if grp in ['year','day','month']:
                    qtids = qtids.filter("{0} = :{0}".format(grp))
                else:
                    qdata = qdata.filter("{0} = :{0}".format(grp))
            ## pull the keys out for later use
            keys = group_objs[0].keys()
            
            for group_obj in group_objs:
                qtids_dict = {}
                qdata_dict = {}
                for grp in grouping:
                    if grp in ['year','day','month']:
                        qtids_dict.update({grp:getattr(group_obj,grp)})
                    else:
                        qdata_dict.update({grp:getattr(group_obj,grp)})
                iqdata = qdata.filter(db.Value.tid.in_(qtids.params(**qtids_dict)))
                iqdata = iqdata.params(**qdata_dict)
                attrs = OrderedDict(zip(keys,[getattr(group_obj,key) for key in keys]))
                attrs['value'] = [d[0] for d in iqdata.all()]
                all_attrs.append(attrs)
        finally:
            s.close()
    
    @timing     
    def get_groups(self):
        s = self.db.Session()
        try:
            distinct_groups = array_split(self.get_distinct_groups(),self.procs)
            ## process in parallel
            all_attrs = Manager().list()
            processes = [Process(target=self.get_group,
                                 args=(all_attrs,self.db,group_objs,self.grouping))
                         for group_objs in distinct_groups]
            self.run_parallel(processes)
            return(all_attrs)
        finally:
            s.close()
    
    @timing
    def get_distinct_groups(self):
        s = self.db.Session()
        try:
            ## return the date subquery
            sq = self.get_date_query(s)
            ## retrieve the unique groups over which to iterate
            columns = [getattr(sq.c,grp) for grp in self.grouping]
            qdistinct = s.query(*columns).distinct()
            return(qdistinct.all())
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
        
    
    @staticmethod
    def f_calculate(all_attrs,groups,funcs):
        """
        funcs -- dict[] {'function':sum,'name':'foo','kwds':{}} - kwds optional
        """ 
        
        ## always count the data
        funcs = [{'function':len,'name':'count'}] + funcs
        ## construct the base variable dictionary
        pvars = []
        keys = groups[0].keys() + [f.get('name',f['function'].__name__) for f in funcs]
        for key in keys:
            if key != 'value':
                pvars.append(pl.ParallelVariable(key))
        ## get indices for fast inserting?
        idx = dict([pvar.name,ii] for ii,pvar in enumerate(pvars))

        ## loop through the groups adding the data
        for group in groups:
            for key,value in group.iteritems():
                if key != 'value':
                    pvars[idx[key]].append(value)
            for f in funcs:
                kwds = f.get('kwds',{})
                args = [group['value']] + f.get('args',[])
                name = f.get('name',f['function'].__name__)
                ivalue = f['function'](*args,**kwds)
                if type(ivalue) == np.float_:
                    ivalue = float(ivalue)
                elif type(ivalue) == np.int_:
                    ivalue = int(ivalue)
                pvars[idx[name]].append(ivalue)

        all_attrs.append(pvars)
            
    @timing
    def calculate(self,funcs):
        ## the shared list
        all_attrs = Manager().list()
        if self.procs > 1:
            ## split groups into distinct groupings
#            print('  splitting into distinct groups...')
            distinct_groups = array_split(self.groups,self.procs)
#            print('  init processes...')
            processes = [Process(target=self.f_calculate,
                                 args=(all_attrs,
                                       groups,
                                       funcs))
                         for groups in distinct_groups]
#            print('  exec processes...')
            self.run_parallel(processes)
        else:
            self.f_calculate(all_attrs,self.groups,funcs)
#        print('    done.')
        return(all_attrs)
    
    @timing   
    def calculate_load(self,funcs):
        all_attrs = self.calculate(funcs)
        self.load(all_attrs)
    
    @timing
    def load(self,all_attrs):
        self.set_table(all_attrs[0])
        pmodels = [pl.ParallelModel(self.db.Stat,pvars) for pvars in all_attrs]
        ploader = pl.ParallelLoader(procs=self.procs)
        ploader.load_models(pmodels)
            
    def set_table(self,pvars):
        attrs = OrderedDict({'__tablename__':'stats',
                             'ocgid':Column(Integer,primary_key=True),
                             'gid':Column(Integer,ForeignKey(self.db.Geometry.gid)),
                             'level':Column(Integer,nullable=False,index=True)})
        for pvar in pvars:
            if pvar.name in ['gid','level']: continue
            if pvar.name in ['day','month','year']:
                index = True
            else:
                index = False
            attrs.update({pvar.name:Column(self.__types[type(pvar.data[0])],
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
        
        
class SubOcgStat(OcgStat):
    
    def __init__(self,db,grouping,sub,**kwds):
        self.sub = sub
        
        super(SubOcgStat,self).__init__(db,grouping,**kwds)
    
    @staticmethod
    def get_group(all_attrs,distinct_groups,sub,time_grouping,grouping,time_conv):
        ## get the time indices for each group
        for dgrp in distinct_groups:
            dgrp = OrderedDict(zip(grouping,[getattr(dgrp,grp) for grp in grouping]))
            ## get comparator
            cmp = [dgrp.get(grp) for grp in time_grouping]
            ## loop through time vector selecting the time indices
            tids = [ii for ii,time in enumerate(time_conv) if cmp == time]
            ## create the output
            cell_id = sub.cell_id == dgrp['gid']
            dgrp['value'] = sub.value[tids,dgrp['level']-1,cell_id].tolist()
            all_attrs.append(dgrp)
    
    @timing
    def get_groups(self):
        ## convert the time vector for grouping comparison
        time_conv = [[getattr(time,grp) for grp in self.time_grouping] 
                     for time in self.sub.timevec]
        ## set up parallel processing
        distinct_groups = array_split(self.get_distinct_groups(),self.procs)
        ## process in parallel
        all_attrs = Manager().list()
        processes = [Process(target=self.get_group,
                             args=(all_attrs,
                                   distinct_group,
                                   self.sub,
                                   self.time_grouping,
                                   self.grouping,
                                   time_conv))
                     for distinct_group in distinct_groups]
        self.run_parallel(processes)
        return(all_attrs)


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