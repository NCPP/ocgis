from sqlalchemy.types import Integer, Float
from sqlalchemy.schema import Column, ForeignKey
import numpy as np
from util.ncconv.experimental.ordered_dict import OrderedDict
import re
import inspect
from sqlalchemy.exc import InvalidRequestError, OperationalError
from util.ncconv.experimental.helpers import timing, array_split,\
    check_function_dictionary
from multiprocessing import Manager
from multiprocessing.process import Process
import ploader as pl


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
    
    def get_date_query(self,session):
        qdate = session.query(self.db.Value,
                              self.db.Time.day,
                              self.db.Time.month,
                              self.db.Time.year)
        qdate = qdate.join(self.db.Time)
        return(qdate.subquery())
    
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
    
    @staticmethod
    def f_calculate(all_attrs,sub,groups,funcs,time_conv,time_grouping):
        """
        funcs -- dict[] {'function':sum,'name':'foo','kwds':{}} - kwds optional
        """ 
        ## construct the base variable dictionary
        keys = list(groups[0]._labels) + [f.get('name') for f in funcs]
        pvars = OrderedDict([[key,list()] for key in keys])
        ## loop through the groups adding the data
        for group in groups:
            ## append the grouping information
            for label in group._labels:
                pvars[label].append(getattr(group,label))
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
        ## precalc the function name if none is provided
        for f in funcs:
            if 'name' not in f:
                ## just the function name if the argument test exception was not
                ## triggered.
                f.update({'name':f.get('name',f['function'].__name__)})
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
                                       self.time_grouping))
                         for groups in self.groups]
            self.run_parallel(processes)
        else:
            self.f_calculate(all_attrs,self.sub,self.groups[0],funcs,time_conv,self.time_grouping)
            
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


class OcgStatFunction(object):
    """
    >>> functions = ['mean','median','max','min','gt2','between1,2']
    >>> stat = OcgStatFunction()
    >>> function_list = stat.get_function_list(functions)
    >>> potentials = stat.get_potentials()
    """
    
    _descs = {
        'min': 'Minimum value in the series',
        'max': 'Maximum value in the series',
        'mean': 'Mean value for the series',
        'median': 'Median value for the series',
        'std': 'Standard deviation for the series',
        'gt':'Count of values greater than {0} in the series (exclusive).',
        'between':'Count of values between {0} and {1} (inclusive).',
        'len':'Sample size of the series.'
#        'gt_thirty_two_point_two': 'Count of values greater than 32.2',
#        'gt_thirty_five': 'Count of values greater than 35',
#        'gt_thirty_seven_point_eight': 'Count of values greater than 37.8',
#        'lt_zero': 'Count of values less than 0',
#        'lt_negative_twelve_point_two': 'Count of values less than -12.2',
#        'lt_negative_seventeen_point_seven': 'Count of values less than -17.7',
              }
    
    def get_function_list(self,functions):
        funcs = []
        for f in functions:
            fname = re.search('([A-Za-z_]+)',f).group(1)
            try:
                args = re.search('([\d,]+)',f).group(1)
            except AttributeError:
                args = None
            attrs = {'function':getattr(self,fname)}
            if args is not None:
                args = [float(a) for a in args.split(',')]
                attrs.update({'args':args})
            if ':' in f:
                attrs.update({'name':f.split(':')[1]})
            funcs.append(attrs)
        return(funcs)
    
    @classmethod
    def get_potentials(cls):
        filters = ['_','get_'] # filter out methods that start with these strings
        ret = []
        for member in inspect.getmembers(cls):
            if inspect.isfunction(member[1]):
                test = [member[0].startswith(filter) for filter in filters]
                if not any(test):
                    ret.append([member[0],cls._descs.get(member[0],member[0])])
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
#    def gt_thirty_two_point_two(values):
#        days = filter(lambda x: x > 32.2, values)
#        return(len(days))
#    
#    @staticmethod
#    def gt_thirty_five(values):
#        days = filter(lambda x: x > 35, values)
#        return(len(days))
#    
#    @staticmethod
#    def gt_thirty_seven_point_eight(values):
#        days = filter(lambda x: x > 37.8, values)
#        return(len(days))
#    
#    @staticmethod
#    def lt_zero(values):
#        days = filter(lambda x: x < 0, values)
#        return(len(days))
#    
#    @staticmethod
#    def lt_negative_twelve_point_two(values):
#        days = filter(lambda x: x < -12.2, values)
#        return(len(days))
#    
#    @staticmethod
#    def lt_negative_seventeen_point_seven(values):
#        days = filter(lambda x: x < -17.7, values)
#        return(len(days))
    
    @staticmethod
    def gt(values,threshold=None):
        if threshold is None:
            raise(ValueError('a threshold must be passed'))
        days = filter(lambda x: x > threshold, values)
        return(len(days))
    
    @staticmethod
    def between(values,lower=None,upper=None):
        if lower is None or upper is None:
            raise(ValueError('a lower and upper limit are required'))
        days = filter(lambda x: x >= lower and x <= upper, values)
        return(len(days))
    
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()