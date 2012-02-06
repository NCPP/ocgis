import itertools
from util.ncconv.experimental.ordered_dict import OrderedDict
import numpy as np
from util.ncconv.experimental.helpers import timing, check_function_dictionary,\
    array_split, merge_dict_list
from util.ncconv.experimental.pmanager import ProcessManager
from multiprocessing import Manager
from multiprocessing.process import Process
from util.ncconv.experimental.ploader import ParallelModel, ParallelLoader,\
    ParallelGenerator
from sqlalchemy.schema import Column, ForeignKey
from sqlalchemy.types import Integer, Float
from sqlalchemy.exc import InvalidRequestError, OperationalError
from django.conf import settings


class SubOcgStat(object):
    __types = {int:Integer,
               float:Float}
    
    def __init__(self,sub,grouping=None,procs=1):
        self.sub = sub
        self.procs = procs
        self.time_grouping = grouping or []
        self.grouping = ['gid','level'] + self.time_grouping
        self.stats = None
        self._date_parts = None
        self._groups = None
        
    def iter_stats(self,caps=True,keep_geom=True,wkt=False,wkb=False):
        if caps:
            keys = [k.upper() for k in self.stats.keys()]
        else:
            keys = self.stats.keys()
        if not keep_geom:
            keys.remove('GEOMETRY')
        keys.insert(0,'OCGID')
        for ii in range(0,len(self.stats['gid'])):
            d = OrderedDict(zip(keys,
                            [ii+1] + [self.stats[k.lower()][ii] for k in keys[1:]]))
            if wkt:
                d.update({'WKT':self.stats['geometry'][ii][0].wkt})
            if wkb:
                d.update({'WKT':self.stats['geometry'][ii][0].wkb})
            yield(d)
        
    @property
    def date_parts(self):
        if self._date_parts is None:
            year = []
            month = []
            day = []
            for t in self.sub.timevec:
                year.append(t.year)
                month.append(t.month)
                day.append(t.day)
            self._date_parts = {'year':year,
                                'month':month,
                                'day':day}
        return(self._date_parts)
    
    @property
    def groups(self):
        if self._groups is None:
            self._groups = array_split(self.get_distinct_groups(),self.procs)
        return(self._groups)
    
    @timing
    def get_distinct_groups(self):
        gids = self.sub.gid.tolist()
        levels = self.sub.levelvec.tolist()
        years = [None]
        months = [None]
        days = [None]
        if 'year' in self.grouping:
            years = list(set((self.date_parts['year'])))
        if 'month' in self.grouping:
            months = list(set((self.date_parts['month'])))
        if 'day' in self.grouping:
            days = list(set((self.date_parts['day'])))
        groups = []
        for gid,level,year,month,day in itertools.product(gids,levels,years,months,days):
            groups.append(dict(gid=gid,level=level,year=year,day=day,month=month))
        return(groups)
    
    @timing
    def calculate(self,funcs):
        ## always count the data
        base_funcs = [{'function':len,'name':'count_agg','raw':False}]
        ## check if there are raw calculation functions
        if any([f['raw'] for f in funcs]):
            has_raw = True
            if self.sub.value_set == {}:
                raise(ValueError('Raw aggregate statistics requested with no "value_set_coll"!!'))
        else:
            has_raw = False
        ## need to count the raw data values if raw value are present
        if has_raw:
            base_funcs.append({'function':len,'name':'count_raw','raw':True})
        ## append the rest of the functions
        funcs = base_funcs + funcs
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
                                       self.grouping,
                                       has_raw))
                         for groups in self.groups]
            pmanager = ProcessManager(processes,self.procs)
            pmanager.run()
        else:
            self.f_calculate(all_attrs,
                             self.sub,
                             self.groups[0],
                             funcs,
                             time_conv,
                             self.time_grouping,
                             self.grouping,
                             has_raw)
        self.stats = merge_dict_list(list(all_attrs))
    
    @staticmethod
    def f_calculate(all_attrs,sub,groups,funcs,time_conv,
                    time_grouping,grouping,has_raw):
        """
        funcs -- dict[] {'function':sum,'name':'foo','kwds':{}} - kwds optional
        """ 
        ## construct the base variable dictionary
        keys = list(grouping) + [f.get('name') for f in funcs] + ['geometry']
        pvars = OrderedDict([[key,list()] for key in keys])
        ## loop through the groups adding the data
        for group in groups:
            ## append the grouping information
            for grp in grouping:
                pvars[grp].append(group.get(grp))
            ## extract the time indices of the values
            cmp = [group[grp] for grp in time_grouping]
            ## loop through time vector selecting the time indices
            tids = [ii for ii,time in enumerate(time_conv) if cmp == time]
            ## get the index for the particular geometry
            gid_idx = sub.gid == group['gid']
            ## append the geometry
            pvars['geometry'].append(sub.geometry[gid_idx])
            ## extract the values
            values = sub.value[tids,group['level']-1,gid_idx].tolist()
            if has_raw:
                raw_values = sub.value_set[group['gid']][tids,group['level']-1,:].compressed().tolist()
            ## calculate function value and update variables
            for f in funcs:
                if f['raw']:
                    args = [raw_values]
                else:
                    args = [values]
                kwds = f.get('kwds',{})
                args += f.get('args',[])
                ivalue = f['function'](*args,**kwds)
                if type(ivalue) == np.float_:
                    ivalue = float(ivalue)
                elif type(ivalue) == np.int_:
                    ivalue = int(ivalue)
                pvars[f.get('name')].append(ivalue)
        all_attrs.append(pvars)
        
    @timing
    def load(self,db):
        self.set_table(OrderedDict(zip(self.stats.keys(),
                                       [self.stats[key][0] for key in self.stats.keys()])),db)
        
        def f(idx,stats={}):
            d = dict(zip(stats.keys(),
                        [stats[key][idx] for key in stats.keys()]))
            return(d)
        
        fkwds = dict(stats=self.stats)
        gen = ParallelGenerator(db.Stat,
                                range(0,len(self.stats['gid'])),
                                f,
                                fkwds=fkwds,
                                procs=settings.MAXPROCESSES,
                                use_lock=True)
        gen.load()
        
#        pmodels = [ParallelModel(db.Stat,data) for data in self.stats]
#        ploader = ParallelLoader(procs=self.procs)
#        ploader.load_models(pmodels)
            
    def set_table(self,arch,db):
        attrs = OrderedDict({'__tablename__':'stats',
                             'ocgid':Column(Integer,primary_key=True),
                             'gid':Column(Integer,ForeignKey(db.Geometry.gid)),
                             'level':Column(Integer,nullable=False,index=True)})
        for key,value in arch.iteritems():
            if key in ['gid','level','geometry']: continue
            if key in ['day','month','year']:
                index = True
            else:
                index = False
            attrs.update({key:Column(self.__types[type(value)],
                                           nullable=False,
                                           index=index)})
        try:
            db.Stat = type('Stat',
                                (db.AbstractValue,db.Base),
                                attrs)
        except InvalidRequestError:
            db.metadata.remove(db.Stat.__table__)
            db.Stat = type('Stat',
                                (db.AbstractValue,db.Base),
                                attrs)
        try:
            db.Stat.__table__.create()
        except OperationalError:
            db.Stat.__table__.drop()
            db.Stat.__table__.create()