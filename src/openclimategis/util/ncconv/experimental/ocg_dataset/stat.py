import itertools
from util.ncconv.experimental.ordered_dict import OrderedDict
import numpy as np
from util.ncconv.experimental.helpers import timing, check_function_dictionary,\
    array_split, merge_dict_list
from util.ncconv.experimental.pmanager import ProcessManager
from multiprocessing import Manager
from multiprocessing.process import Process


class SubOcgStat(object):
    
    def __init__(self,sub,grouping,procs=1):
        self.sub = sub
        self.procs = procs
        self.grouping = ['gid','level'] + list(grouping)
        self.time_grouping = grouping
        self.stats = None
        self._date_parts = None
        self._groups = None
        
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
            pmanager = ProcessManager(processes,self.procs)
            pmanager.run()
        else:
            self.f_calculate(all_attrs,
                             self.sub,
                             self.groups[0],
                             funcs,
                             time_conv,
                             self.time_grouping,
                             self.grouping)
        self.stats = merge_dict_list(list(all_attrs))
    
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
                pvars[grp].append(group.get(grp))
            ## extract the time indices of the values
            cmp = [group[grp] for grp in time_grouping]
            ## loop through time vector selecting the time indices
            tids = [ii for ii,time in enumerate(time_conv) if cmp == time]
            ## get the index for the particular geometry
            gid_idx = sub.gid == group['gid']
            ## extract the values
            values = sub.value[tids,group['level']-1,gid_idx].tolist()
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