import numpy as np
import itertools
from base import OcgFunctionTree, OcgCvArgFunction
import library


class OcgCalculationEngine(object):
    '''
    grouping : list of bool ndarray : temporal groups
    timevec : ndarray of datetime.datetime : actual datetime objects
    funcs : dict : dictionary of function definitions
    raw=False : bool : true if calculation should be on raw data values
    agg=False : bool : true if calculations should be performed on aggregated values
    time_range : list of datetime.datetime : bounding range for time selection
    '''
    
    def __init__(self,grouping,funcs,raw=False,agg=False,snippet=False):
        self.raw = raw
        self.agg = agg
        self.snippet = snippet
#        self.time_range = time_range
#        ## subset timevec if time_range is passed
#        if self.time_range is not None:
#            self.timevec = timevec[(timevec>=time_range[0])*
#                                   (timevec<=time_range[1])]
#            if len(self.timevec) == 0:
#                raise(IndexError('time range returned no data.'))
#        else:
#            self.timevec = timevec
        ## convert solitary grouping arguments to list
        if type(grouping) == str: grouping = [grouping]
        self.grouping = grouping or ['day']
        ## always calculate the sample size. do a copy so functions list cannot
        ## grow in memory. only a problem when testing.
#        funcs_copy = copy(funcs)
#        funcs_copy.insert(0,{'func':'n'})
        self.funcs = self.set_funcs(funcs)
        self.funcs = funcs
        ## get the time groups
#        self.dgroups,self.dtime = self.get_distinct_groups()
        ## select which value data to pull based on raw and agg arguments
        if self.raw:
            self.use_agg = False
        elif self.raw is False and self.agg is True:
            self.use_agg = True
        else:
            self.use_agg = False
        
    def set_funcs(self,funcs):
        potentials = OcgFunctionTree.get_potentials()
        for f in funcs:
            for p in potentials:
                if p[0] == f['func']:
                    f['ref'] = getattr(library,p[1])
                    break
            if 'name' not in f:
                f['name'] = f['func']
            if 'kwds' not in f:
                f['kwds'] = {}
        return(funcs)
        
#    def get_distinct_groups(self):
#        ## holds date components
#        dparts = {'year':[],'month':[],'day':[],'idx':[]}
#        ## pull date parts from date objects and append to date part dictionary
#        for ii,dt in enumerate(self.timevec):
#            for grp in self.grouping:
#                dparts[grp].append(getattr(dt,grp))
#            dparts['idx'].append(ii)
#        ## convert to numpy arrays
#        for key in dparts.keys(): dparts[key] = np.array(dparts[key],dtype=int)
#        ## replace empty list with a list containing NoneType for nested
#        ## iterator and find unique combinations.
#        duni = {}
#        for key in dparts.keys():
#            if key is 'idx':
#                continue
#            elif len(dparts[key]) is 0:
#                duni[key] = np.array([None])
#            else:
#                duni[key] = np.unique(dparts[key]).astype(int)
#                
#        ## make the unique group combinations
#        
#        ## will hold idx to groups
#        dgroups = []
#        dtime = {'tgid':[],'month':[],'day':[],'year':[]}
#        ## the default select all array
#        bidx = np.ones(len(dparts['idx']),dtype=bool)
#        ## loop for combinations
#        tgid = 1
#        for year,month,day in itertools.product(duni['year'],duni['month'],duni['day']):
#            ## idx arrays that when combined provide a group set
#            check = dict(zip(['year','month','day'],[year,month,day]))
#            yidx,midx,didx = [self._get_date_idx_(bidx,dparts,part,value) 
#                              for part,value in check.iteritems()]
#            idx = yidx*midx*didx
#            ## if dates are drilling down to day, it is possible to return date
#            ## combinations that are unreasonable.
#            if idx.sum() == 0:
#                continue
#            for key,value in check.iteritems():
#                dtime[key].append(value)
#            dgroups.append(idx)
#            dtime['tgid'].append(tgid)
#            tgid += 1
#        return(dgroups,dtime)
#            
#    def _get_date_idx_(self,bidx,dparts,part,value):
#        if part in self.grouping:
#            idx = dparts[part] == value
#        else:
#            idx = bidx
#        return(idx)
    
    def execute(self,coll):
        
        ## group the variables
        for ocg_variable in coll.variables.itervalues():
            ocg_variable.group(self.grouping)
            if self.snippet:
                ocg_variable.snippet()
        
#        ## tell collection which data to return
#        coll._use_agg = self.use_agg
        ## flag used for sample size calculation for multivariate calculations
        has_multi = False
        ## iterate over functions
        for f in self.funcs:
            ## change behavior for multivariate functions
            if issubclass(f['ref'],OcgCvArgFunction):
                raise(NotImplementedError)
                has_multi = True
                ## cv-controlled multivariate functions require collecting
                ## data arrays before passing to function.
                kwds = f['kwds'].copy()
                for key in f['ref'].keys:
                    ## the name of the variable passed in the request
                    ## that should be mapped to the named argument
                    backref = kwds[key]
                    ## pull associated data
                    dref = coll._get_value_(backref)
                    ## update dict with properly reference data
                    kwds.update({key:dref})
                ## function object instance
                ref = f['ref'](agg=self.agg,groups=self.dgroups,kwds=kwds,weights=coll.weights)
                ## store calculation value
                coll.calc_multi[f['name']] = ref.calculate()
                coll.cid.add(f['name'])
            else:
                ## perform calculation on each variable
                for var_name,var in coll.variables.iteritems():
                    if not self.use_agg and var.raw_value is not None:
                        value = var.raw_value
                    else:
                        value = var.value
                    ## instance of function object
                    ref = f['ref'](values=value,agg=self.agg,groups=var.temporal_group.dgroups,kwds=f['kwds'],weights=var.spatial.weights)
                    ## calculate the values
                    calc = ref.calculate()
                    ## update calculation identifier
#                    coll.variables[var_name].cid = np.append(coll.variables[var_name].cid,cid)
                    if f['name'] == 'n':
                        add_name = f['name'] + '_' + var_name
                    else:
                        add_name = f['name']
                    ## store the values
                    var.calc_value.update({add_name:calc})
#                    coll.cid.add(add_name)
                    coll.add_calculation(var)
        ## calculate sample size for multivariate calculation
        if has_multi:
            for ii,(key,value) in enumerate(coll.variables.iteritems()):
                if ii == 0:
                    n = value.calc_value['n_'+key].copy()
                else:
                    n += value.calc_value['n_'+key]
            coll.calc_multi['n_multi'] = n
            coll.cid.add('n_multi')
#        return(coll)
