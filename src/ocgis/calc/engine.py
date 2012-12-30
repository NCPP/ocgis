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
    
    def __init__(self,grouping,funcs,raw=False,agg=False):
        self.raw = raw
        self.agg = agg
        ## convert solitary grouping arguments to list
        if type(grouping) == str: grouping = [grouping]
        self.grouping = grouping or ['day']
        ## always calculate the sample size. do a copy so functions list cannot
        ## grow in memory. only a problem when testing.
#        funcs_copy = copy(funcs)
#        funcs_copy.insert(0,{'func':'n'})
        self.funcs = self.set_funcs(funcs)
        self.funcs = funcs
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
    
    def execute(self,coll):
        
        ## group the variables
        for ocg_variable in coll.variables.itervalues():
            ocg_variable.group(self.grouping)
        
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
