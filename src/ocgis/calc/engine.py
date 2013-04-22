from base import OcgCvArgFunction
from ocgis.calc.library import SampleSize


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
        self.grouping = grouping
        ## always calculate the sample size. do a copy so functions list cannot
        ## grow in memory. only a problem when testing.
#        funcs_copy = copy(funcs)
#        funcs_copy.insert(0,{'func':'n'})
#        self.funcs = self.set_funcs(funcs)
        self.funcs = funcs
        ## select which value data to pull based on raw and agg arguments
        if self.raw:
            self.use_agg = False
        elif self.raw is False and self.agg is True:
            self.use_agg = True
        else:
            self.use_agg = False
        
#    def set_funcs(self,funcs):
#        potentials = OcgFunctionTree.get_potentials()
#        for f in funcs:
#            for p in potentials:
#                if p[0] == f['func']:
#                    f['ref'] = getattr(library,p[1])
#                    break
#            if 'name' not in f:
#                f['name'] = f['func']
#            if 'kwds' not in f:
#                f['kwds'] = {}
#        return(funcs)

    def _get_value_(self,ocg_variable):
        ## select the value source based on raw or aggregated switches
        if not self.use_agg and ocg_variable.raw_value is not None:
            value = ocg_variable.raw_value
        else:
            value = ocg_variable.value
        return(value)
    
    def execute(self,coll):
        
        ## group the variables. if grouping is None, calculations are performed
        ## on each element. array computations are taken advantage of.
        if self.grouping is not None:
            for ocg_variable in coll.variables.itervalues():
                ocg_variable.group(self.grouping)
        
#        ## flag used for sample size calculation for multivariate calculations
#        has_multi = False
        ## iterate over functions
        for f in self.funcs:
            ## change behavior for multivariate functions
            if issubclass(f['ref'],OcgCvArgFunction):
#                has_multi = True
                ## cv-controlled multivariate functions require collecting
                ## data arrays before passing to function.
                kwds = f['kwds'].copy()
                for ii,key in enumerate(f['ref'].keys):
                    ## the name of the variable passed in the request
                    ## that should be mapped to the named argument
                    backref = kwds[key]
                    ## pull associated data
                    dref = coll.variables[backref]
                    value = self._get_value_(dref)
                    ## get the calculation groups and weights.
                    if ii == 0:
                        arch = dref
                        weights = dref.spatial.weights
                        if self.grouping is None:
                            dgroups = None
                        else:
                            dgroups = dref.temporal_group.dgroups
                    ## update dict with properly reference data
                    kwds.update({key:value})
                ## function object instance
                ref = f['ref'](agg=self.agg,groups=dgroups,kwds=kwds,weights=weights)
                calc = ref.calculate()
                ## store calculation value
                var = OcgMultivariateCalculationVariable(f['name'],calc,arch.temporal,arch.spatial,arch.level)
                var.temporal_group = arch.temporal_group
                coll.add_multivariate_calculation_variable(var)
            else:
                ## perform calculation on each variable
                for var in coll.variables.itervalues():
                    value = self._get_value_(var)
                    ## make the function instance
                    try:
                        ref = f['ref'](values=value,agg=self.agg,
                                       groups=var.temporal_group.dgroups,
                                       kwds=f['kwds'],weights=var.spatial.weights)
                    except AttributeError:
                        ## if there is no grouping, there is no need to calculate
                        ## sample size.
                        if self.grouping is None and f['ref'] == SampleSize:
                            break
                        elif self.grouping is None:
                            raise(NotImplementedError('Univariate calculations must have a temporal grouping.'))
                        else:
                            raise
                    ## calculate the values
                    calc = ref.calculate()
                    ## update calculation identifier
                    add_name = f['name']
                    ## store the values
                    var.calc_value.update({add_name:calc})
#                    coll.cid.add(add_name)
                    coll.add_calculation(var)
#        ## calculate sample size for multivariate calculation
#        if has_multi:
#            import ipdb;ipdb.set_trace()
#            for ii,(key,value) in enumerate(coll.variables.iteritems()):
#                if ii == 0:
#                    n = value.calc_value['n_'+key].copy()
#                else:
#                    n += value.calc_value['n_'+key]
#            coll.calc_multi['n_multi'] = n
#            coll.cid.add('n_multi')
