from base import OcgCvArgFunction
from ocgis.calc.library import SampleSize
from ocgis.api.collection import CalcCollection, MultivariateCalcCollection
from collections import OrderedDict
import numpy as np
from ocgis import constants, env
from warnings import warn


class OcgCalculationEngine(object):
    '''
    :type grouping: list of temporal groupings (e.g. ['month','year'])
    :type funcs: list of function dictionaries
    :type raw: bool
    :type agg: bool
    '''
    
    def __init__(self,grouping,funcs,raw=False,agg=False):
        self.raw = raw
        self.agg = agg
        self.grouping = grouping
        self.funcs = funcs
        ## select which value data to pull based on raw and agg arguments
        if self.raw:
            self.use_agg = False
        elif self.raw is False and self.agg is True:
            self.use_agg = True
        else:
            self.use_agg = False
            
        ## check for multivariate functions
        check = [issubclass(f['ref'],OcgCvArgFunction) for f in funcs]
        self.has_multi = True if any(check) else False

    def _get_value_weights_(self,ds):
        ## select the value source based on raw or aggregated switches
        if not self.use_agg:
            try:
                value = ds.raw_value
                weights = ds.spatial.vector.raw_weights
            except AttributeError:
                value = ds.value
                weights = ds.spatial.vector.weights
        else:
            value = ds.value
            weights = ds.spatial.vector.weights
        return(value,weights)
    
    def execute(self,coll,file_only=False):
        ## switch collection type based on the presence of a multivariate
        ## calculation
        if self.has_multi:
            ret = MultivariateCalcCollection(coll)
        else:
            ret = CalcCollection(coll)

        ## group the variables. if grouping is None, calculations are performed
        ## on each element. array computations are taken advantage of.
        if self.grouping is not None:
            for ds in coll.variables.itervalues():
                ds.temporal.set_grouping(self.grouping)

        ## iterate over functions
        for f in self.funcs:
            ## change behavior for multivariate functions
            if issubclass(f['ref'],OcgCvArgFunction) or (self.has_multi and f['ref'] == SampleSize):
                ## do not calculated sample size for multivariate calculations
                ## yet
                if f['ref'] == SampleSize:
                    if not env.VERBOSE:
                        warn('sample size calculations not implemented for multivariate calculations yet')
                    continue
                ## cv-controlled multivariate functions require collecting
                ## data arrays before passing to function.
                kwds = f['kwds'].copy()
                for ii,key in enumerate(f['ref'].keys):
                    ## the name of the variable passed in the request
                    ## that should be mapped to the named argument
                    backref = kwds[key]
                    ## pull associated data
                    dref = coll.variables[backref]
                    value,weights = self._get_value_weights_(dref)
                    ## get the calculation groups and weights.
                    if ii == 0:
                        if self.grouping is None:
                            dgroups = None
                        else:
                            dgroups = dref.temporal.group.dgroups
                    ## update dict with properly reference data
                    kwds.update({key:value})
                ## function object instance
                ref = f['ref'](agg=self.agg,groups=dgroups,kwds=kwds,weights=weights)
                calc = ref.calculate()
                ## store calculation value
                ret.calc[f['name']] = calc
            else:
                ## perform calculation on each variable
                for alias,var in coll.variables.iteritems():
                    if alias not in ret.calc:
                        ret.calc[alias] = OrderedDict()
                    if file_only:
                        calc = np.ma.array(np.empty(0,dtype=f['ref'].dtype),fill_value=constants.fill_value)
                    else:
                        value,weights = self._get_value_weights_(var)
                        ## make the function instance
                        try:
                            ref = f['ref'](values=value,agg=self.agg,
                                           groups=var.temporal.group.dgroups,
                                           kwds=f['kwds'],weights=weights)
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
                    ## store the values
                    ret.calc[alias][f['name']] = calc
        return(ret)
