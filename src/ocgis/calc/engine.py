from base import OcgCvArgFunction
from ocgis.calc.library import SampleSize
from ocgis.api.collection import CalcCollection, MultivariateCalcCollection, \
    KeyedOutputCalcCollection
from collections import OrderedDict
import numpy as np
from ocgis import constants
from ocgis.util.logging_ocgis import ocgis_lh
import logging
from ocgis.calc.base import KeyedFunctionOutput


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
    
    def _check_calculation_members_(self,funcs,klass):
        '''
        Return True if a subclass of type `klass` is contained in the calculation
        list.
        
        :param funcs: Sequence of calculation dictionaries.
        :param klass: `ocgis.calc.base.OcgFunction`
        '''
        check = [issubclass(f['ref'],klass) for f in funcs]
        ret = True if any(check) else False
        return(ret)
        
    def execute(self,coll,file_only=False):
        ## switch collection type based on the types of calculations present
        if self._check_calculation_members_(self.funcs,OcgCvArgFunction):
            klass = MultivariateCalcCollection
        elif self._check_calculation_members_(self.funcs,KeyedFunctionOutput):
            klass = KeyedOutputCalcCollection
        else:
            klass = CalcCollection
        ret = klass(coll,funcs=self.funcs)
        ocgis_lh(msg='returning collection of type {0}'.format(coll.__class__),
                 logger='calc.engine')

        ## group the variables. if grouping is None, calculations are performed
        ## on each element. array computations are taken advantage of.
        if self.grouping is not None:
            ocgis_lh('setting temporal grouping(s)','calc.engine')
            for ds in coll.variables.itervalues():
                ds.temporal.set_grouping(self.grouping)

        ## iterate over functions
        for f in self.funcs:
            ocgis_lh('calculating: {0}'.format(f),logger='calc.engine')
            ## change behavior for multivariate functions
            if issubclass(f['ref'],OcgCvArgFunction) or (isinstance(ret,MultivariateCalcCollection) and f['ref'] == SampleSize):
                ## do not calculated sample size for multivariate calculations
                ## yet
                if f['ref'] == SampleSize:
                    ocgis_lh('sample size calculations not implemented for multivariate calculations yet',
                             'calc.engine',level=logging.WARN)
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
                                e = NotImplementedError('Univariate calculations must have a temporal grouping.')
                                ocgis_lh(exc=e,logger='calc.engine')
                            else:
                                raise
                        ## calculate the values
                        calc = ref.calculate()
                    ## store the values
                    ret.calc[alias][f['name']] = calc
        return(ret)
