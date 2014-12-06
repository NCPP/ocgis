from ocgis.calc.base import AbstractUnivariateFunction
from ocgis import constants
import numpy as np
import re
from copy import deepcopy


class EvalFunction(AbstractUnivariateFunction):
    '''
    A function that parses and evaluates string representations of calculations.
    
    :param str expr: The string function to evaluate. The function must have an
     equals sign. The function may contain multiple variables aliases. Mathematical
     operators include standard arithmetic symbols and NumPy functions. The list of
     enabled functions is contained in :attr:~`ocgis.constants.enabled_numpy_ufuncs`.
    '''
    description = None
    dtype = None
    key = None
    standard_name = ''
    long_name = ''
    
    def __init__(self,**kwargs):
        self.expr = kwargs.pop('expr')
        AbstractUnivariateFunction.__init__(self,**kwargs)
    
    def calculate(self):
        raise(NotImplementedError)
    
    def _execute_(self):
        ## get the variable aliases that will map to variables in the string
        ## expresssion
        map_vars = {}
        for variable in self.field.variables.itervalues():
            map_vars[variable.alias] = '_exec_' + variable.alias + '.value'
        ## parse the string filling in the local variable names
        expr,out_variable_name = self._get_eval_string_(self.expr,map_vars)
        ## update the output alias and key used to create the variable collection
        ## later
        self.alias,self.key = out_variable_name,out_variable_name
        ## update the local variable dictionary so when the string expression is
        ## evaluated they will be available
        for k,v in map_vars.iteritems():
            locals()[v.split('.')[0]] = self.field.variables[k]
        
        ## determine the output data type for the calculation. the output data
        ## type has the largest data type measured in number of bytes. also set
        ## the fill value based on the output data type.
        dtype = {v.dtype:np.array([1],dtype=v.dtype).nbytes for v in self.field.variables.itervalues()}
        max_nbytes = np.max(dtype.values())
        for k,v in dtype.iteritems():
            if v == max_nbytes:
                dtype = k
                fill_value = dtype.type(np.ma.array([1],dtype=dtype).fill_value)
                break
        
        ## if the output is file only, do no perform any calculations.
        if self.file_only:
            fill = self._empty_fill
        ## evaluate the expression and update the data type.

        #todo: with numpy 1.8.+ you can do the type modification inplace. this
        ## will make the type conversion operation less memory intensive.
        else:
            fill = eval(expr)
            fill = fill.astype(dtype)
        
        self._add_to_collection_(value=fill,parent_variables=self.field.variables.values(),
                                 dtype=dtype,fill_value=fill_value,units=None,alias=self.alias)
    
    @staticmethod
    def is_multivariate(expr):
        '''
        Return ``True`` if ``expr`` is a multivariate string function.
        
        :param str expr: The string function to evaluate. The function must have an
          equals sign. The function may contain multiple variables aliases. Mathematical
          operators include standard arithmetic symbols and NumPy functions. The list of
          enabled functions is contained in :attr:~`ocgis.constants.enabled_numpy_ufuncs`.
        :returns bool:
        '''
        ## do not count the output variable name
        expr = expr.split('=')[1]
        ## count the number of variable names in the right hand side of the string
        ## expression.
        strings = set(re.findall('[A-Za-z0-9_]+',expr))
        strings = set([s for s in strings if re.search('[A-Za-z]',s) is not None])
        strings_left = deepcopy(strings)
        for s in strings:
            if s in constants.enabled_numpy_ufuncs:
                strings_left.remove(s)
        ## if there are more than one variable alias in the equation, the expression
        ## is multivariate
        if len(strings_left) > 1:
            ret = True
        else:
            ret = False
        return(ret)
    
    @staticmethod
    def _get_eval_string_(expr,map_vars):
        '''
        :param str expr: The string function to evaluate. The function must have an
          equals sign. The function may contain multiple variables aliases. Mathematical
          operators include standard arithmetic symbols and NumPy functions. The list of
          enabled functions is contained in :attr:~`ocgis.constants.enabled_numpy_ufuncs`.
        :param dict map_vars: Maps variable aliases to their output representation
         in ``expr``.
         
        >>> map_vars = {'tas':'tas.value',...}
        
        :returns tuple: A tuple composed of two elements ``(expr, out_variable_name)``.
         * ``expr``: The output string expression to evaluate.
         * ``out_variable_name``: The variable name to assign to the output. This
          is the left-hand side of the expression.
        :raises ValueError:
        '''
        ## standard arithmetic mathematical operators
        math_set = '[(\-+*/)]'
        ## regular expressions use to find the variable aliases
        re_expr_base = ['{1}{0}{1}','^{0}{1}','{1}{0}$','^{0}$']
        ## attempt to split the expression at the equals sign.
        try:
            out_variable_name,expr = expr.split('=')
        except ValueError:
            raise(ValueError('Unable to parse expression string: "{0}". The equals sign is likely missing.'.format(expr)))
        ## find the variable aliases and function names in the string expression
        strings = set(re.findall('[A-Za-z0-9_]+',expr))
        strings = set([s for s in strings if re.search('[A-Za-z]',s) is not None])
        ## "strings" must be entirely composed of enabled numpy functions and the
        ## variable aliases originating from the keys in "map_vars"
        for s in strings:
            if s not in constants.enabled_numpy_ufuncs and s not in map_vars.keys():
                raise(ValueError('Unable to parse expression string: "{0}". '
                                 'Ensure the NumPy functions are enabled and appropriate '
                                 'variables have been requested. The problem string value is "{1}".'.format(expr,s)))
        ## update the names of the numpy functions to use the module prefix "np"
        for np_func in constants.enabled_numpy_ufuncs:
            expr = expr.replace(np_func,'np.'+np_func)
        ## update the variable aliases to match the key-value relationship in
        ## "map_vars"
        max_ctr = len(expr)
        for k,v in map_vars.iteritems():
            for r in re_expr_base:
                re_expr = r.format(k,math_set)
                ctr = 0
                while True:
                    def repl(mo):
                        return(mo.group(0).replace(k,v))
                    expr = re.sub(re_expr,repl,expr)
                    if re.search(re_expr,expr) is None:
                        break
                    ctr += 1
                    if ctr > max_ctr:
                        break
        return(expr,out_variable_name)
    
    def _set_derived_variable_alias_(self,*args,**kwargs):
        pass


class MultivariateEvalFunction(EvalFunction):
    '''
    Dummy class to help the software distinguish between univariate and multivariate
    function string expressions.
    '''
    pass
