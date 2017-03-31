import itertools
import re
from copy import deepcopy

import numpy as np

from ocgis import constants
from ocgis.calc.base import AbstractUnivariateFunction


class EvalFunction(AbstractUnivariateFunction):
    """
    A function that parses and evaluates string representations of calculations. If ``file_only`` is ``True``, the
    output array's data type is :attr:~`env.NP_FLOAT` if ``dtype`` is ``None``. If ``file_only`` is
    ``False``, the output data type is determined by the NumPy calculation.

    .. note:: Accepts all parameters to :class:`~ocgis.calc.base.AbstractUnivariateFunction`.

    :param str expr: The string function to evaluate. The function must have an equals sign. The function may contain
     multiple variables aliases. Mathematical operators include standard arithmetic symbols and NumPy functions. The
     list of enabled functions is contained in :attr:~`ocgis.constants.ENABLED_NUMPY_UFUNCS`.
    """

    description = None
    key = None
    standard_name = ''
    long_name = ''

    def __init__(self, **kwargs):
        # Tricks PyCharm into not removing the import on import optimizations.
        assert np

        self.expr = kwargs.pop('expr')
        AbstractUnivariateFunction.__init__(self, **kwargs)

    def calculate(self):
        raise NotImplementedError

    def _execute_(self):

        # get the variable aliases that will map to variables in the string expresssion
        map_vars = {}
        calculation_targets = {}
        for variable in self.iter_calculation_targets(yield_calculation_name=False, validate_units=False):
            # for variable in self.field.variables.itervalues():
            map_vars[variable.name] = '_exec_' + variable.name
            calculation_targets[variable.name] = variable
            # map_vars[variable.name] = '_exec_' + variable.name + '.value'
            # map_vars[variable.alias] = '_exec_' + variable.alias + '.value'
        # parse the string filling in the local variable names
        expr, out_variable_name = self._get_eval_string_(self.expr, map_vars)
        # update the output alias and key used to create the variable collection later
        self.alias, self.key = out_variable_name, out_variable_name

        # Construct conformed array iterator.
        keys = map_vars.keys()
        crosswalks = [self._get_dimension_crosswalk_(calculation_targets[k]) for k in keys]
        variable_shapes = [calculation_targets[k].shape for k in keys]
        arrs = [self.get_variable_value(calculation_targets[k]) for k in keys]
        archetype = calculation_targets[keys[0]]
        fill = self.get_fill_variable(archetype, self.alias, archetype.dimensions, self.file_only,
                                      dtype=archetype.dtype)
        fill.units = None

        if not self.file_only:
            arr_fill = self.get_variable_value(fill)

            itrs = [self._iter_conformed_arrays_(crosswalks[idx], variable_shapes[idx], arrs[idx], arr_fill, None)
                    for idx in range(len(crosswalks))]

            for yld in itertools.izip(*itrs):
                for idx in range(len(keys)):
                    locals()[map_vars[keys[idx]]] = yld[idx][0]
                res = eval(expr)
                carr_fill = yld[0][1]
                carr_fill.data[:] = res.data
                carr_fill.mask[:] = res.mask

        self._add_to_collection_({'fill': fill})

    @staticmethod
    def is_multivariate(expr):
        """
        Return ``True`` if ``expr`` is a multivariate string function.

        :param str expr: The string function to evaluate. The function must have an equals sign. The function may
         contain multiple variables aliases. Mathematical operators include standard arithmetic symbols and NumPy
         functions. The list of enabled functions is contained in :attr:~`ocgis.constants.ENABLED_NUMPY_UFUNCS`.
        :returns bool:
        """

        # do not count the output variable name
        expr = expr.split('=')[1]
        # count the number of variable names in the right hand side of the string expression.
        strings = set(re.findall("[A-Za-z0-9_]+", expr))
        strings = set([s for s in strings if re.search("[A-Za-z]", s) is not None])
        strings_left = deepcopy(strings)
        for s in strings:
            if s in constants.ENABLED_NUMPY_UFUNCS:
                strings_left.remove(s)
        # if there are more than one variable alias in the equation, the expression is multivariate
        if len(strings_left) > 1:
            ret = True
        else:
            ret = False
        return ret

    @staticmethod
    def _get_eval_string_(expr, map_vars):
        """
        :param str expr: The string function to evaluate. The function must have an equals sign. The function may
         contain multiple variables aliases. Mathematical operators include standard arithmetic symbols and NumPy
         functions. The list of enabled functions is contained in :attr:~`ocgis.constants.ENABLED_NUMPY_UFUNCS`.
        :param dict map_vars: Maps variable aliases to their output representation in ``expr``.

        >>> map_vars = {'tas':'tas.value',...}

        :returns tuple: A tuple composed of two elements ``(expr, out_variable_name)``.
         * ``expr``: The output string expression to evaluate.
         * ``out_variable_name``: The variable name to assign to the output. This
          is the left-hand side of the expression.
        :raises ValueError:
        """

        # standard arithmetic mathematical operators
        math_set = '[(\-+*/)]'
        # regular expressions use to find the variable aliases
        re_expr_base = ['{1}{0}{1}', '^{0}{1}', '{1}{0}$', '^{0}$', '{1}{0},']
        # attempt to split the expression at the equals sign.
        try:
            out_variable_name, expr = expr.split('=')
        except ValueError:
            msg = 'Unable to parse expression string: "{0}". The equals sign is likely missing.'
            raise ValueError(msg.format(expr))
        # find the variable aliases and function names in the string expression
        strings = set(re.findall("[A-Za-z0-9_]+", expr))
        strings = set([s for s in strings if re.search("[A-Za-z]", s) is not None])
        # "strings" must be entirely composed of enabled numpy functions and the variable aliases originating from the
        # keys in "map_vars"
        for s in strings:
            if s not in constants.ENABLED_NUMPY_UFUNCS and s not in map_vars.keys():
                raise ValueError('Unable to parse expression string: "{0}". '
                                 'Ensure the NumPy functions are enabled and appropriate '
                                 'variables have been requested. The problem string value is "{1}".'.format(expr, s))
        # update the names of the numpy functions to use the module prefix "np"
        for np_func in constants.ENABLED_NUMPY_UFUNCS:
            expr = expr.replace(np_func, 'np.' + np_func)
        # update the variable aliases to match the key-value relationship in "map_vars"
        max_ctr = len(expr)
        for k, v in map_vars.iteritems():
            for r in re_expr_base:
                re_expr = r.format(k, math_set)
                ctr = 0
                while True:
                    def repl(mo):
                        return mo.group(0).replace(k, v)

                    expr = re.sub(re_expr, repl, expr)
                    if re.search(re_expr, expr) is None:
                        break
                    ctr += 1
                    if ctr > max_ctr:
                        break
        return expr, out_variable_name

    def _set_derived_variable_alias_(self, *args, **kwargs):
        pass


class MultivariateEvalFunction(EvalFunction):
    """
    Dummy class to help the software distinguish between univariate and multivariate function string expressions.
    """
    pass
