from ocgis.calc import base
import numpy as np
from ocgis import constants


class Divide(base.AbstractMultivariateFunction):
    key = 'divide'
    description = 'Divide arr1 by arr2.'
    required_variables = ['arr1','arr2']
    dtype = constants.np_float
    
    def calculate(self,arr1=None,arr2=None):
        return(arr1/arr2)
    
    
class NaturalLogarithm(base.AbstractUnivariateFunction):
    key = 'ln'
    description = 'Compute the natural logarithm.'
    dtype = constants.np_float
    
    def calculate(self,values):
        return(np.ma.log(values))
    
    def get_output_units(self,*args,**kwds):
        return(None)


class Sum(base.AbstractUnivariateSetFunction):
    key = 'sum'
    description = 'Compute the algebraic sum of a series.'
    dtype = constants.np_float

    def calculate(self, values):
        return np.ma.sum(values, axis=0)