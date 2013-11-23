from ocgis.calc import base
import numpy as np


class Divide(base.AbstractMultivariateFunction):
    key = 'divide'
    description = 'Divide arr1 by arr2.'
    required_variables = ['arr1','arr2']
    
    def calculate(self,arr1=None,arr2=None):
        return(arr1/arr2)
    
    
class NaturalLogarithm(base.AbstractUnivariateFunction):
    key = 'ln'
    description = 'Compute the natural logarithm.'
    
    def calculate(self,values):
        return(np.ma.log(values))
