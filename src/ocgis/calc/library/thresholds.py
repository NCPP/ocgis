from ocgis.calc import base
import numpy as np


class Between(base.AbstractUnivariateSetFunction,base.AbstractParameterizedFunction):
    description = 'Count of values falling within the limits lower and upper (inclusive).'
    parms_definition = {'lower':float,'upper':float}
    dtype = np.int32
    key = 'between'
    
    def calculate(self,values,lower=None,upper=None):
        '''
        :param lower: The lower value of the range.
        :type lower: float
        :param upper: The upper value of the range.
        :type upper: float
        '''
        assert(lower <= upper)
        idx = (values >= float(lower))*(values <= float(upper))
        return(np.ma.sum(idx,axis=0))
    
    
class Threshold(base.AbstractUnivariateSetFunction,base.AbstractParameterizedFunction):
    description = 'Count of values where the logical operation returns TRUE.'
    parms_definition = {'threshold':float,'operation':str}
    dtype = np.int32
    key = 'threshold'
    
    def calculate(self,values,threshold=None,operation=None):
        '''
        :param threshold: The threshold value to use for the logical operation.
        :type threshold: float
        :param operation: The logical operation. One of 'gt','gte','lt', or 'lte'.
        :type operation: str
        '''
        
        ## perform requested logical operation
        if operation == 'gt':
            idx = values > threshold
        elif operation == 'lt':
            idx = values < threshold
        elif operation == 'gte':
            idx = values >= threshold
        elif operation == 'lte':
            idx = values <= threshold
        else:
            raise(NotImplementedError('The operation "{0}" was not recognized.'.format(operation)))
        
        ret = np.ma.sum(idx,axis=0)
        return(ret)
        
    def _aggregate_spatial_(self,values,weights):
        return(np.ma.sum(values))
