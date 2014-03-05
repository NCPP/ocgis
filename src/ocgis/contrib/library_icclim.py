from ocgis.calc.base import AbstractUnivariateSetFunction
import icclim
from ocgis import constants
from icclim import set_longname_units as slu
from icclim import set_globattr
import abc


_icclim_function_map = {
                        'icclim_TG':{'func':icclim.TG_calculation,'meta':slu.TG_setvarattr},
                        'icclim_SU':{'func':icclim.SU_calculation,'meta':slu.SU_setvarattr}
                        }


class NcVariableSimulator(object):
    
    def __init__(self,meta):
        self.meta = meta
        
    def setncattr(self,key,value):
        self.meta['attrs'][key] = value
        
        
class NcDatasetSimulator(NcVariableSimulator):
    
    def __getattr__(self,name):
        return(self.meta['dataset'][name])
    
    def setncattr(self,key,value):
        self.meta['dataset'][key] = value


class AbstractIcclimFunction(AbstractUnivariateSetFunction):
    __metaclass__ = abc.ABCMeta
    description = None
    standard_name = 'ECA_indice'
    
    def calculate(self,values):
        return(_icclim_function_map[self.key]['func'](values,values.fill_value))
    
    def set_field_metadata(self):
        sim = NcDatasetSimulator(self.field.meta)
        indice_name = self.key.split('_')[1]
        set_globattr.set_history_globattr(sim,
                                          self.tgd.grouping,
                                          indice_name,
                                          [self.field.temporal.value_datetime.min(),
                                           self.field.temporal.value_datetime.max()])
        set_globattr.set_title_globattr(sim,indice_name)
    
    def set_variable_metadata(self,variable):
        sim = NcVariableSimulator(variable.meta)
        _icclim_function_map[self.key]['meta'](sim)
        
        
class IcclimTG(AbstractIcclimFunction):
    dtype = constants.np_float
    key = 'icclim_TG'


class IcclimSU(AbstractIcclimFunction):
    dtype = constants.np_int
    key = 'icclim_SU'
    required_units = ['K','kelvin']
