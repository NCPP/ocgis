from ocgis.calc.base import AbstractUnivariateSetFunction,\
    AbstractMultivariateFunction
from ocgis import constants
from icclim import calc_indice
from icclim import set_longname_units as slu
from icclim import set_globattr
import abc
import numpy as np
from collections import OrderedDict
from copy import deepcopy
import json


_icclim_function_map = {
                        'icclim_TG':{'func':calc_indice.TG_calculation,'meta':slu.TG_setvarattr},
                        'icclim_TN':{'func':calc_indice.TN_calculation,'meta':slu.TN_setvarattr},
                        'icclim_TX':{'func':calc_indice.TX_calculation,'meta':slu.TX_setvarattr},
                        'icclim_SU':{'func':calc_indice.SU_calculation,'meta':slu.SU_setvarattr},
                        'icclim_DTR':{'func':calc_indice.DTR_calculation,'meta':slu.DTR_setvarattr},
                        'icclim_ETR':{'func':calc_indice.ETR_calculation,'meta':slu.ETR_setvarattr}
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
        
        
class AbstractIcclimFunction(object):
    __metaclass__ = abc.ABCMeta
    description = None
    standard_name = 'ECA_indice'
    _global_attributes_maintain = ['history']
    _global_attribute_source_name = 'source_data_global_attributes'
    
    def set_field_metadata(self):
        sim = NcDatasetSimulator(self.field.meta)
        
        ## we are going to strip the metadata elements and store in a dictionary
        ## JSON representation
        
        def _get_value_(key,target):
            try:
                ret = target[key]
                ret_key = key
                return(ret,ret_key)
            except KeyError:
                for method in ['lower','upper','title']:
                    try:
                        ret_key = getattr(str,method)(key)
                        ret = target[ret_key]
                        return(ret,ret_key)
                    except KeyError:
                        pass
            return('',key)
        
        ## reorganize the output metadata pushing source global attributes to a
        ## new attribute. the old attributes are serialized to a JSON string
        original = deepcopy(sim.meta['dataset'])
        sim.meta['dataset'] = OrderedDict()
        sim.meta['dataset'][self._global_attribute_source_name] = original
        ## copy attributes from the original dataset
        for key in self._global_attributes_maintain:
            value,value_key = _get_value_(key,sim.meta['dataset'][self._global_attribute_source_name])
            sim.meta['dataset'][value_key] = value
        ref = sim.meta['dataset'][self._global_attribute_source_name]
        sim.meta['dataset'][self._global_attribute_source_name] = self._get_json_string_(ref)
        
        ## update global attributes using ICCLIM functions
        indice_name = self.key.split('_')[1]
        set_globattr.history(sim,
                             self.tgd.grouping,
                             indice_name,
                             [self.field.temporal.value_datetime.min(),
                             self.field.temporal.value_datetime.max()])
        set_globattr.title(sim,indice_name)
        set_globattr.references(sim)
        set_globattr.institution(sim,'Climate impact portal (http://climate4impact.eu)')
        set_globattr.comment(sim,indice_name)
    
    def set_variable_metadata(self,variable):
        sim = NcVariableSimulator(variable.meta)
        _icclim_function_map[self.key]['meta'](sim)
        ## update the variable's units from the metadata as this is modified
        ## inside ICCLIM
        variable.units = variable.meta['attrs']['units']
    
    @staticmethod
    def _get_json_string_(dct):
        '''
        Prepare a dictionary for conversion to JSON. The serializer does not
        understand NumPy types so those must be converted to native Python types
        first.
        '''
        dct = deepcopy(dct)
        for k,v in dct.iteritems():
            try:
                v = v.tolist()
            except AttributeError:
                pass
            dct[k] = v
        return(json.dumps(dct))
        

class AbstractIcclimUnivariateSetFunction(AbstractIcclimFunction,AbstractUnivariateSetFunction):
    __metaclass__ = abc.ABCMeta
    
    def calculate(self,values):
        return(_icclim_function_map[self.key]['func'](values,values.fill_value))
    
    
class AbstractIcclimMultivariateFunction(AbstractIcclimFunction,AbstractMultivariateFunction):
    __metaclass__ = abc.ABCMeta
    
        
class IcclimTG(AbstractIcclimUnivariateSetFunction):
    dtype = constants.np_float
    key = 'icclim_TG'


class IcclimTN(IcclimTG):
    key = 'icclim_TN'


class IcclimTX(IcclimTG):
    key = 'icclim_TX'


class IcclimSU(AbstractIcclimUnivariateSetFunction):
    dtype = constants.np_int
    key = 'icclim_SU'
    required_units = ['K','kelvin']
    
    
class IcclimDTR(AbstractIcclimMultivariateFunction):
    key = 'icclim_DTR'
    dtype = constants.np_float
    required_variables = ['tasmin','tasmax']
    time_aggregation_external = False
    
    def calculate(self,tasmax=None,tasmin=None):
        ret = _icclim_function_map[self.key]['func'](tasmax,tasmin,tasmax.fill_value,tasmin.fill_value)
        ## convert output to a masked array
        ret_mask = ret == tasmax.fill_value
        ret = np.ma.array(ret,mask=ret_mask,fill_value=tasmax.fill_value)
        return(ret)
    

class IcclimETR(IcclimDTR):
    key = 'icclim_ETR'
