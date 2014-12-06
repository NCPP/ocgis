from ocgis.interface.base.attributes import Attributes
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
from ocgis.exc import DefinitionValidationError
from ocgis.api.parms.definition import CalcGrouping


_icclim_function_map = {
                        'icclim_TG':{'func':calc_indice.TG_calculation,'meta':slu.TG_setvarattr},
                        'icclim_TN':{'func':calc_indice.TN_calculation,'meta':slu.TN_setvarattr},
                        'icclim_TX':{'func':calc_indice.TX_calculation,'meta':slu.TX_setvarattr},
                        'icclim_SU':{'func':calc_indice.SU_calculation,'meta':slu.SU_setvarattr},
                        'icclim_DTR':{'func':calc_indice.DTR_calculation,'meta':slu.DTR_setvarattr},
                        'icclim_ETR':{'func':calc_indice.ETR_calculation,'meta':slu.ETR_setvarattr},
                        'icclim_TXx':{'func':calc_indice.TXx_calculation,'meta':slu.TXx_setvarattr},
                        'icclim_TXn':{'func':calc_indice.TXn_calculation,'meta':slu.TXn_setvarattr},
                        'icclim_TNx':{'func':calc_indice.TNx_calculation,'meta':slu.TNx_setvarattr},
                        'icclim_TNn':{'func':calc_indice.TNn_calculation,'meta':slu.TNn_setvarattr},
                        'icclim_CSU':{'func':calc_indice.CSU_calculation,'meta':slu.CSU_setvarattr},
                        'icclim_TR':{'func':calc_indice.TR_calculation,'meta':slu.TR_setvarattr},
                        'icclim_FD':{'func':calc_indice.FD_calculation,'meta':slu.FD_setvarattr},
                        'icclim_CFD':{'func':calc_indice.CFD_calculation,'meta':slu.CFD_setvarattr},
                        'icclim_ID':{'func':calc_indice.ID_calculation,'meta':slu.ID_setvarattr},
                        'icclim_HD17':{'func':calc_indice.HD17_calculation,'meta':slu.HD17_setvarattr},
                        'icclim_GD4':{'func':calc_indice.GD4_calculation,'meta':slu.GD4_setvarattr},
                        'icclim_vDTR':{'func':calc_indice.vDTR_calculation,'meta':slu.vDTR_setvarattr},
                        'icclim_RR':{'func':calc_indice.RR_calculation,'meta':slu.RR_setvarattr},
                        'icclim_RR1':{'func':calc_indice.RR1_calculation,'meta':slu.RR1_setvarattr},
                        'icclim_CWD':{'func':calc_indice.CWD_calculation,'meta':slu.CWD_setvarattr},
                        'icclim_SDII':{'func':calc_indice.SDII_calculation,'meta':slu.SDII_setvarattr},
                        'icclim_R10mm':{'func':calc_indice.R10mm_calculation,'meta':slu.R10mm_setvarattr},
                        'icclim_R20mm':{'func':calc_indice.R20mm_calculation,'meta':slu.R20mm_setvarattr},
                        'icclim_RX1day':{'func':calc_indice.RX1day_calculation,'meta':slu.RX1day_setvarattr},
                        'icclim_RX5day':{'func':calc_indice.RX5day_calculation,'meta':slu.RX5day_setvarattr},
                        'icclim_SD':{'func':calc_indice.SD_calculation,'meta':slu.SD_setvarattr},
                        'icclim_SD1':{'func':calc_indice.SD1_calculation,'meta':slu.SD1_setvarattr},
                        'icclim_SD5cm':{'func':calc_indice.SD5cm_calculation,'meta':slu.SD5cm_setvarattr},
                        'icclim_SD50cm':{'func':calc_indice.SD50cm_calculation,'meta':slu.SD50cm_setvarattr},
                        'icclim_CDD':{'func':calc_indice.CDD_calculation,'meta':slu.CDD_setvarattr},
                        }


class NcAttributesSimulator(object):

    def __init__(self, attrs):
        self.attrs = attrs

    def __getattr__(self, name):
        return self.attrs[name]

    def setncattr(self, key, value):
        self.attrs[key] = value
        
        
class AbstractIcclimFunction(object):
    __metaclass__ = abc.ABCMeta
    description = None
    standard_name = 'ECA_indice'
    long_name = ''
    _global_attributes_maintain = ['history']
    _global_attribute_source_name = 'source_data_global_attributes'
    _allowed_temporal_groupings = [('month',),('month','year'),('year',)]
    
    def set_field_metadata(self):
        # we are going to strip the metadata elements and store in a dictionary JSON representation
        
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

        # reorganize the output metadata pushing source global attributes to a new attribute. the old attributes are
        # serialized to a JSON string
        original = deepcopy(self.field.attrs)
        self.field.attrs = OrderedDict()
        sim = NcAttributesSimulator(self.field.attrs)
        sim.attrs[self._global_attribute_source_name] = original
        # copy attributes from the original dataset
        for key in self._global_attributes_maintain:
            value,value_key = _get_value_(key,sim.attrs[self._global_attribute_source_name])
            sim.attrs[value_key] = value
        ref = sim.attrs[self._global_attribute_source_name]
        sim.attrs[self._global_attribute_source_name] = self._get_json_string_(ref)
        
        # update global attributes using ICCLIM functions
        indice_name = self.key.split('_')[1]
        set_globattr.history(sim,self.tgd.grouping,indice_name,[self.field.temporal.value_datetime.min(),self.field.temporal.value_datetime.max()])
        set_globattr.title(sim,indice_name)
        set_globattr.references(sim)
        set_globattr.institution(sim,'Climate impact portal (http://climate4impact.eu)')
        set_globattr.comment(sim,indice_name)

    def set_variable_metadata(self, variable):
        sim = NcAttributesSimulator(variable.attrs)
        _icclim_function_map[self.key]['meta'](sim)
        # update the variable's units from the metadata as this is modified inside ICCLIM
        variable.units = variable.attrs['units']
    
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
    
    @staticmethod
    def validate_icclim(klass,ops):
        should_raise = False
        allowed = [set(_) for _ in klass._allowed_temporal_groupings]
        try:
            if set(ops.calc_grouping) not in allowed:
                should_raise = True
        except TypeError:
            ## this is a seasonal grouping
            should_raise = True
        if should_raise:
            msg = 'The following temporal groupings are supported for ICCLIM: {0}. '.format(klass._allowed_temporal_groupings)
            msg += 'The requested temporal group is: {0}.'.format(ops.calc_grouping)
            raise(DefinitionValidationError(CalcGrouping,msg))
        

class AbstractIcclimUnivariateSetFunction(AbstractIcclimFunction,AbstractUnivariateSetFunction):
    __metaclass__ = abc.ABCMeta
    
    def calculate(self,values):
        return(_icclim_function_map[self.key]['func'](values,values.fill_value))
    
    @classmethod
    def validate(cls,ops):
        cls.validate_icclim(cls,ops)
        super(AbstractIcclimUnivariateSetFunction,cls).validate(ops)
    
    
class AbstractIcclimMultivariateFunction(AbstractIcclimFunction,AbstractMultivariateFunction):
    __metaclass__ = abc.ABCMeta
    
    @classmethod
    def validate(cls,ops):
        cls.validate_icclim(cls,ops)
        super(AbstractIcclimMultivariateFunction,cls).validate(ops)
    
        
class IcclimTG(AbstractIcclimUnivariateSetFunction):
    dtype = constants.np_float
    key = 'icclim_TG'


class IcclimTN(IcclimTG):
    key = 'icclim_TN'


class IcclimTX(IcclimTG):
    key = 'icclim_TX'
    
    
class IcclimTXx(IcclimTG):
    key = 'icclim_TXx'
    
    
class IcclimTXn(IcclimTG):
    key = 'icclim_TXn'


class IcclimTNx(IcclimTG):
    key = 'icclim_TNx'


class IcclimTNn(IcclimTG):
    key = 'icclim_TNn'


class IcclimCSU(AbstractIcclimUnivariateSetFunction):
    dtype = constants.np_int
    key = 'icclim_CSU'


class IcclimTR(IcclimCSU):
    key = 'icclim_TR'


class IcclimFD(IcclimCSU):
    key = 'icclim_FD'


class IcclimCFD(IcclimCSU):
    key = 'icclim_CFD'


class IcclimID(IcclimCSU):
    key = 'icclim_ID'


class IcclimHD17(IcclimTG):
    dtype = constants.np_float
    key = 'icclim_HD17'
    required_units = ['K','kelvin']


class IcclimGD4(IcclimTG):
    dtype = constants.np_float
    key = 'icclim_GD4'
    required_units = ['K','kelvin']


class IcclimSU(IcclimCSU):
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
    
    
class IcclimvDTR(IcclimDTR):
    key = 'icclim_vDTR'


class IcclimRR(IcclimCSU):
    dtype = constants.np_float
    key = 'icclim_RR'


class IcclimRR1(IcclimCSU):
    dtype = constants.np_float
    key = 'icclim_RR1'


class IcclimCWD(IcclimCSU):
    dtype = constants.np_float
    key = 'icclim_CWD'


class IcclimSDII(IcclimCSU):
    dtype = constants.np_float
    key = 'icclim_SDII'


class IcclimR10mm(IcclimCSU):
    dtype = constants.np_float
    key = 'icclim_R10mm'


class IcclimR20mm(IcclimCSU):
    dtype = constants.np_float
    key = 'icclim_R20mm'


class IcclimRX1day(IcclimCSU):
    dtype = constants.np_float
    key = 'icclim_RX1day'


class IcclimRX5day(IcclimCSU):
    dtype = constants.np_float
    key = 'icclim_RX5day'


class IcclimSD(IcclimCSU):
    dtype = constants.np_float
    key = 'icclim_SD'


class IcclimSD1(IcclimCSU):
    dtype = constants.np_float
    key = 'icclim_SD1'


class IcclimSD5(IcclimCSU):
    dtype = constants.np_float
    key = 'icclim_SD5cm'


class IcclimSD50(IcclimCSU):
    dtype = constants.np_float
    key = 'icclim_SD50cm'


class IcclimCDD(IcclimCSU):
    dtype = constants.np_float
    key = 'icclim_CDD'
