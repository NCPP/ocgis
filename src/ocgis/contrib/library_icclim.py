import abc
from collections import OrderedDict
from copy import deepcopy
import json
import numpy as np

from icclim.percentile_dict import get_percentile_dict
from icclim import calc_indice, calc_indice_perc
from icclim import set_longname_units as slu
from icclim import set_globattr

from ocgis.calc.base import AbstractUnivariateSetFunction, AbstractMultivariateFunction, AbstractParameterizedFunction
from ocgis import constants
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
                        'icclim_TG10p':{'func':calc_indice_perc.TG10p_calculation,'meta':slu.TG10p_setvarattr},
                        'icclim_TX10p':{'func':calc_indice_perc.TX10p_calculation,'meta':slu.TX10p_setvarattr},
                        'icclim_TN10p':{'func':calc_indice_perc.TN10p_calculation,'meta':slu.TN10p_setvarattr},
                        'icclim_TG90p':{'func':calc_indice_perc.TG90p_calculation,'meta':slu.TG90p_setvarattr},
                        'icclim_TX90p':{'func':calc_indice_perc.TX90p_calculation,'meta':slu.TX90p_setvarattr},
                        'icclim_TN90p':{'func':calc_indice_perc.TN90p_calculation,'meta':slu.TN90p_setvarattr},
                        'icclim_WSDI':{'func':calc_indice_perc.WSDI_calculation,'meta':slu.WSDI_setvarattr},
                        'icclim_CSDI':{'func':calc_indice_perc.CSDI_calculation,'meta':slu.CSDI_setvarattr},
                        'icclim_R75p':{'func':calc_indice_perc.R75p_calculation,'meta':slu.R75p_setvarattr},
                        'icclim_R75TOT':{'func':calc_indice_perc.R75TOT_calculation,'meta':slu.R75TOT_setvarattr},
                        'icclim_R95p':{'func':calc_indice_perc.R95p_calculation,'meta':slu.R95p_setvarattr},
                        'icclim_R95TOT':{'func':calc_indice_perc.R95TOT_calculation,'meta':slu.R95TOT_setvarattr},
                        'icclim_R99p':{'func':calc_indice_perc.R99p_calculation,'meta':slu.R99p_setvarattr},
                        'icclim_R99TOT':{'func':calc_indice_perc.R99TOT_calculation,'meta':slu.R99TOT_setvarattr},
                        'icclim_CD': {'func': calc_indice_perc.CD_calculation, 'meta': slu.CD_setvarattr},
                        'icclim_CW': {'func': calc_indice_perc.CW_calculation, 'meta': slu.CW_setvarattr},
                        'icclim_WD': {'func': calc_indice_perc.WD_calculation, 'meta': slu.WD_setvarattr},
                        'icclim_WW': {'func': calc_indice_perc.WW_calculation, 'meta': slu.WW_setvarattr},
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
        return self._get_icclim_func_()(values, values.fill_value)
    
    @classmethod
    def validate(cls,ops):
        cls.validate_icclim(cls, ops)
        super(AbstractIcclimUnivariateSetFunction, cls).validate(ops)

    def _get_icclim_func_(self):
        return _icclim_function_map[self.key]['func']
    
    
class AbstractIcclimMultivariateFunction(AbstractIcclimFunction,AbstractMultivariateFunction):
    __metaclass__ = abc.ABCMeta
    
    @classmethod
    def validate(cls,ops):
        cls.validate_icclim(cls,ops)
        super(AbstractIcclimMultivariateFunction,cls).validate(ops)


class AbstractIcclimPercentileIndice(AbstractIcclimUnivariateSetFunction, AbstractParameterizedFunction):
    __metaclass__ = abc.ABCMeta
    parms_definition = {'percentile_dict': dict}
    window_width = 5
    only_leap_years = False

    def __init__(self, *args, **kwargs):
        self._storage_percentile_dict = {}
        AbstractIcclimUnivariateSetFunction.__init__(self, *args, **kwargs)

        if self.field is not None:
            assert(self.field.shape[0] == 1)
            assert(self.field.shape[2] == 1)

    @abc.abstractproperty
    def percentile(self):
        """
        The percentile value to use for computing the percentile basis. Value is between 0 and 100.

        :type: int
        """
        pass

    def calculate(self, values, percentile_dict=None):

        # if the percentile dictionary is not provided compute it
        if percentile_dict is None:
            try:
                percentile_dict = self._storage_percentile_dict[self._curr_variable.alias]
            except KeyError:
                variable = self.field.variables[self._curr_variable.alias]
                value = variable.value[0, :, 0, :, :]
                assert(value.ndim == 3)
                percentile_dict = get_percentile_dict(value, self.field.temporal.value_datetime, self.percentile,
                                                      self.window_width, only_leap_years=self.only_leap_years)
                self._storage_percentile_dict[self._curr_variable.alias] = percentile_dict

        dt_arr = self.field.temporal.value_datetime[self._curr_group]
        ret = _icclim_function_map[self.key]['func'](values, dt_arr, percentile_dict, fill_val=values.fill_value)
        return ret

    @staticmethod
    def get_percentile_dict(*args, **kwargs):
        """See :func:`icclim.percentile_dict.get_percentile_dict` documentation."""

        return get_percentile_dict(*args, **kwargs)


class IcclimTG(AbstractIcclimUnivariateSetFunction):
    dtype = constants.NP_FLOAT
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
    dtype = constants.NP_INT
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
    dtype = constants.NP_FLOAT
    key = 'icclim_HD17'
    required_units = ['K','kelvin']


class IcclimGD4(IcclimTG):
    dtype = constants.NP_FLOAT
    key = 'icclim_GD4'
    required_units = ['K','kelvin']


class IcclimSU(IcclimCSU):
    dtype = constants.NP_INT
    key = 'icclim_SU'
    required_units = ['K','kelvin']
    
    
class IcclimDTR(AbstractIcclimMultivariateFunction):
    key = 'icclim_DTR'
    dtype = constants.NP_FLOAT
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
    dtype = constants.NP_FLOAT
    key = 'icclim_RR'


class IcclimRR1(IcclimCSU):
    dtype = constants.NP_FLOAT
    key = 'icclim_RR1'


class IcclimCWD(IcclimCSU):
    dtype = constants.NP_FLOAT
    key = 'icclim_CWD'


class IcclimSDII(IcclimCSU):
    dtype = constants.NP_FLOAT
    key = 'icclim_SDII'


class IcclimR10mm(IcclimCSU):
    dtype = constants.NP_FLOAT
    key = 'icclim_R10mm'


class IcclimR20mm(IcclimCSU):
    dtype = constants.NP_FLOAT
    key = 'icclim_R20mm'


class IcclimRX1day(IcclimCSU):
    dtype = constants.NP_FLOAT
    key = 'icclim_RX1day'


class IcclimRX5day(IcclimCSU):
    dtype = constants.NP_FLOAT
    key = 'icclim_RX5day'


class IcclimSD(IcclimCSU):
    dtype = constants.NP_FLOAT
    key = 'icclim_SD'


class IcclimSD1(IcclimCSU):
    dtype = constants.NP_FLOAT
    key = 'icclim_SD1'


class IcclimSD5(IcclimCSU):
    dtype = constants.NP_FLOAT
    key = 'icclim_SD5cm'


class IcclimSD50(IcclimCSU):
    dtype = constants.NP_FLOAT
    key = 'icclim_SD50cm'


class IcclimCDD(IcclimCSU):
    dtype = constants.NP_FLOAT
    key = 'icclim_CDD'


class IcclimTG10p(AbstractIcclimPercentileIndice):
    key = 'icclim_TG10p'
    dtype = constants.NP_FLOAT
    percentile = 10


class IcclimTX10p(AbstractIcclimPercentileIndice):
    key = 'icclim_TX10p'
    dtype = constants.NP_FLOAT
    percentile = 10


class IcclimTN10p(AbstractIcclimPercentileIndice):
    key = 'icclim_TN10p'
    dtype = constants.NP_FLOAT
    percentile = 10


class IcclimTG90p(AbstractIcclimPercentileIndice):
    key = 'icclim_TG90p'
    dtype = constants.NP_FLOAT
    percentile = 90


class IcclimTX90p(AbstractIcclimPercentileIndice):
    key = 'icclim_TX90p'
    dtype = constants.NP_FLOAT
    percentile = 10


class IcclimTN90p(AbstractIcclimPercentileIndice):
    key = 'icclim_TN90p'
    dtype = constants.NP_FLOAT
    percentile = 10


class IcclimWSDI(AbstractIcclimPercentileIndice):
    key = 'icclim_WSDI'
    dtype = constants.NP_FLOAT
    percentile = 90


class IcclimCSDI(AbstractIcclimPercentileIndice):
    key = 'icclim_CSDI'
    dtype = constants.NP_FLOAT
    percentile = 10


class IcclimR75p(AbstractIcclimPercentileIndice):
    key = 'icclim_R75p'
    dtype = constants.NP_FLOAT
    percentile = 75


class IcclimR75TOT(AbstractIcclimPercentileIndice):
    key = 'icclim_R75TOT'
    dtype = constants.NP_FLOAT
    percentile = 75


class IcclimR95p(AbstractIcclimPercentileIndice):
    key = 'icclim_R95p'
    dtype = constants.NP_FLOAT
    percentile = 95


class IcclimR95TOT(AbstractIcclimPercentileIndice):
    key = 'icclim_R95TOT'
    dtype = constants.NP_FLOAT
    percentile = 95


class IcclimR99p(AbstractIcclimPercentileIndice):
    key = 'icclim_R99p'
    dtype = constants.NP_FLOAT
    percentile = 99


class IcclimR99TOT(AbstractIcclimPercentileIndice):
    key = 'icclim_R99TOT'
    dtype = constants.NP_FLOAT
    percentile = 99


class IcclimCD(AbstractIcclimMultivariateFunction, AbstractParameterizedFunction):
    key = 'icclim_CD'
    dtype = constants.NP_FLOAT
    required_variables = ['tas', 'pr']
    time_aggregation_external = False
    parms_definition = {'tas_25th_percentile_dict': dict, 'pr_25th_percentile_dict': dict}
    window_width = 5
    percentile_tas = 25
    percentile_pr = 25

    def __init__(self, *args, **kwargs):
        self._storage_percentile_dict = {}
        super(IcclimCD, self).__init__(*args, **kwargs)

    def calculate(self, tas=None, pr=None, tas_25th_percentile_dict=None, pr_25th_percentile_dict=None):
        """
        See documentation for :func:`icclim.calc_indice_perc.CD_calculation`.
        """

        return self._calculate_(tas=tas, pr=pr, tas_percentile_dict=tas_25th_percentile_dict,
                                pr_percentile_dict=pr_25th_percentile_dict)

    def _calculate_(self, tas=None, pr=None, tas_percentile_dict=None, pr_percentile_dict=None):
        """
        Allows subclasses to overload parameter definitions for `calculate`.
        """

        assert(tas.ndim == 3)
        assert(pr.ndim == 3)

        try:
            dt_arr = self.field.temporal.value_datetime[self._curr_group]
        except AttributeError:
            if not hasattr(self, '_curr_group'):
                dt_arr = self.field.temporal.value_datetime
            else:
                raise

        if tas_percentile_dict is None:
            try:
                tas_percentile_dict = self._storage_percentile_dict['tas']
                pr_percentile_dict = self._storage_percentile_dict['pr']
            except KeyError:
                dt_arr_perc = self.field.temporal.value_datetime
                alias_tas = self.parms['tas']
                alias_pr = self.parms['pr']
                t_arr_perc = self.field.variables[alias_tas].value.squeeze()
                p_arr_perc = self.field.variables[alias_pr].value.squeeze()
                tas_percentile_dict = get_percentile_dict(t_arr_perc, dt_arr_perc, self.percentile_tas, self.window_width)
                pr_percentile_dict = get_percentile_dict(p_arr_perc, dt_arr_perc, self.percentile_pr, self.window_width)
                self._storage_percentile_dict['tas'] = tas_percentile_dict
                self._storage_percentile_dict['pr'] = pr_percentile_dict

        ret = _icclim_function_map[self.key]['func'](tas, tas_percentile_dict, pr, pr_percentile_dict, dt_arr,
                                                     fill_val=tas.fill_value)
        # convert output to a masked array
        ret_mask = ret == tas.fill_value
        ret = np.ma.array(ret, mask=ret_mask, fill_value=tas.fill_value)
        return ret


class IcclimCW(IcclimCD):
    key = 'icclim_CW'
    parms_definition = {'tas_25th_percentile_dict': dict, 'pr_75th_percentile_dict': dict}
    percentile_tas = 25
    percentile_pr = 75

    def calculate(self, tas=None, pr=None, tas_25th_percentile_dict=None, pr_75th_percentile_dict=None):
        """
        See documentation for :func:`icclim.calc_indice_perc.CW_calculation`.
        """

        return self._calculate_(tas=tas, pr=pr, tas_percentile_dict=tas_25th_percentile_dict,
                                pr_percentile_dict=pr_75th_percentile_dict)


class IcclimWD(IcclimCD):
    key = 'icclim_WD'
    parms_definition = {'tas_75th_percentile_dict': dict, 'pr_25th_percentile_dict': dict}
    percentile_tas = 75
    percentile_pr = 25

    def calculate(self, tas=None, pr=None, tas_75th_percentile_dict=None, pr_25th_percentile_dict=None):
        """
        See documentation for :func:`icclim.calc_indice_perc.WD_calculation`.
        """

        return self._calculate_(tas=tas, pr=pr, tas_percentile_dict=tas_75th_percentile_dict,
                                pr_percentile_dict=pr_25th_percentile_dict)


class IcclimWW(IcclimCD):
    key = 'icclim_WW'
    parms_definition = {'tas_75th_percentile_dict': dict, 'pr_75th_percentile_dict': dict}
    percentile_tas = 75
    percentile_pr = 75

    def calculate(self, tas=None, pr=None, tas_75th_percentile_dict=None, pr_75th_percentile_dict=None):
        """
        See documentation for :func:`icclim.calc_indice_perc.WW_calculation`.
        """

        return self._calculate_(tas=tas, pr=pr, tas_percentile_dict=tas_75th_percentile_dict,
                                pr_percentile_dict=pr_75th_percentile_dict)
