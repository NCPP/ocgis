import abc
import json
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import six
from icclim import calc_indice
from icclim import set_globattr
from icclim import set_longname_units as slu
from icclim.calc_percentiles import get_percentile_dict, get_percentile_arr
from numpy.core.multiarray import ndarray

from ocgis.calc.base import AbstractUnivariateSetFunction, AbstractMultivariateFunction, AbstractParameterizedFunction
from ocgis.calc.temporal_groups import SeasonalTemporalGroup

_icclim_function_map = {
    'icclim_TG': {'func': calc_indice.TG_calculation, 'meta': slu.TG_setvarattr},
    'icclim_TN': {'func': calc_indice.TN_calculation, 'meta': slu.TN_setvarattr},
    'icclim_TX': {'func': calc_indice.TX_calculation, 'meta': slu.TX_setvarattr},
    'icclim_SU': {'func': calc_indice.SU_calculation, 'meta': slu.SU_setvarattr},
    'icclim_DTR': {'func': calc_indice.DTR_calculation, 'meta': slu.DTR_setvarattr},
    'icclim_ETR': {'func': calc_indice.ETR_calculation, 'meta': slu.ETR_setvarattr},
    'icclim_TXx': {'func': calc_indice.TXx_calculation, 'meta': slu.TXx_setvarattr},
    'icclim_TXn': {'func': calc_indice.TXn_calculation, 'meta': slu.TXn_setvarattr},
    'icclim_TNx': {'func': calc_indice.TNx_calculation, 'meta': slu.TNx_setvarattr},
    'icclim_TNn': {'func': calc_indice.TNn_calculation, 'meta': slu.TNn_setvarattr},
    'icclim_CSU': {'func': calc_indice.CSU_calculation, 'meta': slu.CSU_setvarattr},
    'icclim_TR': {'func': calc_indice.TR_calculation, 'meta': slu.TR_setvarattr},
    'icclim_FD': {'func': calc_indice.FD_calculation, 'meta': slu.FD_setvarattr},
    'icclim_CFD': {'func': calc_indice.CFD_calculation, 'meta': slu.CFD_setvarattr},
    'icclim_ID': {'func': calc_indice.ID_calculation, 'meta': slu.ID_setvarattr},
    'icclim_HD17': {'func': calc_indice.HD17_calculation, 'meta': slu.HD17_setvarattr},
    'icclim_GD4': {'func': calc_indice.GD4_calculation, 'meta': slu.GD4_setvarattr},
    'icclim_vDTR': {'func': calc_indice.vDTR_calculation, 'meta': slu.vDTR_setvarattr},
    'icclim_PRCPTOT': {'func': calc_indice.PRCPTOT_calculation, 'meta': slu.PRCPTOT_setvarattr},
    'icclim_RR1': {'func': calc_indice.RR1_calculation, 'meta': slu.RR1_setvarattr},
    'icclim_CWD': {'func': calc_indice.CWD_calculation, 'meta': slu.CWD_setvarattr},
    'icclim_SDII': {'func': calc_indice.SDII_calculation, 'meta': slu.SDII_setvarattr},
    'icclim_R10mm': {'func': calc_indice.R10mm_calculation, 'meta': slu.R10mm_setvarattr},
    'icclim_R20mm': {'func': calc_indice.R20mm_calculation, 'meta': slu.R20mm_setvarattr},
    'icclim_RX1day': {'func': calc_indice.RX1day_calculation, 'meta': slu.RX1day_setvarattr},
    'icclim_RX5day': {'func': calc_indice.RX5day_calculation, 'meta': slu.RX5day_setvarattr},
    'icclim_SD': {'func': calc_indice.SD_calculation, 'meta': slu.SD_setvarattr},
    'icclim_SD1': {'func': calc_indice.SD1_calculation, 'meta': slu.SD1_setvarattr},
    'icclim_SD5cm': {'func': calc_indice.SD5cm_calculation, 'meta': slu.SD5cm_setvarattr},
    'icclim_SD50cm': {'func': calc_indice.SD50cm_calculation, 'meta': slu.SD50cm_setvarattr},
    'icclim_CDD': {'func': calc_indice.CDD_calculation, 'meta': slu.CDD_setvarattr},
    'icclim_TG10p': {'func': calc_indice.TG10p_calculation, 'meta': slu.TG10p_setvarattr},
    'icclim_TX10p': {'func': calc_indice.TX10p_calculation, 'meta': slu.TX10p_setvarattr},
    'icclim_TN10p': {'func': calc_indice.TN10p_calculation, 'meta': slu.TN10p_setvarattr},
    'icclim_TG90p': {'func': calc_indice.TG90p_calculation, 'meta': slu.TG90p_setvarattr},
    'icclim_TX90p': {'func': calc_indice.TX90p_calculation, 'meta': slu.TX90p_setvarattr},
    'icclim_TN90p': {'func': calc_indice.TN90p_calculation, 'meta': slu.TN90p_setvarattr},
    'icclim_WSDI': {'func': calc_indice.WSDI_calculation, 'meta': slu.WSDI_setvarattr},
    'icclim_CSDI': {'func': calc_indice.CSDI_calculation, 'meta': slu.CSDI_setvarattr},
    'icclim_R75p': {'func': calc_indice.R75p_calculation, 'meta': slu.R75p_setvarattr},
    'icclim_R75pTOT': {'func': calc_indice.R75pTOT_calculation, 'meta': slu.R75pTOT_setvarattr},
    'icclim_R95p': {'func': calc_indice.R95p_calculation, 'meta': slu.R95p_setvarattr},
    'icclim_R95pTOT': {'func': calc_indice.R95pTOT_calculation, 'meta': slu.R95pTOT_setvarattr},
    'icclim_R99p': {'func': calc_indice.R99p_calculation, 'meta': slu.R99p_setvarattr},
    'icclim_R99pTOT': {'func': calc_indice.R99pTOT_calculation, 'meta': slu.R99pTOT_setvarattr},
    # 'icclim_CD': {'func': calc_indice.CD_calculation, 'meta': slu.CD_setvarattr},
    # 'icclim_CW': {'func': calc_indice.CW_calculation, 'meta': slu.CW_setvarattr},
    # 'icclim_WD': {'func': calc_indice.WD_calculation, 'meta': slu.WD_setvarattr},
    # 'icclim_WW': {'func': calc_indice.WW_calculation, 'meta': slu.WW_setvarattr},
}


class NcAttributesSimulator(object):
    def __init__(self, attrs):
        self.attrs = attrs

    def __getattr__(self, name):
        return self.attrs[name]

    def setncattr(self, key, value):
        self.attrs[key] = value


@six.add_metaclass(abc.ABCMeta)
class AbstractIcclimFunction(object):
    description = None
    standard_name = 'ECA_indice'
    long_name = ''
    _global_attributes_maintain = ['history']
    _global_attribute_source_name = 'source_data_global_attributes'

    def set_field_metadata(self):
        # we are going to strip the metadata elements and store in a dictionary JSON representation

        def _get_value_(key, target):
            try:
                ret = target[key]
                ret_key = key
                return ret, ret_key
            except KeyError:
                for method in ['lower', 'upper', 'title']:
                    try:
                        ret_key = getattr(str, method)(key)
                        ret = target[ret_key]
                        return ret, ret_key
                    except KeyError:
                        pass
            return '', key

        # reorganize the output metadata pushing source global attributes to a new attribute. the old attributes are
        # serialized to a JSON string
        original = deepcopy(self.field.attrs)
        self.field.attrs = OrderedDict()
        sim = NcAttributesSimulator(self.field.attrs)
        sim.attrs[self._global_attribute_source_name] = original
        # copy attributes from the original dataset
        for key in self._global_attributes_maintain:
            value, value_key = _get_value_(key, sim.attrs[self._global_attribute_source_name])
            sim.attrs[value_key] = value
        ref = sim.attrs[self._global_attribute_source_name]
        sim.attrs[self._global_attribute_source_name] = self._get_json_string_(ref)

        # update global attributes using ICCLIM functions
        indice_name = self.key.split('_')[1]

        # Find the minimum and maximum numeric times using those indices to extract the datetime value min and max time
        # ranges.
        # time_range = [self.field.temporal.value_datetime.min(), self.field.temporal.value_datetime.max()]
        min_idx = np.argmin(self.field.temporal.value_numtime)
        max_idx = np.argmax(self.field.temporal.value_numtime)
        time_range = [self.field.temporal.value_datetime[min_idx], self.field.temporal.value_datetime[max_idx]]

        args = [sim, self.tgd.grouping, indice_name, time_range]
        try:
            set_globattr.history(*args)
        except TypeError:
            # temporal grouping is likely a season. convert to a season object and try again
            args[1] = SeasonalTemporalGroup(self.tgd.grouping)
            set_globattr.history(*args)
        set_globattr.title(sim, indice_name)
        set_globattr.references(sim)
        set_globattr.institution(sim, 'Climate impact portal (http://climate4impact.eu)')
        set_globattr.comment(sim, indice_name)

    def set_variable_metadata(self, variable):
        sim = NcAttributesSimulator(variable.attrs)
        _icclim_function_map[self.key]['meta'](sim)
        # update the variable's units from the metadata as this is modified inside ICCLIM
        variable.units = variable.attrs['units']

    @classmethod
    def validate_icclim(cls, ops):
        """
        :type ops: :class:`ocgis.OcgOperations`
        """
        pass

    @staticmethod
    def _get_json_string_(dct):
        """
        Prepare a dictionary for conversion to JSON. The serializer does not understand NumPy types so those must be
        converted to native Python types first.
        """

        dct = deepcopy(dct)
        for k, v in dct.items():
            try:
                v = v.tolist()
            except AttributeError:
                pass
            dct[k] = v
        return json.dumps(dct)


@six.add_metaclass(abc.ABCMeta)
class AbstractIcclimUnivariateSetFunction(AbstractIcclimFunction, AbstractUnivariateSetFunction):
    def calculate(self, values):
        return self._get_icclim_func_()(values, values.fill_value)

    @classmethod
    def validate(cls, ops):
        cls.validate_icclim(ops)
        super(AbstractIcclimUnivariateSetFunction, cls).validate(ops)

    def _get_icclim_func_(self):
        return _icclim_function_map[self.key]['func']


@six.add_metaclass(abc.ABCMeta)
class AbstractIcclimMultivariateFunction(AbstractIcclimFunction, AbstractMultivariateFunction):
    @classmethod
    def validate(cls, ops):
        cls.validate_icclim(ops)
        super(AbstractIcclimMultivariateFunction, cls).validate(ops)


@six.add_metaclass(abc.ABCMeta)
class AbstractIcclimPercentileIndice(AbstractIcclimUnivariateSetFunction, AbstractParameterizedFunction):
    window_width = 5

    def __init__(self, *args, **kwargs):
        self._storage_percentile = {}
        AbstractIcclimUnivariateSetFunction.__init__(self, *args, **kwargs)

        # if self.field is not None:
        #     assert self.field.shape[0] == 1
        #     assert self.field.shape[2] == 1

    @abc.abstractproperty
    def percentile(self):
        """
        The percentile value to use for computing the percentile basis. Value is between 0 and 100.

        :type: int
        """

    def calculate(self, values, percentile_basis=None):

        if percentile_basis is None:
            try:
                # Make an attempt to find the previous computation. We do not want to recompute the basis for every time
                # grouping.
                percentile_basis = self._storage_percentile[self._curr_variable.name]
            except KeyError:
                value = self._current_conformed_array[0, :, 0, :, :]
                assert value.ndim == 3
                percentile_basis = self._get_percentile_basis_(value)
                self._storage_percentile[self._curr_variable.name] = percentile_basis

        ret = self._get_icclim_function_return_(values, percentile_basis)
        return ret

    @abc.abstractmethod
    def _get_icclim_function_return_(self, values, percentile_basis):
        """Execute the mapped ICCLIM function and return the value."""

    @abc.abstractmethod
    def _get_percentile_basis_(self, value):
        """Return the percentile basis for the subclass."""


@six.add_metaclass(abc.ABCMeta)
class AbstractIcclimPercentileDictionaryIndice(AbstractIcclimPercentileIndice):
    parms_definition = {'percentile_dict': dict}
    only_leap_years = False

    def calculate(self, values, percentile_dict=None):
        return AbstractIcclimPercentileIndice.calculate(self, values, percentile_basis=percentile_dict)

    @staticmethod
    def get_percentile_dict(*args, **kwargs):
        """See :func:`icclim.percentile_dict.get_percentile_dict` documentation."""
        return get_percentile_dict(*args, **kwargs)

    def _get_icclim_function_return_(self, values, percentile_basis):
        dt_arr = self.field.temporal.value_datetime[self._curr_group]
        ret = _icclim_function_map[self.key]['func'](values, dt_arr, percentile_basis, fill_val=values.fill_value)
        return ret

    def _get_percentile_basis_(self, value):
        percentile_basis = get_percentile_dict(value, self.field.temporal.value_datetime, self.percentile,
                                               self.window_width, self.field.time.calendar, self.field.time.units,
                                               only_leap_years=self.only_leap_years)
        return percentile_basis


@six.add_metaclass(abc.ABCMeta)
class AbstractIcclimPercentileArrayIndice(AbstractIcclimPercentileIndice):
    parms_definition = {'percentile_arr': ndarray}

    def calculate(self, values, percentile_arr=None):
        ret = AbstractIcclimPercentileIndice.calculate(self, values, percentile_basis=percentile_arr)
        return ret

    @staticmethod
    def get_percentile_arr(*args, **kwargs):
        """See :func:`icclim.percentile_dict.get_percentile_arr` documentation."""
        return get_percentile_arr(*args, **kwargs)

    def _get_icclim_function_return_(self, values, percentile_basis):
        ret = _icclim_function_map[self.key]['func'](values, percentile_basis, fill_val=values.fill_value)
        return ret

    def _get_percentile_basis_(self, value):
        assert value.ndim == 3
        percentile_basis = get_percentile_arr(value, self.percentile, self.window_width, fill_val=value.fill_value)
        return percentile_basis


class IcclimTG(AbstractIcclimUnivariateSetFunction):
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
    dtype_default = 'int'
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
    key = 'icclim_HD17'
    required_units = ['K', 'kelvin']


class IcclimGD4(IcclimTG):
    key = 'icclim_GD4'
    required_units = ['K', 'kelvin']


class IcclimSU(IcclimCSU):
    dtype_default = 'int'
    key = 'icclim_SU'
    required_units = ['K', 'kelvin']


class IcclimDTR(AbstractIcclimMultivariateFunction):
    key = 'icclim_DTR'
    required_variables = ['tasmin', 'tasmax']
    time_aggregation_external = False

    def calculate(self, tasmax=None, tasmin=None):
        ret = _icclim_function_map[self.key]['func'](tasmax, tasmin, tasmax.fill_value, tasmin.fill_value)
        # convert output to a masked array
        ret_mask = ret == tasmax.fill_value
        ret = np.ma.array(ret, mask=ret_mask, fill_value=tasmax.fill_value)
        return ret


class IcclimETR(IcclimDTR):
    key = 'icclim_ETR'


class IcclimvDTR(IcclimDTR):
    key = 'icclim_vDTR'


class IcclimPRCPTOT(IcclimCSU):
    key = 'icclim_PRCPTOT'


class IcclimRR1(IcclimCSU):
    key = 'icclim_RR1'


class IcclimCWD(IcclimCSU):
    key = 'icclim_CWD'


class IcclimSDII(IcclimCSU):
    key = 'icclim_SDII'


class IcclimR10mm(IcclimCSU):
    key = 'icclim_R10mm'


class IcclimR20mm(IcclimCSU):
    key = 'icclim_R20mm'


class IcclimRX1day(IcclimCSU):
    key = 'icclim_RX1day'


class IcclimRX5day(IcclimCSU):
    key = 'icclim_RX5day'


class IcclimSD(IcclimCSU):
    key = 'icclim_SD'


class IcclimSD1(IcclimCSU):
    key = 'icclim_SD1'


class IcclimSD5(IcclimCSU):
    key = 'icclim_SD5cm'


class IcclimSD50(IcclimCSU):
    key = 'icclim_SD50cm'


class IcclimCDD(IcclimCSU):
    key = 'icclim_CDD'


class IcclimTG10p(AbstractIcclimPercentileDictionaryIndice):
    key = 'icclim_TG10p'

    percentile = 10


class IcclimTX10p(AbstractIcclimPercentileDictionaryIndice):
    key = 'icclim_TX10p'

    percentile = 10


class IcclimTN10p(AbstractIcclimPercentileDictionaryIndice):
    key = 'icclim_TN10p'

    percentile = 10


class IcclimTG90p(AbstractIcclimPercentileDictionaryIndice):
    key = 'icclim_TG90p'

    percentile = 90


class IcclimTX90p(AbstractIcclimPercentileDictionaryIndice):
    key = 'icclim_TX90p'

    percentile = 90


class IcclimTN90p(AbstractIcclimPercentileDictionaryIndice):
    key = 'icclim_TN90p'

    percentile = 90


class IcclimWSDI(AbstractIcclimPercentileDictionaryIndice):
    key = 'icclim_WSDI'

    percentile = 90


class IcclimCSDI(AbstractIcclimPercentileDictionaryIndice):
    key = 'icclim_CSDI'

    percentile = 10


class IcclimR75p(AbstractIcclimPercentileArrayIndice):
    key = 'icclim_R75p'

    percentile = 75


class IcclimR75pTOT(AbstractIcclimPercentileArrayIndice):
    key = 'icclim_R75pTOT'

    percentile = 75

    def set_variable_metadata(self, variable):
        super(IcclimR75pTOT, self).set_variable_metadata(variable)
        variable.units = 'mm/day'


class IcclimR95p(AbstractIcclimPercentileArrayIndice):
    key = 'icclim_R95p'

    percentile = 95


class IcclimR95pTOT(IcclimR75pTOT):
    key = 'icclim_R95pTOT'

    percentile = 95


class IcclimR99p(AbstractIcclimPercentileArrayIndice):
    key = 'icclim_R99p'

    percentile = 99


class IcclimR99pTOT(IcclimR75pTOT):
    key = 'icclim_R99pTOT'

    percentile = 99
