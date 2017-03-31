import json
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from unittest import SkipTest

import ocgis
from ocgis import TemporalVariable
from ocgis.base import orphaned
from ocgis.calc.library.register import FunctionRegistry, register_icclim
from ocgis.calc.library.statistics import Mean
from ocgis.calc.library.thresholds import Threshold
from ocgis.calc.temporal_groups import SeasonalTemporalGroup
from ocgis.constants import TagNames
from ocgis.contrib import library_icclim
from ocgis.contrib.library_icclim import IcclimTG, IcclimSU, AbstractIcclimFunction, IcclimDTR, IcclimETR, IcclimTN, \
    IcclimTX, AbstractIcclimUnivariateSetFunction, AbstractIcclimMultivariateFunction, IcclimTG10p, \
    AbstractIcclimPercentileArrayIndice, IcclimR75pTOT
from ocgis.exc import DefinitionValidationError, UnitsValidationError
from ocgis.ops.core import OcgOperations
from ocgis.ops.parms.definition import Calc, CalcGrouping
from ocgis.test import strings
from ocgis.test.base import TestBase, nc_scope, attr
from ocgis.util.helpers import itersubclasses
from ocgis.util.large_array import compute
from ocgis.util.units import get_units_object, get_are_units_equivalent
from ocgis.variable.base import VariableCollection


class FakeAbstractIcclimFunction(AbstractIcclimFunction):
    key = 'icclim_fillme'

    def __init__(self, field, tgd):
        self.field = field
        self.tgd = tgd


@attr('icclim')
class TestAbstractIcclimFunction(TestBase):
    def setUp(self):
        FakeAbstractIcclimFunction.key = 'icclim_TG'
        super(TestAbstractIcclimFunction, self).setUp()

    def tearDown(self):
        FakeAbstractIcclimFunction.key = 'icclim_fillme'
        super(TestAbstractIcclimFunction, self).tearDown()

    def get(self, grouping=None):
        field = self.get_field()
        temporal = TemporalVariable(value=self.get_time_series(datetime(2000, 1, 1), datetime(2001, 12, 31)),
                                    dimensions='time')
        grouping = grouping or [[12, 1, 2]]
        tgd = temporal.get_grouping(grouping)
        aa = FakeAbstractIcclimFunction(field, tgd)
        return aa

    def test_init(self):
        f = self.get()
        self.assertIsInstance(f, AbstractIcclimFunction)

    def test_set_field_metadata(self):
        # test with a seasonal grouping
        aa = self.get()
        aa.set_field_metadata()
        self.assertIn(SeasonalTemporalGroup(aa.tgd.grouping).icclim_mode, aa.field.attrs['history'])

        # test with a day calculation grouping
        aa = self.get(grouping=['day'])
        aa.set_field_metadata()
        self.assertIn(str(['day']), aa.field.attrs['history'])


@attr('icclim')
class TestAbstractIcclimPercentileArrayIndice(TestBase):
    def test_calculate(self):
        klasses = list(itersubclasses(AbstractIcclimPercentileArrayIndice))
        # There are six classes to test.
        self.assertEqual(len(klasses), 6)

        for mod in (1, 2):
            field = self.get_field(ntime=365)
            # Values less than 1 mm/day will be masked inside icclim.
            # var = field.variables.first()
            var = field.get_by_tag(TagNames.DATA_VARIABLES)[0]
            var.get_value()[:] = var.get_value() * mod
            tgd = field.temporal.get_grouping(['month'])

            for klass in klasses:
                c = klass(field=field, tgd=tgd)
                res = c.execute()
                self.assertIsInstance(res, VariableCollection)
                dv = res.first()
                # Output units are always mm/day.
                if isinstance(c, IcclimR75pTOT):
                    self.assertTrue(get_are_units_equivalent((dv.cfunits, get_units_object('mm/day'))))


@attr('icclim')
class TestLibraryIcclim(TestBase):
    def test_bad_icclim_key_to_operations(self):
        value = [{'func': 'icclim_TG_bad', 'name': 'TG'}]
        with self.assertRaises(DefinitionValidationError):
            Calc(value)

    def test_calc_argument_to_operations(self):
        value = [{'func': 'icclim_TG', 'name': 'TG'}]
        calc = Calc(value)
        self.assertEqual(len(calc.value), 1)
        self.assertEqual(calc.value[0]['ref'], IcclimTG)

    @attr('slow')
    def test_icclim_combinatorial(self):
        raise SkipTest('release only')
        shapes = ([('month',), 12], [('month', 'year'), 24], [('year',), 2])
        ocgis.env.OVERWRITE = True
        keys = set(library_icclim._icclim_function_map.keys())
        for klass in [AbstractIcclimUnivariateSetFunction, AbstractIcclimMultivariateFunction]:
            for subclass in itersubclasses(klass):

                if subclass.__name__.startswith('Abstract'):
                    continue

                keys.remove(subclass.key)

                for cg in CalcGrouping.iter_possible():
                    # print cg
                    calc = [{'func': subclass.key, 'name': subclass.key.split('_')[1]}]
                    if klass == AbstractIcclimUnivariateSetFunction:
                        rd = self.test_data.get_rd('cancm4_tas')
                        rd.time_region = {'year': [2001, 2002]}
                        calc = [{'func': subclass.key, 'name': subclass.key.split('_')[1]}]
                    else:
                        tasmin = self.test_data.get_rd('cancm4_tasmin_2001')
                        tasmax = self.test_data.get_rd('cancm4_tasmax_2001')
                        rd = [tasmin, tasmax]
                        for r in rd:
                            r.time_region = {'year': [2001, 2002]}
                        kwds = {'tasmin': 'tasmin', 'tasmax': 'tasmax'}
                        calc[0].update({'kwds': kwds})
                    try:
                        ops = ocgis.OcgOperations(dataset=rd, output_format='nc', calc=calc, calc_grouping=cg,
                                                  geom=[35.39, 45.62, 42.54, 52.30])
                        ret = ops.execute()
                        to_test = None
                        for shape in shapes:
                            if shape[0] == cg:
                                to_test = shape[1]
                        with nc_scope(ret) as ds:
                            var = ds.variables[calc[0]['name']]
                            if to_test is not None:
                                self.assertEqual(var.shape, (to_test, 3, 3))
                    except DefinitionValidationError as e:
                        if e.message.startswith(strings.S4):
                            pass
                        else:
                            raise e
        self.assertEqual(len(keys), 0)

    def test_register_icclim(self):
        fr = FunctionRegistry()
        self.assertNotIn('icclim_TG', fr)
        register_icclim(fr)
        self.assertIn('icclim_TG', fr)
        self.assertIn('icclim_vDTR', fr)

    @attr('data')
    def test_seasonal_calc_grouping(self):
        """Test seasonal calculation grouping with an ICCLIM function."""

        rd = self.test_data.get_rd('cancm4_tas')
        slc = [None, [0, 600], None, [0, 10], [0, 10]]
        calc_icclim = [{'func': 'icclim_TG', 'name': 'TG'}]
        calc_ocgis = [{'func': 'mean', 'name': 'mean'}]
        cg = [[12, 1, 2], 'unique']
        ops_ocgis = OcgOperations(calc=calc_ocgis, calc_grouping=cg, slice=slc, dataset=rd)
        ret_ocgis = ops_ocgis.execute()
        ops_icclim = OcgOperations(calc=calc_icclim, calc_grouping=cg, slice=slc, dataset=rd)
        ret_icclim = ops_icclim.execute()
        desired = ret_ocgis.get_element(variable_name='mean').get_masked_value()
        actual = ret_icclim.get_element(variable_name='TG').get_masked_value()
        self.assertNumpyAll(desired, actual)


@attr('icclim')
class TestTG10p(TestBase):
    def test_init(self):
        IcclimTG10p()

    @attr('data', 'slow')
    def test_execute(self):
        tas = self.test_data.get_rd('cancm4_tas').get()
        tas = tas.get_field_slice({'y': slice(10, 12), 'x': slice(20, 22)})
        tgd = tas.temporal.get_grouping(['month'])
        tg = IcclimTG10p(field=tas, tgd=tgd)
        ret = tg.execute()
        self.assertEqual(ret['icclim_TG10p'].shape, (12, 2, 2))
        self.assertEqual(ret['icclim_TG10p'].get_value().mean(), 30.0625)

        # Test with a percentile dictionary.
        field_pd = tas.get_field_slice({'time': slice(0, 800)})
        arr = field_pd['tas'].masked_value.squeeze()
        dt_arr = field_pd.temporal.value_datetime
        percentile = 10
        window_width = 5
        pd = tg.get_percentile_dict(arr, dt_arr, percentile, window_width)
        tg = IcclimTG10p(field=tas, tgd=tgd, parms={'percentile_dict': pd})
        ret = tg.execute()
        self.assertEqual(ret['icclim_TG10p'].shape, (12, 2, 2))
        # This value should change since we are using a different base period.
        self.assertEqual(ret['icclim_TG10p'].get_value().mean(), 31.0)

    @attr('data', 'slow')
    def test_large_array_compute_local(self):
        """Test tiling works for percentile-based indice on a local dataset."""

        calc = [{'func': 'icclim_TG10p', 'name': 'itg'}]
        calc_grouping = ['month']
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, output_format='nc',
                                  geom='state_boundaries',
                                  select_ugid=[24])
        ret = compute(ops, 5, verbose=False)

        with nc_scope(ret) as ds:
            self.assertAlmostEqual(ds.variables['itg'][:].sum(), 2121.0, 6)


@attr('icclim')
class TestDTR(TestBase):
    @attr('data')
    def test_calculate(self):
        tasmin = self.test_data.get_rd('cancm4_tasmin_2001')
        tasmax = self.test_data.get_rd('cancm4_tasmax_2001')
        field = tasmin.get()
        field_tasmax = tasmax.get()
        variable_to_add = field_tasmax['tasmax']
        with orphaned(variable_to_add):
            field.add_variable(variable_to_add, is_data=True)
        field = field.get_field_slice({'time': slice(0, 600), 'y': slice(25, 50), 'x': slice(25, 50)})
        tgd = field.temporal.get_grouping(['month'])
        dtr = IcclimDTR(field=field, tgd=tgd)
        ret = dtr.execute()
        self.assertEqual(ret['icclim_DTR'].get_value().shape, (12, 25, 25))

    @attr('data')
    def test_bad_keyword_mapping(self):
        tasmin = self.test_data.get_rd('cancm4_tasmin_2001')
        tas = self.test_data.get_rd('cancm4_tas')
        rds = [tasmin, tas]
        calc = [{'func': 'icclim_DTR', 'name': 'DTR', 'kwds': {'tas': 'tasmin', 'tasmax': 'tasmax'}}]
        with self.assertRaises(DefinitionValidationError):
            ocgis.OcgOperations(dataset=rds, calc=calc, calc_grouping=['month'],
                                output_format='nc')

        calc = [{'func': 'icclim_DTR', 'name': 'DTR'}]
        with self.assertRaises(DefinitionValidationError):
            ocgis.OcgOperations(dataset=rds, calc=calc, calc_grouping=['month'],
                                output_format='nc')

    @attr('data')
    def test_calculation_operations(self):
        # note the kwds must contain a map of the required variables to their associated aliases.
        calc = [{'func': 'icclim_DTR', 'name': 'DTR', 'kwds': {'tasmin': 'tasmin', 'tasmax': 'tasmax'}}]
        tasmin = self.test_data.get_rd('cancm4_tasmin_2001')
        tasmin.time_region = {'year': [2002]}
        tasmax = self.test_data.get_rd('cancm4_tasmax_2001')
        tasmax.time_region = {'year': [2002]}
        rds = [tasmin, tasmax]
        ops = ocgis.OcgOperations(dataset=rds, calc=calc, calc_grouping=['month'], output_format='nc')
        ops.execute()


@attr('icclim')
class TestETR(TestBase):
    @attr('data')
    def test_calculate(self):
        tasmin = self.test_data.get_rd('cancm4_tasmin_2001')
        tasmax = self.test_data.get_rd('cancm4_tasmax_2001')
        field = tasmin.get()
        field_tasmax = tasmax.get()
        with orphaned(field_tasmax['tasmax']):
            field.add_variable(field_tasmax['tasmax'], is_data=True)
        field = field.get_field_slice({'time': slice(0, 600), 'y': slice(25, 50), 'x': slice(25, 50)})
        tgd = field.temporal.get_grouping(['month'])
        dtr = IcclimETR(field=field, tgd=tgd)
        ret = dtr.execute()
        self.assertEqual(ret['icclim_ETR'].get_value().shape, (12, 25, 25))

    @attr('data')
    def test_calculate_rotated_pole(self):
        tasmin_fake = self.test_data.get_rd('rotated_pole_ichec')
        tasmin_fake.rename_variable = 'tasmin'
        tasmax_fake = deepcopy(tasmin_fake)
        tasmax_fake.rename_variable = 'tasmax'
        rds = [tasmin_fake, tasmax_fake]
        for rd in rds:
            rd.time_region = {'year': [1973]}
        calc_ETR = [{'func': 'icclim_ETR', 'name': 'ETR', 'kwds': {'tasmin': 'tasmin', 'tasmax': 'tasmax'}}]
        ops = ocgis.OcgOperations(dataset=[tasmin_fake, tasmax_fake],
                                  calc=calc_ETR,
                                  calc_grouping=['year', 'month'],
                                  prefix='ETR_ocgis_icclim',
                                  output_format='nc',
                                  add_auxiliary_files=False)
        with nc_scope(ops.execute()) as ds:
            self.assertEqual(ds.variables['ETR'][:].shape, (12, 103, 106))


@attr('icclim')
class TestTx(TestBase):
    @attr('data')
    def test_calculate_operations(self):
        rd = self.test_data.get_rd('cancm4_tas')
        slc = [None, None, None, [0, 10], [0, 10]]
        calc_icclim = [{'func': 'icclim_TG', 'name': 'TG'}]
        calc_ocgis = [{'func': 'mean', 'name': 'mean'}]
        _calc_grouping = [['month'], ['month', 'year']]
        for cg in _calc_grouping:
            ops_ocgis = OcgOperations(calc=calc_ocgis, calc_grouping=cg, slice=slc, dataset=rd)
            ret_ocgis = ops_ocgis.execute()
            ops_icclim = OcgOperations(calc=calc_icclim, calc_grouping=cg, slice=slc, dataset=rd)
            ret_icclim = ops_icclim.execute()
            desired = ret_ocgis.get_element(variable_name='mean').get_masked_value()
            actual = ret_icclim.get_element(variable_name='TG').get_masked_value()
            self.assertNumpyAll(desired, actual)

    @attr('data')
    def test_calculation_operations_to_nc(self):
        rd = self.test_data.get_rd('cancm4_tas')
        slc = [None, None, None, [0, 10], [0, 10]]
        ops_ocgis = OcgOperations(calc=[{'func': 'icclim_TG', 'name': 'TG'}],
                                  calc_grouping=['month'],
                                  slice=slc,
                                  dataset=rd,
                                  output_format='nc')
        ret = ops_ocgis.execute()
        with nc_scope(ret) as ds:
            self.assertIn('Calculation of TG indice (monthly climatology)', ds.history)
            self.assertEqual(ds.title, 'ECA temperature indice TG')
            var = ds.variables['TG']
            # check the JSON serialization
            actual = u'{"institution": "CCCma (Canadian Centre for Climate Modelling and Analysis, Victoria, BC, Canada)", "institute_id": "CCCma", "experiment_id": "decadal2000", "source": "CanCM4 2010 atmosphere: CanAM4 (AGCM15i, T63L35) ocean: CanOM4 (OGCM4.0, 256x192L40) sea ice: CanSIM1 (Cavitating Fluid, T63 Gaussian Grid) land: CLASS2.7", "model_id": "CanCM4", "forcing": "GHG,Oz,SA,BC,OC,LU,Sl,Vl (GHG includes CO2,CH4,N2O,CFC11,effective CFC12)", "parent_experiment_id": "N/A", "parent_experiment_rip": "N/A", "branch_time": 0.0, "contact": "cccma_info@ec.gc.ca", "references": "http://www.cccma.ec.gc.ca/models", "initialization_method": 1, "physics_version": 1, "tracking_id": "fac7bd83-dd7a-425b-b4dc-b5ab2e915939", "branch_time_YMDH": "2001:01:01:00", "CCCma_runid": "DHFP1B_E002_I2001_M01", "CCCma_parent_runid": "DHFP1_E002", "CCCma_data_licence": "1) GRANT OF LICENCE - The Government of Canada (Environment Canada) is the \\nowner of all intellectual property rights (including copyright) that may exist in this Data \\nproduct. You (as \\"The Licensee\\") are hereby granted a non-exclusive, non-assignable, \\nnon-transferable unrestricted licence to use this data product for any purpose including \\nthe right to share these data with others and to make value-added and derivative \\nproducts from it. This licence is not a sale of any or all of the owner\'s rights.\\n2) NO WARRANTY - This Data product is provided \\"as-is\\"; it has not been designed or \\nprepared to meet the Licensee\'s particular requirements. Environment Canada makes no \\nwarranty, either express or implied, including but not limited to, warranties of \\nmerchantability and fitness for a particular purpose. In no event will Environment Canada \\nbe liable for any indirect, special, consequential or other damages attributed to the \\nLicensee\'s use of the Data product.", "product": "output", "experiment": "10- or 30-year run initialized in year 2000", "frequency": "day", "creation_date": "2011-05-08T01:01:51Z", "history": "2011-05-08T01:01:51Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.", "Conventions": "CF-1.4", "project_id": "CMIP5", "table_id": "Table day (28 March 2011) f9d6cfec5981bb8be1801b35a81002f0", "title": "CanCM4 model output prepared for CMIP5 10- or 30-year run initialized in year 2000", "parent_experiment": "N/A", "modeling_realm": "atmos", "realization": 2, "cmor_version": "2.5.4"}'
            self.assertEqual(ds.__dict__[AbstractIcclimFunction._global_attribute_source_name], actual)
            # load the original source attributes from the JSON string
            json.loads(ds.__dict__[AbstractIcclimFunction._global_attribute_source_name])
            actual = {u'units': u'K', 'grid_mapping': 'latitude_longitude',
                      u'standard_name': AbstractIcclimFunction.standard_name,
                      u'long_name': u'Mean of daily mean temperature'}
            self.assertEqual(dict(var.__dict__), actual)

    @attr('data')
    def test_calculate(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        field = field.get_field_slice({'y': slice(0, 10), 'x': slice(0, 10)})
        klasses = [IcclimTG, IcclimTN, IcclimTX]
        for klass in klasses:
            for calc_grouping in [['month'], ['month', 'year']]:
                tgd = field.temporal.get_grouping(calc_grouping)
                itg = klass(field=field, tgd=tgd)
                ret_icclim = itg.execute()
                mean = Mean(field=field, tgd=tgd)
                ret_ocgis = mean.execute()
                self.assertNumpyAll(ret_icclim[klass.key].get_value(), ret_ocgis['mean'].get_value())


@attr('icclim')
class TestSU(TestBase):
    @attr('data')
    def test_calculate(self):
        rd = self.test_data.get_rd('cancm4_tasmax_2011')
        field = rd.get()
        field = field.get_field_slice({'y': slice(0, 10), 'x': slice(0, 10)})
        for calc_grouping in [['month'], ['month', 'year']]:
            tgd = field.temporal.get_grouping(calc_grouping)
            itg = IcclimSU(field=field, tgd=tgd)
            ret_icclim = itg.execute()
            threshold = Threshold(field=field, tgd=tgd, parms={'threshold': 298.15, 'operation': 'gt'})
            ret_ocgis = threshold.execute()
            self.assertNumpyAll(ret_icclim['icclim_SU'].get_value(), ret_ocgis['threshold'].get_value())

    @attr('data')
    def test_calculation_operations_bad_units(self):
        rd = self.test_data.get_rd('daymet_tmax')
        calc_icclim = [{'func': 'icclim_SU', 'name': 'SU'}]
        ops_icclim = OcgOperations(calc=calc_icclim, calc_grouping=['month', 'year'], dataset=rd)
        with self.assertRaises(UnitsValidationError):
            ops_icclim.execute()

    @attr('data')
    def test_calculation_operations_to_nc(self):
        rd = self.test_data.get_rd('cancm4_tasmax_2011')
        slc = [None, None, None, [0, 10], [0, 10]]
        ops_ocgis = OcgOperations(calc=[{'func': 'icclim_SU', 'name': 'SU'}], calc_grouping=['month'], slice=slc,
                                  dataset=rd, output_format='nc')
        ret = ops_ocgis.execute()
        with nc_scope(ret) as ds:
            to_test = deepcopy(ds.__dict__)
            history = to_test.pop('history')
            self.assertEqual(history[111:187],
                             ' Calculation of SU indice (monthly climatology) from 2011-1-1 to 2020-12-31.')
            actual = OrderedDict([(u'source_data_global_attributes',
                                   u'{"institution": "CCCma (Canadian Centre for Climate Modelling and Analysis, Victoria, BC, Canada)", "institute_id": "CCCma", "experiment_id": "decadal2010", "source": "CanCM4 2010 atmosphere: CanAM4 (AGCM15i, T63L35) ocean: CanOM4 (OGCM4.0, 256x192L40) sea ice: CanSIM1 (Cavitating Fluid, T63 Gaussian Grid) land: CLASS2.7", "model_id": "CanCM4", "forcing": "GHG,Oz,SA,BC,OC,LU,Sl,Vl (GHG includes CO2,CH4,N2O,CFC11,effective CFC12)", "parent_experiment_id": "N/A", "parent_experiment_rip": "N/A", "branch_time": 0.0, "contact": "cccma_info@ec.gc.ca", "references": "http://www.cccma.ec.gc.ca/models", "initialization_method": 1, "physics_version": 1, "tracking_id": "64384802-3f0f-4ab4-b569-697bd5430854", "branch_time_YMDH": "2011:01:01:00", "CCCma_runid": "DHFP1B_E002_I2011_M01", "CCCma_parent_runid": "DHFP1_E002", "CCCma_data_licence": "1) GRANT OF LICENCE - The Government of Canada (Environment Canada) is the \\nowner of all intellectual property rights (including copyright) that may exist in this Data \\nproduct. You (as \\"The Licensee\\") are hereby granted a non-exclusive, non-assignable, \\nnon-transferable unrestricted licence to use this data product for any purpose including \\nthe right to share these data with others and to make value-added and derivative \\nproducts from it. This licence is not a sale of any or all of the owner\'s rights.\\n2) NO WARRANTY - This Data product is provided \\"as-is\\"; it has not been designed or \\nprepared to meet the Licensee\'s particular requirements. Environment Canada makes no \\nwarranty, either express or implied, including but not limited to, warranties of \\nmerchantability and fitness for a particular purpose. In no event will Environment Canada \\nbe liable for any indirect, special, consequential or other damages attributed to the \\nLicensee\'s use of the Data product.", "product": "output", "experiment": "10- or 30-year run initialized in year 2010", "frequency": "day", "creation_date": "2012-03-28T15:32:08Z", "history": "2012-03-28T15:32:08Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.", "Conventions": "CF-1.4", "project_id": "CMIP5", "table_id": "Table day (28 March 2011) f9d6cfec5981bb8be1801b35a81002f0", "title": "CanCM4 model output prepared for CMIP5 10- or 30-year run initialized in year 2010", "parent_experiment": "N/A", "modeling_realm": "atmos", "realization": 2, "cmor_version": "2.8.0"}'),
                                  (u'title', u'ECA heat indice SU'), (
                                      u'references',
                                      u'ATBD of the ECA indices calculation (http://eca.knmi.nl/documents/atbd.pdf)'),
                                  (u'institution', u'Climate impact portal (http://climate4impact.eu)'),
                                  (u'comment', u' ')])
            self.assertDictEqual(to_test, actual)
            var = ds.variables['SU']
            to_test = dict(var.__dict__)
            self.assertEqual(to_test, {u'units': u'days',
                                       u'standard_name': AbstractIcclimFunction.standard_name,
                                       u'long_name': 'Summer days (number of days where daily maximum temperature > 25 degrees)',
                                       'grid_mapping': 'latitude_longitude'})
