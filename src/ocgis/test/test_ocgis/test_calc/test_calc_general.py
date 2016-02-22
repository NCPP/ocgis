import datetime
import itertools
import unittest
from datetime import datetime as dt

import numpy as np

import ocgis
from ocgis import constants
from ocgis.api.operations import OcgOperations
from ocgis.calc.engine import OcgCalculationEngine
from ocgis.calc.library.thresholds import Threshold
from ocgis.test.base import TestBase
from ocgis.test.base import attr


class AbstractCalcBase(TestBase):
    
    def get_reshaped(self,arr):
        ret = arr.reshape(arr.shape[0],1,1)
        ret = np.ma.array(ret,mask=False)
        assert(len(ret.shape) == 3)
        return(ret)

    def run_standard_operations(self, calc, capture=False, output_format=None):
        _aggregate = [False, True]
        _calc_grouping = [['month'], ['month', 'year'], 'all']
        _output_format = output_format or [constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_CSV_SHAPEFILE,
                                           constants.OUTPUT_FORMAT_NETCDF]
        captured = []
        for ii, tup in enumerate(itertools.product(_aggregate, _calc_grouping, _output_format)):
            aggregate, calc_grouping, output_format = tup
            if aggregate is True and output_format == constants.OUTPUT_FORMAT_NETCDF:
                continue
            rd = self.test_data.get_rd('cancm4_tas', kwds={'time_region': {'year': [2001, 2002]}})
            try:
                ops = OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[25], calc=calc,
                                    calc_grouping=calc_grouping, output_format=output_format, aggregate=aggregate,
                                    prefix=('standard_ops_' + str(ii)))
                ret = ops.execute()
                if output_format == constants.OUTPUT_FORMAT_NUMPY:
                    ref = ret[25]['tas'].variables[calc[0]['name']].value
                    if aggregate:
                        space_shape = [1, 1]
                    else:
                        space_shape = [5, 4]
                    if calc_grouping == ['month']:
                        shp1 = [12]
                    elif calc_grouping == 'all':
                        raise NotImplementedError('calc_grouping all')
                    else:
                        shp1 = [24]
                    test_shape = [1] + shp1 + [1] + space_shape
                    self.assertEqual(ref.shape, tuple(test_shape))
                    if not aggregate:
                        # ensure the geometry mask is appropriately update by the function
                        try:
                            self.assertTrue(np.ma.is_masked(ref[0, 0, 0, 0, 0]))
                        # likely a structure array where testing requires using the mask property
                        except TypeError:
                            self.assertTrue(ref.mask[0, 0, 0, 0, 0])
            except ValueError:
                raise
            except AssertionError:
                raise
            except Exception as e:
                if capture:
                    parms = dict(aggregate=aggregate, calc_grouping=calc_grouping, output_format=output_format)
                    captured.append({'exception': e, 'parms': parms})
                else:
                    raise
        return captured


class Test(AbstractCalcBase):
    """
    Test guide:
     * Data is required to be 4-dimensional:
         arr.shape = (t,1,m,n)
    """

    @attr('data')
    def test_date_groups_all(self):
        calc = [{'func': 'mean', 'name': 'mean'}]
        rd = self.test_data.get_rd('cancm4_tasmax_2011')

        calc_grouping = 'all'
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, geom='state_boundaries',
                            select_ugid=[25])
        ret = ops.execute()
        field = ret[25]['tasmax']
        variable = field.variables['mean']
        parents = variable.parents['tasmax']
        self.assertEqual(parents.value.shape, (1, 3650, 1, 5, 4))
        self.assertEqual(field.shape, (1, 1, 1, 5, 4))
        lhs = np.ma.mean(parents.value, axis=1).reshape(1, 1, 1, 5, 4).astype(parents.dtype)
        rhs = variable.value
        self.assertNumpyAll(lhs, rhs)

    @attr('data')
    def test_time_region(self):
        kwds = {'time_region':{'year':[2011]}}
        rd = self.test_data.get_rd('cancm4_tasmax_2011',kwds=kwds)
        calc = [{'func':'mean','name':'mean'}]
        calc_grouping = ['year','month']
        
        ops = ocgis.OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                                  geom='state_boundaries',select_ugid=[25])
        ret = ops.execute()
        
        tgroup = ret[25]['tasmax'].temporal.date_parts
        self.assertEqual(set([2011]),set(tgroup['year']))
        self.assertEqual(tgroup['month'][-1],12)
        
        kwds = {'time_region':{'year':[2011,2013],'month':[8]}}
        rd = self.test_data.get_rd('cancm4_tasmax_2011',kwds=kwds)
        calc = [{'func':'threshold','name':'threshold','kwds':{'threshold':0.0,'operation':'gte'}}]
        calc_grouping = ['month']
        aggregate = True
        calc_raw = True
        geom = 'us_counties'
        select_ugid = [2762]
        
        ops = ocgis.OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                                 aggregate=aggregate,calc_raw=calc_raw,geom=geom,
                                 select_ugid=select_ugid,output_format='numpy')
        ret = ops.execute()
        threshold = ret[2762]['tasmax'].variables['threshold'].value
        self.assertEqual(threshold.flatten()[0],62)

    @attr('data')
    def test_computational_nc_output(self):
        """Test writing a computation to netCDF."""

        rd = self.test_data.get_rd('cancm4_tasmax_2011', kwds={
        'time_range': [datetime.datetime(2011, 1, 1), datetime.datetime(2011, 12, 31)]})
        calc = [{'func': 'mean', 'name': 'tasmax_mean'}]
        calc_grouping = ['month', 'year']

        ops = ocgis.OcgOperations(rd, calc=calc, calc_grouping=calc_grouping,
                                  output_format='nc')
        ret = ops.execute()

        with self.nc_scope(ret) as ds:
            ref = ds.variables['time']
            self.assertEqual(ref.climatology, 'climatology_bounds')
            self.assertEqual(len(ref[:]), 12)
            ref = ds.variables['climatology_bounds']
            self.assertEqual(ref[:].shape[0], 12)

        ops = ocgis.OcgOperations(dataset={'uri': ret, 'variable': calc[0]['name']},
                                  output_format='nc', prefix='subset_climatology')
        ret = ops.execute()

        with self.nc_scope(ret) as ds:
            ref = ds.variables['time'][:]
            self.assertEqual(len(ref), 12)
            self.assertEqual(set(ds.variables['tasmax_mean'].ncattrs()),
                             set([u'_FillValue', u'units', u'long_name', u'standard_name', 'grid_mapping']))

    def test_frequency_percentiles(self):
        ## data comes in as 4-dimensional array. (time,level,row,column)
        
        perc = 0.95
        round_method = 'ceil' #floor
        
        ## generate gaussian sequence
        np.random.seed(1)
        seq = np.random.normal(size=(31,1,2,2))
        seq = np.ma.array(seq,mask=False)
        ## sort the data
        cseq = seq.copy()
        cseq.sort(axis=0)
        ## reference the time vector length
        n = cseq.shape[0]
        ## calculate the index
        idx = getattr(np,round_method)(perc*n)
        ## get the percentiles
        ret = cseq[idx,:,:,:]
        self.assertAlmostEqual(7.2835104624617717,ret.sum())
        
        ## generate gaussian sequence
        np.random.seed(1)
        seq = np.random.normal(size=(31,1,2,2))
        mask = np.zeros((31,1,2,2))
        mask[:,:,1,1] = True
        seq = np.ma.array(seq,mask=mask)
        ## sort the data
        cseq = seq.copy()
        cseq.sort(axis=0)
        ## reference the time vector length
        n = cseq.shape[0]
        ## calculate the index
        idx = getattr(np,round_method)(perc*n)
        ## get the percentiles
        ret = cseq[idx,:,:,:]
        self.assertAlmostEqual(5.1832553259829295,ret.sum())

    @attr('data')
    def test_date_groups(self):
        calc = [{'func': 'mean', 'name': 'mean'}]
        rd = self.test_data.get_rd('cancm4_tasmax_2011')

        calc_grouping = ['month']
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, geom='state_boundaries',
                            select_ugid=[25])
        ret = ops.execute()
        ref = ret[25]['tasmax'].temporal
        rdt = ref.value_datetime
        self.assertTrue(np.all(rdt == np.array([dt(2011, month, 16) for month in range(1, 13)])))

        calc_grouping = ['year']
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, geom='state_boundaries',
                            select_ugid=[25])
        ret = ops.execute()
        ref = ret[25]['tasmax'].temporal
        rdt = ref.value_datetime
        self.assertTrue(np.all(rdt == [dt(year, 7, 1) for year in range(2011, 2021)]))

        calc_grouping = ['month', 'year']
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, geom='state_boundaries',
                            select_ugid=[25])
        ret = ops.execute()
        ref = ret[25]['tasmax'].temporal
        rdt = ref.value_datetime
        self.assertTrue(
            np.all(rdt == [dt(year, month, 16) for year, month in itertools.product(range(2011, 2021), range(1, 13))]))

        calc_grouping = ['day']
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, geom='state_boundaries',
                            select_ugid=[25])
        ret = ops.execute()
        ref = ret[25]['tasmax'].temporal
        rdt = ref.value_datetime
        self.assertTrue(np.all(rdt == [dt(2011, 1, day, 12) for day in range(1, 32)]))

        calc_grouping = ['month', 'day']
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, geom='state_boundaries',
                            select_ugid=[25])
        ret = ops.execute()
        ref = ret[25]['tasmax'].temporal
        rdt = ref.value_datetime
        self.assertEqual(rdt[0], dt(2011, 1, 1, 12))
        self.assertEqual(rdt[12], dt(2011, 1, 13, 12))

        calc_grouping = ['year', 'day']
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, geom='state_boundaries',
                            select_ugid=[25])
        ret = ops.execute()
        ref = ret[25]['tasmax'].temporal
        rdt = ref.value_datetime
        self.assertEqual(rdt[0], dt(2011, constants.CALC_YEAR_CENTROID_MONTH, 1, 12))

        rd = self.test_data.get_rd('cancm4_tasmax_2011', kwds={'time_region': {'month': [1], 'year': [2011]}})
        field = rd.get()
        calc_grouping = ['month', 'day', 'year']
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, geom='state_boundaries',
                            select_ugid=[25])
        ret = ops.execute()
        ref = ret[25]['tasmax'].temporal
        rdt = ref.value_datetime
        self.assertTrue(np.all(rdt == ref.value_datetime))
        self.assertTrue(np.all(ref.bounds_datetime == field.temporal.bounds_datetime))


class TestOcgCalculationEngine(TestBase):
    
    def get_collection(self,aggregate=False):
        if aggregate:
            spatial_operation = 'clip'
        else:
            spatial_operation = 'intersects'
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd,geom='state_boundaries',select_ugid=[25],
                            spatial_operation=spatial_operation,aggregate=aggregate)
        ret = ops.execute()
        return(ret)

    @attr('data')
    def test_agg_raw(self):
        """Test using raw values for calculations as opposed to spatially averaged data values."""

        grouping = ['month']
        funcs = [{'func': 'threshold', 'name': 'threshold', 'ref': Threshold,
                  'kwds': {'operation': 'gte', 'threshold': 200}}]
        raws = [True, False]
        aggs = [True, False]
        for raw, agg in itertools.product(raws, aggs):
            coll = self.get_collection(aggregate=agg)
            ce = OcgCalculationEngine(grouping, funcs, raw, agg)
            ret = ce.execute(coll)
            value = ret[25]['tas'].variables['threshold'].value
            # # aggregated data should have a (1,1) spatial dimension
            if agg is True:
                self.assertEqual(value.shape[-2:], (1, 1))


if __name__ == '__main__':
    unittest.main()
