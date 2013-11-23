import unittest
import numpy as np
from ocgis.calc import library
from ocgis.api.operations import OcgOperations
from datetime import datetime as dt
import ocgis
import datetime
from ocgis.test.base import TestBase
import netCDF4 as nc
import itertools
from ocgis.exc import DefinitionValidationError
import webbrowser
from ocgis.calc.engine import OcgCalculationEngine
from unittest.case import SkipTest
from ocgis.util.inspect import Inspect
from ocgis.calc.library.statistics import StandardDeviation, Mean
from ocgis.calc.library.thresholds import Threshold
from ocgis.calc.library.index.duration import Duration



class AbstractCalcBase(TestBase):
    
    def get_reshaped(self,arr):
        ret = arr.reshape(arr.shape[0],1,1)
        ret = np.ma.array(ret,mask=False)
        assert(len(ret.shape) == 3)
        return(ret)
    
    def run_standard_operations(self,calc,capture=False,output_format=None):
        _aggregate = [False,True]
        _calc_grouping = [['month'],['month','year']]
        _output_format = output_format or ['numpy','csv+','nc']
        captured = []
        for ii,tup in enumerate(itertools.product(_aggregate,_calc_grouping,_output_format)):
            aggregate,calc_grouping,output_format = tup
            if aggregate is True and output_format == 'nc':
                continue
            rd = self.test_data.get_rd('cancm4_tas',kwds={'time_region':{'year':[2001,2002]}})
            try:
                ops = OcgOperations(dataset=rd,geom='state_boundaries',select_ugid=[25],
                       calc=calc,calc_grouping=calc_grouping,output_format=output_format,
                       aggregate=aggregate,prefix=('standard_ops_'+str(ii)))
                ret = ops.execute()
                if output_format == 'numpy':
                    ref = ret[25]['tas'].variables[calc[0]['name']+'_tas'].value
                    if aggregate:
                        space_shape = [1,1]
                    else:
                        space_shape = [5,4]
                    if calc_grouping == ['month']:
                        shp1 = [12]
                    else:
                        shp1 = [24]
                    test_shape = [1] + shp1 + [1] + space_shape
                    self.assertEqual(ref.shape,tuple(test_shape))
                    if not aggregate:
                        ## ensure the geometry mask is appropriately update by the function
                        try:
                            self.assertTrue(np.ma.is_masked(ref[0,0,0,0,0]))
                        ## likely a structure array with multiple masked elements per index
                        except TypeError:
                            self.assertTrue(np.all([np.ma.is_masked(element) for element in ref[0,0,0,0,0]]))
            except ValueError:
                raise
            except Exception as e:
                if capture:
                    parms = dict(aggregate=aggregate,calc_grouping=calc_grouping,output_format=output_format)
                    captured.append({'exception':e,'parms':parms})
                else:
                    raise
        return(captured)


class Test(AbstractCalcBase):
    '''
    Test guide:
     * Data is required to be 4-dimensional:
         arr.shape = (t,1,m,n)
    '''
    
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
        threshold = ret[2762]['tasmax'].variables['threshold_tasmax'].value
        self.assertEqual(threshold.flatten()[0],62)
    
    def test_heat_index(self):
        ocgis.env.OVERWRITE = True
        kwds = {'time_range':[dt(2011,1,1),dt(2011,12,31,23,59,59)]}
        ds = [self.test_data.get_rd('cancm4_tasmax_2011',kwds=kwds),self.test_data.get_rd('cancm4_rhsmax',kwds=kwds)]
        calc = [{'func':'heat_index','name':'heat_index','kwds':{'tas':'tasmax','rhs':'rhsmax','units':'k'}}]
        select_ugid = [25]
        
        ## operations on entire data arrays
        ops = OcgOperations(dataset=ds,calc=calc)
        self.assertEqual(ops.calc_grouping,None)
        ret = ops.execute()
        ref = ret[1]
        self.assertEqual(ref.keys(),['tasmax_rhsmax'])
        self.assertEqual(ref['tasmax_rhsmax'].variables.keys(),['heat_index'])
        hi = ref['tasmax_rhsmax'].variables['heat_index'].value
        self.assertEqual(hi.shape,(1,365,1,64,128))
        
        ## confirm no masked geometries
        self.assertFalse(ref['tasmax_rhsmax'].spatial.geom.point.value.mask.any())
        ## confirm some masked data in calculation output
        self.assertTrue(hi.mask.any())
                
        # try temporal grouping
        ops = OcgOperations(dataset=ds,calc=calc,calc_grouping=['month'],geom='state_boundaries',select_ugid=select_ugid)
        ret = ops.execute()
        self.assertEqual(ret[25]['tasmax_rhsmax'].variables['heat_index'].value.shape,(1,12,1,5,4))
        
    def test_computational_nc_output(self):
        rd = self.test_data.get_rd('cancm4_tasmax_2011',kwds={'time_range':[datetime.datetime(2011,1,1),datetime.datetime(2011,12,31)]})
        calc = [{'func':'mean','name':'tasmax_mean'}]
        calc_grouping = ['month','year']

        ops = ocgis.OcgOperations(rd,calc=calc,calc_grouping=calc_grouping,
                                  output_format='nc')
        ret = ops.execute()
        ds = nc.Dataset(ret,'r')
        ref = ds.variables['time']
        self.assertEqual(ref.climatology,'climatology_bnds')
        self.assertEqual(len(ref[:]),12)
        ref = ds.variables['climatology_bnds']
        self.assertEqual(ref[:].shape[0],12)
        ds.close()

        ops = ocgis.OcgOperations(dataset={'uri':ret,'variable':calc[0]['name']+'_tasmax'},
                                  output_format='nc',prefix='subset_climatology')
        ret = ops.execute()
        
        ds = nc.Dataset(ret,'r')
        ref = ds.variables['time'][:]
        self.assertEqual(len(ref),12)
        self.assertEqual(set(ds.variables['tasmax_mean_tasmax'].ncattrs()),
                         set([u'_FillValue', u'units', u'long_name', u'standard_name']))
        ds.close()
        
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
        
    def test_date_groups(self):
        calc = [{'func':'mean','name':'mean'}]
        rd = self.test_data.get_rd('cancm4_tasmax_2011')
        
        calc_grouping = ['month']
        ops = OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                            geom='state_boundaries',select_ugid=[25])
        ret = ops.execute()
        ref = ret[25]['tasmax'].temporal
        rdt = ref.value_datetime
        self.assertTrue(np.all(rdt == np.array([dt(2011,month,16) for month in range(1,13)])))
        
        calc_grouping = ['year']
        ops = OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                            geom='state_boundaries',select_ugid=[25])
        ret = ops.execute()
        ref = ret[25]['tasmax'].temporal
        rdt = ref.value_datetime
        self.assertTrue(np.all(rdt == [dt(year,7,1) for year in range(2011,2021)]))

        calc_grouping = ['month','year']
        ops = OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                            geom='state_boundaries',select_ugid=[25])
        ret = ops.execute()
        ref = ret[25]['tasmax'].temporal
        rdt = ref.value_datetime
        self.assertTrue(np.all(rdt == [dt(year,month,16) for year,month in itertools.product(range(2011,2021),range(1,13))]))

        calc_grouping = ['day']
        ops = OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                            geom='state_boundaries',select_ugid=[25])
        ret = ops.execute()
        ref = ret[25]['tasmax'].temporal
        rdt = ref.value_datetime
        self.assertTrue(np.all(rdt == [dt(2011,1,day,12) for day in range(1,32)]))
        
        calc_grouping = ['month','day']
        ops = OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                            geom='state_boundaries',select_ugid=[25])
        ret = ops.execute()
        ref = ret[25]['tasmax'].temporal
        rdt = ref.value_datetime
        self.assertEqual(rdt[0],dt(2011,1,1,12))
        self.assertEqual(rdt[12],dt(2011,1,13,12))

        calc_grouping = ['year','day']
        ops = OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                            geom='state_boundaries',select_ugid=[25])
        ret = ops.execute()
        ref = ret[25]['tasmax'].temporal
        rdt = ref.value_datetime
        self.assertEqual(rdt[0],dt(2011,1,1,12))

        rd = self.test_data.get_rd('cancm4_tasmax_2011',kwds={'time_region':{'month':[1],'year':[2011]}})
        field = rd.get()
        calc_grouping = ['month','day','year']
        ops = OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                            geom='state_boundaries',select_ugid=[25])
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
    
    def test_agg_raw(self):
        grouping = ['month']
        funcs = [{'func':'threshold','name':'threshold','ref':Threshold,'kwds':{'operation':'gte','threshold':200}}]
        raws = [True,False]
        aggs = [True,False]
        for raw,agg in itertools.product(raws,aggs):
            coll = self.get_collection(aggregate=agg)
            ce = OcgCalculationEngine(grouping,funcs,raw,agg)
            ret = ce.execute(coll)
            value = ret[25]['tas'].variables['threshold_tas'].value
            ## aggregated data should have a (1,1) spatial dimension
            if agg is True:
                self.assertNumpyAll(value.shape[-2:],(1,1))


#class TestDynamicDailyKernelPercentileThreshold(TestBase):
#    
#    def get_percentile_reference(self):
#        years = [2001,2002,2003]
#        days = [3,4,5,6,7]
#        
#        dates = []
#        for year,day in itertools.product(years,days):
#            dates.append(datetime.datetime(year,6,day,12))
#            
#        ds = nc.Dataset(self.test_data.get_uri('cancm4_tas'))
#        try:
#            calendar = ds.variables['time'].calendar
#            units = ds.variables['time'].units
#            ncdates = nc.num2date(ds.variables['time'][:],units,calendar=calendar)
#            indices = []
#            for ii,ndate in enumerate(ncdates):
#                if ndate in dates:
#                    indices.append(ii)
#            tas = ds.variables['tas'][indices,:,:]
#            ret = np.percentile(tas,10,axis=0)
#        finally:
#            ds.close()
#            
#        return(ret)
#    
#    def test_constructor(self):
#        DynamicDailyKernelPercentileThreshold()
#        
#    def test_get_calendar_day_window_(self):
#        cday_index = np.arange(0,365)
#        rr = DynamicDailyKernelPercentileThreshold._get_calendar_day_window_
#        width = 5
#        
#        target_cday_index = 0
#        ret = rr(cday_index,target_cday_index,width)
#        self.assertNumpyAll(ret,np.array([0,363,364,1,2]))
#        
#        target_cday_index = 15
#        ret = rr(cday_index,target_cday_index,width)
#        self.assertNumpyAll(ret,np.array([15,13,14,16,17]))
#        
#        target_cday_index = 363
#        ret = rr(cday_index,target_cday_index,width)
#        self.assertNumpyAll(ret,np.array([363,361,362,0,364]))
#    
#    def test_calculate(self):
#        ## daily data for three years is wanted for the test. subset a CMIP5
#        ## decadal simulation to use for input into the computation.
#        rd = self.test_data.get_rd('cancm4_tas')
#        ds = rd.ds.get_subset(temporal=[datetime.datetime(2001,1,1),
#                                        datetime.datetime(2003,12,31,23,59)])
#        ## the calculation will be for months and years. set the temporal grouping.
#        ds.temporal.set_grouping(['month','year'])
#        ## create calculation object
#        percentile = 10
#        width = 5
#        operation = 'lt'
#        kwds = dict(percentile=percentile,width=width,operation=operation)
#        dkp = DynamicDailyKernelPercentileThreshold(values=ds.value,groups=ds.temporal.group.dgroups,
#                                                    kwds=kwds,dataset=ds,calc_name='tg10p')
#        
#        dperc = dkp.daily_percentile
#        select = np.logical_and(dperc['month'] == 6,dperc['day'] == 5)
#        to_test = dperc[select]['percentile'][0]
#        ref = self.get_percentile_reference()
#        self.assertNumpyAll(to_test,ref)
#        
#        ret = dkp.calculate()
#        self.assertEqual(ret.shape,(36,1,64,128))
#        
#    def test_operations(self):
#        raise(SkipTest('dev'))
#        uri = self.test_data.get_uri('cancm4_tas')
#        rd = RequestDataset(uri=uri,
#                            variable='tas',
##                            time_range=[datetime.datetime(2001,1,1),datetime.datetime(2003,12,31,23,59)]
#                            )
#        calc_grouping = ['month','year']
#        calc = [{'func':'dynamic_kernel_percentile_threshold','name':'tg10p','kwds':{'percentile':10,'width':5,'operation':'lt'}}]
#        ops = OcgOperations(dataset=rd,calc_grouping=calc_grouping,calc=calc,
#                            output_format='nc')
#        ret = ops.execute()
#        
#    def test_operations_two_steps(self):
#        ## get the request dataset to use as the basis for the percentiles
#        uri = self.test_data.get_uri('cancm4_tas')
#        variable = 'tas'
#        rd = RequestDataset(uri=uri,variable=variable)
#        ## this is the underly OCGIS dataset object
#        nc_basis = rd.ds
#        
#        ## NOTE: if you want to subset the basis by time, this step is necessary
##        nc_basis = nc_basis.get_subset(temporal=[datetime.datetime(2001,1,1),
##                                                 datetime.datetime(2003,12,31,23,59)])
#        
#        ## these are the values to use when calculating the percentile basis. it
#        ## may be good to wrap this in a function to have memory freed after the
#        ## percentile structure array is computed.
#        all_values = nc_basis.value
#        ## these are the datetime objects used for window creation
#        temporal = nc_basis.temporal.value_datetime
#        ## additional parameters for calculating the basis
#        percentile = 10
#        width = 5
#        ## get the structure array
#        from ocgis.calc.library import DynamicDailyKernelPercentileThreshold
#        daily_percentile = DynamicDailyKernelPercentileThreshold.get_daily_percentile(all_values,temporal,percentile,width)
#        
#        ## perform the calculation using the precomputed basis. in this case,
#        ## the basis and target datasets are the same, so the RequestDataset is
#        ## reused.
#        calc_grouping = ['month','year']
#        kwds = {'percentile':percentile,'width':width,'operation':'lt','daily_percentile':daily_percentile}
#        calc = [{'func':'dynamic_kernel_percentile_threshold','name':'tg10p','kwds':kwds}]
#        ops = OcgOperations(dataset=rd,calc_grouping=calc_grouping,calc=calc,
#                            output_format='nc')
#        ret = ops.execute()
#        
#        ## if we want to return the values as a three-dimenional numpy array the
#        ## method below will do this. note the interface arrangement for the next
#        ## release will alter this slightly.
#        ops = OcgOperations(dataset=rd,calc_grouping=calc_grouping,calc=calc,
#                            output_format='numpy')
#        arrs = ops.execute()
#        ## reference the returned numpy data. the first key is the geometry identifier.
#        ## 1 in this case as this is the default for no selection geometry. the second
#        ## key is the variable alias and the third is the calculation name.
#        tg10p = arrs[1].calc['tas']['tg10p']
#        ## if we want the date information for the temporal groups (again will
#        ## change in the next release to be more straightfoward)
#        date_groups = arrs[1].variables['tas'].temporal.group.value
#        assert(date_groups.shape[0] == tg10p.shape[0])
#        ## these are the representative datetime objects
#        rep_dt = arrs[1].variables['tas'].temporal.group.representative_datetime
#        ## and these are the lower and upper time bounds on the date groups
#        bin_bounds = arrs[1].variables['tas'].temporal.group.bounds
#        
#        ## confirm we have values for each month and year (12*10)
#        ret_ds = nc.Dataset(ret)
#        try:
#            self.assertEqual(ret_ds.variables['tg10p'].shape,(120,64,128))
#        finally:
#            ret_ds.close()
            

if __name__ == '__main__':
    unittest.main()
