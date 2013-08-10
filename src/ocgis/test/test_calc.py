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
from ocgis.calc.library import Duration, FrequencyDuration, StandardDeviation
from ocgis.exc import DefinitionValidationError
import webbrowser
from ocgis.calc.engine import OcgCalculationEngine


class Test(TestBase):
    '''
    Test guide:
     * Data is required to be 4-dimensional:
         arr.shape = (t,1,m,n)
    '''
    
    def get_reshaped(self,arr):
        ret = arr.reshape(arr.shape[0],1,1,1)
        assert(len(ret.shape) == 4)
        ret = np.ma.array(ret,mask=False)
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
                    ref = ret[25].calc['tas'][calc[0]['name']]
                    if aggregate:
                        space_shape = [1,1]
                    else:
                        space_shape = [5,4]
                    if calc_grouping == ['month']:
                        shp1 = [12]
                    else:
                        shp1 = [24]
                    test_shape = shp1 + [1] + space_shape
                    self.assertEqual(ref.shape,tuple(test_shape))
                    if not aggregate:
                        self.assertTrue(np.ma.is_masked(ref[0,0,0,0]))
            except Exception as e:
                if capture:
                    parms = dict(aggregate=aggregate,calc_grouping=calc_grouping,output_format=output_format)
                    captured.append({'exception':e,'parms':parms})
                else:
                    raise
        return(captured)
    
    def test_frequency_duration(self):
        fduration = FrequencyDuration()
        
        values = np.array([1,2,3,3,3,1,1,3,3,3,4,4,1,4,4,1,10,10],dtype=float)
        values = self.get_reshaped(values)
        ret = fduration._calculate_(values,threshold=2,operation='gt')
        self.assertEqual(ret.flatten()[0].dtype.names,('duration','count'))
        self.assertNumpyAll(np.array([2,3,5],dtype=np.int32),ret.flatten()[0]['duration'])
        self.assertNumpyAll(np.array([2,1,1],dtype=np.int32),ret.flatten()[0]['count'])
        
        calc = [{'func':'freq_duration','name':'freq_duration','kwds':{'operation':'gt','threshold':280}}]
        ocgis.env.VERBOSE = False
        ret = self.run_standard_operations(calc,capture=True,output_format=None)
        for dct in ret:
            if isinstance(dct['exception'],NotImplementedError) and dct['parms']['aggregate']:
                pass
            elif isinstance(dct['exception'],DefinitionValidationError):
                if dct['parms']['output_format'] == 'nc' or dct['parms']['calc_grouping'] == ['month']:
                    pass
            else:
                raise(dct['exception'])
            
    def test_frequency_duration_real_data(self):
        uri = 'Maurer02new_OBS_tasmax_daily.1971-2000.nc'
        variable = 'tasmax'
        ocgis.env.DIR_DATA = '/usr/local/climate_data'
        
        for output_format in ['csv+','shp','csv']:
            ops = OcgOperations(dataset={'uri':uri,
                                         'variable':variable,
                                         'time_region':{'year':[1991],'month':[7]}},
                                output_format=output_format,prefix=output_format,
                                calc=[{'name': 'Frequency Duration', 'func': 'freq_duration', 'kwds': {'threshold': 25.0, 'operation': 'gte'}}],
                                calc_grouping=['month','year'],
                                geom='us_counties',select_ugid=[2778],aggregate=True,
                                calc_raw=False,spatial_operation='clip',
                                headers=['did', 'ugid', 'gid', 'year', 'month', 'day', 'variable', 'calc_name', 'value'],)
            ret = ops.execute()
    
    def test_duration(self):
        duration = Duration()
        
        ## three consecutive days over 3
        values = np.array([1,2,3,3,3,1,1],dtype=float)
        values = self.get_reshaped(values)
        ret = duration._calculate_(values,2,operation='gt',summary='max')
        self.assertEqual(3.0,ret.flatten()[0])
        
        ## no duration over the threshold
        values = np.array([1,2,1,2,1,2,1],dtype=float)
        values = self.get_reshaped(values)
        ret = duration._calculate_(values,2,operation='gt',summary='max')
        self.assertEqual(0.,ret.flatten()[0])
        
        ## no duration over the threshold
        values = np.array([1,2,1,2,1,2,1],dtype=float)
        values = self.get_reshaped(values)
        ret = duration._calculate_(values,2,operation='gte',summary='max')
        self.assertEqual(1.,ret.flatten()[0])
        
        ## average duration
        values = np.array([1,5,5,2,5,5,5],dtype=float)
        values = self.get_reshaped(values)
        ret = duration._calculate_(values,4,operation='gte',summary='mean')
        self.assertEqual(2.5,ret.flatten()[0])
        
        ## add some masked values
        values = np.array([1,5,5,2,5,5,5],dtype=float)
        mask = [0,0,0,0,0,1,0]
        values = np.ma.array(values,mask=mask)
        values = self.get_reshaped(values)
        ret = duration._calculate_(values,4,operation='gte',summary='max')
        self.assertEqual(2.,ret.flatten()[0])
        
        ## test with an actual matrix
        values = np.array([1,5,5,2,5,5,5,4,4,0,2,4,4,4,3,3,5,5,6,9],dtype=float)
        values = values.reshape(5,1,2,2)
        ret = duration._calculate_(values,4,operation='gte',summary='mean')
        self.assertNumpyAll(np.array([ 4. ,  2. ,  1.5,  1.5]),ret.flatten())
        
        ret = self.run_standard_operations(
         [{'func':'duration','name':'max_duration','kwds':{'operation':'gt','threshold':2,'summary':'max'}}],
         capture=True)
        for cap in ret:
            reraise = True
            if isinstance(cap['exception'],DefinitionValidationError):
                if cap['parms']['calc_grouping'] == ['month']:
                    reraise = False
            if reraise:
                raise(cap['exception'])
    
    def test_time_region(self):
        kwds = {'time_region':{'year':[2011]}}
        rd = self.test_data.get_rd('cancm4_tasmax_2011',kwds=kwds)
        calc = [{'func':'mean','name':'mean'}]
        calc_grouping = ['year','month']
        
        ops = ocgis.OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                                  geom='state_boundaries',select_ugid=[25])
        ret = ops.execute()
        
        tgroup = ret[25].variables['tasmax'].temporal.group.value
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
        threshold = ret[2762].calc['tasmax']['threshold']
        self.assertEqual(threshold.flatten()[0],62)
    
    def test_heat_index(self):
        ocgis.env.OVERWRITE = True
        kwds = {'time_range':[dt(2011,1,1),dt(2011,12,31,23,59,59)]}
        ds = [self.test_data.get_rd('cancm4_tasmax_2011',kwds=kwds),self.test_data.get_rd('cancm4_rhsmax',kwds=kwds)]
        calc = [{'func':'heat_index','name':'heat_index','kwds':{'tas':'tasmax','rhs':'rhsmax','units':'k'}}]
        geom = 'state_boundaries'
        select_ugid = [25]
        
        ## operations on entire data arrays
        ops = OcgOperations(dataset=ds,calc=calc)
        self.assertEqual(ops.calc_grouping,None)
        ret = ops.execute()
        ref = ret[1]
        self.assertEqual(ref.variables.keys(),['tasmax','rhsmax'])
        self.assertEqual(ref.calc.keys(),['heat_index'])
        hi = ref.calc['heat_index']
        self.assertEqual(hi.shape,(365,1,64,128))
        
        ## confirm no masked geometries
        self.assertFalse(ref._archetype.spatial.vector.geom.mask.any())
        ## confirm some masked data in calculation output
        self.assertTrue(hi.mask.any())
        
        ## snippet-based testing
        ops = OcgOperations(dataset=ds,calc=calc,snippet=True,geom=geom,select_ugid=select_ugid)
        ret = ops.execute()
        self.assertEqual(ret[25].calc['heat_index'].shape,(1,1,5,4))
        ops = OcgOperations(dataset=ds,calc=calc,snippet=True,output_format='csv')
        ret = ops.execute()
                
        # try temporal grouping
        ops = OcgOperations(dataset=ds,calc=calc,calc_grouping=['month'],geom='state_boundaries',select_ugid=select_ugid)
        ret = ops.execute()
        self.assertEqual(ret[25].calc['heat_index'].shape,(12,1,5,4))
        ret = OcgOperations(dataset=ds,calc=calc,calc_grouping=['month'],
                            output_format='csv',snippet=True).execute()

    def test_Mean(self):
        agg = True
        weights = None
        values = np.ones((36,2,4,4))
        values = np.ma.array(values,mask=False)
        
        on = np.ones(12,dtype=bool)
        off = np.zeros(12,dtype=bool)
        
        groups = []
        base_groups = [[on,off,off],[off,on,off],[off,off,on]]
        for bg in base_groups:
            groups.append(np.concatenate(bg))
        
        mean = library.Mean(values=values,agg=agg,weights=weights,groups=groups)
        ret = mean.calculate()
        
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

        ops = ocgis.OcgOperations(dataset={'uri':ret,'variable':calc[0]['name']},
                                  output_format='nc',prefix='subset_climatology')
        ret = ops.execute()
#        subprocess.check_call(['ncdump','-h',ret])
        ip = ocgis.Inspect(ret,variable='n')
        
        ds = nc.Dataset(ret,'r')
        ref = ds.variables['time'][:]
        self.assertEqual(len(ref),12)
        self.assertEqual(set(ds.variables['tasmax_mean'].ncattrs()),
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
        ref = ret[25].variables['tasmax'].temporal
        rdt = ref.group.representative_datetime
        self.assertTrue(np.all(rdt == np.array([dt(2011,month,16) for month in range(1,13)])))
        
        calc_grouping = ['year']
        ops = OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                            geom='state_boundaries',select_ugid=[25])
        ret = ops.execute()
        ref = ret[25].variables['tasmax'].temporal
        rdt = ref.group.representative_datetime
        self.assertTrue(np.all(rdt == [dt(year,7,1) for year in range(2011,2021)]))

        calc_grouping = ['month','year']
        ops = OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                            geom='state_boundaries',select_ugid=[25])
        ret = ops.execute()
        ref = ret[25].variables['tasmax'].temporal
        rdt = ref.group.representative_datetime
        self.assertTrue(np.all(rdt == [dt(year,month,16) for year,month in itertools.product(range(2011,2021),range(1,13))]))

        calc_grouping = ['day']
        ops = OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                            geom='state_boundaries',select_ugid=[25])
        ret = ops.execute()
        ref = ret[25].variables['tasmax'].temporal
        rdt = ref.group.representative_datetime
        self.assertTrue(np.all(rdt == [dt(2011,1,day,12) for day in range(1,32)]))
        
        calc_grouping = ['month','day']
        ops = OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                            geom='state_boundaries',select_ugid=[25])
        ret = ops.execute()
        ref = ret[25].variables['tasmax'].temporal
        rdt = ref.group.representative_datetime
        self.assertEqual(rdt[0],dt(2011,1,1,12))
        self.assertEqual(rdt[12],dt(2011,1,13,12))

        calc_grouping = ['year','day']
        ops = OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                            geom='state_boundaries',select_ugid=[25])
        ret = ops.execute()
        ref = ret[25].variables['tasmax'].temporal
        rdt = ref.group.representative_datetime
        self.assertEqual(rdt[0],dt(2011,1,1,12))

        rd = self.test_data.get_rd('cancm4_tasmax_2011',kwds={'time_region':{'month':[1],'year':[2011]}})
        calc_grouping = ['month','day','year']
        ops = OcgOperations(dataset=rd,calc=calc,calc_grouping=calc_grouping,
                            geom='state_boundaries',select_ugid=[25])
        ret = ops.execute()
        ref = ret[25].variables['tasmax'].temporal
        rdt = ref.group.representative_datetime
        self.assertTrue(np.all(rdt == ref.value_datetime))
        self.assertTrue(np.all(ref.bounds_datetime == ref.group.bounds))


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
        return(ret[25])
    
    def test_agg_raw(self):
        grouping = ['month']
        funcs = [{'func':'std','name':'std','ref':StandardDeviation,'kwds':{}}]
        raws = [True,False]
        aggs = [True,False]
        for raw,agg in itertools.product(raws,aggs):
            coll = self.get_collection(aggregate=agg)
            ce = OcgCalculationEngine(grouping,funcs,raw,agg)
            ret = ce.execute(coll)
            shape = ret.calc['tas']['std'].shape
            value,weights = ce._get_value_weights_(coll.variables['tas'])
            ## aggregated data should have a (1,1) spatial dimension
            if agg is True:
                self.assertNumpyAll(shape[-2:],(1,1))
            ## if raw data is used, the input values to a calculation should be
            ## returned with a different shape - aggregated spatial dimension
            if raw is True and agg is True:
                self.assertNumpyAll(value.shape[-2:],weights.shape)
                self.assertNumpyNotAll(value.shape[-2:],shape[-2:])
            if raw is True and agg is False:
                self.assertNumpyAll(shape[-3:],value.shape[-3:])

if __name__ == '__main__':
    unittest.main()
