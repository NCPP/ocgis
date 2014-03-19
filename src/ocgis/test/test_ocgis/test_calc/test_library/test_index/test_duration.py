from ocgis.test.base import TestBase
from ocgis.calc.library.index.duration import Duration, FrequencyDuration
import numpy as np
from ocgis.exc import DefinitionValidationError
import ocgis
from ocgis.api.operations import OcgOperations
import csv
from ocgis.api.request.base import RequestDataset
import webbrowser
from ocgis.test.test_ocgis.test_calc.test_calc_general import AbstractCalcBase
from ocgis.test.test_base import longrunning


class TestDuration(AbstractCalcBase):

    def test_duration(self):
        duration = Duration()
        
        ## three consecutive days over 3
        values = np.array([1,2,3,3,3,1,1],dtype=float)
        values = self.get_reshaped(values)
        ret = duration.calculate(values,2,operation='gt',summary='max')
        self.assertEqual(3.0,ret.flatten()[0])
        
        ## no duration over the threshold
        values = np.array([1,2,1,2,1,2,1],dtype=float)
        values = self.get_reshaped(values)
        ret = duration.calculate(values,2,operation='gt',summary='max')
        self.assertEqual(0.,ret.flatten()[0])
        
        ## no duration over the threshold
        values = np.array([1,2,1,2,1,2,1],dtype=float)
        values = self.get_reshaped(values)
        ret = duration.calculate(values,2,operation='gte',summary='max')
        self.assertEqual(1.,ret.flatten()[0])
        
        ## average duration
        values = np.array([1,5,5,2,5,5,5],dtype=float)
        values = self.get_reshaped(values)
        ret = duration.calculate(values,4,operation='gte',summary='mean')
        self.assertEqual(2.5,ret.flatten()[0])
        
        ## add some masked values
        values = np.array([1,5,5,2,5,5,5],dtype=float)
        mask = [0,0,0,0,0,1,0]
        values = np.ma.array(values,mask=mask)
        values = self.get_reshaped(values)
        ret = duration.calculate(values,4,operation='gte',summary='max')
        self.assertEqual(2.,ret.flatten()[0])
        
        ## test with an actual matrix
        values = np.array([1,5,5,2,5,5,5,4,4,0,2,4,4,4,3,3,5,5,6,9],dtype=float)
        values = values.reshape(5,2,2)
        values = np.ma.array(values,mask=False)
        ret = duration.calculate(values,4,operation='gte',summary='mean')
        self.assertNumpyAll(np.array([ 4. ,  2. ,  1.5,  1.5]),ret.flatten())
    
    def test_standard_operations(self):
        ret = self.run_standard_operations(
         [{'func':'duration','name':'max_duration','kwds':{'operation':'gt','threshold':2,'summary':'max'}}],
         capture=True)
        for cap in ret:
            reraise = True
            if isinstance(cap['exception'],DefinitionValidationError):
                if cap['parms']['calc_grouping'] in [['month'],'all']:
                    reraise = False
            if reraise:
                raise(cap['exception'])
            
            
class TestFrequencyDuration(AbstractCalcBase):
    
    def test_constructor(self):
        FrequencyDuration()
    
    def test_calculate(self):        
        fduration = FrequencyDuration()
        
        values = np.array([1,2,3,3,3,1,1,3,3,3,4,4,1,4,4,1,10,10],dtype=float)
        values = self.get_reshaped(values)
        ret = fduration.calculate(values,threshold=2,operation='gt')
        self.assertEqual(ret.flatten()[0].dtype.names,('duration','count'))
        self.assertNumpyAll(np.array([2,3,5],dtype=np.int32),ret.flatten()[0]['duration'])
        self.assertNumpyAll(np.array([2,1,1],dtype=np.int32),ret.flatten()[0]['count'])
        
        calc = [{'func':'freq_duration','name':'freq_duration','kwds':{'operation':'gt','threshold':280}}]
        ret = self.run_standard_operations(calc,capture=True,output_format=None)
        for dct in ret:
            if isinstance(dct['exception'],NotImplementedError) and dct['parms']['aggregate']:
                pass
            elif isinstance(dct['exception'],DefinitionValidationError):
                if dct['parms']['output_format'] == 'nc' or dct['parms']['calc_grouping'] == ['month']:
                    pass
            else:
                raise(dct['exception'])
    
    @longrunning
    def test_real_data_multiple_datasets(self):
        ocgis.env.DIR_DATA = '/usr/local/climate_data'
        
        rd_tasmax = RequestDataset(uri='Maurer02new_OBS_tasmax_daily.1971-2000.nc',
                                   variable='tasmax',
                                   time_region={'year':[1991],'month':[7]})
        rd_tasmin = RequestDataset(uri='Maurer02new_OBS_tasmin_daily.1971-2000.nc',
                                   variable='tasmin',
                                   time_region={'year':[1991],'month':[7]})
        
        ops = OcgOperations(dataset=[rd_tasmax,rd_tasmin],
                            output_format='csv+',
                            calc=[{'name': 'Frequency Duration', 'func': 'freq_duration', 'kwds': {'threshold': 25.0, 'operation': 'gte'}}],
                            calc_grouping=['month','year'],
                            geom='us_counties',select_ugid=[2778],aggregate=True,
                            calc_raw=False,spatial_operation='clip',
                            headers=['did', 'ugid', 'gid', 'year', 'month', 'day', 'variable', 'calc_key', 'value'],)
        ret = ops.execute()
        
        with open(ret,'r') as f:
            reader = csv.DictReader(f)
            variables = [row['VARIABLE'] for row in reader]
        self.assertEqual(set(variables),set(['tasmax','tasmin']))
    
    @longrunning
    def test_real_data(self):
        uri = 'Maurer02new_OBS_tasmax_daily.1971-2000.nc'
        variable = 'tasmax'
        ocgis.env.DIR_DATA = '/usr/local/climate_data'
        
        for output_format in ['numpy','csv+','shp','csv']:
            ops = OcgOperations(dataset={'uri':uri,
                                         'variable':variable,
                                         'time_region':{'year':[1991],'month':[7]}},
                                output_format=output_format,prefix=output_format,
                                calc=[{'name': 'Frequency Duration', 'func': 'freq_duration', 'kwds': {'threshold': 15.0, 'operation': 'gte'}}],
                                calc_grouping=['month','year'],
                                geom='us_counties',select_ugid=[2778],aggregate=True,
                                calc_raw=False,spatial_operation='clip',
                                headers=['did', 'ugid', 'gid', 'year', 'month', 'day', 'variable', 'calc_key', 'value'],)
            ret = ops.execute()
            
            if output_format == 'numpy':
                ref = ret[2778]['tasmax'].variables['Frequency Duration'].value
                self.assertEqual(ref.compressed()[0].shape,(2,))
            
            if output_format == 'csv+':
                real = [{'COUNT': '1', 'UGID': '2778', 'DID': '1', 'CALC_KEY': 'freq_duration', 'MONTH': '7', 'DURATION': '7', 'GID': '2778', 'YEAR': '1991', 'VARIABLE': 'tasmax', 'DAY': '16'}, {'COUNT': '1', 'UGID': '2778', 'DID': '1', 'CALC_KEY': 'freq_duration', 'MONTH': '7', 'DURATION': '23', 'GID': '2778', 'YEAR': '1991', 'VARIABLE': 'tasmax', 'DAY': '16'}]
                with open(ret,'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                for row,real_row in zip(rows,real):
                    self.assertDictEqual(row,real_row)
