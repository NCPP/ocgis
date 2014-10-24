from ocgis.conv.nc import NcConverter
import numpy as np
from ocgis.test.test_simple.test_simple import nc_scope
from ocgis.test.test_ocgis.test_conv.test_base import AbstractTestConverter
import ocgis


class Test(AbstractTestConverter):
    
    def test_fill_value_modified(self):
        ## test the fill value is appropriately copied if reset inside the field
        coll = self.get_spatial_collection()
        ref = coll[1]['tas'].variables['tas']
        ref._dtype = np.int32
        ref._value = ref.value.astype(np.int32)
        ref._fill_value = None
        ncconv = NcConverter([coll],self.current_dir_output,'ocgis_output')
        ret = ncconv.write()
        with nc_scope(ret) as ds:
            var = ds.variables['tas']
            self.assertEqual(var.dtype,np.dtype('int32'))
            self.assertEqual(var.shape,(1,1,1))
            self.assertEqual(var._FillValue,np.ma.array([],dtype=np.dtype('int32')).fill_value)
        
    def test_fill_value_copied(self):
        rd = self.test_data.get_rd('cancm4_tas')
        with nc_scope(rd.uri) as ds:
            fill_value_test = ds.variables['tas']._FillValue
        ops = ocgis.OcgOperations(dataset=rd,snippet=True,output_format='nc')
        ret = ops.execute()
        with nc_scope(ret) as ds:
            self.assertEqual(fill_value_test,ds.variables['tas']._FillValue)