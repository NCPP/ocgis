from ocgis.conv.nc import NcConverter
import numpy as np
from ocgis.test.test_simple.test_simple import nc_scope
from ocgis.test.test_ocgis.test_conv.test_base import AbstractTestConverter


class Test(AbstractTestConverter):
    
    def test_overflow_with_fill_value(self):
        ## ensure that if the default fill value is not convertible to the default
        ## for OCGIS that the default netCDF4-python fill value is used.
        coll = self.get_spatial_collection()
        ref = coll[1]['tas'].variables['tas']
        ref._dtype = np.int32
        ref._value = ref.value.astype(np.int32)
        ncconv = NcConverter([coll],self._test_dir,'ocgis_output')
        ret = ncconv.write()
        with nc_scope(ret) as ds:
            var = ds.variables['tas']
            self.assertEqual(var.dtype,np.dtype('int32'))
            self.assertEqual(var.shape,(1,1,1))