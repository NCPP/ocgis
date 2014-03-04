from ocgis.test.base import TestBase
from ocgis.api.collection import SpatialCollection
from ocgis.conv.nc import NcConverter
import numpy as np
from ocgis.test.test_simple.test_simple import nc_scope


class Test(TestBase):
    
    def test_overflow_with_fill_value(self):
        ## ensure that if the default fill value is not convertible to the default
        ## for OCGIS that the default netCDF4-python fill value is used.
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()[:,0,:,0,0]
        coll = SpatialCollection()
        coll.add_field(1,None,'tas',field)
        ref = coll[1]['tas'].variables['tas']
        ref._dtype = np.int32
        ref._value = ref.value.astype(np.int32)
        ncconv = NcConverter([coll],self._test_dir,'ocgis_output')
        ret = ncconv.write()
        with nc_scope(ret) as ds:
            var = ds.variables['tas']
            self.assertEqual(var.dtype,np.dtype('int32'))
            self.assertEqual(var.shape,(1,1,1))