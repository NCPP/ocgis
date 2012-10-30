from django.test import TestCase
import unittest
from util.parms import *
import exc


class TestParms(TestCase):
    fixtures = ['cdata.json']

    def test_queryparm(self):
        name_map = {'actual_foo':'renamed_foo'}
        query = {'renamed_foo':'4|9'}
        
        self.assertRaises(exc.QueryParmError,QueryParm,
                          *(query,'actual_foo'),**{'nullable':False})
        
        qp = QueryParm(query,'actual_foo',name_map=name_map)
        self.assertEqual(qp.value,['4','9'])
        
        self.assertRaises(exc.ScalarError,
                          QueryParm,
                          *(query,'actual_foo'),
                          **{'name_map':name_map,'scalar':True})
        
        qp = QueryParm(query,'actual_foo',name_map=name_map,dtype=int)
        self.assertEqual(qp.value,[4,9])
        
    def test_uidparm(self):
        query = {'uid':'4'}
        uid = UidParm(query,'uid')
        self.assertEqual(uid.value,[u'/usr/local/climate_data/CanCM4/tas_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc'])
        
        query = {'uid':'4|5'}
        uid = UidParm(query,'uid')
        self.assertEqual(uid.value,[u'/usr/local/climate_data/CanCM4/tas_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',
                                    u'/usr/local/climate_data/CanCM4/tasmin_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc'])
        
        query = {'uid':'none'}
        self.assertRaises(exc.NotNullableError,UidParm,*(query,'uid'))
        

if __name__ == "__main__":
#    import sys;sys.argv = ['', 'TestParms.test_uidparm']
    unittest.main()