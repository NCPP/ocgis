import unittest
from util.parms import QueryParm
import exc


class TestParms(unittest.TestCase):

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

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()