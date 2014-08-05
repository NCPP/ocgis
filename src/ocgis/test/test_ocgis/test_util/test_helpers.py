from collections import OrderedDict
import os
import numpy as np
from ocgis.util.helpers import format_bool, iter_array, validate_time_subset,\
    get_formatted_slice, get_is_date_between, get_trimmed_array_by_mask,\
    get_added_slice, get_iter, get_ordered_dicts_from_records_array, get_sorted_uris_by_time_dimension
import itertools
from ocgis.test.base import TestBase
from datetime import datetime as dt


class Test(TestBase):

    def test_get_sorted_uris_by_time_dimension(self):
        rd_2001 = self.test_data.get_rd('cancm4_tasmax_2001')
        rd_2011 = self.test_data.get_rd('cancm4_tasmax_2011')
        not_sorted = [rd_2011.uri, rd_2001.uri]

        actual = ['tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc',
                  'tasmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc']

        for variable in [None, 'tasmax']:
            is_sorted = get_sorted_uris_by_time_dimension(not_sorted, variable=variable)
            to_test = [os.path.split(ii)[1] for ii in is_sorted]
            self.assertEqual(actual, to_test)

    def test_get_ordered_dicts_from_records_array(self):
        """Test converting a records array to a list of ordered dictionaries."""

        arr = np.zeros(2, dtype=[('UGID', int), ('NAME', object)])
        arr[0] = (1, 'Name1')
        arr[1] = (2, 'Name2')
        ret = get_ordered_dicts_from_records_array(arr)
        self.assertListEqual(ret, [OrderedDict([('UGID', 1), ('NAME', 'Name1')]),
                                   OrderedDict([('UGID', 2), ('NAME', 'Name2')])])

    def test_get_iter_str(self):
        """Test whole string returned as opposed to its immutable elements."""

        itr = get_iter('hi')
        self.assertEqual(list(itr), ['hi'])

    def test_get_iter_numpy(self):
        """Test entire NumPy array returned versus its individual elements."""

        arr = np.array([1, 2, 3, 4])
        itr = get_iter(arr)
        self.assertNumpyAll(list(itr)[0], arr)

    def test_get_iter_dtype(self):
        """Test the dtype is properly used when determining how to iterate over elements."""

        class foo(object):
            pass

        dtypes = [dict, (dict,), (dict, foo)]

        data = {'hi': 'there'}

        for dtype in dtypes:
            itr = get_iter(data, dtype=dtype)
            self.assertDictEqual(list(itr)[0], data)

        ## a foo object should also be okay
        f = foo()
        itr = get_iter(f, dtype=foo)
        self.assertEqual(list(itr), [f])

        ## if no dtype is passed, then the builtin iterator of the element will be used
        itr = get_iter(data)
        self.assertEqual(list(itr), ['hi'])
    
    def test_get_added_slice(self):
        slice1 = slice(46,47)
        slice2 = slice(0,None)
        ret = get_added_slice(slice1,slice2)
        self.assertEqual(ret,slice(46,47))
        
        slice1 = slice(46,47)
        slice2 = slice(0,-1)
        ret = get_added_slice(slice1,slice2)
        self.assertEqual(ret,slice(46,46))
        
        slice1 = slice(0,47)
        slice2 = slice(2,-3)
        ret = get_added_slice(slice1,slice2)
        self.assertEqual(ret,slice(2,44))
        
        slice1 = slice(0,47,3)
        slice2 = slice(2,-3)
        with self.assertRaises(AssertionError):
            get_added_slice(slice1,slice2)
    
    def test_get_trimmed_array_by_mask_by_bool(self):
        arr = np.zeros((4,4),dtype=bool)
        arr[-1,:] = True
        ret = get_trimmed_array_by_mask(arr)
        self.assertFalse(ret.any())
        
    def test_get_trimmed_array_by_mask_bad_type(self):
        arr = np.zeros((4,4))
        with self.assertRaises(NotImplementedError):
            get_trimmed_array_by_mask(arr)
    
    def test_get_trimmed_array_by_mask_row_only(self):
        arr = np.random.rand(4,4)
        arr = np.ma.array(arr,mask=False)
        arr.mask[0,:] = True
        arr.mask[-1,:] = True
        ret = get_trimmed_array_by_mask(arr)
        self.assertNumpyAll(ret,arr[1:-1,:])
        self.assertTrue(np.may_share_memory(ret,arr))
    
    def test_get_trimmed_array_by_mask_rows_and_columns(self):
        arr = np.random.rand(4,4)
        arr = np.ma.array(arr,mask=False)
        arr.mask[0,:] = True
        arr.mask[-1,:] = True
        arr.mask[:,0] = True
        arr.mask[:,-1] = True
        ret = get_trimmed_array_by_mask(arr)
        self.assertNumpyAll(ret,arr[1:-1,1:-1])
        self.assertTrue(np.may_share_memory(ret,arr))
        ret,adjust = get_trimmed_array_by_mask(arr,return_adjustments=True)
        self.assertEqual(adjust,{'col': slice(1, -1), 'row': slice(1, -1)})
    
    def test_get_trimmed_array_by_mask_none_masked(self):
        arr = np.random.rand(4,4)
        arr = np.ma.array(arr,mask=False)
        ret,adjust = get_trimmed_array_by_mask(arr,return_adjustments=True)
        self.assertNumpyAll(ret,arr)
        self.assertTrue(np.may_share_memory(ret,arr))
        self.assertEqual(adjust,{'col': slice(0, None), 'row': slice(0, None)})
    
    def test_get_trimmed_array_by_mask_interior_masked(self):
        arr = np.random.rand(4,4)
        arr = np.ma.array(arr,mask=False)
        arr[2,:] = True
        arr[1,:] = True
        ret = get_trimmed_array_by_mask(arr)
        self.assertNumpyAll(ret,arr)
        self.assertTrue(np.may_share_memory(ret,arr))
    
    def test_get_trimmed_array_by_mask_all_masked(self):
        arr = np.random.rand(4,4)
        arr = np.ma.array(arr,mask=True)
        ret,adjust = get_trimmed_array_by_mask(arr,return_adjustments=True)
        self.assertEqual(ret.shape,(0,0))
        self.assertEqual(adjust,{'col': slice(4, -5), 'row': slice(4, -5)})
    
    def test_get_is_date_between(self):
        lower = dt(1971,1,1)
        upper = dt(2000,2,1)
        self.assertFalse(get_is_date_between(lower,upper,month=6))
        self.assertFalse(get_is_date_between(lower,upper,month=2))
        self.assertTrue(get_is_date_between(lower,upper,month=1))
        
        self.assertFalse(get_is_date_between(lower,upper,year=1968))
        self.assertTrue(get_is_date_between(lower,upper,year=1995))
        
        lower = dt(2013, 1, 1, 0, 0)
        upper = dt(2013, 1, 2, 0, 0)
        self.assertTrue(get_is_date_between(lower,upper,year=2013))
            
    def test_get_formatted_slc(self):
        ret = get_formatted_slice(slice(None,None,None),10)
        self.assertEqual(ret,[slice(None,None,None)]*10)
        
        ret = get_formatted_slice(0,1)
        self.assertEqual(slice(0,1),ret)
        with self.assertRaises(IndexError):
            get_formatted_slice(slice(0,1),2)
            
        ret = get_formatted_slice((slice(0,1),0),2)
        self.assertEqual(ret,[slice(0,1,None),slice(0,1,None)])
        
        ret = get_formatted_slice([(1,2,3),slice(None)],2)
        self.assertNumpyAll(ret[0],np.arange(1,4))
        self.assertEqual(ret[1],slice(None))
        self.assertEqual(len(ret),2)
        
        ret = get_formatted_slice((1,2),1)
        self.assertNumpyAll(ret,np.array([1,2]))
        
        ret = get_formatted_slice((1,),1)
        self.assertEqual(ret,slice(1))
    
    def test_validate_time_subset(self):
        time_range = [dt(2000,1,1),dt(2001,1,1)]
        self.assertTrue(validate_time_subset(time_range,{'year':[2000,2001]}))
        self.assertFalse(validate_time_subset(time_range,{'year':[2000,2001,2002]}))
        self.assertTrue(validate_time_subset(time_range,{'month':[6,7,8]}))
        self.assertTrue(validate_time_subset(time_range,{'month':[6,7,8],'year':[2000]}))
        self.assertFalse(validate_time_subset(time_range,{'month':[6,7,8],'year':[2008]}))
        self.assertFalse(validate_time_subset([dt(2000,1,1),dt(2000,2,1)],{'month':[6,7,8],'year':[2008]}))
        self.assertTrue(validate_time_subset([dt(2000,1,1),dt(2000,2,1)],None))

    def test_iter_array(self):
        arrays = [
                  1,
                  [[1,2],[1,2]],
                  np.array([1,2,3]),
                  np.array(1),
                  np.ma.array([1,2,3],mask=False),
                  np.ma.array([[1,2],[3,4]],mask=[[True,False],[False,True]]),
                  np.ma.array([[1,2],[3,4]],mask=True),
                 ]
        _flag1 = [
                  True,
                  False
                  ]
        _flag2 = [
                  True,
                  False
                  ]
        
        for arr,flag1,flag2 in itertools.product(arrays,_flag1,_flag2):
            for ret in iter_array(arr,use_mask=flag1,return_value=flag2):
                pass

        arr = np.ma.array([1,2,3],mask=True)
        ret = list(iter_array(arr))
        self.assertEqual(len(ret),0)
        arr = np.ma.array([1,2,3],mask=False)
        ret = list(iter_array(arr))
        self.assertEqual(len(ret),3)
        
        values = np.random.rand(2,2,4,4)
        mask = np.random.random_integers(0,1,values.shape)
        values = np.ma.array(values,mask=mask)
        for idx in iter_array(values):
            self.assertFalse(values.mask[idx])
        self.assertEqual(len(list(iter_array(values,use_mask=True))),len(values.compressed()))
        self.assertEqual(len(list(iter_array(values,use_mask=False))),len(values.data.flatten()))
        
    def test_format_bool(self):
        mmap = {0:False,1:True,'t':True,'True':True,'f':False,'False':False}
        for key,value in mmap.iteritems():
            ret = format_bool(key)
            self.assertEqual(ret,value)
