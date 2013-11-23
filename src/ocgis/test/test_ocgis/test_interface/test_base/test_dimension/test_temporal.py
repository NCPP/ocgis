from ocgis.test.base import TestBase
from datetime import datetime as dt
from ocgis.interface.base.dimension.temporal import TemporalDimension
import numpy as np
from ocgis.util.helpers import get_date_list
import datetime


class TestTemporalDimension(TestBase):
    
    def test_get_time_region_value_only(self):
        dates = get_date_list(dt(2002,1,31),dt(2009,12,31),1)
        td = TemporalDimension(value=dates)
        
        ret,indices = td.get_time_region({'month':[8]},return_indices=True)
        self.assertEqual(set([8]),set([d.month for d in ret.value.flat]))
        
        ret,indices = td.get_time_region({'year':[2008,2004]},return_indices=True)
        self.assertEqual(set([2008,2004]),set([d.year for d in ret.value.flat]))
        
        ret,indices = td.get_time_region({'day':[20,31]},return_indices=True)
        self.assertEqual(set([20,31]),set([d.day for d in ret.value.flat]))
        
        ret,indices = td.get_time_region({'day':[20,31],'month':[9,10],'year':[2003]},return_indices=True)
        self.assertNumpyAll(ret.value,[dt(2003,9,20),dt(2003,10,20),dt(2003,10,31,)])
        self.assertEqual(ret.shape,indices.shape)
        
        self.assertEqual(ret.extent,(datetime.datetime(2003,9,20),datetime.datetime(2003,10,31)))


class TestTemporalGroupDimension(TestBase):
    
    def test_constructor_by_temporal_dimension(self):
        value = [dt(2012,1,1),dt(2012,1,2)]
        td = TemporalDimension(value=value)
        tgd = td.get_grouping(['month'])
        self.assertEqual(tuple(tgd.date_parts[0]),(None,1,None,None,None,None,None))
        self.assertTrue(tgd.dgroups[0].all())
        self.assertNumpyAll(tgd.uid,np.array([1]))
