from ocgis.test.base import TestBase
from datetime import datetime as dt
from ocgis.interface.base.dimension.temporal import TemporalDimension
import numpy as np
from ocgis.util.helpers import get_date_list
import datetime


class TestTemporalDimension(TestBase):
    
    def test_time_range_subset(self):
        dt1 = datetime.datetime(1950,01,01,12)
        dt2 = datetime.datetime(1950,12,31,12)
        dates = np.array(get_date_list(dt1,dt2,1))
        r1 = datetime.datetime(1950,01,01)
        r2 = datetime.datetime(1950,12,31)
        td = TemporalDimension(value=dates)
        ret = td.get_between(r1,r2)
        self.assertEqual(ret.value[-1],datetime.datetime(1950,12,30,12,0))
        delta = datetime.timedelta(hours=12)
        lower = dates - delta
        upper = dates + delta
        bounds = np.empty((lower.shape[0],2),dtype=object)
        bounds[:,0] = lower
        bounds[:,1] = upper
        td = TemporalDimension(value=dates,bounds=bounds)
        ret = td.get_between(r1,r2)
        self.assertEqual(ret.value[-1],datetime.datetime(1950,12,31,12,0))
    
    def test_seasonal_get_grouping(self):
        calc_grouping = [[6,7,8]]
        dates = get_date_list(dt(2012,4,1),dt(2012,10,31),1)
        td = TemporalDimension(value=dates)
        tg = td.get_grouping(calc_grouping)
        self.assertEqual(len(tg.value),1)
        selected_months = [s.month for s in td.value[tg.dgroups[0]].flat]
        not_selected_months = [s.month for s in td.value[np.invert(tg.dgroups[0])]]
        self.assertEqual(set(calc_grouping[0]),set(selected_months))
        self.assertFalse(set(not_selected_months).issubset(set(calc_grouping[0])))
        
        calc_grouping = [[4,5,6,7],[8,9,10]]
        tg = td.get_grouping(calc_grouping)
        self.assertEqual(len(tg.value),2)
        self.assertNumpyAll(tg.dgroups[0],np.invert(tg.dgroups[1]))
        
        calc_grouping = [[11,12,1]]
        dates = get_date_list(dt(2012,10,1),dt(2013,3,31),1)
        td = TemporalDimension(value=dates)
        tg = td.get_grouping(calc_grouping)
        selected_months = [s.month for s in td.value[tg.dgroups[0]].flat]
        self.assertEqual(set(calc_grouping[0]),set(selected_months))
        self.assertEqual(tg.value[0],dt(2012,12,16))
        
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        td = TemporalDimension(value=field.temporal.value_datetime)
        tg = td.get_grouping([[3,4,5]])
        self.assertEqual(tg.value[0],dt(2005,4,16))
    
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
