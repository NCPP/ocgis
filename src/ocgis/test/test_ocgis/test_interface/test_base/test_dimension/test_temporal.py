from ocgis.test.base import TestBase
from datetime import datetime as dt
from ocgis.interface.base.dimension.temporal import TemporalDimension,\
    get_is_interannual, get_sorted_seasons, get_time_regions,\
    iter_boolean_groups_from_time_regions
import numpy as np
from ocgis.util.helpers import get_date_list
import datetime
from collections import deque
import itertools
from ocgis.exc import IncompleteSeasonError


class TestTemporalDimension(TestBase):
    
    def get_temporal_dimension(self,add_bounds=True,start=None,stop=None,days=1):
        start = start or datetime.datetime(1899,1,1,12)
        stop = stop or datetime.datetime(1901,12,31,12)
        dates = get_date_list(start,stop,days=days)
        if add_bounds:
            delta = datetime.timedelta(hours=12)
            lower = np.array(dates) - delta
            upper = np.array(dates) + delta
            bounds = np.empty((lower.shape[0],2),dtype=object)
            bounds[:,0] = lower
            bounds[:,1] = upper
        else:
            bounds = None
        td = TemporalDimension(value=dates,bounds=bounds)
        return(td)
    
    def test_get_grouping(self):
        td = self.get_temporal_dimension()
        td = td.get_between(datetime.datetime(1900,1,1),datetime.datetime(1900,12,31,23,59))
        tgd = td.get_grouping(['year'])
        self.assertEqual(tgd.value,np.array([datetime.datetime(1900,7,1)]))
        
    def test_get_grouping_for_all(self):
        for b in [True,False]:
            td = self.get_temporal_dimension(add_bounds=b)
            tgd = td.get_grouping('all')
            self.assertEqual(tgd.dgroups,[slice(None)])
            self.assertEqual(td.value[546],tgd.value[0])
            if b:
                self.assertNumpyAll(tgd.bounds,np.array([[datetime.datetime(1899,1,1),
                                                          datetime.datetime(1902,1,1)]]))
            else:
                self.assertNumpyAll(tgd.bounds,np.array([[datetime.datetime(1899,1,1,12),
                                                          datetime.datetime(1901,12,31,12)]]))
    
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
        dates = get_date_list(dt(2012,4,1),dt(2012,10,31),1)
        td = TemporalDimension(value=dates)
        
        ## standard seasonal group
        calc_grouping = [[6,7,8]]
        tg = td.get_grouping(calc_grouping)
        self.assertEqual(len(tg.value),1)
        selected_months = [s.month for s in td.value[tg.dgroups[0]].flat]
        not_selected_months = [s.month for s in td.value[np.invert(tg.dgroups[0])]]
        self.assertEqual(set(calc_grouping[0]),set(selected_months))
        self.assertFalse(set(not_selected_months).issubset(set(calc_grouping[0])))
        
        ## seasons different sizes
        calc_grouping = [[4,5,6,7],[8,9,10]]
        tg = td.get_grouping(calc_grouping)
        self.assertEqual(len(tg.value),2)
        self.assertNumpyAll(tg.dgroups[0],np.invert(tg.dgroups[1]))
        
        ## crosses year boundary
        calc_grouping = [[11,12,1]]
        dates = get_date_list(dt(2012,10,1),dt(2013,3,31),1)
        td = TemporalDimension(value=dates)
        tg = td.get_grouping(calc_grouping)
        selected_months = [s.month for s in td.value[tg.dgroups[0]].flat]
        self.assertEqual(set(calc_grouping[0]),set(selected_months))
        self.assertEqual(tg.value[0],dt(2012,12,16))
        
        ## grab real data
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        td = TemporalDimension(value=field.temporal.value_datetime)
        tg = td.get_grouping([[3,4,5]])
        self.assertEqual(tg.value[0],dt(2005,4,16))
        
    def test_seasonal_get_grouping_year_flag(self):
        ## test with year flag
        dates = get_date_list(dt(2012,1,1),dt(2013,12,31),1)
        td = TemporalDimension(value=dates)
        calc_grouping = [[6,7,8],'year']
        tg = td.get_grouping(calc_grouping)
        self.assertEqual(tg.value.shape[0],2)
        self.assertEqual(str(tg.value),'[2012-07-16 00:00:00 2013-07-16 00:00:00]')
        sub0 = td.value[tg.dgroups[0]]
        self.assertEqual(str(sub0),'[2012-06-01 00:00:00 2012-06-02 00:00:00 2012-06-03 00:00:00\n 2012-06-04 00:00:00 2012-06-05 00:00:00 2012-06-06 00:00:00\n 2012-06-07 00:00:00 2012-06-08 00:00:00 2012-06-09 00:00:00\n 2012-06-10 00:00:00 2012-06-11 00:00:00 2012-06-12 00:00:00\n 2012-06-13 00:00:00 2012-06-14 00:00:00 2012-06-15 00:00:00\n 2012-06-16 00:00:00 2012-06-17 00:00:00 2012-06-18 00:00:00\n 2012-06-19 00:00:00 2012-06-20 00:00:00 2012-06-21 00:00:00\n 2012-06-22 00:00:00 2012-06-23 00:00:00 2012-06-24 00:00:00\n 2012-06-25 00:00:00 2012-06-26 00:00:00 2012-06-27 00:00:00\n 2012-06-28 00:00:00 2012-06-29 00:00:00 2012-06-30 00:00:00\n 2012-07-01 00:00:00 2012-07-02 00:00:00 2012-07-03 00:00:00\n 2012-07-04 00:00:00 2012-07-05 00:00:00 2012-07-06 00:00:00\n 2012-07-07 00:00:00 2012-07-08 00:00:00 2012-07-09 00:00:00\n 2012-07-10 00:00:00 2012-07-11 00:00:00 2012-07-12 00:00:00\n 2012-07-13 00:00:00 2012-07-14 00:00:00 2012-07-15 00:00:00\n 2012-07-16 00:00:00 2012-07-17 00:00:00 2012-07-18 00:00:00\n 2012-07-19 00:00:00 2012-07-20 00:00:00 2012-07-21 00:00:00\n 2012-07-22 00:00:00 2012-07-23 00:00:00 2012-07-24 00:00:00\n 2012-07-25 00:00:00 2012-07-26 00:00:00 2012-07-27 00:00:00\n 2012-07-28 00:00:00 2012-07-29 00:00:00 2012-07-30 00:00:00\n 2012-07-31 00:00:00 2012-08-01 00:00:00 2012-08-02 00:00:00\n 2012-08-03 00:00:00 2012-08-04 00:00:00 2012-08-05 00:00:00\n 2012-08-06 00:00:00 2012-08-07 00:00:00 2012-08-08 00:00:00\n 2012-08-09 00:00:00 2012-08-10 00:00:00 2012-08-11 00:00:00\n 2012-08-12 00:00:00 2012-08-13 00:00:00 2012-08-14 00:00:00\n 2012-08-15 00:00:00 2012-08-16 00:00:00 2012-08-17 00:00:00\n 2012-08-18 00:00:00 2012-08-19 00:00:00 2012-08-20 00:00:00\n 2012-08-21 00:00:00 2012-08-22 00:00:00 2012-08-23 00:00:00\n 2012-08-24 00:00:00 2012-08-25 00:00:00 2012-08-26 00:00:00\n 2012-08-27 00:00:00 2012-08-28 00:00:00 2012-08-29 00:00:00\n 2012-08-30 00:00:00 2012-08-31 00:00:00]')
        sub1 = td.value[tg.dgroups[1]]
        self.assertEqual(str(sub1),'[2013-06-01 00:00:00 2013-06-02 00:00:00 2013-06-03 00:00:00\n 2013-06-04 00:00:00 2013-06-05 00:00:00 2013-06-06 00:00:00\n 2013-06-07 00:00:00 2013-06-08 00:00:00 2013-06-09 00:00:00\n 2013-06-10 00:00:00 2013-06-11 00:00:00 2013-06-12 00:00:00\n 2013-06-13 00:00:00 2013-06-14 00:00:00 2013-06-15 00:00:00\n 2013-06-16 00:00:00 2013-06-17 00:00:00 2013-06-18 00:00:00\n 2013-06-19 00:00:00 2013-06-20 00:00:00 2013-06-21 00:00:00\n 2013-06-22 00:00:00 2013-06-23 00:00:00 2013-06-24 00:00:00\n 2013-06-25 00:00:00 2013-06-26 00:00:00 2013-06-27 00:00:00\n 2013-06-28 00:00:00 2013-06-29 00:00:00 2013-06-30 00:00:00\n 2013-07-01 00:00:00 2013-07-02 00:00:00 2013-07-03 00:00:00\n 2013-07-04 00:00:00 2013-07-05 00:00:00 2013-07-06 00:00:00\n 2013-07-07 00:00:00 2013-07-08 00:00:00 2013-07-09 00:00:00\n 2013-07-10 00:00:00 2013-07-11 00:00:00 2013-07-12 00:00:00\n 2013-07-13 00:00:00 2013-07-14 00:00:00 2013-07-15 00:00:00\n 2013-07-16 00:00:00 2013-07-17 00:00:00 2013-07-18 00:00:00\n 2013-07-19 00:00:00 2013-07-20 00:00:00 2013-07-21 00:00:00\n 2013-07-22 00:00:00 2013-07-23 00:00:00 2013-07-24 00:00:00\n 2013-07-25 00:00:00 2013-07-26 00:00:00 2013-07-27 00:00:00\n 2013-07-28 00:00:00 2013-07-29 00:00:00 2013-07-30 00:00:00\n 2013-07-31 00:00:00 2013-08-01 00:00:00 2013-08-02 00:00:00\n 2013-08-03 00:00:00 2013-08-04 00:00:00 2013-08-05 00:00:00\n 2013-08-06 00:00:00 2013-08-07 00:00:00 2013-08-08 00:00:00\n 2013-08-09 00:00:00 2013-08-10 00:00:00 2013-08-11 00:00:00\n 2013-08-12 00:00:00 2013-08-13 00:00:00 2013-08-14 00:00:00\n 2013-08-15 00:00:00 2013-08-16 00:00:00 2013-08-17 00:00:00\n 2013-08-18 00:00:00 2013-08-19 00:00:00 2013-08-20 00:00:00\n 2013-08-21 00:00:00 2013-08-22 00:00:00 2013-08-23 00:00:00\n 2013-08-24 00:00:00 2013-08-25 00:00:00 2013-08-26 00:00:00\n 2013-08-27 00:00:00 2013-08-28 00:00:00 2013-08-29 00:00:00\n 2013-08-30 00:00:00 2013-08-31 00:00:00]')
        
        ## test crossing year boundary
        for calc_grouping in [[[12,1,2],'year'],['year',[12,1,2]]]:
            tg = td.get_grouping(calc_grouping)
            self.assertEqual(str(tg.value),'[2012-01-16 00:00:00 2013-01-16 00:00:00]')
            self.assertEqual(str(tg.bounds),'[[2012-01-01 00:00:00 2012-12-31 00:00:00]\n [2013-01-01 00:00:00 2013-12-31 00:00:00]]')
            self.assertEqual(str(td.value[tg.dgroups[1]]),'[2013-01-01 00:00:00 2013-01-02 00:00:00 2013-01-03 00:00:00\n 2013-01-04 00:00:00 2013-01-05 00:00:00 2013-01-06 00:00:00\n 2013-01-07 00:00:00 2013-01-08 00:00:00 2013-01-09 00:00:00\n 2013-01-10 00:00:00 2013-01-11 00:00:00 2013-01-12 00:00:00\n 2013-01-13 00:00:00 2013-01-14 00:00:00 2013-01-15 00:00:00\n 2013-01-16 00:00:00 2013-01-17 00:00:00 2013-01-18 00:00:00\n 2013-01-19 00:00:00 2013-01-20 00:00:00 2013-01-21 00:00:00\n 2013-01-22 00:00:00 2013-01-23 00:00:00 2013-01-24 00:00:00\n 2013-01-25 00:00:00 2013-01-26 00:00:00 2013-01-27 00:00:00\n 2013-01-28 00:00:00 2013-01-29 00:00:00 2013-01-30 00:00:00\n 2013-01-31 00:00:00 2013-02-01 00:00:00 2013-02-02 00:00:00\n 2013-02-03 00:00:00 2013-02-04 00:00:00 2013-02-05 00:00:00\n 2013-02-06 00:00:00 2013-02-07 00:00:00 2013-02-08 00:00:00\n 2013-02-09 00:00:00 2013-02-10 00:00:00 2013-02-11 00:00:00\n 2013-02-12 00:00:00 2013-02-13 00:00:00 2013-02-14 00:00:00\n 2013-02-15 00:00:00 2013-02-16 00:00:00 2013-02-17 00:00:00\n 2013-02-18 00:00:00 2013-02-19 00:00:00 2013-02-20 00:00:00\n 2013-02-21 00:00:00 2013-02-22 00:00:00 2013-02-23 00:00:00\n 2013-02-24 00:00:00 2013-02-25 00:00:00 2013-02-26 00:00:00\n 2013-02-27 00:00:00 2013-02-28 00:00:00 2013-12-01 00:00:00\n 2013-12-02 00:00:00 2013-12-03 00:00:00 2013-12-04 00:00:00\n 2013-12-05 00:00:00 2013-12-06 00:00:00 2013-12-07 00:00:00\n 2013-12-08 00:00:00 2013-12-09 00:00:00 2013-12-10 00:00:00\n 2013-12-11 00:00:00 2013-12-12 00:00:00 2013-12-13 00:00:00\n 2013-12-14 00:00:00 2013-12-15 00:00:00 2013-12-16 00:00:00\n 2013-12-17 00:00:00 2013-12-18 00:00:00 2013-12-19 00:00:00\n 2013-12-20 00:00:00 2013-12-21 00:00:00 2013-12-22 00:00:00\n 2013-12-23 00:00:00 2013-12-24 00:00:00 2013-12-25 00:00:00\n 2013-12-26 00:00:00 2013-12-27 00:00:00 2013-12-28 00:00:00\n 2013-12-29 00:00:00 2013-12-30 00:00:00 2013-12-31 00:00:00]')
    
    def test_seasonal_get_grouping_unique_flag(self):
        ## test with year flag
        dates = get_date_list(dt(2012,1,1),dt(2013,12,31),1)
        td = TemporalDimension(value=dates)
        calc_grouping = [[6,7,8],'unique']
        tg = td.get_grouping(calc_grouping)

        time_region = {'year':[2012],'month':[6,7,8]}
        sub1,idx1 = td.get_time_region(time_region,return_indices=True)
        time_region = {'year':[2013],'month':[6,7,8]}
        sub2,idx2 = td.get_time_region(time_region,return_indices=True)
        base_select = np.zeros(td.shape[0],dtype=bool)
        dgroups = deque()
        
        for software,manual in itertools.izip(tg.dgroups,dgroups):
            self.assertNumpyAll(software,manual)
        self.assertEqual(len(tg.dgroups),2)
        self.assertEqual(tg.value.tolist(),[datetime.datetime(2012, 7, 17, 0, 0), datetime.datetime(2013, 7, 17, 0, 0)])
        self.assertEqual(tg.bounds.tolist(),[[datetime.datetime(2012, 6, 1, 0, 0), datetime.datetime(2012, 8, 31, 0, 0)], [datetime.datetime(2013, 6, 1, 0, 0), datetime.datetime(2013, 8, 31, 0, 0)]])
        
        dgroup1 = base_select.copy()
        dgroup1[idx1] = True
        dgroup2 = base_select.copy()
        dgroup2[idx2] = True
        
        dgroups.append(dgroup1)
        dgroups.append(dgroup2)
        
        tg = td.get_grouping([[6,7,8],'year'])
        for ii in range(len(tg.dgroups)):
            self.assertNumpyAll(tg.dgroups[ii],dgroups[ii])
        self.assertEqual(len(tg.dgroups),len(dgroups))
        
    def test_seasonal_get_grouping_unique_flag_winter_season(self):
        dt1 = datetime.datetime(1900,01,01)
        dt2 = datetime.datetime(1902,12,31)
        dates = get_date_list(dt1,dt2,days=1)
        td = TemporalDimension(value=dates)
        group = [[12,1,2],'unique']
        tg = td.get_grouping(group)
        self.assertEqual(tg.value.shape[0],2)
        self.assertEqual(tg.bounds.tolist(),[[datetime.datetime(1901, 1, 1, 0, 0), datetime.datetime(1901, 2, 28, 0, 0)], [datetime.datetime(1902, 1, 1, 0, 0), datetime.datetime(1902, 2, 28, 0, 0)]])
    
    def test_empty_season_with_year_missing_month(self):
        dt1 = datetime.datetime(1900,01,01)
        dt2 = datetime.datetime(1903,1,31)
        dates = get_date_list(dt1,dt2,days=1)
        td = TemporalDimension(value=dates)
        group = [[12,1,2],'unique']
        tg = td.get_grouping(group)
        ## there should be a month missing from the last season (february) and it should not be
        ## considered complete
        self.assertEqual(tg.value.shape[0],2)
        
    def test_get_boolean_groups_from_time_regions(self):
        dates = get_date_list(dt(2012,1,1),dt(2013,12,31),1)
        seasons = [[3,4,5],[6,7,8],[9,10,11],[12,1,2]]
        td = TemporalDimension(value=dates)
        time_regions = get_time_regions(seasons,dates,raise_if_incomplete=False)
                
        dgroups = list(iter_boolean_groups_from_time_regions(time_regions,td))
        ## the last winter season is not complete as it does not have enough years
        self.assertEqual(len(dgroups),7)
        
        to_test = []
        for dgroup in dgroups:
            sub = td[dgroup]
            ## (upper and lower values of time vector, count of elements in time group, the middle value of the vector)
            to_test.append([sub.extent, sub.shape[0], sub[sub.shape[0]/2].value[0]])
        correct = [[(datetime.datetime(2012, 3, 1, 0, 0), datetime.datetime(2012, 5, 31, 0, 0)), 92, datetime.datetime(2012, 4, 16, 0, 0)], [(datetime.datetime(2012, 6, 1, 0, 0), datetime.datetime(2012, 8, 31, 0, 0)), 92, datetime.datetime(2012, 7, 17, 0, 0)], [(datetime.datetime(2012, 9, 1, 0, 0), datetime.datetime(2012, 11, 30, 0, 0)), 91, datetime.datetime(2012, 10, 16, 0, 0)], [(datetime.datetime(2012, 12, 1, 0, 0), datetime.datetime(2013, 2, 28, 0, 0)), 90, datetime.datetime(2013, 1, 15, 0, 0)], [(datetime.datetime(2013, 3, 1, 0, 0), datetime.datetime(2013, 5, 31, 0, 0)), 92, datetime.datetime(2013, 4, 16, 0, 0)], [(datetime.datetime(2013, 6, 1, 0, 0), datetime.datetime(2013, 8, 31, 0, 0)), 92, datetime.datetime(2013, 7, 17, 0, 0)], [(datetime.datetime(2013, 9, 1, 0, 0), datetime.datetime(2013, 11, 30, 0, 0)), 91, datetime.datetime(2013, 10, 16, 0, 0)]]
        self.assertEqual(to_test,correct)
        
    def test_seasonal_get_time_regions(self):
        dates = get_date_list(dt(2012,1,1),dt(2013,12,31),1)
        
        ## two simple seasons
        calc_grouping = [[6,7,8],[9,10,11]]
        time_regions = get_time_regions(calc_grouping,dates)
        correct = [[{'month': [6, 7, 8], 'year': [2012]}], [{'month': [9, 10, 11], 'year': [2012]}], [{'month': [6, 7, 8], 'year': [2013]}], [{'month': [9, 10, 11], 'year': [2013]}]]
        self.assertEqual(time_regions,correct)
        
        ## add an interannual season at the back
        calc_grouping = [[6,7,8],[9,10,11],[12,1,2]]
        with self.assertRaises(IncompleteSeasonError):
            get_time_regions(calc_grouping,dates)
        time_regions = get_time_regions(calc_grouping,dates,raise_if_incomplete=False)
        correct = [[{'month': [6, 7, 8], 'year': [2012]}], [{'month': [9, 10, 11], 'year': [2012]}], [{'month': [12], 'year': [2012]}, {'month': [2, 1], 'year': [2013]}], [{'month': [6, 7, 8], 'year': [2013]}], [{'month': [9, 10, 11], 'year': [2013]}]]
        self.assertEqual(time_regions,correct)
        
        ## put the interannual season in the middle
        calc_grouping = [[9,10,11],[12,1,2],[6,7,8]]
        with self.assertRaises(IncompleteSeasonError):
            get_time_regions(calc_grouping,dates)
        time_regions = get_time_regions(calc_grouping,dates,raise_if_incomplete=False)
        correct = [[{'month': [9, 10, 11], 'year': [2012]}], [{'month': [12], 'year': [2012]}, {'month': [2, 1], 'year': [2013]}], [{'month': [6, 7, 8], 'year': [2012]}], [{'month': [9, 10, 11], 'year': [2013]}], [{'month': [6, 7, 8], 'year': [2013]}]]
        self.assertEqual(time_regions,correct)
        
        ## odd seasons, but covering the whole year
        calc_grouping = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
        time_regions = get_time_regions(calc_grouping,dates)
        correct = [[{'month': [1, 2, 3], 'year': [2012]}], [{'month': [4, 5, 6], 'year': [2012]}], [{'month': [7, 8, 9], 'year': [2012]}], [{'month': [10, 11, 12], 'year': [2012]}], [{'month': [1, 2, 3], 'year': [2013]}], [{'month': [4, 5, 6], 'year': [2013]}], [{'month': [7, 8, 9], 'year': [2013]}], [{'month': [10, 11, 12], 'year': [2013]}]]
        self.assertEqual(time_regions,correct)
        
        ## standard seasons
        calc_grouping = [[3,4,5],[6,7,8],[9,10,11],[12,1,2]]
        time_regions = get_time_regions(calc_grouping,dates,raise_if_incomplete=False)
        correct = [[{'month': [3, 4, 5], 'year': [2012]}], [{'month': [6, 7, 8], 'year': [2012]}], [{'month': [9, 10, 11], 'year': [2012]}], [{'month': [12], 'year': [2012]}, {'month': [2, 1], 'year': [2013]}], [{'month': [3, 4, 5], 'year': [2013]}], [{'month': [6, 7, 8], 'year': [2013]}], [{'month': [9, 10, 11], 'year': [2013]}]]
        self.assertEqual(time_regions,correct)
        
    def test_get_sorted_seasons(self):
        calc_grouping = [[9,10,11],[12,1,2],[6,7,8]]
        
        for perm in itertools.permutations(calc_grouping,r=3):
            ret = get_sorted_seasons(perm)
            self.assertEqual(ret,[[6, 7, 8], [9, 10, 11], [12, 1, 2]])
        
    def test_get_is_interannual(self):       
        self.assertTrue(get_is_interannual([11,12,1]))
        self.assertFalse(get_is_interannual([10,11,12]))
    
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
        self.assertNumpyAll(ret.value,np.array([dt(2003,9,20),dt(2003,10,20),dt(2003,10,31,)]))
        self.assertEqual(ret.shape,indices.shape)
        
        self.assertEqual(ret.extent,(datetime.datetime(2003,9,20),datetime.datetime(2003,10,31)))


class TestTemporalGroupDimension(TestBase):
    
    def test_constructor_by_temporal_dimension(self):
        value = [dt(2012,1,1),dt(2012,1,2)]
        td = TemporalDimension(value=value)
        tgd = td.get_grouping(['month'])
        self.assertEqual(tuple(tgd.date_parts[0]),(None,1,None,None,None,None))
        self.assertTrue(tgd.dgroups[0].all())
        self.assertNumpyAll(tgd.uid,np.array([1]))
