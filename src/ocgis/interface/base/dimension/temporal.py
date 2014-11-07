import base
import numpy as np
from collections import deque
import itertools
import datetime
from ocgis import constants
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.exc import EmptySubsetError, IncompleteSeasonError
from ocgis.util.helpers import get_is_date_between
from copy import deepcopy


class TemporalDimension(base.VectorDimension):
    _date_parts = ('year','month','day','hour','minute','second')
    _axis = 'T'
    
    def get_grouping(self,grouping):
        ## there is no need to go through the process of breaking out datetime
        ## parts when the grouping is 'all'.
        if grouping == 'all':
            new_bounds,date_parts,repr_dt,dgroups = self._get_grouping_all_()
        ## the process for getting "unique" seasons is also specialized
        elif 'unique' in grouping:
            new_bounds,date_parts,repr_dt,dgroups = self._get_grouping_seasonal_unique_(grouping)
        ## for standard groups ("['month']") or seasons across entire time range
        else:
            new_bounds,date_parts,repr_dt,dgroups = self._get_grouping_other_(grouping)
        
        tgd = self._get_temporal_group_dimension_(
                    grouping=grouping,date_parts=date_parts,bounds=new_bounds,
                    dgroups=dgroups,value=repr_dt,name_value='time',name_uid='tid',
                    name=self.name,meta=self.meta,units=self.units)
        
        return(tgd)
    
    def _get_grouping_seasonal_unique_(self, grouping):
        """
        :param list grouping: A seasonal list containing the unique flag.

        >>> grouping = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], 'unique']

        :returns: A tuple of elements necessary to create a :class:`ocgis.interface.base.dimension.temporal.TemporalGroupDimension`
         object.
        :rtype: tuple
        """

        # remove the unique keyword from the list
        grouping = list(deepcopy(grouping))
        grouping.remove('unique')
        grouping = get_sorted_seasons(grouping)
        # turn the seasons into time regions
        time_regions = get_time_regions(grouping, self._get_datetime_value_(), raise_if_incomplete=False)
        # holds the boolean selection arrays
        dgroups = deque()
        new_bounds = np.array([], dtype=object).reshape(-1, 2)
        repr_dt = np.array([], dtype=object)
        # return temporal dimensions and convert to groups
        for dgroup, sub in iter_boolean_groups_from_time_regions(time_regions, self, yield_subset=True,
                                                                 raise_if_incomplete=False):
            dgroups.append(dgroup)
            sub_value_datetime = sub._get_datetime_value_()
            new_bounds = np.vstack((new_bounds, [min(sub_value_datetime), max(sub_value_datetime)]))
            repr_dt = np.append(repr_dt, sub_value_datetime[int(sub.shape[0] / 2)])
        # no date parts yet...
        date_parts = None

        return new_bounds, date_parts, repr_dt, dgroups
    
    def _get_grouping_all_(self):
        '''
        Applied when the grouping is 'all'.
        '''
        
        value = self._get_datetime_value_()
        bounds = self._get_datetime_bounds_()
        try:
            lower = bounds.min()
            upper = bounds.max()
        ## bounds may be None
        except AttributeError:
            lower = value.min()
            upper = value.max()
        
        ## new bounds are simply the minimum and maximum values chosen either from
        ## the value or bounds array. bounds are given preference.
        new_bounds = np.array([lower,upper]).reshape(-1,2)
        ## date parts are not needed for the all case
        date_parts = None
        ## the group should be set to select all data.
        dgroups = [slice(None)]
        ## the representative datetime is the center of the value array.
        repr_dt = np.array([value[int((self.value.shape[0]/2)-1)]])
        
        return(new_bounds,date_parts,repr_dt,dgroups)
    
    def _get_grouping_other_(self,grouping):
        '''
        Applied to groups other than 'all'.
        '''
        
        ## map date parts to index positions in date part storage array and flip
        ## they key-value pairs
        group_map = dict(zip(range(0,len(self._date_parts)),self._date_parts,))
        group_map_rev = dict(zip(self._date_parts,range(0,len(self._date_parts)),))
        
        ## this array will hold the value data constructed differently depending
        ## on if temporal bounds are present
        value = np.empty((self.value.shape[0],3),dtype=object)
        
        ## reference the value and bounds datetime object arrays
        value_datetime = self._get_datetime_value_()
        value_datetime_bounds = self._get_datetime_bounds_()
        
        ## populate the value array depending on the presence of bounds
        if self.bounds is None:
            value[:,:] = value_datetime.reshape(-1,1)
        ## bounds are currently not used for the grouping mechanism
        else:
            value[:,0] = value_datetime_bounds[:,0]
            value[:,1] = value_datetime
            value[:,2] = value_datetime_bounds[:,1]
        
        def _get_attrs_(dt):
            return([dt.year,dt.month,dt.day,dt.hour,dt.minute,dt.second])
        
        ## extract the date parts
        parts = np.empty((len(self.value),len(self._date_parts)),dtype=int)
        for row in range(parts.shape[0]):
            parts[row,:] = _get_attrs_(value[row,1])
        
        ## grouping is different for date part combinations v. seasonal
        ## aggregation.
        if all([isinstance(ii,basestring) for ii in grouping]):
            unique = deque()
            for idx in range(parts.shape[1]):
                if group_map[idx] in grouping:
                    fill = np.unique(parts[:,idx])
                else:
                    fill = [None]
                unique.append(fill)
    
            select = deque()
            idx2_seq = range(len(self._date_parts))
            for idx in itertools.product(*[range(len(u)) for u in unique]):
                select.append([unique[idx2][idx[idx2]] for idx2 in idx2_seq])
            select = np.array(select)
            dgroups = deque()
            
            idx_cmp = [group_map_rev[group] for group in grouping]
                
            keep_select = []
            for idx in range(select.shape[0]):
                match = select[idx,idx_cmp] == parts[:,idx_cmp]
                dgrp = match.all(axis=1)
                if dgrp.any():
                    keep_select.append(idx)
                    dgroups.append(dgrp)
            select = select[keep_select,:]
            assert(len(dgroups) == select.shape[0])
            
            dtype = [(dp,object) for dp in self._date_parts]
        ## this is for seasonal aggregations
        else:
            ## we need to remove the year string from the grouping and do
            ## not want to modify the original list
            grouping = deepcopy(grouping)
            ## search for a year flag, which will break the temporal groups by
            ## years
            if 'year' in grouping:
                has_year = True
                grouping = list(grouping)
                grouping.remove('year')
                years = np.unique(parts[:,0])
            else:
                has_year = False
                years = [None]
            
            dgroups = deque()
            grouping_season = deque()

            # sort the arrays to ensure the ordered in ascending order
            years.sort()
            grouping = get_sorted_seasons(grouping, method='min')

            for year,season in itertools.product(years,grouping):
                subgroup = np.zeros(value.shape[0],dtype=bool)
                for idx in range(value.shape[0]):
                    if has_year:
                        if parts[idx,1] in season and year == parts[idx,0]:
                            subgroup[idx] = True
                    else:
                        if parts[idx,1] in season:
                            subgroup[idx] = True
                dgroups.append(subgroup)
                grouping_season.append([season,year])
            dtype = [('months',object),('year',int)]
            grouping = grouping_season
        
        ## init arrays to hold values and bounds for the grouped data
        new_value = np.empty((len(dgroups),),dtype=dtype)
        new_bounds = np.empty((len(dgroups),2),dtype=object)
        
        for idx,dgrp in enumerate(dgroups):
            ## tuple conversion is required for structure arrays: http://docs.scipy.org/doc/numpy/user/basics.rec.html#filling-structured-arrays
            try:
                new_value[idx] = tuple(select[idx])
            ## likely a seasonal aggregation with a different group representation
            except UnboundLocalError:
                try:
                    new_value[idx] = (grouping[idx][0],grouping[idx][1])
                ## there is likely no year associated with the seasonal aggregation
                ## and it is a Nonetype
                except TypeError:
                    new_value[idx]['months'] = grouping[idx][0]
            sel = value[dgrp][:,(0,2)]
            new_bounds[idx,:] = [sel.min(),sel.max()]
        
        new_bounds = np.atleast_2d(new_bounds).reshape(-1,2)
        date_parts = np.atleast_1d(new_value)
        ## this is the representative center time for the temporal group
        repr_dt = self._get_grouping_representative_datetime_(grouping,new_bounds,date_parts)
        
        return(new_bounds,date_parts,repr_dt,dgroups)
        
    def get_iter(self,*args,**kwds):
        r_name_value = self.name_value
        r_set_date_parts = self._set_date_parts_
        for ii,yld in super(TemporalDimension,self).get_iter(*args,**kwds):
            r_value = yld[r_name_value]
            r_set_date_parts(yld,r_value)
            yield(ii,yld)
            
    def _set_date_parts_(self,yld,value):
        yld['year'],yld['month'],yld['day'] = value.year,value.month,value.day
        
    def get_time_region(self,time_region,return_indices=False):
        assert(isinstance(time_region,dict))
        
        ## return the values to use for the temporal region subsetting.
        value = self._get_datetime_value_()
        bounds = self._get_datetime_bounds_()
        
        ## switch to indicate if bounds or centroid datetimes are to be used.
        use_bounds = False if bounds is None else True
        
        ## remove any none values in the time_region dictionary. this will save
        ## time in iteration.
        time_region = time_region.copy()
        time_region = {k:v for k,v in time_region.iteritems() if v is not None}
        assert(len(time_region) > 0)
        
        ## this is the boolean selection array.
        select = np.zeros(self.shape[0],dtype=bool)
        
        ## for each row, determine if the date criterion are met updating the
        ## select matrix accordingly.
        row_check = np.zeros(len(time_region),dtype=bool)
        
        for idx_row in range(select.shape[0]):
            ## do the comparison for each time_region element.
            if use_bounds:
                row = bounds[idx_row,:]
            else:
                row = value[idx_row]
            for ii,(k,v) in enumerate(time_region.iteritems()):
                if use_bounds:
                    to_include = []
                    for element in v:
                        kwds = {k:element}
                        to_include.append(get_is_date_between(row[0],row[1],**kwds))
                    fill = any(to_include)
                else:
                    part = getattr(row,k)
                    fill = True if part in v else False
                row_check[ii] = fill
            if row_check.all():
                select[idx_row] = True
                
        if not select.any():
            ocgis_lh(logger='nc.temporal',exc=EmptySubsetError(origin='temporal'))

        ret = self[select]
        
        if return_indices:
            raw_idx = np.arange(0,self.shape[0])[select]
            ret = (ret,raw_idx)
        
        return(ret)
    
    def _get_datetime_bounds_(self):
        '''Intended for subclasses to overload the method for accessing the datetime
        value. For example, netCDF times are floats that must be converted.'''
        return(self.bounds)
    
    def _get_datetime_value_(self):
        '''Intended for subclasses to overload the method for accessing the datetime
        value. For example, netCDF times are floats that must be converted.'''
        return(self.value)
    
    def _get_grouping_representative_datetime_(self,grouping,bounds,value):
        ref_value = value
        ref_bounds = bounds
        ret = np.empty((ref_value.shape[0],),dtype=object)
        try:
            set_grouping = set(grouping)
            if set_grouping == set(['month']):
                ref_calc_month_centroid = constants.calc_month_centroid
                for idx in range(ret.shape[0]):
                    month = ref_value[idx]['month']
                    ## get the start year from the bounds data
                    start_year = ref_bounds[idx][0].year
                    ## create the datetime object
                    ret[idx] = datetime.datetime(start_year,month,ref_calc_month_centroid)
            elif set_grouping == set(['year']):
                ref_calc_year_centroid_month = constants.calc_year_centroid_month
                ref_calc_year_centroid_day = constants.calc_year_centroid_day
                for idx in range(ret.shape[0]):
                    year = ref_value[idx]['year']
                    ## create the datetime object
                    ret[idx] = datetime.datetime(year,ref_calc_year_centroid_month,ref_calc_year_centroid_day)
            elif set_grouping == set(['month','year']):
                ref_calc_month_centroid = constants.calc_month_centroid
                for idx in range(ret.shape[0]):
                    year,month = ref_value[idx]['year'],ref_value[idx]['month']
                    ret[idx] = datetime.datetime(year,month,ref_calc_month_centroid)
            elif set_grouping == set(['day']):
                for idx in range(ret.shape[0]):
                    start_year,start_month = ref_bounds[idx][0].year,ref_bounds[idx][0].month
                    ret[idx] = datetime.datetime(start_year,start_month,ref_value[idx]['day'],12)
            elif set_grouping == set(['day','month']):
                for idx in range(ret.shape[0]):
                    start_year = ref_bounds[idx][0].year
                    day,month = ref_value[idx]['day'],ref_value[idx]['month']
                    ret[idx] = datetime.datetime(start_year,month,day,12)
            elif set_grouping == set(['day','year']):
                for idx in range(ret.shape[0]):
                    day,year = ref_value[idx]['day'],ref_value[idx]['year']
                    ret[idx] = datetime.datetime(year,1,day,12)
            elif set_grouping == set(['day','year','month']):
                for idx in range(ret.shape[0]):
                    day,year,month = ref_value[idx]['day'],ref_value[idx]['year'],ref_value[idx]['month']
                    ret[idx] = datetime.datetime(year,month,day,12)
            else:
                ocgis_lh(logger='interface.temporal',exc=NotImplementedError('grouping: {0}'.format(self.grouping)))
        ## likely a seasonal aggregation
        except TypeError:
            ## set for testing if seasonal group crosses the end of a year
            cross_months_set = set([12,1])
            for idx in range(ret.shape[0]):
                r_bounds = bounds[idx,:]
                ## if the season crosses into a new year, find the middles differently
                r_value_months = value[idx]['months']
                if cross_months_set.issubset(r_value_months):
                    middle_index = int(np.floor(len(r_value_months)/2))
                    center_month = r_value_months[middle_index]
                else:
                    center_month = int(np.floor(np.mean([r_bounds[0].month,r_bounds[1].month])))
                center_year = int(np.floor(np.mean([r_bounds[0].year,r_bounds[1].year])))
                fill = datetime.datetime(center_year,center_month,constants.calc_month_centroid)
                ret[idx] = fill
        return(ret)
    
    def _get_iter_value_bounds_(self):
        return(self._get_datetime_value_(),self._get_datetime_bounds_())
    
    def _get_temporal_group_dimension_(self,*args,**kwds):
        return(TemporalGroupDimension(*args,**kwds))


class TemporalGroupDimension(TemporalDimension):
    
    def __init__(self,*args,**kwds):
        self.grouping = kwds.pop('grouping')
        self.dgroups = kwds.pop('dgroups')
        self.date_parts = kwds.pop('date_parts')
                
        TemporalDimension.__init__(self,*args,**kwds)


def iter_boolean_groups_from_time_regions(time_regions, temporal_dimension, yield_subset=False,
                                          raise_if_incomplete=True):
    """
    :param time_regions: Sequence of nested time region dictionaries.

    >>> [[{'month':[1,2],'year':[2024]},...],...]

    :param temporal_dimension: A temporal dimension object.
    :type temporal_dimension: :class:`ocgis.interface.base.dimension.temporal.TemporalDimension`
    :param bool yield_subset: If ``True``, yield a tuple with the subset of ``temporal_dimension``.
    :param bool raise_if_incomplete: If ``True``, raise an exception if the season is incomplete.
    :returns: boolean ndarray vector with yld.shape == temporal_dimension.shape
    :raises: IncompleteSeasonError
    """

    for sub_time_regions in time_regions:
        # incomplete seasons are searched for in the nested loop. this indicates if a time region group should be
        # considered a season.
        is_complete = True
        idx_append = np.array([], dtype=int)
        for time_region in sub_time_regions:
            sub, idx = temporal_dimension.get_time_region(time_region, return_indices=True)
            ## insert a check to ensure there are months present for each time region
            months = set([d.month for d in sub._get_datetime_value_()])
            try:
                assert (months == set(time_region['month']))
            except AssertionError:
                if raise_if_incomplete:
                    for m in time_region['month']:
                        if m not in months:
                            raise IncompleteSeasonError(time_region, month=m)
                else:
                    is_complete = False
            idx_append = np.append(idx_append, idx)

        # if the season is complete append, otherwise pass to next iteration.
        if is_complete:
            dgroup = np.zeros(temporal_dimension.shape[0], dtype=bool)
            dgroup[idx_append] = True
        else:
            continue

        if yield_subset:
            yld = (dgroup, temporal_dimension[dgroup])
        else:
            yld = dgroup

        yield yld
        
def get_is_interannual(sequence):
    '''
    Returns ``True`` if an integer sequence representing a season crosses a year
    boundary.
    
    >>> sequence = [11,12,1]
    >>> get_is_interannual(sequence)
    True
    '''
    
    if 12 in sequence and 1 in sequence:
        ret = True
    else:
        ret = False
    return(ret)


def get_sorted_seasons(seasons, method='max'):
    """
    Sorts ``seasons`` sequence by ``method`` of season elements.

    >>> seasons = [[9,10,11],[12,1,2],[6,7,8]]
    >>> get_sorted_seasons(seasons)
    [[6,7,8],[9,10,11],[12,1,2]]

    :type seasons: list[list[int]]
    :type method: str
    :rtype: list[list[int]]
    """

    methods = {'min': min, 'max': max}

    season_map = {}
    for ii,season in enumerate(seasons):
        season_map[ii] = season
    max_map = {}
    for key,value in season_map.iteritems():
        max_map[methods[method](value)] = key
    sorted_maxes = sorted(max_map)
    ret = [seasons[max_map[s]] for s in sorted_maxes]
    ret = deepcopy(ret)
    return(ret)


def get_time_regions(seasons,dates,raise_if_incomplete=True):
    '''
    >>> seasons = [[6,7,8],[9,10,11],[12,1,2]]
    >>> dates = <vector of datetime objects>
    '''
    ## extract the years from the data vector collapsing them to a unique
    ## set then sort in ascending order
    years = list(set([d.year for d in dates]))
    years.sort()
    ## determine if any of the seasons are interannual
    interannual_check = map(get_is_interannual,seasons)
    ## holds the return value
    time_regions = []
    ## the interannual cases requires two time region sequences to
    ## properly extract
    if any(interannual_check):
        ## loop over years first to ensure each year is accounted for
        ## in the time region output
        for ii_year,year in enumerate(years):
            ## the interannual flag is used internally for simple optimization
            for ic,cg in itertools.izip(interannual_check,seasons):
                ## if no exception is raised for an incomplete season,
                ## this flag indicate whether to append to the output
                append_to_time_regions = True
                if ic:
                    ## copy and sort in descending order the season because
                    ## december of the current year should be first.
                    _cg = deepcopy(cg)
                    _cg.sort()
                    _cg.reverse()
                    ## look for the interannual break and split the season
                    ## into the current year and next year.
                    diff = np.abs(np.diff(_cg))
                    split_base = np.arange(1,len(_cg))
                    split_indices = split_base[diff > 1]
                    split = np.split(_cg,split_indices)
                    ## will hold the sub-element time regions
                    sub_time_region = []
                    for ii_split,s in enumerate(split):
                        try:
                            to_append_sub = {'year':[years[ii_year+ii_split]],'month':s.tolist()}
                            sub_time_region.append(to_append_sub)
                        ## there may not be another year of data for an
                        ## interannual season. we DO NOT keep incomplete
                        ## seasons.
                        except IndexError:
                            ## don't just blow through an incomplete season
                            ## unless asked to
                            if raise_if_incomplete:
                                raise(IncompleteSeasonError(_cg,year))
                            else:
                                append_to_time_regions = False
                                continue
                    to_append = sub_time_region
                else:
                    to_append = [{'year':[year],'month':cg}]
                if append_to_time_regions:
                    time_regions.append(to_append)
    ## without interannual seasons the time regions are unique combos of
    ## the years and seasons designations
    else:
        for year,season in itertools.product(years,seasons):
            time_regions.append([{'year':[year],'month':season}])
            
    return(time_regions)