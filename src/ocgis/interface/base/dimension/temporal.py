from collections import deque
import itertools
from copy import deepcopy
import netCDF4 as nc
import numpy as np

import netcdftime

import datetime
import base
from ocgis import constants
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.exc import EmptySubsetError, IncompleteSeasonError, CannotFormatTimeError
from ocgis.util.helpers import get_is_date_between, iter_array, get_none_or_slice


class TemporalDimension(base.VectorDimension):
    """
    .. note:: Accepts all parameters to :class:`~ocgis.interface.base.dimension.base.VectorDimension`.

    :keyword str calendar: (``='standard'``) The calendar to use when converting from float to datetime objects. Any of
     the netCDF-CF calendar tyes: http://unidata.github.io/netcdf4-python/netCDF4-module.html#num2date
    :keyword bool format_time: (``=True``) If ``False``, do not allow access to ``value_datetime``,
     ``bounds_datetime``, and ``extent_datetime``. If these properties are accessed raise
     :class:``~ocgis.exc.CannotFormatTimeError``.
    :keyword str units: (``='days since 0000-01-01 00:00:00'``) The units string to use when converting from float to
     datetime objects. See: http://unidata.github.io/netcdf4-python/netCDF4-module.html#num2date
    """

    _attrs_slice = ('uid', '_value', '_src_idx', '_value_datetime', '_value_numtime')
    _date_parts = ('year', 'month', 'day', 'hour', 'minute', 'second')

    def __init__(self, *args, **kwargs):
        self.calendar = kwargs.pop('calendar', constants.DEFAULT_TEMPORAL_CALENDAR)
        self.format_time = kwargs.pop('format_time', True)

        kwargs['axis'] = kwargs.get('axis') or 'T'
        kwargs['name'] = kwargs.get('name') or 'time'
        kwargs['name_uid'] = kwargs.get('name_uid') or 'tid'
        kwargs['units'] = kwargs.get('units') or constants.DEFAULT_TEMPORAL_UNITS

        super(TemporalDimension, self).__init__(*args, **kwargs)

        # test if the units are the special case with months in the time units
        if self.units.startswith('months'):
            self._has_months_units = True
        else:
            self._has_months_units = False

        self._value_datetime = None
        self._bounds_datetime = None
        self._value_numtime = None
        self._bounds_numtime = None

    @property
    def bounds_datetime(self):
        if not self.format_time:
            raise CannotFormatTimeError('bounds_datetime')
        if self.bounds is not None:
            if self._bounds_datetime is None:
                if get_datetime_conversion_state(self.bounds[0, 0]):
                    self._bounds_datetime = np.atleast_2d(self.get_datetime(self.bounds))
                else:
                    self._bounds_datetime = self.bounds
        return self._bounds_datetime

    @property
    def bounds_numtime(self):
        if self.bounds is not None:
            if self._bounds_numtime is None:
                if not get_datetime_conversion_state(self.bounds[0, 0]):
                    self._bounds_numtime = np.atleast_2d(self.get_numtime(self.bounds))
                else:
                    self._bounds_numtime = self.bounds
        return self._bounds_numtime

    @base.VectorDimension.conform_units_to.setter
    def conform_units_to(self, value):
        base.VectorDimension._conform_units_to_setter_(self, value)
        if self._conform_units_to is not None:
            self._conform_units_to.calendar = self.calendar

    @property
    def cfunits(self):
        ret = super(TemporalDimension, self).cfunits
        ret.calendar = self.calendar
        return ret

    @property
    def extent_datetime(self):
        if not self.format_time:
            raise CannotFormatTimeError('extent_datetime')
        extent = self.extent
        if get_datetime_conversion_state(extent[0]):
            extent = self.get_datetime(extent)
        return tuple(extent)

    @property
    def extent_numtime(self):
        extent = self.extent
        if not get_datetime_conversion_state(extent[0]):
            extent = self.get_numtime(extent)
        return tuple(extent)

    @property
    def value_datetime(self):
        if not self.format_time:
            raise CannotFormatTimeError('value_datetime')
        if self._value_datetime is None:
            if get_datetime_conversion_state(self.value[0]):
                self._value_datetime = np.atleast_1d(self.get_datetime(self.value))
            else:
                self._value_datetime = self.value
        return self._value_datetime

    @property
    def value_numtime(self):
        if self._value_numtime is None:
            if not get_datetime_conversion_state(self.value[0]):
                self._value_numtime = np.atleast_1d(self.get_numtime(self.value))
            else:
                self._value_numtime = self.value
        return self._value_numtime

    def get_between(self, lower, upper, return_indices=False):
        if get_datetime_conversion_state(self.value[0]):
            lower, upper = tuple(self.get_numtime([lower, upper]))
        return super(TemporalDimension, self).get_between(lower, upper, return_indices=return_indices)

    def get_datetime(self, arr):
        """
        :param arr: An array of floats to convert to datetime objects.
        :type arr: :class:`numpy.ndarray`
        :returns: An object array of the same shape as ``arr`` with float objects converted to datetime objects.
        :rtype: :class:`numpy.ndarray`
        """

        # if there are month units, call the special procedure to convert those to datetime objects
        if not self._has_months_units:
            arr = np.atleast_1d(nc.num2date(arr, self.units, calendar=self.calendar))
            dt = datetime.datetime
            for idx, t in iter_array(arr, return_value=True):
                # attempt to convert times to datetime objects
                try:
                    arr[idx] = dt(t.year, t.month, t.day, t.hour, t.minute, t.second)
                # this may fail for some calendars, in that case maintain the instance object returned from
                # netcdftime see: http://netcdf4-python.googlecode.com/svn/trunk/docs/netcdftime.netcdftime.datetime-class.html
                except ValueError:
                    arr[idx] = arr[idx]
        else:
            arr = get_datetime_from_months_time_units(arr, self.units, month_centroid=constants.CALC_MONTH_CENTROID)
        return arr

    def get_grouping(self, grouping):
        """
        :param sequence grouping: The temporal grouping to use when creating the temporal group dimension.

        >>> grouping = ['month']

        :returns: A temporal group dimension.
        :rtype: :class:`~ocgis.interface.base.dimension.temporal.TemporalGroupDimension`
        """

        # there is no need to go through the process of breaking out datetime parts when the grouping is 'all'.
        if grouping == 'all':
            new_bounds, date_parts, repr_dt, dgroups = self._get_grouping_all_()
        # the process for getting "unique" seasons is also specialized
        elif 'unique' in grouping:
            new_bounds, date_parts, repr_dt, dgroups = self._get_grouping_seasonal_unique_(grouping)
        # for standard groups ("['month']") or seasons across entire time range
        else:
            new_bounds, date_parts, repr_dt, dgroups = self._get_grouping_other_(grouping)

        tgd = self._get_temporal_group_dimension_(grouping=grouping, date_parts=date_parts, bounds=new_bounds,
                                                  dgroups=dgroups, value=repr_dt, name_value='time', name_uid='tid',
                                                  name=self.name, meta=self.meta, units=self.units)

        return tgd

    def get_iter(self, *args, **kwargs):
        r_name_value = self.name_value
        r_set_date_parts = self._set_date_parts_
        for ii, yld in super(TemporalDimension, self).get_iter(*args, **kwargs):
            r_value = yld[r_name_value]
            r_set_date_parts(yld, r_value)
            yield (ii, yld)

    def get_numtime(self, arr):
        """
        :param arr: An array of datetime objects to convert to numeric time.
        :type array: :class:`numpy.array`
        :returns: An array of numeric values with same shape as ``arr``.
        :rtype: :class:`numpy.array`
        """

        try:
            ret = np.atleast_1d(nc.date2num(arr, self.units, calendar=self.calendar))
        except ValueError:
            # special behavior for conversion of time units with months
            if self._has_months_units:
                ret = get_num_from_months_time_units(arr, self.units, dtype=None)
            else:
                raise
        return ret

    def get_time_region(self,time_region,return_indices=False):
        assert(isinstance(time_region,dict))

        ## return the values to use for the temporal region subsetting.
        value = self.value_datetime
        bounds = self.bounds_datetime

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

    def write_to_netcdf_dataset(self, dataset, **kwargs):
        """
        Calls superclass write method then adds ``calendar`` and ``units`` attributes to time variable and time bounds
        variable. See documentation for :meth:`~ocgis.interface.base.dimension.base.VectorDimension#write_to_netcdf_dataset`.
        """

        # swap the value/bounds references from datetime to numtime for the duration for the write
        if not get_datetime_conversion_state(self.value[0]):
            self._value = self.value_numtime
            self._bounds = self.bounds_numtime
            swapped_value_bounds = True
        else:
            swapped_value_bounds = False

        super(TemporalDimension, self).write_to_netcdf_dataset(dataset, **kwargs)

        # return the value and bounds to their original state
        if swapped_value_bounds:
            self._value = self.value_datetime
            self._bounds = self.bounds_datetime

        for name in [self.name_value, self.name_bounds]:
            try:
                variable = dataset.variables[name]
            except KeyError:
                # bounds are likely missing
                if self.bounds is not None:
                    raise
            variable.calendar = self.calendar
            variable.units = self.units

    def _format_slice_state_(self, state, slc):
        state = super(TemporalDimension, self)._format_slice_state_(state, slc)
        state._bounds_datetime = get_none_or_slice(state._bounds_datetime, (slc, slice(None)))
        state._bounds_numtime = get_none_or_slice(state._bounds_numtime, (slc, slice(None)))
        return state

    def _get_grouping_all_(self):
        '''
        Applied when the grouping is 'all'.
        '''

        value = self.value_datetime
        bounds = self.bounds_datetime
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
        value_datetime = self.value_datetime
        value_datetime_bounds = self.bounds_datetime

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

    def _get_grouping_representative_datetime_(self,grouping,bounds,value):
        ref_value = value
        ref_bounds = bounds
        ret = np.empty((ref_value.shape[0],),dtype=object)
        try:
            set_grouping = set(grouping)
            if set_grouping == set(['month']):
                ref_calc_month_centroid = constants.CALC_MONTH_CENTROID
                for idx in range(ret.shape[0]):
                    month = ref_value[idx]['month']
                    ## get the start year from the bounds data
                    start_year = ref_bounds[idx][0].year
                    ## create the datetime object
                    ret[idx] = datetime.datetime(start_year,month,ref_calc_month_centroid)
            elif set_grouping == set(['year']):
                ref_calc_year_centroid_month = constants.CALC_YEAR_CENTROID_MONTH
                ref_calc_year_centroid_day = constants.CALC_YEAR_CENTROID_DAY
                for idx in range(ret.shape[0]):
                    year = ref_value[idx]['year']
                    ## create the datetime object
                    ret[idx] = datetime.datetime(year,ref_calc_year_centroid_month,ref_calc_year_centroid_day)
            elif set_grouping == set(['month','year']):
                ref_calc_month_centroid = constants.CALC_MONTH_CENTROID
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
                fill = datetime.datetime(center_year,center_month,constants.CALC_MONTH_CENTROID)
                ret[idx] = fill
        return(ret)
    
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
        time_regions = get_time_regions(grouping, self.value_datetime, raise_if_incomplete=False)
        # holds the boolean selection arrays
        dgroups = deque()
        new_bounds = np.array([], dtype=object).reshape(-1, 2)
        repr_dt = np.array([], dtype=object)
        # return temporal dimensions and convert to groups
        for dgroup, sub in iter_boolean_groups_from_time_regions(time_regions, self, yield_subset=True,
                                                                 raise_if_incomplete=False):
            dgroups.append(dgroup)
            sub_value_datetime = sub.value_datetime
            new_bounds = np.vstack((new_bounds, [min(sub_value_datetime), max(sub_value_datetime)]))
            repr_dt = np.append(repr_dt, sub_value_datetime[int(sub.shape[0] / 2)])
        # no date parts yet...
        date_parts = None

        return new_bounds, date_parts, repr_dt, dgroups

    def _get_iter_value_bounds_(self):
        if self.format_time:
            ret = self.value_datetime, self.bounds_datetime
        else:
            ret = self.value_numtime, self.bounds_numtime
        return ret

    def _get_temporal_group_dimension_(self, *args, **kwargs):
        return TemporalGroupDimension(*args, **kwargs)

    def _set_date_parts_(self, yld, value):
        if self.format_time:
            fill = (value.year, value.month, value.day)
        else:
            fill = [None]*3
        yld['year'], yld['month'], yld['day'] = fill


class TemporalGroupDimension(TemporalDimension):

    def __init__(self, *args, **kwargs):
        self.grouping = kwargs.pop('grouping')
        self.dgroups = kwargs.pop('dgroups')
        self.date_parts = kwargs.pop('date_parts')

        TemporalDimension.__init__(self, *args, **kwargs)

    def write_to_netcdf_dataset(self, dataset, **kwargs):
        """
        For CF-compliance, ensures climatology bounds are correctly attributed.
        """

        previous_name_bounds = self.name_bounds
        self.name_bounds = 'climatology_bounds'
        try:
            super(TemporalGroupDimension, self).write_to_netcdf_dataset(dataset, **kwargs)
            variable = dataset.variables[self.name_value]
            variable.climatology = variable.bounds
            variable.delncattr('bounds')
        finally:
            self.name_bounds = previous_name_bounds


def get_datetime_conversion_state(archetype):
    """
    :param archetype: The object to test for conversion to datetime.
    :type archetyp: float, :class:`datetime.datetime`, or :class:`netcdftime.datetime`
    :returns: ``True`` if the object should be converted to datetime.
    :rtype: bool
    """

    if isinstance(archetype, (datetime.datetime, netcdftime.datetime)):
        ret = False
    else:
        ret = True
    return ret


def get_datetime_from_months_time_units(vec, units, month_centroid=16):
    """
    Convert a vector of months offsets into :class:``datetime.datetime`` objects.

    :param vec: Vector of integer month offsets.
    :type vec: :class:``np.ndarray``
    :param str units: Source units to parse.
    :param month_centroid: The center day of the month to use when creating the :class:``datetime.datetime`` objects.

    >>> units = "months since 1978-12"
    >>> vec = np.array([0,1,2,3])
    >>> get_datetime_from_months_time_units(vec,units)
    array([1978-12-16 00:00:00, 1979-01-16 00:00:00, 1979-02-16 00:00:00,
           1979-03-16 00:00:00], dtype=object)
    """

    # only work with integer inputs
    vec = np.array(vec, dtype=int)

    def _get_datetime_(current_year, origin_month, offset_month, current_month_correction, month_centroid):
        return datetime.datetime(current_year, (origin_month + offset_month) - current_month_correction, month_centroid)

    origin = get_origin_datetime_from_months_units(units)
    origin_month = origin.month
    current_year = origin.year
    current_month_correction = 0
    ret = np.ones(len(vec), dtype=object)
    for ii, offset_month in enumerate(vec):
        try:
            fill = _get_datetime_(current_year, origin_month, offset_month, current_month_correction, month_centroid)
        except ValueError:
            current_month_correction += 12
            current_year += 1
            fill = _get_datetime_(current_year, origin_month, offset_month, current_month_correction, month_centroid)
        ret[ii] = fill
    return ret


def get_difference_in_months(origin, target):
    """
    Get the integer difference in months between an origin and target datetime.

    :param :class:``datetime.datetime`` origin: The origin datetime object.
    :param :class:``datetime.datetime`` target: The target datetime object.

    >>> get_difference_in_months(datetime.datetime(1978,12,1),datetime.datetime(1979,3,1))
    3
    >>> get_difference_in_months(datetime.datetime(1978,12,1),datetime.datetime(1978,7,1))
    -5
    """

    def _count_(start_month, stop_month, start_year, stop_year, direction):
        count = 0
        curr_month = start_month
        curr_year = start_year
        while True:
            if curr_month == stop_month and curr_year == stop_year:
                break
            else:
                pass

            if direction == 'forward':
                curr_month += 1
            elif direction == 'backward':
                curr_month -= 1
            else:
                raise (NotImplementedError(direction))

            if curr_month == 13:
                curr_month = 1
                curr_year += 1
            if curr_month == 0:
                curr_month = 12
                curr_year -= 1

            if direction == 'forward':
                count += 1
            else:
                count -= 1

        return count

    origin_month, origin_year = origin.month, origin.year
    target_month, target_year = target.month, target.year

    if origin <= target:
        direction = 'forward'
    else:
        direction = 'backward'

    diff_months = _count_(origin_month, target_month, origin_year, target_year, direction)
    return diff_months


def get_is_interannual(sequence):
    """
    Returns ``True`` if an integer sequence representing a season crosses a year boundary.

    >>> sequence = [11,12,1]
    >>> get_is_interannual(sequence)
    True
    """

    if 12 in sequence and 1 in sequence:
        ret = True
    else:
        ret = False
    return ret


def get_num_from_months_time_units(vec, units, dtype=None):
    """
    Convert a vector of :class:``datetime.datetime`` objects into an integer vector.

    :param vec: Input vector to convert.
    :type vec: :class:``np.ndarray``
    :param str units: Source units to parse.
    :param type dtype: Output vector array type.

    >>> units = "months since 1978-12"
    >>> vec = np.array([datetime.datetime(1978,12,1),datetime.datetime(1979,1,1)])
    >>> get_num_from_months_time_units(vec,units)
    array([0, 1])
    """

    origin = get_origin_datetime_from_months_units(units)
    ret = [get_difference_in_months(origin, target) for target in vec]
    return np.array(ret, dtype=dtype)


def get_origin_datetime_from_months_units(units):
    """
    Get the origin Python :class:``datetime.datetime`` object from a month string.

    :param str units: Source units to parse.
    :returns: :class:``datetime.datetime``

    >>> units = "months since 1978-12"
    >>> get_origin_datetime_from_months_units(units)
    datetime.datetime(1978, 12, 1, 0, 0)
    """

    origin = ' '.join(units.split(' ')[2:])
    to_try = ['%Y-%m', '%Y-%m-%d %H']
    converted = False
    for tt in to_try:
        try:
            origin = datetime.datetime.strptime(origin, tt)
            converted = True
            break
        except ValueError as e:
            continue
    if not converted:
        raise e
    return origin


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
    for ii, season in enumerate(seasons):
        season_map[ii] = season
    max_map = {}
    for key, value in season_map.iteritems():
        max_map[methods[method](value)] = key
    sorted_maxes = sorted(max_map)
    ret = [seasons[max_map[s]] for s in sorted_maxes]
    ret = deepcopy(ret)
    return ret


def get_time_regions(seasons, dates, raise_if_incomplete=True):
    """
    >>> seasons = [[6,7,8],[9,10,11],[12,1,2]]
    >>> dates = <vector of datetime objects>
    """

    # extract the years from the data vector collapsing them to a unique set then sort in ascending order
    years = list(set([d.year for d in dates]))
    years.sort()
    # determine if any of the seasons are interannual
    interannual_check = map(get_is_interannual, seasons)
    # holds the return value
    time_regions = []
    # the interannual cases requires two time region sequences to properly extract
    if any(interannual_check):
        # loop over years first to ensure each year is accounted for in the time region output
        for ii_year, year in enumerate(years):
            # the interannual flag is used internally for simple optimization
            for ic, cg in itertools.izip(interannual_check, seasons):
                # if no exception is raised for an incomplete season, this flag indicate whether to append to the output
                append_to_time_regions = True
                if ic:
                    # copy and sort in descending order the season because december of the current year should be first.
                    _cg = deepcopy(cg)
                    _cg.sort()
                    _cg.reverse()
                    # look for the interannual break and split the season into the current year and next year.
                    diff = np.abs(np.diff(_cg))
                    split_base = np.arange(1, len(_cg))
                    split_indices = split_base[diff > 1]
                    split = np.split(_cg, split_indices)
                    # will hold the sub-element time regions
                    sub_time_region = []
                    for ii_split, s in enumerate(split):
                        try:
                            to_append_sub = {'year': [years[ii_year + ii_split]], 'month': s.tolist()}
                            sub_time_region.append(to_append_sub)
                        # there may not be another year of data for an interannual season. we DO NOT keep incomplete
                        # seasons.
                        except IndexError:
                            # don't just blow through an incomplete season unless asked to
                            if raise_if_incomplete:
                                raise (IncompleteSeasonError(_cg, year))
                            else:
                                append_to_time_regions = False
                                continue
                    to_append = sub_time_region
                else:
                    to_append = [{'year': [year], 'month': cg}]
                if append_to_time_regions:
                    time_regions.append(to_append)
    # without interannual seasons the time regions are unique combos of the years and seasons designations
    else:
        for year, season in itertools.product(years, seasons):
            time_regions.append([{'year': [year], 'month': season}])

    return time_regions


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
            months = set([d.month for d in sub.value_datetime])
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
