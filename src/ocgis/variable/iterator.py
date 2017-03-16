from collections import OrderedDict
from itertools import izip, product

import numpy as np

from ocgis.base import AbstractOcgisObject
from ocgis.base import get_dimension_names, get_variable_names
from ocgis.constants import HeaderNames


class Iterator(AbstractOcgisObject):
    _should_skip_flag = -999

    def __init__(self, variable, followers=None, value=None, mask=None, allow_masked=True, primary_mask=None,
                 slice_remap=None, shape=None, melted=None, repeaters=None, formatter=None, clobber_masked=True):
        if melted is not None and followers is None:
            raise ValueError('"melted" must be None if there are no "followers".')

        self.variable = variable
        self.formatter = formatter
        self.allow_masked = allow_masked
        self.slice_remap = slice_remap
        self.clobber_masked = clobber_masked

        if variable.repeat_record is not None:
            if repeaters is None:
                repeaters = variable.repeat_record
            else:
                repeaters += variable.repeat_record
        self.repeaters = repeaters

        self._is_lead = True

        if melted is None:
            melted_repeaters = None
        else:
            melted_repeaters = {}
            for m in melted:
                try:
                    if m.repeat_record is not None:
                        melted_repeaters[m.name] = m.repeat_record
                except AttributeError:
                    if m.repeaters is not None:
                        melted_repeaters[m.name] = m.repeat_record
        self.melted_repeaters = melted_repeaters

        if melted is not None:
            melted = get_variable_names(melted)
        self.melted = melted

        if shape is None:
            shape = self.variable.shape
        self.shape = shape

        if primary_mask is None:
            primary_mask = variable.name
        else:
            primary_mask = get_variable_names(primary_mask)[0]
        self.primary_mask = primary_mask

        if value is None:
            self.value = variable.get_value()
        else:
            self.value = value

        if mask is None:
            self.mask = variable.get_mask()
        else:
            self.mask = mask

        if followers is not None:
            dimensions = get_dimension_names(self.variable.dimensions)
            followers = get_followers(followers)
            for fidx, follower in enumerate(followers):
                if isinstance(follower, self.__class__):
                    iterator = follower
                    follower = follower.variable
                else:
                    iterator = Iterator(follower, allow_masked=allow_masked, primary_mask=primary_mask)

                follower_dimensions = get_dimension_names(follower.dimensions)
                set_follower_dimensions = set(follower_dimensions)
                set_dimensions = set(dimensions)
                if not set_follower_dimensions.issubset(set_dimensions):
                    msg = 'Follower variable "{}" dimensions are not a subset of the lead variable.'.format(
                        follower.name)
                    raise ValueError(msg)
                if follower_dimensions != dimensions:
                    follower_slice_remap = []
                    for d in set_dimensions.intersection(set_follower_dimensions):
                        follower_slice_remap.append(dimensions.index(d))
                    iterator.slice_remap = follower_slice_remap
                    iterator.shape = self.shape
                iterator._is_lead = False
                followers[fidx] = iterator
            self.iterators = [self] + followers
            self.followers = followers
        else:
            self.iterators = [self]
            self.followers = None

        self._is_recursing = False

    def __iter__(self):
        name = self.variable.name
        mask = self.mask
        value = self.value
        formatter = self.formatter
        allow_masked = self.allow_masked
        primary_mask = self.primary_mask
        slice_remap = self.slice_remap
        is_lead = self._is_lead
        should_skip_flag = self._should_skip_flag
        melted = self.melted
        header_value = HeaderNames.VALUE
        header_variable = HeaderNames.VARIABLE
        repeaters = self.repeaters
        clobber_masked = self.clobber_masked
        melted_repeaters = self.melted_repeaters

        if formatter is None:
            has_formatter = False
        else:
            has_formatter = True

        if repeaters is None:
            has_repeaters = False
        else:
            has_repeaters = True

        if len(self.iterators) > 1:
            has_followers = True
        else:
            has_followers = False

        if melted is None:
            len_melted = None
        else:
            len_melted = len(melted)

        if self._is_recursing:
            for idx in product(*[range(s) for s in self.shape]):
                if slice_remap is not None:
                    idx = tuple([idx[ii] for ii in slice_remap])

                try:
                    yld_value, yld_mask = get_record(idx, value, mask)
                except IndexError as e:
                    msg = 'Current iteration variable is: "{}". {}'.format(name, e.message)
                    raise IndexError(msg)

                if yld_mask and clobber_masked:
                    yld_value = None

                if not allow_masked and yld_mask and primary_mask == name:
                    to_yld = should_skip_flag
                else:
                    if has_formatter:
                        to_yld = formatter(name, yld_value, yld_mask)
                    else:
                        to_yld = [(name, yld_value)]
                    if has_repeaters:
                        to_yld = repeaters + to_yld

                yield to_yld
        else:
            try:
                for itr in self.iterators:
                    itr._is_recursing = True
                for record in izip(*self.iterators):
                    if not allow_masked and any([r == should_skip_flag for r in record]):
                        continue
                    else:
                        collected = OrderedDict()
                        for r in record:
                            for rsub in r:
                                collected[rsub[0]] = rsub[1]
                        record = collected
                        if is_lead and has_followers:
                            if melted is not None:
                                melted_values = [None] * len_melted
                                for midx, m in enumerate(melted):
                                    melted_values[midx] = record.pop(m)
                                for midx, m in enumerate(melted):
                                    # record = record.copy()
                                    if melted_repeaters is not None:
                                        for mr in melted_repeaters.get(m, []):
                                            record[mr[0]] = mr[1]
                                    record[header_variable] = m
                                    record[header_value] = melted_values[midx]
                                    yield record.copy()
                        if melted is None:
                            yield record
            finally:
                for itr in self.iterators:
                    itr._is_recursing = False

    def get_repeaters(self, headers_only=False, found=None):
        return get_repeaters(self, headers_only=headers_only, found=found)


def get_record(idx, value, mask):
    asscalar = np.asscalar

    if mask is None:
        ret_mask = False
    else:
        ret_mask = asscalar(mask.__getitem__(idx))
    ret_value = value.__getitem__(idx)
    try:
        ret_value = asscalar(ret_value)
    except AttributeError:
        # Assume this is an object type and allow the object to be returned as is. This happens with geometry variables.
        pass

    return ret_value, ret_mask


def get_followers(followers, found=None):
    if found is None:
        found = []
    for follower in followers:
        found.append(follower)
        if isinstance(follower, Iterator) and follower.followers is not None:
            found = get_followers(follower.followers, found=found)
    return found


def get_repeaters(iterator, headers_only=False, found=None):
    if found is None:
        found = []
    if iterator.repeaters is not None:
        for r in iterator.repeaters:
            if headers_only:
                if r[0] not in found:
                    found.append(r[0])
            else:
                if r not in found:
                    found.append(r)
    if iterator.followers is not None:
        for f in iterator.followers:
            found = get_repeaters(f, headers_only=headers_only, found=found)
    return found
