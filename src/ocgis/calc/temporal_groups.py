from abc import ABCMeta


class AbstractTemporalGroup(object):
    __metaclass__ = ABCMeta


class SeasonalTemporalGroup(list, AbstractTemporalGroup):
    _integer_name_map = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July',
                         8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    _season_type_flags = ('unique', 'year')

    def __init__(self, iterable):
        list.__init__(self, iterable)

    @property
    def icclim_mode(self):
        ret = []
        flag = None
        for element in self:
            sub = []
            if element not in self._season_type_flags:
                for sub_element in element:
                    sub.append(self._integer_name_map[sub_element][0])
                ret.append(''.join(sub))
            else:
                flag = element
        ret = '-'.join(ret)
        if flag is not None:
            ret = '{0} ({1})'.format(ret, flag)
        return ret
