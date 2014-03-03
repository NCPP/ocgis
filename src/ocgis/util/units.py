import itertools


def get_are_units_equivalent(units_sequence):
    if len(units_sequence) < 2:
        raise(ValueError('Units sequence must have length >= 2.'))
    
    ret = True
    for test_units,dest_units in itertools.permutations(units_sequence,2):
        if not test_units.equivalent(dest_units):
            ret = False
            break
    return(ret)

def get_are_units_equal(units_sequence):
    if len(units_sequence) < 2:
        raise(ValueError('Units sequence must have length >= 2.'))
    
    ret = True
    for test_units,dest_units in itertools.permutations(units_sequence,2):
        if test_units != dest_units:
            ret = False
            break
    return(ret)

def get_are_units_equal_by_string_or_cfunits(source,target,try_cfunits=True):
    '''
    Test if unit definitions are equal.
    
    :param str source: String definition of the units to compare against.
    :param str target: Target units to test for equality.
    :param bool try_cfunits: If ``True`` attempt to import and use
     :class:`cfunits.Units` for equality operation.
     
    >>> get_are_units_equal_by_string_or_cfunits('K','K',try_cfunits=True)
    True
    >>> get_are_units_equal_by_string_or_cfunits('K','kelvin',try_cfunits=False)
    False
    >>> get_are_units_equal_by_string_or_cfunits('K','kelvin',try_cfunits=True)
    True
    '''
    try:
        if try_cfunits:
            from cfunits import Units
            source_cfunits = Units(source)
            match = source_cfunits.equals(Units(target))
        else:
            raise(ImportError)
    except ImportError:
        match = source.lower() == target.lower()
    return(match)


if __name__ == '__main__':
    import doctest
    doctest.testmod()