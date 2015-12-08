import itertools

import numpy as np


def get_are_units_equivalent(units_sequence):
    if len(units_sequence) < 2:
        raise ValueError('Units sequence must have length >= 2.')

    is_equivalent = []
    for test_units, dest_units in itertools.permutations(units_sequence, 2):
        try:
            is_equivalent_app = test_units.is_convertible(dest_units)
        except AttributeError:
            is_equivalent_app = test_units.equivalent(dest_units)
        is_equivalent.append(is_equivalent_app)
    return all(is_equivalent)


def get_are_units_equal(units_sequence):
    if len(units_sequence) < 2:
        raise ValueError('Units sequence must have length >= 2.')

    ret = True
    for test_units, dest_units in itertools.permutations(units_sequence, 2):
        if test_units != dest_units:
            ret = False
            break
    return ret


def get_are_units_equal_by_string_or_cfunits(source, target, try_cfunits=True):
    """
    Test if unit definitions are equal.

    :param str source: String definition of the units to compare against.
    :param str target: Target units to test for equality.
    :param bool try_cfunits: If ``True`` attempt to import and use :class:`cfunits.Units` for equality operation.

    >>> get_are_units_equal_by_string_or_cfunits('K', 'K', try_cfunits=True)
    True
    >>> get_are_units_equal_by_string_or_cfunits('K', 'kelvin', try_cfunits=False)
    False
    >>> get_are_units_equal_by_string_or_cfunits('K', 'kelvin', try_cfunits=True)
    True
    """

    units_sequence = [get_units_object(e) for e in (source, target)]
    try:
        if try_cfunits:
            match = get_are_units_equal(units_sequence)
        else:
            raise ImportError
    except ImportError:
        match = source.lower() == target.lower()
    return match


def get_conformed_units(value, units_src, units_dst):
    assert (isinstance(value, np.ndarray))

    units_src = get_units_object(units_src)
    units_dst = get_units_object(units_dst)
    try:
        converted = units_src.convert(value, units_dst)
    except AttributeError:
        # Assume using "cfunits".
        converted = units_src.conform(value, units_src, units_dst, inplace=True)
    return converted


def get_units_class(should_raise=True):
    try:
        from cf_units import Unit as u
    except ImportError:
        try:
            from cfunits import Units as u
        except ImportError:
            if not should_raise:
                u = None
            else:
                raise
    return u


def get_units_object(*args, **kwargs):
    u = get_units_class()
    # Check if the argument is already a units object. In this case, pass through.
    if isinstance(args[0], u):
        ret = args[0]
    else:
        ret = u(*args, **kwargs)
    return ret
