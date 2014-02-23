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


