import numpy as np

from ocgis.util.helpers import get_swap_chain


def conform_array_by_dimension_names(arr, src_names, dst_names):
    # These dimensions need to be inserted into the output array as singleton dimensions.
    singletons = [(idx, dn) for idx, dn in enumerate(dst_names) if dn not in src_names]
    # Names in the destination and source name sequences.
    matching_dst_names = [dn for dn in dst_names if dn in src_names]
    # These dimensions should be squeeze prior to swapping.
    to_squeeze = [dn for dn in src_names if dn not in matching_dst_names]
    # Squeeze out any dimensions.
    if len(to_squeeze) > 0:
        to_squeeze = [src_names.index(t) for t in to_squeeze]
        src_names = [dn for dn in src_names if dn in dst_names]
        arr = np.squeeze(arr, axis=to_squeeze)
    # The swapping chain for the output data array.
    swap_chain = get_swap_chain(src_names, matching_dst_names)
    for sc in swap_chain:
        new_mask = None
        if hasattr(arr, 'mask'):
            new_mask = arr.__dict__['_mask'].swapaxes(*sc)
        arr = arr.swapaxes(*sc)
        if new_mask is not None:
            arr.__dict__['_mask'] = new_mask
    # Insert the singleton dimensions in the output array.
    arr_shape = list(arr.shape)
    for s in singletons:
        arr_shape.insert(s[0], 1)
    arr = arr.reshape(*arr_shape)
    return arr
