from contextlib import contextmanager
from copy import deepcopy

import numpy as np

from ocgis.util.helpers import get_swap_chain


def broadcast_array_by_dimension_names(arr, src_names, dst_names):
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


@contextmanager
def broadcast_scope(src, dst_names):
    """Scope a variable's broadcast, returning the original order when finished. See :func:`ocgis.util.broadcaster.broadcast_variable`."""
    original_names = deepcopy(src.dimension_names)
    try:
        yield broadcast_variable(src, dst_names)
    finally:
        broadcast_variable(src, original_names)


def broadcast_variable(src, dst_names):
    """
    Broadcast a variable's value array in-place (variable is returned for convenience) to match destination dimension.

    :param src: The variable to broadcast. This variable's dimensions must match the dimensions to broadcast to.
    :type src: :class:`~ocgis.Variable`
    :param sequence dst_names: Dimension names to broadcast to.
    :return: :class:`~ocgis.Variable`
    """
    src_names = list(src.dimension_names)

    new_value = broadcast_array_by_dimension_names(src.get_value(), src_names, dst_names)
    has_mask = src.has_mask
    if has_mask:
        new_mask = broadcast_array_by_dimension_names(src.get_mask(), src_names, dst_names)
    else:
        new_mask = None

    ddict = src.dimensions_dict
    new_dimensions = [ddict[n] for n in dst_names]

    src.remove_value()
    src.set_dimensions(None)
    src.set_mask(None)

    src.set_dimensions(new_dimensions)
    src.set_value(new_value)
    if has_mask:
        src.set_mask(new_mask)

    return src
