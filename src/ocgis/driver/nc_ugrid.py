import numpy as np

from ocgis import vm
from ocgis.constants import MPITag
from ocgis.util.helpers import create_unique_global_array


def reduce_reindex_coordinate_variables(cindex, start_index=0):
    """
    Reindex a subset of global coordinate indices contained in the ``cindex`` variable. The coordinate values contained
    in ``coords`` will be reduced to match the coordinates required by the indices in ``cindex``.

    The starting index value (``0`` or ``1``) is set by ``start_index`` for the re-indexing procedure.

    Function will not respect masks.

    The function returns a two-element tuple:

     * First element --> A :class:`numpy.ndarray` with the same dimension as ``cindex`` containing the new indexing.
     * Second element --> A :class:`numpy.ndarray` containing the unique indices that may be used to reduce an external
       coordinate storage variable or array.

    :param cindex: A variable containing coordinate index integer values. This variable may be distributed. This may
     also be a NumPy array.
    :type cindex: :class:`~ocgis.Variable` || :class:`~numpy.ndarray`
    :param int start_index: The first index to use for the re-indexing of ``cindex``. This may be ``0`` or ``1``.
    :rtype: tuple
    """

    # Get the coordinate index values as a NumPy array.
    try:
        cindex = cindex.get_value()
    except AttributeError:
        # Assume this is already a NumPy array.
        pass

    # Create the unique coordinte index array.
    u = np.array(create_unique_global_array(cindex))

    # Holds re-indexed values.
    new_cindex = np.empty_like(cindex)
    # Caches the local re-indexing for the process.
    cache = {}
    # Increment the indexing values based on its presence in the cache.
    curr_idx = 0
    for idx, to_reindex in enumerate(u.flat):
        if to_reindex not in cache:
            cache[to_reindex] = curr_idx
            curr_idx += 1

    # MPI communication tags.
    tag_cache_create = MPITag.REINDEX_CACHE_CREATE
    tag_cache_get_recv = MPITag.REINDEX_CACHE_GET_RECV
    tag_cache_get_send = MPITag.REINDEX_CACHE_GET_SEND

    # This is the local offset to move sequentially across processes. If the local cache is empty, there is no
    # offsetting to move between tasks.
    if len(cache) > 0:
        offset = max(cache.values()) + 1
    else:
        offset = 0

    # Synchronize the processes with the appropriate local offset.
    for idx, rank in enumerate(vm.ranks):
        try:
            dest_rank = vm.ranks[idx + 1]
        except IndexError:
            break
        else:
            if vm.rank == rank:
                vm.comm.send(start_index + offset, dest=dest_rank, tag=tag_cache_create)
            elif vm.rank == dest_rank:
                offset_previous = vm.comm.recv(source=rank, tag=tag_cache_create)
                start_index = offset_previous
    vm.barrier()

    # Find any missing local coordinate indices that are not mapped by the local cache.
    is_missing = False
    is_missing_indices = []
    for idx, to_reindex in enumerate(cindex.flat):
        try:
            local_new_cindex = cache[to_reindex]
        except KeyError:
            is_missing = True
            is_missing_indices.append(idx)
        else:
            new_cindex[idx] = local_new_cindex + start_index

    # Check if there are any processors missing their new index values.
    is_missing_global = vm.gather(is_missing)
    if vm.rank == 0:
        is_missing_global = any(is_missing_global)
    is_missing_global = vm.bcast(is_missing_global)

    # Execute a search across the process caches for any missing coordinate index values.
    if is_missing_global:
        for rank in vm.ranks:
            is_missing_rank = vm.bcast(is_missing, root=rank)
            if is_missing_rank:
                n_missing = vm.bcast(len(is_missing_indices), root=rank)
                if vm.rank == rank:
                    for imi in is_missing_indices:
                        for subrank in vm.ranks:
                            if vm.rank != subrank:
                                vm.comm.send(cindex[imi], dest=subrank, tag=tag_cache_get_recv)
                                new_cindex_element = vm.comm.recv(source=subrank, tag=tag_cache_get_send)
                                if new_cindex_element is not None:
                                    new_cindex[imi] = new_cindex_element
                else:
                    for _ in range(n_missing):
                        curr_missing = vm.comm.recv(source=rank, tag=tag_cache_get_recv)
                        new_cindex_element = cache.get(curr_missing)
                        if new_cindex_element is not None:
                            new_cindex_element += start_index
                        vm.comm.send(new_cindex_element, dest=rank, tag=tag_cache_get_send)

    return new_cindex, u
