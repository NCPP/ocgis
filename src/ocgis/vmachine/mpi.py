import itertools
from contextlib import contextmanager
from functools import reduce

import numpy as np
import six

from ocgis import constants
from ocgis.base import AbstractOcgisObject, raise_if_empty
from ocgis.constants import DataType, MPITag
from ocgis.exc import DimensionNotFound
from ocgis.util.helpers import get_optimal_slice_from_array, get_iter, get_group

try:
    from mpi4py import MPI
    from mpi4py.MPI import COMM_NULL
except ImportError:
    MPI_ENABLED = False
else:
    if MPI.COMM_WORLD.Get_size() > 1:
        MPI_ENABLED = True
    else:
        MPI_ENABLED = False


class DummyMPIComm(object):
    def Barrier(self):
        pass

    def bcast(self, *args, **kwargs):
        return args[0]

    def Create(self, *args, **kwargs):
        return self

    def Free(self, *args, **kwargs):
        pass

    def gather(self, *args, **kwargs):
        return [args[0]]

    def Get_group(self):
        return self

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Incl(self, *args, **kwargs):
        return self

    def recv(self, *args, **kwargs):
        pass

    def scatter(self, *args, **kwargs):
        return args[0][0]

    def send(self, *args, **kwargs):
        pass


if MPI_ENABLED and MPI.COMM_WORLD.Get_size() > 1:
    MPI_COMM = MPI.COMM_WORLD
else:
    MPI_COMM = DummyMPIComm()
    COMM_NULL = constants.MPI_COMM_NULL_VALUE
MPI_SIZE = MPI_COMM.Get_size()
MPI_RANK = MPI_COMM.Get_rank()


class OcgDist(AbstractOcgisObject):
    def __init__(self, size=None, ranks=None):
        if size is None:
            size = MPI_SIZE
        if ranks is None:
            ranks = tuple(range(size))

        self.ranks = ranks
        self.size = size
        self.mapping = {}
        for rank in self.ranks:
            self.mapping[rank] = get_template_rank_dict()

        self.has_updated_dimensions = False

    def add_dimension(self, dim, group=None, force=False):
        from ocgis import Dimension

        if not isinstance(dim, Dimension):
            raise ValueError('"dim" must be a "Dimension" object.')

        for rank in self.ranks:
            if rank != MPI_RANK:
                to_add_dim = dim.copy()
            else:
                to_add_dim = dim
            the_group = self._create_or_get_group_(group, rank=rank)
            if not force and to_add_dim.name in the_group['dimensions']:
                msg = 'Dimension with name "{}" already in group "{}" and "force=False".'
                raise ValueError(msg.format(to_add_dim.name, group))
            else:
                the_group['dimensions'][to_add_dim.name] = to_add_dim

    def add_dimensions(self, dims, **kwargs):
        for dim in dims:
            self.add_dimension(dim, **kwargs)

    def add_variable(self, name_or_variable, force=False, dimensions=None, group=None):
        """
        Add a variable to the distribution mapping.

        :param name_or_variable: The variable or variable name to add to the distribution.
        :type name_or_variable: :class:`~ocgis.new_interface.variable.Variable`
        :param force: If ``True``, overwrite any variables with the same name.
        :param sequence dimensions: A sequence of dimension names if ``name_or_variable`` is a name. Otherwise,
         dimensions are pulled from the variable object.
        :raises: ValueError
        """
        from ocgis.variable.base import Variable
        from ocgis.variable.dimension import Dimension

        if isinstance(name_or_variable, Variable):
            group = group or name_or_variable.group
            name = name_or_variable.name
            dimensions = name_or_variable.dimensions
        else:
            name = name_or_variable
            dimensions = list(get_iter(dimensions, dtype=(str, Dimension)))

        if dimensions is not None and len(dimensions) > 0:
            if isinstance(dimensions[0], Dimension):
                dimensions = [dim.name for dim in dimensions]
        else:
            dimensions = []
        dimensions = tuple(dimensions)

        for rank_home in self.ranks:
            the_group = self._create_or_get_group_(group, rank_home)
            if not force and name in the_group['variables']:
                msg = 'Variable with name "{}" already in group "{}" and "force=False".'
                raise ValueError(msg.format(name, group))
            else:
                the_group['variables'][name] = {'dimensions': dimensions}

    def add_variables(self, vars, **kwargs):
        for var in vars:
            self.add_variable(var, **kwargs)

    def create_dimension(self, *args, **kwargs):
        from ocgis import Dimension

        group = kwargs.pop('group', None)
        dim = Dimension(*args, **kwargs)
        self.add_dimension(dim, group=group)

        # If rank counts are not the same, you have to get your own dimension. Ranks may not be contained in the
        # mapping.
        if self.size != MPI_SIZE:
            ret = None
        else:
            ret = self.get_dimension(dim.name, group=group)
        return ret

    def create_variable(self, *args, **kwargs):
        from ocgis import Variable
        group = kwargs.pop('group', None)
        var = Variable(*args, **kwargs)
        self.add_variable(var, group=group)
        return var

    def get_bounds_local(self, group=None, rank=MPI_RANK):
        the_group = self.get_group(group=group, rank=rank)
        ret = [dim.bounds_local for dim in list(the_group['dimensions'].values())]
        return tuple(ret)

    def get_dimension(self, name, group=None, rank=MPI_RANK):
        group_data = self.get_group(group=group, rank=rank)
        return group_data['dimensions'][name]

    def get_dimensions(self, names, **kwargs):
        ret = [self.get_dimension(name, **kwargs) for name in names]
        return ret

    def get_empty_ranks(self, group=None, inverse=False):
        ret = []
        for rank in range(self.size):
            the_group = self.get_group(group, rank=rank)
            if any([dim.is_empty for dim in list(the_group['dimensions'].values())]):
                ret.append(rank)
        if inverse:
            ret = set(range(self.size)) - set(ret)
            ret = list(ret)
            ret.sort()
        return tuple(ret)

    def get_variable(self, name_or_variable, group=None, rank=MPI_RANK):
        if group is None:
            try:
                group = name_or_variable.group
            except AttributeError:
                pass
        try:
            name = name_or_variable.source_name
        except AttributeError:
            name = name_or_variable

        group_data = self.get_group(group, rank=rank)
        return group_data['variables'][name]

    def get_group(self, group=None, rank=MPI_RANK):
        return self._create_or_get_group_(group, rank=rank)

    def iter_groups(self, rank=MPI_RANK):
        from ocgis.driver.base import iter_all_group_keys
        mapping = self.mapping[rank]
        for group_key in iter_all_group_keys(mapping):
            group_data = get_group(mapping, group_key)
            yield group_key, group_data

    def update_dimension_bounds(self, rank='all', min_elements=2):
        """
        :param rank: If ``'all'``, update across all ranks. Otherwise, update for the integer rank provided.
        :type rank: str/int
        :param int min_elements: The minimum number of elements per rank. It must be >= 2.
        """
        if self.has_updated_dimensions:
            raise ValueError('Dimensions already updated.')
        if min_elements < 1:
            raise ValueError('"min_elements" must be >= 1.')

        if rank == 'all':
            ranks = list(range(self.size))
        else:
            ranks = [rank]

        for rank in ranks:
            for _, group_data in self.iter_groups(rank=rank):
                dimdict = group_data['dimensions']

                # If there are no distributed dimensions, there is no work to be dome with MPI bounds.
                if not any([dim.dist for dim in list(dimdict.values())]):
                    continue

                # Get dimension lengths.
                lengths = {dim.name: len(dim) for dim in list(dimdict.values()) if dim.dist}
                # Choose the size of the distribution group. There needs to be at least one element per rank. First,
                # get the longest distributed dimension.
                max_length = max(lengths.values())
                for k, v in list(lengths.items()):
                    if v == max_length:
                        distributed_dimension = dimdict[k]
                # Ensure only the distributed dimension remains distributed.
                for dim in list(dimdict.values()):
                    if dim.name != distributed_dimension.name:
                        dim.dist = False
                # Adjust the MPI distributed size if the length of the longest dimension is less than the rank count.
                # Dimensions on higher ranks will be considered empty.
                the_size = self.size
                if len(distributed_dimension) < the_size:
                    the_size = len(distributed_dimension)
                # Use the minimum number of elements per rank to potentially adjust the size.
                min_size = int(np.floor(len(distributed_dimension) / float(min_elements)))
                if self.size > 1 and min_size < the_size:
                    the_size = min_size

                # Fix the global bounds.
                distributed_dimension.bounds_global = (0, len(distributed_dimension))
                # Use this to calculate the local bounds for a dimension.
                bounds_local = get_rank_bounds(len(distributed_dimension), the_size, rank)
                if bounds_local is not None:
                    start, stop = bounds_local
                    if distributed_dimension._src_idx is None:
                        src_idx = None
                    else:
                        src_idx = distributed_dimension._src_idx[start:stop]
                    distributed_dimension.set_size(stop - start, src_idx=src_idx)
                    distributed_dimension.bounds_local = bounds_local
                else:
                    # If there are no local bounds, the dimension is empty.
                    distributed_dimension.convert_to_empty()

        self.has_updated_dimensions = True

    def _create_or_get_group_(self, group, rank=MPI_RANK):
        # Allow None and single string group selection.
        if group is None or isinstance(group, six.string_types):
            group = [group]
        # Always start with None, the root group, when searching for data groups.
        if group[0] is not None:
            group.insert(0, None)

        ret = self.mapping[rank]
        for ctr, g in enumerate(group):
            # No group nesting for the first iteration.
            if ctr > 0:
                ret = ret['groups']
            # This is the default fill for the group.
            if g not in ret:
                ret[g] = {'groups': {}, 'dimensions': {}, 'variables': {}}
            ret = ret[g]
        return ret


def barrier_print(*args):
    if len(args) == 1:
        args = args[0]
    ranks = list(range(MPI_SIZE))
    MPI_COMM.Barrier()
    for rank in ranks:
        if rank == MPI_RANK:
            print('')
            print('(rank={}, barrier=True) {}'.format(MPI_RANK, args))
            print('')
        MPI_COMM.Barrier()


def barrier_ranks(ranks, comm=None):
    comm, rank, _ = get_standard_comm_state(comm=comm)
    if len(ranks) > 1:
        if rank in ranks:
            root = ranks[0]
            if rank == root:
                received = [root]
                for irank in ranks:
                    if irank != root:
                        comm.recv(source=irank, tag=MPITag.BARRIER)
                        received.append(irank)
                for irank in ranks:
                    if irank != root:
                        comm.send(True, dest=irank, tag=MPITag.BARRIER)
            else:
                comm.send(True, dest=root, tag=MPITag.BARRIER)
                comm.recv(source=root, tag=MPITag.BARRIER)


def create_slices(length, size):
    # tdk: optimize: remove np.arange
    r = np.arange(length)
    sections = np.array_split(r, size)
    sections = [get_optimal_slice_from_array(s, check_diff=False) for s in sections]
    return sections


def dgather(elements):
    grow = elements[0]
    for idx in range(1, len(elements)):
        for k, v in elements[idx].items():
            grow[k] = v
    return grow


def get_global_to_local_slice(start_stop, bounds_local):
    """
    :param start_stop: Two-element, integer sequence for the start and stop global indices.
    :type start_stop: tuple
    :param bounds_local: Two-element, integer sequence describing the local bounds.
    :type bounds_local: tuple
    :return: Two-element integer sequence mapping the global to the local slice. If the local bounds are outside the
     global slice, ``None`` will be returned.
    :rtype: tuple or None
    """
    start, stop = start_stop
    lower, upper = bounds_local

    if start is None or stop is None:
        raise ValueError('Start and/or stop may not be None.')

    new_start = start
    if start >= upper:
        new_start = None
    else:
        if new_start < lower:
            new_start = lower

    if stop <= lower:
        new_stop = None
    elif upper < stop:
        new_stop = upper
    else:
        new_stop = stop

    if new_start is None or new_stop is None:
        ret = None
    else:
        ret = (new_start - lower, new_stop - lower)
    return ret


def dict_get_or_create(ddict, key, default):
    try:
        ret = ddict[key]
    except KeyError:
        ret = ddict[key] = default
    return ret


def gather_ranks(ranks, value, root=0, comm=None):
    comm, rank, size = get_standard_comm_state(comm=comm)

    if rank == root:
        gathered = [None] * len(ranks)
    else:
        gathered = None

    for idx, curr_rank in enumerate(ranks):
        if rank == root:
            if rank == curr_rank:
                gathered[idx] = value
        elif rank == curr_rank:
            comm.send(value, dest=root, tag=MPITag.GATHER)

    if rank == root:
        for idx, recv_rank in enumerate(ranks):
            if recv_rank != root:
                recv = comm.recv(source=recv_rank, tag=MPITag.GATHER)
                gathered[idx] = recv

    return gathered


def get_rank_bounds(nelements, size, rank, esplit=None):
    """
    :param nelements: The number of elements in the sequence to split.
    :param size: Processor count. If ``None`` use MPI size.
    :param rank: The process's rank. If ``None`` use the MPI rank.
    :param esplit: The split size. If ``None``, compute this internally.
    :return: A tuple of lower and upper bounds using Python slicing rules. Returns ``None`` if no bounds are available
     for the rank. Also returns ``None`` in the case of zero length.
    :rtype: tuple or None

    >>> get_rank_bounds(5, 4, 2, esplit=None)
    (3, 4)
    """
    # This is the edge case for zero-length.
    if nelements == 0:
        return

    # Set defaults for the rank and size.
    # This is the edge case for ranks outside the size. Possible with an overloaded size not related to the MPI
    # environment.
    if rank >= size:
        return

    # Case with more length than size. Do not take this route of a default split is provided.
    if nelements > size and esplit is None:
        nelements = int(nelements)
        size = int(size)
        esplit, remainder = divmod(nelements, size)
        if remainder > 0:
            # Find the rank bounds with no remainder.
            ret = get_rank_bounds(nelements - remainder, size, rank)
            # Adjust the returned slices accounting for the remainder.
            if rank + 1 <= remainder:
                ret = (ret[0] + rank, ret[1] + rank + 1)
            else:
                ret = (ret[0] + remainder, ret[1] + remainder)
        elif remainder == 0:
            # Provide the default split to compute the bounds and avoid the recursion.
            ret = get_rank_bounds(nelements, size, rank, esplit=esplit)
        else:
            raise NotImplementedError
    # Case with equal length and size or more size than length.
    else:
        if esplit is None:
            if nelements < size:
                esplit = int(np.ceil(float(nelements) / float(size)))
            elif nelements == size:
                esplit = 1
            else:
                raise NotImplementedError
        else:
            esplit = int(esplit)

        if rank == 0:
            lbound = 0
        else:
            lbound = rank * esplit
        ubound = lbound + esplit

        if ubound >= nelements:
            ubound = nelements

        if lbound >= ubound:
            # The lower bound is outside the vector length
            ret = None
        else:
            ret = (lbound, ubound)

    return ret


@contextmanager
def mpi_group_scope(ranks, comm=None):
    from mpi4py.MPI import COMM_NULL

    comm, rank, size = get_standard_comm_state(comm=comm)

    base_group = comm.Get_group()
    sub_group = base_group.Incl(ranks)
    new_comm = comm.Create(sub_group)

    try:
        yield new_comm
    finally:
        if new_comm != COMM_NULL:
            new_comm.Free()
            sub_group.Free()


def bcast_ranks(ranks, value, root=0, comm=None):
    comm, rank, size = get_standard_comm_state(comm=comm)

    if rank == root:
        for target_rank in ranks:
            if target_rank != root:
                comm.send(value, dest=target_rank, tag=MPITag.BCAST)
    elif rank in ranks:
        value = comm.recv(source=root, tag=MPITag.BCAST)
    else:
        value = None

    return value


def scatter_ranks(ranks, value, root=0, comm=None):
    comm, rank, size = get_standard_comm_state(comm=comm)

    if rank == root:
        for target_rank, target_value in zip(ranks, value):
            if target_rank != root:
                comm.send(target_value, dest=target_rank, tag=MPITag.SCATTER)
            else:
                ret_value = target_value
    elif rank in ranks:
        ret_value = comm.recv(source=root, tag=MPITag.SCATTER)
    else:
        ret_value = None

    return ret_value


def ogather(elements):
    ret = np.array(elements, dtype=object)
    return ret


def hgather(elements):
    n = sum([e.shape[0] for e in elements])
    fill = np.zeros(n, dtype=elements[0].dtype)
    start = 0
    for e in elements:
        shape_e = e.shape[0]
        if shape_e == 0:
            continue
        stop = start + shape_e
        fill[start:stop] = e
        start = stop
    return fill


def create_nd_slices(splits, shape):
    ret = [None] * len(shape)
    for idx, (split, shp) in enumerate(zip(splits, shape)):
        ret[idx] = create_slices(shp, split)
    ret = [slices for slices in itertools.product(*ret)]
    return tuple(ret)


def find_dimension_in_sequence(dimension_name, dimensions):
    ret = None
    for dim in dimensions:
        if dimension_name == dim.name:
            ret = dim
            break
    if ret is None:
        raise DimensionNotFound('Dimension not found: {}'.format(dimension_name))
    return ret


def get_nonempty_ranks(target, the_vm):
    """Collective!"""

    gathered = the_vm.gather(target.is_empty)
    if the_vm.rank == 0:
        ret = []
        for idx, rank in enumerate(the_vm.ranks):
            if not gathered[idx]:
                ret.append(rank)
        ret = tuple(ret)
    else:
        ret = None
    ret = the_vm.bcast(ret)
    return ret


def get_optimal_splits(size, shape):
    n_elements = reduce(lambda x, y: x * y, shape)
    if size >= n_elements:
        splits = shape
    else:
        if size <= shape[0]:
            splits = [1] * len(shape)
            splits[0] = size
        else:
            even_split = int(np.power(size, 1.0 / float(len(shape))))
            splits = [None] * len(shape)
            for idx, shp in enumerate(shape):
                if even_split > shp:
                    fill = shp
                else:
                    fill = even_split
                splits[idx] = fill
    return tuple(splits)


def get_standard_comm_state(comm=None):
    comm = comm or MPI_COMM
    return comm, comm.Get_rank(), comm.Get_size()


def get_template_rank_dict():
    return {None: {'dimensions': {}, 'groups': {}, 'variables': {}}}


def rank_print(*args):
    if len(args) == 1:
        args = args[0]
    msg = '(rank={}) {}'.format(MPI_RANK, args)
    print(msg)


def variable_gather(variable, root=0):
    from ocgis import vm

    if variable.is_empty:
        raise ValueError('No empty variables allowed.')

    if vm.size > 1:
        if vm.rank == root:
            new_variable = variable.copy()
            new_variable.dtype = variable.dtype
            new_variable.set_mask(None)
            new_variable.set_value(None)
            new_variable.set_dimensions(None)
            assert not new_variable.has_allocated_value
        else:
            new_variable = None
    else:
        new_variable = variable
        for dim in new_variable.dimensions:
            dim.dist = False

    if vm.size > 1:
        if vm.rank == root:
            new_dimensions = [None] * variable.ndim
        else:
            new_dimensions = None

        for idx, dim in enumerate(variable.dimensions):
            if dim.dist:
                parts = vm.gather(dim)
            if vm.rank == root:
                new_dim = dim.copy()
                if dim.dist:
                    new_size = 0
                    has_src_idx = False
                    for part in parts:
                        has_src_idx = part._src_idx is not None
                        new_size += len(part)
                    if has_src_idx:
                        new_src_idx = np.zeros(new_size, dtype=DataType.DIMENSION_SRC_INDEX)
                        for part in parts:
                            new_src_idx[part.bounds_local[0]: part.bounds_local[1]] = part._src_idx
                    else:
                        new_src_idx = None
                    new_dim = dim.copy()
                    new_dim.set_size(new_size, src_idx=new_src_idx)
                    new_dim.dist = False
                new_dimensions[idx] = new_dim

        if vm.rank == root:
            new_variable.set_dimensions(new_dimensions, force=True)

        gathered_variables = vm.gather(variable)

        if vm.rank == root:
            for idx, gv in enumerate(gathered_variables):
                destination_slice = [slice(*dim.bounds_local) for dim in gv.dimensions]
                new_variable.__setitem__(destination_slice, gv)
            return new_variable
        else:
            return
    else:
        return new_variable


def variable_scatter(variable, dest_dist, root=0):
    from ocgis import vm

    if variable is not None:
        raise_if_empty(variable)

    if vm.rank == root:
        if variable.dist:
            raise ValueError('Only variables with no prior distribution may be scattered.')
        if not dest_dist.has_updated_dimensions:
            raise ValueError('The destination distribution must have updated dimensions.')

    # Synchronize distribution across processors.
    dest_dist = vm.bcast(dest_dist)

    # Find the appropriate group for the dimensions.
    if vm.rank == root:
        group = variable.group
        dimension_names = [dim.name for dim in variable.dimensions]
    else:
        group = None
        dimension_names = None

    # Synchronize the processes with the MPI distribution and the group containing the dimensions.
    dest_dist = vm.bcast(dest_dist)
    group = vm.bcast(group)
    dimension_names = vm.bcast(dimension_names)

    # These are the dimensions for the local process.
    dest_dimensions = dest_dist.get_dimensions(dimension_names, group=group)

    # barrier_print('dest_dimensions bounds_global: ', [(d.name, d.bounds_global) for d in dest_dimensions])

    # Slice the variables collecting the sequence to scatter to the MPI procs.
    if vm.rank == root:
        size = dest_dist.size
        if size > 1:
            slices = [None] * size
            # Get the slices need to scatter the variables. These are essentially the local bounds on each dimension.
            empty_ranks = dest_dist.get_empty_ranks()
            empty_variable = variable.copy()
            empty_variable.convert_to_empty()
            for current_rank in range(size):
                if current_rank in empty_ranks:
                    slices[current_rank] = None
                else:
                    current_dimensions = list(
                        dest_dist.get_group(group=group, rank=current_rank)['dimensions'].values())
                    slices[current_rank] = {dim.name: slice(*dim.bounds_local) for dim in current_dimensions if
                                            dim.name in variable.parent.dimensions}
            # Slice the variables. These sliced variables are the scatter targets.
            variables_to_scatter = [None] * size
            for idx, slc in enumerate(slices):
                if slc is None:
                    variables_to_scatter[idx] = empty_variable
                else:
                    variables_to_scatter[idx] = variable.parent[slc][variable.name]
        else:
            variables_to_scatter = [variable]
    else:
        variables_to_scatter = None

    # Scatter the variable across processes.
    scattered_variable = vm.scatter(variables_to_scatter, root=root)
    # Update the scattered variable dimensions with the destination dimensions on the process. Everything should align
    # shape-wise. If they don't, an exception will be raised.
    scattered_variable.set_dimensions(dest_dimensions, force=True)

    return scattered_variable


def variable_collection_scatter(variable_collection, dest_dist):
    from ocgis import vm

    if vm.rank == 0:
        target = variable_collection.first()
    else:
        target = None
    sv = variable_scatter(target, dest_dist)
    svc = sv.parent

    if vm.rank == 0:
        n_children = len(variable_collection.children)
    else:
        n_children = None
    n_children = vm.bcast(n_children)
    if vm.rank == 0:
        children = list(variable_collection.children.values())
    else:
        children = [None] * n_children
    for child in children:
        scattered_child = variable_collection_scatter(child, dest_dist)[0]
        svc.add_child(scattered_child, force=True)

    return svc


def vgather(elements):
    n = sum([e.shape[0] for e in elements])
    fill = np.zeros((n, elements[0].shape[1]), dtype=elements[0].dtype)
    start = 0
    for e in elements:
        shape_e = e.shape
        if shape_e[0] == 0:
            continue
        stop = start + shape_e[0]
        fill[start:stop, :] = e
        start = stop
    return fill
