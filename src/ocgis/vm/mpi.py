import itertools
from contextlib import contextmanager

import numpy as np

from ocgis.base import AbstractOcgisObject
from ocgis.constants import MPIDistributionMode, DataTypes, MPITags
from ocgis.exc import DimensionNotFound
from ocgis.util.helpers import get_optimal_slice_from_array, get_iter

try:
    from mpi4py import MPI
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
MPI_SIZE = MPI_COMM.Get_size()
MPI_RANK = MPI_COMM.Get_rank()


class OcgMpi(AbstractOcgisObject):
    def __init__(self, size=MPI_SIZE):
        self.size = size
        self.mapping = {}
        for rank in range(size):
            self.mapping[rank] = get_template_rank_dict()

        self.has_updated_dimensions = False

    def add_dimension(self, dim, group=None, force=False):
        from ocgis import Dimension
        if not isinstance(dim, Dimension):
            raise ValueError('"dim" must be a "Dimension" object.')

        for rank in range(self.size):
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

    def add_variable(self, name_or_variable, ranks='all', dist=MPIDistributionMode.REPLICATED, force=False,
                     dimensions=None, group=None):
        """
        Add a variable to the distribution mapping.

        :param name_or_variable: The variable or variable name to add to the distribution.
        :type name_or_variable: :class:`~ocgis.new_interface.variable.Variable`
        :param ranks: If ``'all'``, add variable to all ranks. Otherwise, add to the integer ranks. Should be ``'all'``
         except for isolated variables which require a sequence of integer home ranks.
        :type ranks: str/int
        :param dist: The distribution mode.
        :type dist: :class:`~ocgis.constants.MPIDistributionMode`
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
            dimensions = list(get_iter(dimensions, dtype=(basestring, Dimension)))

        if dimensions is not None and len(dimensions) > 0:
            if isinstance(dimensions[0], Dimension):
                dimensions = [dim.name for dim in dimensions]
        else:
            dimensions = []
        dimensions = tuple(dimensions)

        for rank_home in range(self.size):
            the_group = self._create_or_get_group_(group, rank_home)
            if not force and name in the_group['variables']:
                msg = 'Variable with name "{}" already in group "{}" and "force=False".'
                raise ValueError(msg.format(name, group))
            else:
                the_group['variables'][name] = {'dist': dist, 'dimensions': dimensions}

            if dist == MPIDistributionMode.ISOLATED:
                if ranks == 'all':
                    raise ValueError('Isolated variables should not be added to "all" ranks.')
                the_group['variables'][name]['ranks'] = tuple(get_iter(ranks, dtype=int))

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

        dist = kwargs.get('dist', MPIDistributionMode.REPLICATED)
        kwargs['dist'] = dist

        group = kwargs.pop('group', None)
        ranks = kwargs.get('ranks', None)

        if ranks is None:
            if dist in (MPIDistributionMode.REPLICATED, MPIDistributionMode.DISTRIBUTED):
                ranks = 'all'
            elif dist == MPIDistributionMode.ISOLATED:
                ranks = (0,)
            kwargs['ranks'] = ranks

        var = Variable(*args, **kwargs)

        # Variables with distributed dimensions are always distributed.
        if var.has_dimensions:
            for dim in var.dimensions:
                if dim.dist:
                    var.dist = MPIDistributionMode.DISTRIBUTED
                    break

        self.add_variable(var, group=group, ranks=ranks, dist=dist)
        return var

    def gather_dimensions(self, group=None, root=0):
        target_group = self.get_group(group=group)
        new_dimensions = [None] * len(target_group['dimensions'])
        for idx, dim in enumerate(target_group['dimensions']):
            if dim.dist:
                dim = self._gather_dimension_(dim.name, group=group, root=root)
            else:
                pass
            new_dimensions[idx] = dim

        if self.rank == root:
            target_group['dimensions'] = []
            for dim in new_dimensions:
                self.add_dimension(dim, group=group)
        else:
            pass

        if self.rank == root:
            return self.get_group(group=group)
        else:
            return None

    def get_bounds_local(self, group=None, rank=MPI_RANK):
        the_group = self.get_group(group=group, rank=rank)
        ret = [dim.bounds_local for dim in the_group['dimensions'].values()]
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
            if any([dim.is_empty for dim in the_group['dimensions'].values()]):
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

    def get_variable_ranks(self, *args, **kwargs):
        variable_dist = self.get_variable(*args, **kwargs)
        return variable_dist.get('ranks', tuple(range(self.size)))

    def get_group(self, group=None, rank=MPI_RANK):
        return self._create_or_get_group_(group, rank=rank)

    def iter_groups(self, rank=MPI_RANK):
        from ocgis.driver.base import iter_all_group_keys, get_group
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
            ranks = range(self.size)
        else:
            ranks = [rank]

        for rank in ranks:
            for _, group_data in self.iter_groups(rank=rank):
                dimdict = group_data['dimensions']

                # If there are no distributed dimensions, there is no work to be dome with MPI bounds.
                if not any([dim.dist for dim in dimdict.values()]):
                    continue

                # Get dimension lengths.
                lengths = {dim.name: len(dim) for dim in dimdict.values() if dim.dist}
                # Choose the size of the distribution group. There needs to be at least one element per rank. First,
                # get the longest distributed dimension.
                max_length = max(lengths.values())
                for k, v in lengths.items():
                    if v == max_length:
                        distributed_dimension = dimdict[k]
                # Ensure only the distributed dimension remains distributed.
                for dim in dimdict.values():
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

                # Update distributed variables. If a variable has a distributed dimension, it must be distributed.
                # That's it.
                for variable_name, variable_data in group_data['variables'].items():
                    if len(variable_data['dimensions']) > 0:
                        for dimension_name in variable_data['dimensions']:
                            if group_data['dimensions'][dimension_name].dist:
                                variable_data['dist'] = MPIDistributionMode.DISTRIBUTED
                                break

        self.has_updated_dimensions = True

    def _create_or_get_group_(self, group, rank=MPI_RANK):
        # Allow None and single string group selection.
        if group is None or isinstance(group, basestring):
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

    def _gather_dimension_(self, name, group=None, root=0):
        dim = self.get_dimension(name, group=group)
        parts = self.comm.gather(dim, root=root)
        if self.rank == root:
            new_size = 0
            for part in parts:
                if not part.is_empty:
                    new_size += len(part)
                else:
                    pass

            # Only update the size if it is set.
            if dim.size is not None:
                dim._size = new_size
            else:
                pass
            dim._size_current = new_size

            if dim._src_idx is not None:
                new_src_idx = np.zeros(new_size, dtype=dim._src_idx.dtype)
                for part in parts:
                    if not part.is_empty:
                        lower, upper = part.bounds_local
                        new_src_idx[lower:upper] = part._src_idx
                    else:
                        pass
                dim._src_idx = new_src_idx
            else:
                pass

            # Dimension is no longer distributed and should not have local bounds.
            dim._bounds_local = None
            # Dimension is no longer empty as its component parts have been gathered across ranks.
            dim.is_empty = False

            ret = dim
        else:
            ret = None
        return ret


def barrier_print(*args):
    if len(args) == 1:
        args = args[0]
    ranks = range(MPI_SIZE)
    MPI_COMM.Barrier()
    for rank in ranks:
        if rank == MPI_RANK:
            print ''
            print '(rank={}, barrier=True) {}'.format(MPI_RANK, args)
            print ''
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
                        comm.recv(source=irank, tag=MPITags.BARRIER)
                        received.append(irank)
                for irank in ranks:
                    if irank != root:
                        comm.send(True, dest=irank, tag=MPITags.BARRIER)
            else:
                comm.send(True, dest=root, tag=MPITags.BARRIER)
                comm.recv(source=root, tag=MPITags.BARRIER)


def create_slices(length, size):
    # tdk: optimize: remove np.arange
    r = np.arange(length)
    sections = np.array_split(r, size)
    sections = [get_optimal_slice_from_array(s, check_diff=False) for s in sections]
    return sections


def dgather(elements):
    grow = elements[0]
    for idx in range(1, len(elements)):
        for k, v in elements[idx].iteritems():
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
            comm.send(value, dest=root)

    if rank == root:
        for idx, recv_rank in enumerate(ranks):
            if recv_rank != root:
                recv = comm.recv(source=recv_rank)
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


def scatter_ranks(ranks, value, root=0, comm=None):
    comm, rank, size = get_standard_comm_state(comm=comm)

    if rank == root:
        for target_rank in ranks:
            if target_rank != root:
                comm.send(value, dest=target_rank, tag=MPITags.SCATTER)
    elif rank in ranks:
        value = comm.recv(source=root, tag=MPITags.SCATTER)
    else:
        value = None

    return value


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


def get_nonempty_ranks(target, ranks=None, comm=None, root=0):
    comm, rank, size = get_standard_comm_state(comm)
    if ranks is None:
        ranks = range(size)

    if rank == root:
        is_empty = [None] * len(ranks)
        is_empty[rank] = target.is_empty
        for curr_rank in ranks:
            if curr_rank != root:
                is_empty[curr_rank] = comm.recv(source=curr_rank)
    else:
        comm.send(target.is_empty, dest=root)

    if rank == root:
        select_not_empty = np.invert(is_empty)
        ret = tuple(np.arange(size)[select_not_empty].tolist())
    else:
        ret = None

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
    print msg


def variable_gather(variable, root=0, comm=None):
    comm = comm or MPI_COMM
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size > 1:
        new_variable = variable.copy()
        new_variable.dtype = variable.dtype
        new_variable.set_mask(None)
        new_variable.set_value(None)
        new_variable.set_dimensions(None)
    else:
        new_variable = variable
        for dim in new_variable.dimensions:
            dim.dist = False

    if size > 1:
        if rank == root:
            new_dimensions = [None] * variable.ndim
        else:
            new_dimensions = None
        assert not new_variable.has_allocated_value

        for idx, dim in enumerate(variable.dimensions):
            if dim.dist:
                parts = comm.gather(dim, root=root)
                variable_is_empty = comm.gather(variable.is_empty, root=root)
            if rank == root:
                new_dim = dim.copy()
                if dim.dist:
                    new_size = 0
                    has_src_idx = False
                    for vempty, part in zip(variable_is_empty, parts):
                        if not part.is_empty and not vempty:
                            has_src_idx = part._src_idx is not None
                            new_size += len(part)
                    if has_src_idx:
                        new_src_idx = np.zeros(new_size, dtype=DataTypes.DIMENSION_SRC_INDEX)
                        for part in parts:
                            if not part.is_empty:
                                new_src_idx[part.bounds_local[0]: part.bounds_local[1]] = part._src_idx
                    else:
                        new_src_idx = None
                    new_dim = dim.copy()
                    new_dim.set_size(new_size, src_idx=new_src_idx)
                    new_dim.dist = False
                new_dimensions[idx] = new_dim

        if rank == root:
            new_variable.set_dimensions(new_dimensions, force=True)

        gathered_variables = comm.gather(variable, root=root)

        if rank == root:
            for idx, gv in enumerate(gathered_variables):
                if not gv.is_empty:
                    destination_slice = [slice(*dim.bounds_local) for dim in gv.dimensions]
                    new_variable.__setitem__(destination_slice, gv)

            new_variable.dist = None
            new_variable.ranks = None

            return new_variable
        else:
            return
    else:
        new_variable.dist = None
        new_variable.ranks = None
        return new_variable


def variable_scatter(variable, dest_mpi, root=0, comm=None):
    comm = comm or MPI_COMM
    rank = comm.Get_rank()

    if rank == root:
        if variable.dist is not None:
            raise ValueError('Only variables with no prior distribution may be scattered.')
        if not dest_mpi.has_updated_dimensions:
            raise ValueError('The destination distribution must have updated dimensions.')

    # Synchronize distribution across processors.
    dest_mpi = comm.bcast(dest_mpi, root=root)

    # Find the appropriate group for the dimensions.
    if rank == root:
        group = variable.group
        dimension_names = [dim.name for dim in variable.dimensions]
    else:
        group = None
        dimension_names = None

    # Synchronize the processes with the MPI distribution and the group containing the dimensions.
    dest_mpi = comm.bcast(dest_mpi, root=root)
    group = comm.bcast(group, root=root)
    dimension_names = comm.bcast(dimension_names, root=root)

    # These are the dimensions for the local process.
    dest_dimensions = dest_mpi.get_dimensions(dimension_names, group=group)

    # barrier_print('dest_dimensions bounds_global: ', [(d.name, d.bounds_global) for d in dest_dimensions])

    # Slice the variables collecting the sequence to scatter to the MPI procs.
    if rank == root:
        size = dest_mpi.size
        if size > 1:
            slices = [None] * size
            # Get the slices need to scatter the variables. These are essentially the local bounds on each dimension.
            empty_ranks = dest_mpi.get_empty_ranks()
            empty_variable = variable.copy()
            empty_variable.convert_to_empty()
            for current_rank in range(size):
                if current_rank in empty_ranks:
                    slices[current_rank] = None
                else:
                    current_dimensions = dest_mpi.get_group(group=group, rank=current_rank)['dimensions'].values()
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
    scattered_variable = comm.scatter(variables_to_scatter, root=root)
    # Update the scattered variable dimensions with the destination dimensions on the process. Everything should align
    # shape-wise. If they don't, an exception will be raised.
    scattered_variable.set_dimensions(dest_dimensions, force=True)
    # The variable is now distributed.
    if scattered_variable.has_distributed_dimension:
        scattered_variable.dist = MPIDistributionMode.DISTRIBUTED
    else:
        scattered_variable.dist = MPIDistributionMode.REPLICATED

    return scattered_variable, dest_mpi


def variable_collection_scatter(variable_collection, dest_mpi, root=0, comm=None):
    comm = comm or MPI_COMM
    rank = comm.Get_rank()

    if rank == root:
        target = variable_collection.first()
    else:
        target = None
    sv, dest_mpi = variable_scatter(target, dest_mpi, root=root, comm=comm)
    svc = sv.parent

    # The variable is now distributed.
    for scattered_variable in svc.values():
        if not scattered_variable.is_empty:
            if scattered_variable.has_distributed_dimension:
                scattered_variable.dist = MPIDistributionMode.DISTRIBUTED
            else:
                scattered_variable.dist = MPIDistributionMode.REPLICATED

    if rank == root:
        n_children = len(variable_collection.children)
    else:
        n_children = None
    n_children = comm.bcast(n_children, root=root)
    if rank == root:
        children = variable_collection.children.values()
    else:
        children = [None] * n_children
    for child in children:
        scattered_child = variable_collection_scatter(child, dest_mpi, root=root, comm=comm)[0]
        svc.add_child(scattered_child, force=True)

    return svc, dest_mpi


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
