import itertools

import numpy as np
from shapely.geometry import box

from ocgis.base import AbstractOcgisObject
from ocgis.collection.field import OcgField
from ocgis.vm.mpi import OcgMpi


class GridSplitter(AbstractOcgisObject):
    """
    Splits source and destination grids into separate netCDF files. "Source" is intended to mean the source data for a
    regridding operation. "Destination" is the destination grid for regridding operation.

    The destination subset extents are buffered to ensure full overlap with the source destination subset. Hence,
    elements in the destination subset are globally unique and source subset elements are not necessarily globally
    unique.

    .. note:: All function calls are collective.

    :param src_grid: The source grid for a regridding operation.
    :type src_grid: :class:`ocgis.GridXY`
    :param dst_grid: The destination grid for a regridding operation.
    :type dst_grid: :class:`ocgis.GridXY`
    :param tuple nsplits_dst: The split count for the grid. Tuple length must match the dimension count of the grid.

    >>> npslits_dst = (2, 3)

    :param bool check_contains: If ``True``, check that the source subset bounding box fully contains the destination
     subset bounding box.
    :raises: ValueError
    """

    def __init__(self, src_grid, dst_grid, nsplits_dst, check_contains=True):
        if len(nsplits_dst) != dst_grid.ndim:
            raise ValueError('The number of splits must match the grid dimension count.')

        self.src_grid = src_grid
        self.dst_grid = dst_grid
        self.nsplits_dst = nsplits_dst
        self.check_contains = check_contains

    def iter_dst_grid_slices(self):
        """
        Yield global slices for the destination grid using guidance from ``nsplits_dst``.

        :return: A dictionary with keys as the grid dimension names and the values the associated slice for that
         dimension.
        :rtype: dict

        >>> example_yield = {'dimx': slice(2, 4), 'dimy': slice(10, 20)}
        """

        slice_store = []
        ydim_name = self.dst_grid.dimensions[0].name
        xdim_name = self.dst_grid.dimensions[1].name
        dst_grid_shape_global = self.dst_grid.shape_global
        # barrier_print('dst grid shape global', dst_grid_shape_global)
        for idx in range(self.dst_grid.ndim):
            splits = self.nsplits_dst[idx]
            size = dst_grid_shape_global[idx]
            slices = create_slices_for_dimension(size, splits)
            # barrier_print('size splits slices', size, splits, slices)
            slice_store.append(slices)
        for slice_y, slice_x in itertools.product(*slice_store):
            yield {ydim_name: create_slice_from_tuple(slice_y),
                   xdim_name: create_slice_from_tuple(slice_x)}

    def iter_dst_grid_subsets(self):
        """
        Using slices from ``iter_dst_grid_slices``, yield destination grid subsets.

        :return: The sliced grid object.
        :rtype: :class:`ocgis.GridXY`
        """

        for slc in self.iter_dst_grid_slices():
            sub = self.dst_grid.get_distributed_slice(slc)
            yield sub

    def iter_src_grid_subsets(self, yield_dst=False):
        """
        Yield source grid subsets using the extent of its associated destination grid subset.

        :param bool yield_dst: If ``True``, yield the destination subset as well as the source grid subset.
        :return: The source grid if ``yield_dst`` is ``False``, otherwise a two-element tuple in the form
         ``(<source grid subset>, <destination grid subset>)``.
        :rtype: :class:`ocgis.GridXY` or (:class:`ocgis.GridXY`, :class:`ocgis.GridXY`)
        """

        dst_grid_resolution = self.dst_grid.resolution
        src_grid_resolution = self.src_grid.resolution

        if dst_grid_resolution >= src_grid_resolution:
            target_resolution = dst_grid_resolution
        else:
            target_resolution = src_grid_resolution
        buffer_value = 2 * target_resolution

        # barrier_print('before self.iter_dst_grid_subsets')
        for dst_grid_subset in self.iter_dst_grid_subsets():
            # barrier_print('dst_grid_subset ', dst_grid_subset.x.get_value().max())
            sub_box = box(*dst_grid_subset.extent_global).buffer(buffer_value)
            src_grid_subset = self.src_grid.get_intersects(sub_box, keep_touches=True)
            if self.check_contains:
                src_box = box(*src_grid_subset.extent_global)
                dst_box = box(*dst_grid_subset.extent_global)
                if not does_contain(src_box, dst_box):
                    raise ValueError('Contains check failed.')

            if yield_dst:
                yld = (src_grid_subset, dst_grid_subset)
            else:
                yld = src_grid_subset
            yield yld

    def write_subsets(self, src_template, dst_template):
        """
        Write grid subsets to netCDF files using the provided filename templates. The template must contain the full
        file path with a single curly-bracer pair to insert the combination counter.

        >>> template_example = '/path/to/data_{}.nc'

        :param str src_template: The template for the source subset file.
        :param str dst_template: The template for the destination subset file.
        """

        for ctr, (sub_src, sub_dst) in enumerate(self.iter_src_grid_subsets(yield_dst=True)):
            src_path = src_template.format(ctr)
            dst_path = dst_template.format(ctr)
            for target, path in zip([sub_src, sub_dst], [src_path, dst_path]):
                field = OcgField(grid=target)
                field.write(path)


def create_slice_from_tuple(tup):
    return slice(tup[0], tup[1])


def create_slices_for_dimension(size, splits):
    ompi = OcgMpi(size=splits)
    dimname = 'foo'
    ompi.create_dimension(dimname, size, dist=True)
    ompi.update_dimension_bounds()
    slices = []
    for rank in range(splits):
        dimension = ompi.get_dimension(dimname, rank=rank)
        slices.append(dimension.bounds_local)
    return slices


def does_contain(container, containee):
    intersection = container.intersection(containee)
    return np.isclose(intersection.area, containee.area)
