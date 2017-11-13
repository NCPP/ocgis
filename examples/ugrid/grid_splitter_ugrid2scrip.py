import os
import time
from collections import deque

import netCDF4 as nc
import numpy as np
from shapely.geometry import box, Point

import ocgis

# Path to the source grid. This is assumed to be UGRID.
SRC_PATH = '/media/benkoziol/Extra Drive 1/data/bekozi-work/i49-ugrid-cesm/UGRID_1km-merge-10min_HYDRO1K-merge-nomask_c130402.nc'
# SRC_PATH = '/home/ubuntu/data/i49-ugrid-cesm/UGRID_1km-merge-10min_HYDRO1K-merge-nomask_c130402.nc'

# Path to the destination file. This is assumed to be structured or unstructured SCRIP.
# DST_PATH = '/media/benkoziol/Extra Drive 1/data/bekozi-work/i49-ugrid-cesm/0.9x1.25_c110307.nc'
# DST_PATH = '/media/benkoziol/Extra Drive 1/data/bekozi-work/i49-ugrid-cesm/SCRIPgrid_ne16np4_nomask_c110512.nc'
# DST_PATH = '/home/ubuntu/data/i49-ugrid-cesm/0.9x1.25_c110307.nc'
# DST_PATH = '/home/ubuntu/data/i49-ugrid-cesm/SCRIPgrid_ne16np4_nomask_c110512.nc'
DST_PATH = '/home/benkoziol/l/data/bekozi-work/i49-ugrid-cesm/SCRIPgrid_1x1pt_brazil_nomask_c110308.nc'

# Path to the data output directory which will contain chunked grid files, weight files, and the master merged weight
# file.
# WD = '/home/benkoziol/htmp/ugrid_splits'
# WD = '/home/ubuntu/data/i49-ugrid-cesm/splits1'
# WD = '/home/ubuntu/data/i49-ugrid-cesm/splits1.1'
# WD = '/home/ubuntu/data/i49-ugrid-cesm/splits2'
WD = '/home/benkoziol/l/data/bekozi-work/i49-ugrid-cesm/splits/SCRIPgrid_1x1pt_brazil_nomask_c110308'

# Contains chunking information for each grid.
GS_META = {'0.9x1.25_c110307.nc': {'spatial_resolution': 1.25,  # Not really used since the buffer value is set.
                                   'buffer_value': 0.75,
                                   # Distance to buffer around the chunk extent for the source subset.
                                   'nsplits_dst': 96},  # This will allow at least two rows per chunk.
           'UGRID_1km-merge-10min_HYDRO1K-merge-nomask_c130402.nc': {'spatial_resolution': 0.167},
           'SCRIPgrid_ne16np4_nomask_c110512.nc': {'buffer_value': None,
                                                   'nsplits_dst': 50,
                                                   'spatial_resolution': 3.0},
           'SCRIPgrid_1x1pt_brazil_nomask_c110308.nc': {'spatial_resolution': 2.0}}

# WD = '/home/ubuntu/data/i49-ugrid-cesm/splits'
# SRC_PATH = '/home/ubuntu/data/i49-ugrid-cesm/UGRID_1km-merge-10min_HYDRO1K-merge-nomask_c130402.nc'
# DST_PATH = '/home/ubuntu/data/i49-ugrid-cesm/0.9x1.25_c110307.nc'

# Do not set units on bounds variables by default. ESMF does not support units on bounds.
ocgis.env.CLOBBER_UNITS_ON_BOUNDS = False


def create_scrip_grid(path):
    """Create an OCGIS grid from a SCRIP file.

    :param str path: Path to source NetCDF file.
    """
    rfield = ocgis.RequestDataset(path).create_raw_field()
    pgc = ocgis.PointGC(x=rfield['grid_center_lon'], y=rfield['grid_center_lat'], crs=ocgis.crs.Spherical())
    return ocgis.GridUnstruct(geoms=pgc)


def iter_dst(grid_splitter, yield_slice=False):
    """
    Writes a chunk of the destination grid and yields an OCGIS SCRIP grid object. This is called internally by the grid
    splitter.

    :param grid_splitter: A grid splitter object.
    :type grid_splitter: :class:`ocgis.spatial.grid_splitter.GridSplitter`
    :param bool yield_slice: If ``True`` yield the slice that created the grid subset.
    :rtype: :class:`ocgis.GridUnstruct`
    """
    pgc = grid_splitter.dst_grid.abstractions_available['point']

    center_lat = pgc.parent['grid_center_lat'].get_value()
    ucenter_lat = np.unique(center_lat)
    ucenter_splits = np.array_split(ucenter_lat, grid_splitter.nsplits_dst[0])

    for ctr, ucenter_split in enumerate(ucenter_splits, start=1):
        select = np.zeros_like(center_lat, dtype=bool)
        for v in ucenter_split.flat:
            select = np.logical_or(select, center_lat == v)
        sub = pgc.parent[{pgc.node_dim.name: select}]
        split_path = os.path.join(WD, 'split_dst_{}.nc').format(ctr)

        ux = np.unique(sub['grid_center_lon'].get_value()).shape[0]
        uy = np.unique(sub['grid_center_lat'].get_value()).shape[0]
        sub['grid_dims'].get_value()[:] = ux, uy

        with ocgis.vm.scoped('grid write', [0]):
            if not ocgis.vm.is_null:
                sub.write(split_path, driver='netcdf')
        ocgis.vm.barrier()

        yld = create_scrip_grid(split_path)

        if yield_slice:
            yld = yld, ucenter_split
        yield yld


def iter_dst2(grid_splitter, yield_slice=False):
    """
    Writes a chunk of the destination grid and yields an OCGIS SCRIP grid object. This is called internally by the grid
    splitter.

    :param grid_splitter: A grid splitter object.
    :type grid_splitter: :class:`ocgis.spatial.grid_splitter.GridSplitter`
    :param bool yield_slice: If ``True`` yield the slice that created the grid subset.
    :rtype: :class:`ocgis.GridUnstruct`
    """
    pgc = grid_splitter.dst_grid.abstractions_available['point']
    lon_corners = pgc.parent['grid_corner_lon'].get_value()
    lat_corners = pgc.parent['grid_corner_lat'].get_value()
    bounds_global = lon_corners.min(), lat_corners.min(), lon_corners.max(), lat_corners.max()

    bounds_global = box(*bounds_global).buffer(0.01).envelope.bounds
    num = GS_META['SCRIPgrid_ne16np4_nomask_c110512.nc']['nsplits_dst']
    lat_ranges = np.linspace(bounds_global[1], bounds_global[3], num=num)

    for ii in range(lat_ranges.shape[0]):
        if ii == lat_ranges.shape[0] - 1:
            break
        else:
            lr = lat_ranges[ii], lat_ranges[ii + 1]

        subset_bounds = (bounds_global[0], lr[0], bounds_global[2], lr[1])

        subset_geom = ocgis.GeometryVariable.from_shapely(box(*subset_bounds), is_bbox=True, crs=ocgis.crs.Spherical())

        yld = pgc.get_intersects(subset_geom)

        lon_corners = yld.parent['grid_corner_lon'].get_value()
        lat_corners = yld.parent['grid_corner_lat'].get_value()
        new_bounds = lon_corners.min(), lat_corners.min(), lon_corners.max(), lat_corners.max()

        split_path = os.path.join(WD, 'split_dst_{}.nc').format(ii + 1)

        yld.parent.write(split_path, driver='netcdf')

        yld.parent.attrs['extent_global'] = new_bounds

        if yield_slice:
            yld = yld, subset_geom

        yield yld


def assert_weight_file_is_rational(weight_filename):
    """Asserts a weight file is rational."""
    wf = ocgis.RequestDataset(weight_filename).get()
    row = wf['row'].get_value()
    S = wf['S'].get_value()
    # row = wf['col'].get_value()
    processed = deque()
    for row_value in row.flat:
        if row_value not in processed:
            print weight_filename
            print 'current row:', row_value
            row_idx = row == row_value
            curr_S = S[row_idx].sum()
            print 'current S sum:', curr_S
            print '============================='
            processed.append(row_value)
            # assert abs(1.0 - curr_S) <= 1e-6


def create_grid_splitter(src_path, dst_path):
    """Create grid splitter object from a source and destination path."""
    src_filename = os.path.split(src_path)[1]
    dst_filename = os.path.split(dst_path)[1]

    grid_splitter_paths = {'wd': WD}
    grid_abstraction = ocgis.constants.GridAbstraction.POINT
    src_grid = ocgis.RequestDataset(uri=src_path, driver='netcdf-ugrid', grid_abstraction=grid_abstraction).get().grid

    dst_grid = create_scrip_grid(dst_path)

    nsplits_dst = GS_META[dst_filename]['nsplits_dst']
    src_grid_resolution = GS_META[src_filename]['spatial_resolution']
    dst_grid_resolution = GS_META[dst_filename]['spatial_resolution']
    buffer_value = GS_META[dst_filename]['buffer_value']

    if dst_filename == 'SCRIPgrid_ne16np4_nomask_c110512.nc':
        idest = iter_dst2
    else:
        idest = iter_dst

    gs = ocgis.GridSplitter(src_grid, dst_grid, (nsplits_dst,), paths=grid_splitter_paths,
                            src_grid_resolution=src_grid_resolution, check_contains=False,
                            dst_grid_resolution=dst_grid_resolution, iter_dst=idest, buffer_value=buffer_value,
                            redistribute=True)
    return gs


def main(write_subsets=False, merge_weight_files=False):
    # if ocgis.vm.size > 1:
    #     raise RuntimeError('Script does not work in parallel.')
    if ocgis.vm.rank == 0:
        tmain_start = time.time()
        ocgis.vm.rank_print('starting main')

    dst_filename = os.path.split(DST_PATH)[1]
    if dst_filename == 'SCRIPgrid_1x1pt_brazil_nomask_c110308.nc':
        buffer_value = GS_META[dst_filename]['spatial_resolution'] * 1.25
        field = ocgis.RequestDataset(DST_PATH, driver='netcdf').get()
        center_lon = field['grid_center_lon'].get_value()[0]
        center_lat = field['grid_center_lat'].get_value()[0]
        subset_geom = box(*Point(center_lon, center_lat).buffer(buffer_value).envelope.bounds)
        subset_geom = ocgis.GeometryVariable.from_shapely(subset_geom, is_bbox=True, crs=ocgis.crs.Spherical())
        subset_geom.unwrap()
        src_field = ocgis.RequestDataset(SRC_PATH, driver='netcdf-ugrid', crs=ocgis.crs.Spherical(),
                                         grid_abstraction='point').get()
        src_subset = src_field.grid.get_intersects(subset_geom, optimized_bbox_subset=True).parent
        src_subset = src_subset.grid.reduce_global().parent
        out_path = os.path.join(WD, dst_filename[0:-3] + '_UGRID_subset.nc')
        src_subset.write(out_path, driver='netcdf')
    else:
        gs = create_grid_splitter(SRC_PATH, DST_PATH)
        if write_subsets:
            gs.write_subsets()
        if merge_weight_files:
            mwfn = os.path.join(WD, '01-merged_weights.nc')
            gs.create_merged_weight_file(mwfn)
            with nc.Dataset(mwfn, 'a') as ds:
                ds.setncattr('row_input', DST_PATH)
                ds.setncattr('col_input', SRC_PATH)

    ocgis.vm.barrier()
    if ocgis.vm.rank == 0:
        tmain_stop = time.time()
        ocgis.vm.rank_print('timing::main::{}'.format(tmain_stop - tmain_start))


def diagnostics():
    """Miscellaneous diagnostic code."""
    # print('writing source...')
    # from shapely.geometry import box
    # path = '/home/benkoziol/l/data/bekozi-work/i49-ugrid-cesm/splits/split_src_79.nc'
    # rd = ocgis.RequestDataset(path, driver='netcdf-ugrid', grid_abstraction='polygon')
    # field = rd.get()
    # gv = field.grid.archetype.convert_to()
    # gv.write_vector('/home/benkoziol/l/data/bekozi-work/i49-ugrid-cesm/splits/split_src_79.shp')
    #
    # print('writing destination...')
    # path = '/home/benkoziol/l/data/bekozi-work/i49-ugrid-cesm/splits/split_dst_79.nc'
    # rd = ocgis.RequestDataset(path, driver='netcdf')
    # field = rd.get()
    # grid_size = field.dimensions['grid_size'].size
    # lon_corner = field['grid_corner_lon'].get_value()
    # lat_corner = field['grid_corner_lat'].get_value()
    # value = np.ones(grid_size, dtype=object)
    # for idx in range(grid_size):
    #     curr_lon = lon_corner[idx, :]
    #     curr_lat = lat_corner[idx, :]
    #     minx, maxx = curr_lon.min(), curr_lon.max()
    #     miny, maxy = curr_lat.min(), curr_lat.max()
    #     value[idx] = box(minx, miny, maxx, maxy)
    # gv = ocgis.GeometryVariable(name='geom', value=value, crs=ocgis.crs.Spherical(), geom_type='Polygon',
    #                             dimensions='geom')
    # # gv = field.grid.archetype.convert_to()
    # gv.write_vector('/home/benkoziol/l/data/bekozi-work/i49-ugrid-cesm/splits/split_dst_79.shp')

    # path = '/media/benkoziol/Extra Drive 1/data/bekozi-work/i49-ugrid-cesm/SCRIPgrid_ne16np4_nomask_c110512.nc'
    # from shapely.geometry import Polygon, Point
    # field = ocgis.RequestDataset(path).get()
    # lon_corner = field['grid_corner_lon'].get_value()
    # lat_corner = field['grid_corner_lat'].get_value()
    # lon = field['grid_center_lon'].get_value()
    # lat = field['grid_center_lat'].get_value()
    # size = lon_corner.shape[0]
    # size_corners = lon_corner.shape[1]
    # polys = np.zeros(size, dtype=object)
    # points = np.zeros(size, dtype=object)
    # for ii in range(size):
    #     coords = [(lon_corner[ii, jj], lat_corner[ii, jj]) for jj in range(size_corners)]
    #     polys[ii] = Polygon(coords)
    #     points[ii] = Point(lon[ii], lat[ii])

    # point_offset = []
    # for ii, (point, poly) in enumerate(zip(polys, points)):
    #     if not point.intersects(poly):
    #         point_offset.append(ii)
    # pgv = ocgis.GeometryVariable(name='bad_point', value=points[point_offset], dimensions='ngeoms', geom_type='Point')
    # pogv = ocgis.GeometryVariable(name='bad_poly', value=polys[point_offset], dimensions='ngeoms', geom_type='Polygon')
    # name_prefix = 'point_offset_'
    # for p in [pgv, pogv]:
    #     filename = '{}{}.shp'.format(name_prefix, p.name)
    #     print filename
    #     p.write_vector(os.path.join('/media/benkoziol/Extra Drive 1/data/bekozi-work/i49-ugrid-cesm', filename))
    #
    # cx = lon_corner[point_offset[1]]

    # print point_offset
    # import ipdb;
    # ipdb.set_trace()
    # polygons_overlap = []
    # for ii, (p1, p2) in enumerate(itertools.combinations(polys, 2)):
    #     if p1.overlaps(p2):
    #         polygons_overlap.append(ii)
    # print polygons_overlap

    # not_valid = []
    # for ii, poly in enumerate(polys):
    #     if not poly.is_valid:
    #         print '{} - not valid coordinates:'.format(ii)
    #         print 'x={}'.format(lon_corner[ii].tolist())
    #         print 'y={}'.format(lat_corner[ii].tolist())
    #         print '====='
    #         not_valid.append(ii)
    # print not_valid

    # zero_area = []
    # for ii, poly in enumerate(polys):
    #     if poly.area <= 0:
    #         zero_area.append(ii)
    # print zero_area

    # gv = ocgis.GeometryVariable(name='geom', value=polys, dimensions='ngeom', geom_type=polys[0].geom_type,
    #                             crs=ocgis.crs.Spherical())
    # gv.write_vector('/media/benkoziol/Extra Drive 1/data/bekozi-work/i49-ugrid-cesm/SCRIPgrid_ne16np4_nomask_c110512.shp')
    # gv = ocgis.GeometryVariable(name='geom', value=points, dimensions='ngeom', geom_type=points[0].geom_type,
    #                             crs=ocgis.crs.Spherical())
    # gv.write_vector(
    #     '/media/benkoziol/Extra Drive 1/data/bekozi-work/i49-ugrid-cesm/SCRIPgrid_ne16np4_nomask_c110512_centers.shp')
    # import ipdb;ipdb.set_trace()


# def test(splits=True):
#     """Run some miscellaneous testing."""
#     if splits:
#         for f in os.listdir(WD):
#             if f.startswith('esmf_weights'):
#                 path = os.path.join(WD, f)
#                 assert_weight_file_is_rational(path)
#
#     weight_filename = os.path.join(WD, '01-merged_weights.nc')
#     assert_weight_file_is_rational(weight_filename)

def test():
    main(write_subsets=True, merge_weight_files=False)


if __name__ == '__main__':
    main(write_subsets=True, merge_weight_files=False)
    # test()
    # diagnostics()
