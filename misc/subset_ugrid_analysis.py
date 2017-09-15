import os

import matplotlib.pyplot as plt
import numpy as np
import shapely
from descartes import PolygonPatch
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from subset_ugrid_first_principles import IN_PATH, FACE_CENTER_Y, FACE_CENTER_X

from ocgis.driver.request.core import RequestDataset
from ocgis.test.base import nc_scope


def plot_centers(path):
    rd = RequestDataset(path)
    vc = rd.get_raw_field()

    x = vc[FACE_CENTER_X].get_value()  # [::100]
    y = vc[FACE_CENTER_Y].get_value()  # [::100]

    plt.scatter(x, y, marker='o', color='b')


def plot_subset():
    # src_file = '/home/benkoziol/l/data/ocgis/ugrid-cesm-subsetting/UGRID_1km-merge-10min_HYDRO1K-merge-nomask_c130402.nc'
    src_file = '/home/benkoziol/htmp/src_subset_57.nc'
    dst_file = '/home/benkoziol/htmp/dst_subset_57.nc'

    # the_plt = plot_centers(src_file)
    ugrid_corner_plotting(src_file, show=False)

    rd = RequestDataset(dst_file)
    vc = rd.get()
    x = vc['x'].get_value()
    y = vc['y'].get_value()
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    plt.scatter(x, y, marker='x', color='r')
    plt.show()


def resolution():
    rd = RequestDataset(IN_PATH)
    vc = rd.get_raw_field()

    x = vc[FACE_CENTER_X].get_value()
    x = np.sort(x)
    y = vc[FACE_CENTER_Y].get_value()
    y = np.sort(y)

    dx = np.diff(x)
    sx = {dx.min(), dx.mean(), dx.max()}

    dy = np.diff(y)
    sy = {dy.min(), dy.mean(), dy.max()}

    print sx, sy


def ugrid_area():
    # path = '/home/benkoziol/htmp/src_subset_1.nc'
    path = '/home/benkoziol/l/data/ocgis/ugrid-cesm-subsetting/UGRID_1km-merge-10min_HYDRO1K-merge-nomask_c130402.nc'
    rd = RequestDataset(path)
    vc = rd.get_raw_field()
    vc.load()

    face_nodes = vc['landmesh_face_node'].get_value()
    face_node_x = vc['landmesh_node_x'].get_value()
    face_node_y = vc['landmesh_node_y'].get_value()
    face_center_x = vc['landmesh_face_x'].get_value()
    face_center_y = vc['landmesh_face_y'].get_value()

    areas = []
    for ctr, idx in enumerate(range(face_nodes.shape[0])):
        if ctr % 10000 == 0:
            print '{} of {}'.format(ctr, face_nodes.shape[0])

        curr_face_indices = face_nodes[idx, :]
        curr_face_node_x = face_node_x[curr_face_indices]
        curr_face_node_y = face_node_y[curr_face_indices]
        face_coords = np.zeros((4, 2))
        face_coords[:, 0] = curr_face_node_x
        face_coords[:, 1] = curr_face_node_y
        poly = Polygon(face_coords)
        parea = poly.area
        poly = shapely.geometry.box(*poly.bounds)

        pt = Point(face_center_x[idx], face_center_y[idx])

        if not poly.intersects(pt):
            print idx, np.array(pt), poly.bounds

        # if parea > 1:
        #     print idx
        #     print face_nodes[idx, :]
        #     print face_coords
        #     print poly.bounds
        #     sys.exit()

        areas.append(parea)


def ugrid_corner_plotting(path, show=True):
    # path = '/home/benkoziol/htmp/ugrid_splits/src_subset_1.nc'
    # path = '/home/benkoziol/l/data/ocgis/ugrid-cesm-subsetting/UGRID_1km-merge-10min_HYDRO1K-merge-nomask_c130402.nc'

    rd = RequestDataset(path)
    vc = rd.get_raw_field()
    vc.load()

    face_nodes = vc['landmesh_face_node'].get_value()
    face_node_x = vc['landmesh_node_x'].get_value()
    face_node_y = vc['landmesh_node_y'].get_value()

    for ctr, idx in enumerate(range(face_nodes.shape[0])):
        if ctr % 1000 == 0:
            print '{} of {}'.format(ctr, face_nodes.shape[0])

        curr_face_indices = face_nodes[idx, :]
        curr_face_node_x = face_node_x[curr_face_indices]
        curr_face_node_y = face_node_y[curr_face_indices]
        face_coords = np.zeros((4, 2))
        face_coords[:, 0] = curr_face_node_x
        face_coords[:, 1] = curr_face_node_y

        plt.scatter(face_coords[:, 0], face_coords[:, 1], marker='o', color='b')

    if show:
        plt.show()


def check_coords():
    path = '/home/benkoziol/l/data/ocgis/ugrid-cesm-subsetting/UGRID_1km-merge-10min_HYDRO1K-merge-nomask_c130402.nc'
    indices = np.array([73139, 74240, 74241, 74242])
    with nc_scope(path) as ds:
        var = ds.variables['landmesh_node_x']
        var = ds.variables['landmesh_node_y']
        print var[indices]

        var = ds.variables['landmesh_face_node']
        print var[0, :]


def analyze_weights():
    folder = '/home/benkoziol/htmp/esmf_weights_full_20170628'
    for f in os.listdir(folder):
        if f.startswith('esmf_weights'):
            f = os.path.join(folder, f)
            print f
            rd = RequestDataset(f)
            vc = rd.get_raw_field()
            weights = vc['S'].get_value()
            wmin, wmax = weights.min(), weights.max()
            if wmin < 0:
                raise ValueError('min less than 0: {}'.format(f))
            if wmax > 1.0 + 1e-6:
                raise ValueError('max greater than 1 ({}): {}'.format(wmax, f))


def check_spatial_overlap():
    BLUE = '#6699cc'
    GRAY = '#999999'

    src_file = '/home/benkoziol/htmp/src_subset_57.nc'
    dst_file = '/home/benkoziol/htmp/dst_subset_57.nc'

    vc = RequestDataset(src_file).get_raw_field()
    face_node_x = vc['landmesh_node_x'].get_value()
    face_node_y = vc['landmesh_node_y'].get_value()
    minx, maxx = face_node_x.min(), face_node_x.max()
    miny, maxy = face_node_y.min(), face_node_y.max()
    src_box = shapely.geometry.box(minx, miny, maxx, maxy)

    field = RequestDataset(dst_file).get()
    dst_box = field.grid.envelope

    print 'overlap={}'.format(src_box.intersects(dst_box))

    src_patch = PolygonPatch(src_box, fc=BLUE, ec=BLUE, alpha=0.5, zorder=2)
    dst_patch = PolygonPatch(dst_box, fc=GRAY, ec=GRAY, alpha=0.5, zorder=1)

    fig = plt.figure(num=1)
    ax = fig.add_subplot(111)
    ax.add_patch(src_patch)
    ax.add_patch(dst_patch)

    minx, miny, maxx, maxy = dst_box.bounds
    w, h = maxx - minx, maxy - miny
    ax.set_xlim(minx - 0.2 * w, maxx + 0.2 * w)
    ax.set_ylim(miny - 0.2 * h, maxy + 0.2 * h)
    ax.set_aspect(1)

    plt.show()


if __name__ == '__main__':
    # ugrid_area()
    # ugrid_corner_plotting()
    # check_coords()
    # resolution()

    # the_plt = plot_centers('/home/benkoziol/htmp/ugrid_splits/src_subset_1.nc')
    # the_plt.show()

    # plot_subset()

    # analyze_weights()

    check_spatial_overlap()
