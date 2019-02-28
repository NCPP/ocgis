import itertools

import numpy as np
from ocgis.constants import GridAbstraction, DMK, OcgisConvention
from ocgis.driver.nc_ugrid import DriverNetcdfUGRID
from ocgis.driver.request.core import RequestDataset
from ocgis.spatial.geomc import reduce_reindex_coordinate_index
from ocgis.spatial.grid import Grid, GridUnstruct
from ocgis.test.base import TestBase
from ocgis.util.helpers import find_index
from ocgis.variable.base import Variable, VariableCollection


class TestDriverNetcdfUGRID(TestBase):
    def fixture(self, **kwargs):
        path = self.get_temporary_file_path('__testdriverugrid__.nc')
        u = get_ugrid_data_structure()
        u.write(path)
        rd = RequestDataset(path, driver=DriverNetcdfUGRID, **kwargs)
        du = DriverNetcdfUGRID(rd)
        return du

    def test_system_is_isomorphic(self):
        """Test the is_isomorphic property on UGRID drivers."""

        f = self.fixture()
        field = f.create_field()
        self.assertFalse(field.grid.is_isomorphic)

        for g in [True, False]:
            f = self.fixture(grid_is_isomorphic=g)
            field = f.create_field()
            self.assertEqual(field.grid.is_isomorphic, g)

    def test_create_dimension_map(self):
        du = self.fixture()
        group_metadata = du.rd.metadata
        actual = du.create_dimension_map(group_metadata)
        desired = {'attribute_host': {'attrs': {u'cf_role': u'mesh_topology',
                                                u'dimension': 2,
                                                u'face_coordinates': u'face_center_x face_center_y',
                                                u'face_node_connectivity': u'face_node_index',
                                                u'locations': u'face node',
                                                u'node_coordinates': u'face_node_x face_node_y',
                                                u'standard_name': u'mesh_topology'},
                                      'bounds': None,
                                      'dimension': [],
                                      'variable': u'mesh'},
                   'crs': {'variable': 'latitude_longitude'},
                   'driver': 'netcdf-ugrid',
                   'topology': {'point': {'x': {'attrs': {},
                                                'bounds': None,
                                                'dimension': [u'n_face'],
                                                'variable': u'face_center_x'},
                                          'y': {'attrs': {},
                                                'bounds': None,
                                                'dimension': [u'n_face'],
                                                'variable': u'face_center_y'}},
                                'polygon': {'x': {'attrs': {},
                                                  'bounds': None,
                                                  'dimension': [u'n_node'],
                                                  'variable': u'face_node_x'},
                                            'y': {'attrs': {},
                                                  'bounds': None,
                                                  'dimension': [u'n_node'],
                                                  'variable': u'face_node_y'},
                                            'element_node_connectivity': {
                                                'attrs': {'standard_name': 'face_node_connectivity'},
                                                'bounds': None,
                                                'dimension': [u'n_face'],
                                                'variable': u'face_node_index'}
                                            }}}

        self.assertDictEqual(actual.as_dict(), desired)

    def test_create_field(self):
        du = self.fixture(grid_abstraction=GridAbstraction.POINT)
        field = du.create_field()
        self.assertEqual(field.grid_abstraction, GridAbstraction.POINT)
        self.assertEqual(field.driver, DriverNetcdfUGRID)
        self.assertIsInstance(field.grid, GridUnstruct)
        self.assertEqual(field.grid.abstraction, GridAbstraction.POINT)
        self.assertIsNone(field.grid.archetype.cindex)

        # Test retrieving polygons.
        du = self.fixture(grid_abstraction=GridAbstraction.POLYGON)
        field = du.create_field()
        self.assertEqual(field.grid_abstraction, GridAbstraction.POLYGON)
        self.assertEqual(field.driver, DriverNetcdfUGRID)
        self.assertIsInstance(field.grid, GridUnstruct)
        self.assertEqual(field.grid.abstraction, GridAbstraction.POLYGON)
        self.assertIsNotNone(field.grid.archetype.cindex)

    def test_get_distributed_dimension_name(self):
        du = self.fixture()
        actual = du.get_distributed_dimension_name(du.rd.dimension_map, du.rd.metadata['dimensions'])
        self.assertEqual(actual, 'n_face')

    def test_get_grid(self):
        du = self.fixture()
        field = du.create_field()

        grid = DriverNetcdfUGRID.get_grid(field)

        actual = [g.abstraction for g in grid.geoms]
        desired = [GridAbstraction.POINT, GridAbstraction.POLYGON]
        self.assertAsSetEqual(actual, desired)

    def test_get_multi_break_value(self):
        mbv_name = OcgisConvention.Name.MULTI_BREAK_VALUE
        cindex = Variable(name='cindex', attrs={mbv_name: -899})
        actual = DriverNetcdfUGRID.get_multi_break_value(cindex)
        self.assertEqual(actual, -899)

    def test_write(self):
        du = self.fixture()
        field = du.create_field()
        path = self.get_temporary_file_path('foo.nc')
        field.write(path)

        actual = RequestDataset(path, driver=DriverNetcdfUGRID).get()
        attr_host = actual.dimension_map.get_variable(DMK.ATTRIBUTE_HOST, parent=actual)
        actual.remove_variable(attr_host)
        actual.dimension_map.set_variable(DMK.ATTRIBUTE_HOST, None)
        self.assertNotIn(attr_host.name, actual)
        res = actual.driver.create_host_attribute_variable(actual.dimension_map)
        self.assertEqual(res.attrs, attr_host.attrs)

        path2 = self.get_temporary_file_path('foo2.nc')
        actual.write(path2)
        actual2 = RequestDataset(path2, driver=DriverNetcdfUGRID).get()
        attr_host = actual2.dimension_map.get_variable(DMK.ATTRIBUTE_HOST)
        self.assertIsNotNone(attr_host)


def get_ugrid_data_structure():
    x = Variable(name='node_x', value=[10, 20, 30], dtype=float, dimensions='x')
    y = Variable(name='node_y', value=[-60, -55, -50, -45, -40], dimensions='y')
    grid = Grid(x, y)
    grid.set_extrapolated_bounds('x_bounds', 'y_bounds', 'bounds')
    grid.expand()

    cindex = np.zeros((grid.archetype.size, 4), dtype=int)
    xc = grid.x.bounds.get_value().flatten()
    yc = grid.y.bounds.get_value().flatten()

    for eidx, (ridx, cidx) in enumerate(itertools.product(*[range(ii) for ii in grid.shape])):
        curr_element = grid[ridx, cidx]
        curr_xc = curr_element.x.bounds.get_value().flatten()
        curr_yc = curr_element.y.bounds.get_value().flatten()
        for element_node_idx in range(curr_xc.shape[0]):
            found_idx = find_index([xc, yc], [curr_xc[element_node_idx], curr_yc[element_node_idx]])
            cindex[eidx, element_node_idx] = found_idx

    new_cindex, uindices = reduce_reindex_coordinate_index(cindex.flatten(), start_index=0)
    new_cindex = new_cindex.reshape(*cindex.shape)
    xc = xc[uindices]
    yc = yc[uindices]

    centers = grid.get_value_stacked()
    center_xc = centers[1].flatten()
    center_yc = centers[0].flatten()

    longitude_attrs = {'standard_name': 'longitude', 'units': 'degrees_east'}
    latitude_attrs = {'standard_name': 'latitude', 'units': 'degrees_north'}

    vc = VariableCollection(attrs={'conventions': 'CF-1.6, UGRID-1.0'})
    face_center_x = Variable(name='face_center_x', value=center_xc, dimensions='n_face', parent=vc,
                             attrs=longitude_attrs, dtype=float)
    face_center_y = Variable(name='face_center_y', value=center_yc, dimensions='n_face', parent=vc,
                             attrs=latitude_attrs, dtype=float)

    face_node_index = Variable(name='face_node_index', value=new_cindex, dimensions=['n_face', 'max_nodes'],
                               parent=vc,
                               attrs={'standard_name': 'face_node_connectivity', 'order': 'counterclockwise'})

    face_node_x = Variable(name='face_node_x', value=xc, dimensions='n_node', parent=vc, attrs=longitude_attrs,
                           dtype=float)
    face_node_y = Variable(name='face_node_y', value=yc, dimensions='n_node', parent=vc, attrs=latitude_attrs,
                           dtype=float)

    mesh = Variable(name='mesh',
                    attrs={'standard_name': 'mesh_topology', 'cf_role': 'mesh_topology', 'dimension': 2,
                           'locations': 'face node', 'node_coordinates': 'face_node_x face_node_y',
                           'face_coordinates': 'face_center_x face_center_y',
                           'face_node_connectivity': 'face_node_index'},
                    parent=vc)

    # path = self.get_temporary_file_path('foo.nc')
    # vc.write(path)
    # self.ncdump(path)
    #

    # ==============================================================================================================
    # import matplotlib.pyplot as plt
    # from descartes import PolygonPatch
    # from shapely.geometry import Polygon, MultiPolygon
    #
    # BLUE = '#6699cc'
    # GRAY = '#999999'
    #
    # fig = plt.figure(num=1)
    # ax = fig.add_subplot(111)
    #
    # polys = []
    #
    # for face_idx in range(face_node_index.shape[0]):
    #     sub = face_node_index[face_idx, :].parent
    #     curr_cindex = sub[face_node_index.name].get_value().flatten()
    #     fcx = sub[face_node_x.name].get_value()[curr_cindex]
    #     fcy = sub[face_node_y.name].get_value()[curr_cindex]
    #
    #     coords = np.zeros((4, 2))
    #     coords[:, 0] = fcx
    #     coords[:, 1] = fcy
    #
    #     poly = Polygon(coords)
    #     polys.append(poly)
    #     patch = PolygonPatch(poly, fc=BLUE, ec=GRAY, alpha=0.5, zorder=2)
    #     ax.add_patch(patch)
    #
    # minx, miny, maxx, maxy = MultiPolygon(polys).bounds
    # w, h = maxx - minx, maxy - miny
    # ax.set_xlim(minx - 0.2 * w, maxx + 0.2 * w)
    # ax.set_ylim(miny - 0.2 * h, maxy + 0.2 * h)
    # ax.set_aspect(1)
    #
    # plt.scatter(center_xc, center_yc, zorder=1)
    #
    # plt.show()
    # ===============================================================================================================

    return vc
