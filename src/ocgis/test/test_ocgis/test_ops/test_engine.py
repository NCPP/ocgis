import itertools
from copy import deepcopy

import numpy as np
import ocgis
from ocgis import SpatialCollection, Variable
from ocgis import env
from ocgis.collection.field import Field
from ocgis.constants import TagName, DimensionMapKey
from ocgis.conv.numpy_ import NumpyConverter
from ocgis.ops.core import OcgOperations
from ocgis.ops.engine import OperationsEngine
from ocgis.spatial.grid import Grid
from ocgis.test.base import attr, AbstractTestInterface, get_geometry_dictionaries
from ocgis.util.itester import itr_products_keywords
from ocgis.util.logging_ocgis import ProgressOcgOperations
from ocgis.variable.crs import Spherical, WGS84, CoordinateReferenceSystem
from shapely import wkt


class TestOperationsEngine(AbstractTestInterface):
    def get_operations(self):
        rd = self.test_data.get_rd('cancm4_tas')
        slc = [None, [0, 100], None, [0, 10], [0, 10]]
        ops = ocgis.OcgOperations(dataset=rd, slice=slc)
        return ops

    def get_subset_operation(self):
        geom = get_geometry_dictionaries()
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, geom=geom, select_nearest=True)
        subset = OperationsEngine(ops)
        return subset

    @attr('data')
    def test_init(self):
        for rb, p in itertools.product([True, False], [None, ProgressOcgOperations()]):
            sub = OperationsEngine(self.get_operations(), request_base_size_only=rb, progress=p)
            for ii, coll in enumerate(sub):
                self.assertIsInstance(coll, SpatialCollection)
        self.assertEqual(ii, 0)

    @attr('data')
    def test_abstraction_not_available(self):
        """Test appropriate exception is raised when a selected abstraction is not available."""

        dmap = {DimensionMapKey.X: {DimensionMapKey.VARIABLE: 'x'},
                DimensionMapKey.Y: {DimensionMapKey.VARIABLE: 'y'}}
        rd = self.test_data.get_rd('daymet_tmax', kwds={'dimension_map': dmap})
        ops = ocgis.OcgOperations(dataset=rd, abstraction='polygon', geom='state_boundaries', select_ugid=[25])
        with self.assertRaises(ValueError):
            ops.execute()

    @attr('data')
    def test_system_dataset_as_field_from_file(self):
        """Test with dataset argument coming in as a field as opposed to a request dataset collection."""

        rd = self.test_data.get_rd('cancm4_tas')
        geom = 'state_boundaries'
        select_ugid = [23]
        field = rd.get()
        ops = OcgOperations(dataset=field, snippet=True, geom=geom, select_ugid=select_ugid)
        ret = ops.execute()
        field_out_from_field = ret.get_element(container_ugid=23)
        self.assertEqual(field_out_from_field.data_variables[0].shape, (1, 3, 3))
        ops = OcgOperations(dataset=rd, snippet=True, geom=geom, select_ugid=select_ugid)
        ret = ops.execute()
        field_out_from_rd = ret.get_element(container_ugid=23)
        self.assertNumpyAll(field_out_from_field['tas'].get_value(), field_out_from_rd['tas'].get_value())

    @attr('data')
    def test_system_geometry_dictionary(self):
        """Test geometry dictionaries come out properly as collections."""

        subset = self.get_subset_operation()
        conv = NumpyConverter(subset)
        coll = conv.write()
        geom = get_geometry_dictionaries()
        container = coll.children[2]
        self.assertEqual(container['COUNTRY'].get_value()[0], 'Germany')
        self.assertEqual(container.geom.get_value()[0], geom[1]['geom'])
        self.assertEqual(len(coll.children), 3)

    def test_system_process_geometries(self):
        """Test multiple geometries with coordinate system update."""

        a = 'POLYGON((-105.21347987288135073 40.21514830508475313,-104.39928495762711691 40.21514830508475313,-104.3192002118643984 39.5677966101694949,-102.37047139830508513 39.61451271186440692,-102.12354343220337682 37.51896186440677639,-105.16009004237288593 37.51896186440677639,-105.21347987288135073 40.21514830508475313))'
        b = 'POLYGON((-104.15235699152542281 39.02722457627118757,-103.71189088983049942 39.44099576271186436,-102.71750529661017026 39.28082627118644155,-102.35712394067796538 37.63908898305084705,-104.13900953389830306 37.63241525423728717,-104.15235699152542281 39.02722457627118757))'
        geom = [{'geom': wkt.loads(xx), 'properties': {'UGID': ugid}} for ugid, xx in enumerate([a, b])]

        grid_value = [
            [[37.0, 37.0, 37.0, 37.0], [38.0, 38.0, 38.0, 38.0], [39.0, 39.0, 39.0, 39.0], [40.0, 40.0, 40.0, 40.0]],
            [[-105.0, -104.0, -103.0, -102.0], [-105.0, -104.0, -103.0, -102.0], [-105.0, -104.0, -103.0, -102.0],
             [-105.0, -104.0, -103.0, -102.0]]]
        output_crs = CoordinateReferenceSystem(
            value={'a': 6370997, 'lon_0': -100, 'y_0': 0, 'no_defs': True, 'proj': 'laea', 'x_0': 0, 'units': 'm',
                   'b': 6370997, 'lat_0': 45})

        x = Variable('x', grid_value[1], dimensions=['lat', 'lon'])
        y = Variable('y', grid_value[0], dimensions=['lat', 'lon'])
        grid = Grid(x, y)
        field = Field(grid=grid, crs=Spherical())

        ops = OcgOperations(dataset=field, geom=geom, output_crs=output_crs)
        ret = ops.execute()

        expected = {0: -502052.79407259845,
                    1: -510391.37909706926}
        for field, container in ret.iter_fields(yield_container=True):
            self.assertAlmostEqual(field.grid.get_value_stacked().mean(),
                                   expected[container.geom.ugid.get_value()[0]])

    @attr('data', 'esmf')
    def test_system_regridding_bounding_box_wrapped(self):
        """Test subsetting with a wrapped bounding box with the target as a 0-360 global grid."""

        bbox = [-104, 36, -95, 44]
        rd_global = self.test_data.get_rd('cancm4_tas')
        rd_downscaled = self.test_data.get_rd('maurer_bcca_1991')

        ops = ocgis.OcgOperations(dataset=rd_global, regrid_destination=rd_downscaled, geom=bbox, output_format='nc',
                                  snippet=True)
        ret = ops.execute()
        rd = ocgis.RequestDataset(ret)
        field = rd.get()
        self.assertEqual(field['tas'].shape, (1, 64, 72))
        self.assertEqual(field.grid.get_value_stacked().sum(), -274176.0)
        self.assertTrue(field.grid.has_bounds)
        self.assertAlmostEqual(field['tas'].get_value().sum(), 1207697.8, places=0)

    @attr('data', 'esmf')
    def test_system_regridding_same_field(self):
        """Test regridding operations with same field used to regrid the source."""

        rd_dest = self.test_data.get_rd('cancm4_tas')

        keywords = dict(regrid_destination=[rd_dest, rd_dest.get()],
                        geom=['state_boundaries'])

        select_ugid = [25, 41]

        for ctr, k in enumerate(itr_products_keywords(keywords, as_namedtuple=True)):

            rd1 = self.test_data.get_rd('cancm4_tas')
            rd2 = self.test_data.get_rd('cancm4_tas', kwds={'field_name': 'tas2'})

            ops = ocgis.OcgOperations(dataset=[rd1, rd2], geom=k.geom, regrid_destination=k.regrid_destination,
                                      time_region={'month': [1], 'year': [2002]}, select_ugid=select_ugid)
            subset = OperationsEngine(ops)
            colls = list(subset)
            self.assertEqual(len(colls), 4)
            for coll in colls:
                for d in coll.iter_melted(tag=TagName.DATA_VARIABLES):
                    field = d['field']
                    self.assertEqual(field.crs, env.DEFAULT_COORDSYS)
                    self.assertTrue(d['variable'].get_value().mean() > 100)
                    self.assertTrue(np.any(field.grid.get_mask()))
                    self.assertTrue(np.any(d['variable'].get_mask()))

    @attr('data', 'esmf')
    def test_system_regridding_same_field_bad_bounds_without_corners(self):
        """Test bad bounds may be regridded with_corners as False."""

        from ESMF.api.constants import RegridMethod
        rd1 = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd1, regrid_destination=rd1, snippet=True,
                                  regrid_options={'regrid_method': RegridMethod.BILINEAR})
        subset = OperationsEngine(ops)
        ret = list(subset)
        for coll in ret:
            for dd in coll.iter_melted():
                field = dd['field']
                for dv in field.data_variables:
                    self.assertGreater(dv.get_value().sum(), 100)

    @attr('data', 'esmf')
    def test_system_regridding_same_field_value_mask(self):
        """Test with a value mask."""

        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tas', kwds={'field_name': 'tas2'})
        value_mask = np.zeros(rd2.get().grid.shape, dtype=bool)
        value_mask[30, 45] = True
        regrid_options = {'value_mask': value_mask}
        ops = ocgis.OcgOperations(dataset=rd1, regrid_destination=rd2, snippet=True, regrid_options=regrid_options)
        ret = list(OperationsEngine(ops))
        actual = ret[0].get_element(variable_name='tas').get_mask().sum()
        self.assertEqual(1, actual)

    @attr('data', 'esmf')
    def test_system_regridding_different_fields_requiring_wrapping(self):
        """Test with fields requiring wrapping."""

        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('maurer_2010_tas')

        select_ugid = [25]

        ops = ocgis.OcgOperations(dataset=rd2, regrid_destination=rd1, geom=self.path_state_boundaries,
                                  select_ugid=select_ugid, time_region={'month': [2], 'year': [1990]})
        subset = OperationsEngine(ops)
        colls = list(subset)
        self.assertEqual(len(colls), 1)
        for coll in colls:
            for dd in coll.iter_melted(tag=TagName.DATA_VARIABLES):
                self.assertEqual(dd['variable'].shape, (28, 4, 4))

    @attr('data', 'esmf')
    def test_system_regridding_different_fields_variable_regrid_targets(self):
        """Test with a request dataset having regrid_source as False."""

        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('maurer_2010_tas', kwds={'time_region': {'year': [1990], 'month': [2]}})
        rd2.regrid_source = False
        rd3 = deepcopy(rd2)
        rd3.regrid_source = True
        rd3._field_name = 'maurer2'

        geom = 'state_boundaries'
        select_ugid = [25]

        ops = ocgis.OcgOperations(dataset=[rd2, rd3], regrid_destination=rd1, geom=geom, select_ugid=select_ugid)
        subset = OperationsEngine(ops)
        colls = list(subset)
        self.assertEqual(len(colls), 2)
        for coll in colls:
            for dd in coll.iter_melted(tag=TagName.DATA_VARIABLES):
                field = dd['field']
                variable = dd['variable']
                if field.name == 'tas':
                    self.assertEqual(variable.shape, (28, 77, 83))
                elif field.name == 'maurer2':
                    self.assertEqual(variable.shape, (28, 4, 4))
                else:
                    raise NotImplementedError

    @attr('esmf')
    def test_system_regridding_crs(self):
        """Test with coordinate systems."""

        dest_crs = WGS84()

        grid_spherical = self.get_gridxy_global(resolution=10.0, wrapped=False, crs=Spherical())
        self.assertEqual(grid_spherical.crs, Spherical())
        coords = grid_spherical.get_value_stacked()
        data_value = self.get_exact_field_value(coords[1], coords[0])
        desired = data_value.copy()
        data_var = Variable(name='data_src', value=data_value, dimensions=grid_spherical.dimensions)
        source = Field(grid=grid_spherical, is_data=data_var, crs=grid_spherical.crs)
        self.assertEqual(source.crs, Spherical())

        destination = deepcopy(source)
        destination.update_crs(dest_crs)

        source_expanded = deepcopy(source.grid)
        source_expanded.expand()
        diff = np.abs(destination.y.get_value() - source_expanded.y.get_value())
        self.assertAlmostEqual(diff.max(), 0.19231511439)

        for output_crs in [None, WGS84()]:
            ops = OcgOperations(dataset=source, regrid_destination=destination, output_crs=output_crs)
            ret = ops.execute()

            actual = ret.get_element(variable_name=data_var.name)
            if output_crs is None:
                self.assertEqual(actual.parent.crs, Spherical())
            else:
                self.assertEqual(actual.parent.crs, WGS84())
            actual = actual.get_value()
            diff = np.abs(actual - desired)
            self.assertTrue(diff.max() < 1e-5)
