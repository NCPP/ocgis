from shapely.geometry import Point

from ocgis.collection.field import Field
from ocgis.collection.spatial import SpatialCollection
from ocgis.constants import TagName, DimensionMapKey
from ocgis.driver.vector import DriverVector
from ocgis.test.base import AbstractTestInterface
from ocgis.variable.base import Variable
from ocgis.variable.crs import CoordinateReferenceSystem
from ocgis.variable.geom import GeometryVariable


class TestSpatialCollection(AbstractTestInterface):
    @property
    def crs(self):
        return CoordinateReferenceSystem(epsg=2136)

    @property
    def field1(self):
        return self.get_field()

    @property
    def field2(self):
        field = self.get_field()
        field['exact1'].set_name('exact2')
        field['exact2'].get_value()[:] += 2
        field.set_name('exact2')
        field.append_to_tags(TagName.DATA_VARIABLES, 'exact2')
        return field

    @property
    def geoms(self):
        geoms = GeometryVariable(name='geoms', value=[Point(100.972, 41.941), Point(102.898, 40.978)],
                                 dimensions='ngeom', crs=self.crs)
        return geoms

    @property
    def spatial_collection(self):
        poi = self.get_subset_field()
        field1, field2 = self.field1, self.field2

        sc = SpatialCollection()
        for ii in range(poi.geom.shape[0]):
            subset = poi.geom[ii]
            subset_geom = subset.get_value()[0]
            container = poi.get_field_slice({'geom': ii})
            for field in [field1, field2]:
                field.set_abstraction_geom()
                subset_field = field.geom.get_intersects(subset_geom).parent
                sc.add_field(subset_field, container)

        return sc

    def get_field(self):
        gridxy = self.get_gridxy(with_xy_bounds=True)
        coords_stacked = gridxy.get_value_stacked()
        exact = self.get_exact_field_value(coords_stacked[1], coords_stacked[0])
        exact = Variable(name='exact1', value=exact, dimensions=gridxy.dimensions)
        dimension_map = {'x': {'variable': 'x', 'bounds': 'xbounds'}, 'y': {'variable': 'y', 'bounds': 'ybounds'}}
        field1 = Field.from_variable_collection(gridxy.parent, dimension_map=dimension_map)
        field1.add_variable(exact)
        field1.set_name('exact1')
        field1.append_to_tags(TagName.DATA_VARIABLES, 'exact1')
        return field1

    def get_subset_field(self):
        crs = self.crs
        geoms = self.geoms

        gridcode = Variable('gridcode', [110101, 12103], dimensions='ngeom')
        description = Variable('description', ['high point', 'low point'], dimensions='ngeom')
        dimension_map = {'geom': {'variable': 'geoms', DimensionMapKey.DIMENSION: ['ngeom']},
                         'crs': {'variable': crs.name}}
        poi = Field(variables=[geoms, gridcode, description], dimension_map=dimension_map,
                    is_data=[gridcode, description])
        geoms.set_ugid(gridcode)
        return poi

    def test_system_spatial_collection_creation(self):
        """Test creating a spatial collection using a subset of two fields."""

        crs = self.crs
        field1 = self.field1
        field2 = self.field2
        poi = self.get_subset_field()
        geoms = self.geoms
        sc = self.spatial_collection

        self.assertEqual(geoms.crs, crs)
        self.assertEqual(poi.crs, crs)

        self.assertTrue(
            sc.children[110101].children['exact1'].geom.get_value()[0, 0].intersects(Point(100.972, 41.941)))
        self.assertTrue(
            sc.children[110101].children['exact2'].geom.get_value()[0, 0].intersects(Point(100.972, 41.941)))

        self.assertTrue(sc.children[12103].children['exact1'].geom.get_value()[0, 0].intersects(Point(102.898, 40.978)))
        self.assertTrue(sc.children[12103].children['exact2'].geom.get_value()[0, 0].intersects(Point(102.898, 40.978)))

        self.assertEqual(len(sc.properties), 2)
        self.assertEqual(sc.crs, crs)

        path = self.get_temporary_file_path('grid.shp')
        path2 = self.get_temporary_file_path('poi.shp')
        path3 = self.get_temporary_file_path('grid2.shp')

        field2.set_abstraction_geom()
        field2.write(path, driver=DriverVector)

        field1.set_abstraction_geom()
        field1.write(path3, driver=DriverVector)
        poi.write(path2, driver=DriverVector)

    def test_properties(self):
        # Test an empty properties dictionary is created if there are no data variables on the container/parent field.
        gvar = GeometryVariable(name='empty', value=[Point(1, 2)], ugid=20, dimensions='ngeom')
        info = Variable(name='info')
        gvar.parent.add_variable(info)
        sc = SpatialCollection()
        sc.add_field(Field(name='what'), gvar.parent)
        self.assertEqual(len(sc.properties), 1)
