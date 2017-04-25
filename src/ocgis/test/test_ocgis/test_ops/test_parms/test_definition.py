import tempfile
import types

from ocgis.calc.library.statistics import Mean
from ocgis.constants import TagName
from ocgis.conv.numpy_ import NumpyConverter
from ocgis.ops.parms.base import AbstractParameter
from ocgis.ops.parms.definition import *
from ocgis.ops.query import QueryInterface
from ocgis.spatial.geom_cabinet import GeomCabinet
from ocgis.test.base import TestBase, attr
from ocgis.util.helpers import make_poly
from ocgis.util.itester import itr_products_keywords
from ocgis.util.units import get_units_object, get_are_units_equal
from ocgis.variable.crs import WGS84


class Test(TestBase):
    def test_callback(self):
        c = Callback()
        self.assertEqual(c.value, None)

        with self.assertRaises(DefinitionValidationError):
            Callback('foo')

        def callback(percent, message):
            pass

        c = Callback(callback)
        self.assertEqual(callback, c.value)

    def test_optimizations(self):
        o = Optimizations()
        self.assertEqual(o.value, None)
        with self.assertRaises(DefinitionValidationError):
            Optimizations({})
        with self.assertRaises(DefinitionValidationError):
            Optimizations({'foo': 'foo'})
        o = Optimizations({'tgds': {'tas': 'TemporalGroupDimension'}})
        self.assertEqual(o.value, {'tgds': {'tas': 'TemporalGroupDimension'}})

    def test_optimizations_deepcopy(self):
        # # we should not deepcopy optimizations
        arr = np.array([1, 2, 3, 4])
        value = {'tgds': {'tas': arr}}
        o = Optimizations(value)
        self.assertTrue(np.may_share_memory(o.value['tgds']['tas'], arr))

    def test_add_auxiliary_files(self):
        for val in [True, False]:
            p = AddAuxiliaryFiles(val)
            self.assertEqual(p.value, val)
        p = AddAuxiliaryFiles()
        self.assertEqual(p.value, True)

    def test_dir_output(self):
        # # raise an exception if the directory does not exist
        do = '/does/not/exist'
        with self.assertRaises(DefinitionValidationError):
            DirOutput(do)

        ## make sure directory name does not change case
        do = 'Some'
        new_dir = os.path.join(tempfile.gettempdir(), do)
        os.mkdir(new_dir)
        try:
            dd = DirOutput(new_dir)
            self.assertEqual(new_dir, dd.value)
        finally:
            os.rmdir(new_dir)

    def test_slice(self):
        slc = Slice(None)
        self.assertEqual(slc.value, None)

        slc = Slice([None, 0, 0, 0, 0])
        self.assertEqual(slc.value, {'y': slice(0, 1, None), 'x': slice(0, 1, None), 'level': slice(0, 1, None),
                                     'time': slice(0, 1, None), 'realization': slice(None, None, None)})

        slc = Slice([None, 0, None, [0, 1], [0, 100]])
        self.assertEqual(slc.value, {'y': slice(0, 1, None), 'x': slice(0, 100, None), 'level': slice(None, None, None),
                                     'time': slice(0, 1, None), 'realization': slice(None, None, None)})

        with self.assertRaises(DefinitionValidationError):
            slc.value = 4
        with self.assertRaises(DefinitionValidationError):
            slc.value = [None, None]

    def test_snippet(self):
        self.assertFalse(Snippet().value)
        for ii in ['t', 'TRUE', 'tRue', 1, '1', ' 1 ']:
            self.assertTrue(Snippet(ii).value)
        s = Snippet()
        s.value = False
        self.assertFalse(s.value)
        s.value = '0'
        self.assertFalse(s.value)
        with self.assertRaises(DefinitionValidationError):
            s.value = 'none'

        s.get_meta()

    def test_spatial_operation(self):
        so = SpatialOperation()
        self.assertEqual(so.value, 'intersects')
        with self.assertRaises(DefinitionValidationError):
            so.value = 'clips'
        so.value = 'clip'

    def test_output_format(self):
        so = OutputFormat('csv')
        self.assertEqual(so.value, 'csv')
        so.value = 'OCGIS'
        self.assertEqual(so.value, 'ocgis')

    def test_select_ugid(self):
        so = GeomSelectUid()
        self.assertEqual(so.value, None)
        with self.assertRaises(DefinitionValidationError):
            so.value = 98.5
        so.value = 'none'
        self.assertEqual(so.value, None)
        with self.assertRaises(DefinitionValidationError):
            so.value = 1
        so = GeomSelectUid('10')
        self.assertEqual(so.value, (10,))
        with self.assertRaises(DefinitionValidationError):
            so.value = ('1|1|2')
        with self.assertRaises(DefinitionValidationError):
            so.value = '22.5'
        so = GeomSelectUid('22|23|24')
        self.assertEqual(so.value, (22, 23, 24))
        with self.assertRaises(DefinitionValidationError):
            so.value = '22|23.5|24'

    def test_prefix(self):
        pp = Prefix()
        self.assertEqual(pp.value, 'ocgis_output')
        pp.value = ' Old__man '
        self.assertEqual(pp.value, 'Old__man')


class TestAbstraction(TestBase):
    create_dir = False

    def test_init(self):
        K = Abstraction

        k = K()
        self.assertEqual(k.value, 'auto')
        self.assertEqual(str(k), 'abstraction="auto"')

        k = K('point')
        self.assertEqual(k.value, 'point')

        with self.assertRaises(DefinitionValidationError):
            K('pt')


class TestAggregate(TestBase):
    create_dir = False

    def test_init(self):
        A = Aggregate

        a = A(True)
        self.assertEqual(a.value, True)

        a = A(False)
        self.assertEqual(a.value, False)

        a = A('True')
        self.assertEqual(a.value, True)


class TestCalc(TestBase):
    create_dir = False

    def test_init(self):
        calc = [{'func': 'mean', 'name': 'my_mean'}]
        cc = Calc(calc)
        eq = [{'ref': Mean, 'name': 'my_mean', 'func': 'mean', 'kwds': {}, 'meta_attrs': None}]

        self.assertEqual(cc.value, eq)
        cc.value = 'mean~my_mean'
        self.assertEqual(cc.value, eq)

        calc = [{'func': 'mean', 'name': 'my_mean'}]
        cc = Calc(calc)
        eq = [{'ref': Mean, 'name': 'my_mean', 'func': 'mean', 'kwds': {}, 'meta_attrs': None}]

        self.assertEqual(cc.value, eq)
        cc.value = 'mean~my_mean'
        self.assertEqual(cc.value, eq)

    def test_bad_key(self):
        calc = [{'func': 'bad_mean', 'name': 'my_mean'}]
        with self.assertRaises(DefinitionValidationError):
            Calc(calc)

    def test_get_meta(self):
        for poss in Calc._possible:
            cc = Calc(poss)
            cc.get_meta()

    def test_eval_string(self):
        value = [
            'es=tas+4',
            ['es=tas+4']
        ]
        actual = [{'func': 'es=tas+4', 'ref': EvalFunction, 'meta_attrs': None, 'name': None, 'kwds': OrderedDict()}]
        for v in value:
            cc = Calc(v)
            self.assertEqual(cc.value, actual)

    def test_eval_string_malformed(self):
        with self.assertRaises(DefinitionValidationError):
            Calc('estas+4')

    def test_eval_string_multivariate(self):
        value = [
            'es=exp(tas)+tasmax+log(4)',
            ['es=exp(tas)+tasmax+log(4)']
        ]
        actual = [
            {'func': 'es=exp(tas)+tasmax+log(4)', 'ref': MultivariateEvalFunction, 'meta_attrs': None, 'name': None,
             'kwds': OrderedDict()}]
        for v in value:
            cc = Calc(v)
            self.assertEqual(cc.value, actual)

    def test_eval_string_number_after_variable_alias(self):
        value = 'tas2=tas1+tas2'
        self.assertTrue(EvalFunction.is_multivariate(value))

        value = 'tas2=tas1+tas1'
        self.assertFalse(EvalFunction.is_multivariate(value))

    def test_eval_underscores_in_variable_names(self):
        value = 'tas_4=tasmin_2+tasmin'
        self.assertEqual(Calc(value).value, [
            {'func': value, 'ref': MultivariateEvalFunction, 'meta_attrs': None, 'name': None, 'kwds': OrderedDict()}])

    def test_meta_attrs(self):
        """Test various forms for meta_attrs in the calculation definition dictionary."""

        kwds = dict(
            meta_attr=[None, {}, {'something_else': 'is_here with us'}],
            calc=[{'func': 'mean', 'name': 'my_mean'}, 'foo=tas+4',
                  {'func': 'foo=tas+4', 'meta_attrs': {'something': 'special'}}],
            add_meta_attrs_if_none=[True, False]
        )
        for k in itr_products_keywords(kwds, as_namedtuple=True):
            k = deepcopy(k)
            if not k.add_meta_attrs_if_none and k.meta_attr is None:
                pass
            else:
                try:
                    k.calc.update({'meta_attrs': k.meta_attr})
                # likely the string representation
                except AttributeError:
                    if k.calc == 'foo=tas+4':
                        pass
                    else:
                        raise
            calc = Calc([k.calc])
            self.assertEqual(set(calc.value[0].keys()), set(['ref', 'meta_attrs', 'name', 'func', 'kwds']))

    def test_parse(self):
        calc = [{'name': 'mean'}]
        with self.assertRaises(DefinitionValidationError):
            Calc(calc)

        calc = [{'func': 'mean'}]
        with self.assertRaises(DefinitionValidationError):
            Calc(calc)

        calc = [{}]
        with self.assertRaises(DefinitionValidationError):
            Calc(calc)

    def test_str(self):
        calc = [{'func': 'mean', 'name': 'my_mean'}]
        cc = Calc(calc)
        self.assertTrue(len(str(cc)) > 10)

        cc = Calc(None)
        self.assertEqual(str(cc), 'calc=None')

        calc = [{'func': 'mean', 'name': 'my_mean', 'kwds': {'a': np.zeros(1000)}}]
        cc = Calc(calc)
        self.assertIn('ndarray', str(cc))

    def test_validate(self):
        calc = [{'func': 'threshold', 'name': 'threshold'}]
        # this function definition is missing some keyword parameters
        with self.assertRaises(DefinitionValidationError):
            Calc(calc)


class TestCalcGrouping(TestBase):
    create_dir = False

    def init(self):
        cg = CalcGrouping(['day', 'month'])
        self.assertEqual(cg.value, ('day', 'month'))
        with self.assertRaises(DefinitionValidationError):
            cg.value = ['d', 'foo']

    def test_all(self):
        cg = CalcGrouping('all')
        self.assertEqual(cg.value, 'all')

    def test_seasonal_aggregation(self):
        cg = CalcGrouping([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(cg.value, ([1, 2, 3], [4, 5, 6]))

        # # element groups must be composed of unique elements
        with self.assertRaises(DefinitionValidationError):
            CalcGrouping([[1, 2, 3], [4, 4, 6]])

        # # element groups must have an empty intersection
        with self.assertRaises(DefinitionValidationError):
            CalcGrouping([[1, 2, 3], [1, 4, 6]])

        ## months must be between 1 and 12
        with self.assertRaises(DefinitionValidationError):
            CalcGrouping([[1, 2, 3], [4, 5, 66]])

    def test_seasonal_aggregation_with_year(self):
        cg = CalcGrouping([[1, 2, 3], [4, 5, 6], 'year'])
        self.assertEqual(cg.value, ([1, 2, 3], [4, 5, 6], 'year'))

    def test_seasonal_aggregation_with_unique(self):
        cg = CalcGrouping([[1, 2, 3], [4, 5, 6], 'unique'])
        self.assertEqual(cg.value, ([1, 2, 3], [4, 5, 6], 'unique'))

    def test_seasonal_aggregation_with_bad_flag(self):
        with self.assertRaises(DefinitionValidationError):
            CalcGrouping([[1, 2, 3], [4, 5, 6], 'foo'])
        with self.assertRaises(DefinitionValidationError):
            CalcGrouping([[1, 2, 3], [4, 5, 6], 'fod'])


class TestConformUnitsTo(TestBase):
    create_dir = False

    def test_init(self):
        cc = ConformUnitsTo()
        self.assertEqual(cc.value, None)

        if env.USE_CFUNITS:
            cc = ConformUnitsTo('kelvin')
            self.assertEqual(cc.value, 'kelvin')

            with self.assertRaises(DefinitionValidationError):
                ConformUnitsTo('not_a_unit')

            cc = ConformUnitsTo(get_units_object('celsius'))
            target = get_are_units_equal((cc.value, get_units_object('celsius')))
            self.assertTrue(target)

            cc = ConformUnitsTo('hPa')
            target = get_are_units_equal((cc.value, get_units_object('hPa')))
            self.assertTrue(target)
        else:
            with self.assertRaises(ImportError):
                ConformUnitsTo('celsius')


class TestMelted(TestBase):
    create_dir = False

    def test_init(self):
        mm = Melted()
        self.assertEqual(mm.value, Melted.default)


class TestDataset(TestBase):
    @attr('data')
    def test_init(self):
        rd = self.test_data.get_rd('cancm4_tas')
        dd = Dataset(rd)
        # Test the generator may be reused.
        for _ in range(5):
            self.assertIsInstance(list(dd.value)[0], RequestDataset)

        uri = '/a/bad/path'
        with self.assertRaises(ValueError):
            RequestDataset(uri, 'foo')

        # Test it can be initialized from itself.
        rd = self.test_data.get_rd('cancm4_tas')
        dd = Dataset(rd)
        dd2 = Dataset(dd)
        self.assertEqual(list(dd)[0].uri, list(dd2)[0].uri)

        # Test with a dictionary.
        v = {'uri': rd.uri, 'variable': rd.variable}
        dd = Dataset(v)
        self.assertEqual(list(dd)[0].variable, rd.variable)

        # Test with a list/tuple.
        v2 = v.copy()
        dd = Dataset([v, v2])
        for rd in dd:
            self.assertIsInstance(rd, RequestDataset)

        # Test with a request dataset.
        dd = Dataset(rd)
        self.assertIsInstance(dd._value, RequestDataset)

        # Test with an iterator.
        itr = (rd for rd in [self.test_data.get_rd('cancm4_tas')])
        dd = Dataset(itr)
        self.assertIsInstance(dd.value, types.GeneratorType)

        # Test with field does not load anything.
        field = self.test_data.get_rd('cancm4_tas').get()
        for var in list(field.values()):
            var.protected = True
        dd = Dataset(field)
        rfield = list(dd)[0]
        for var in list(rfield.values()):
            self.assertFalse(var.has_allocated_value)

        # Test with a Field object. Field objects should not be deepcopied by the parameter.
        field = self.get_field()
        d = Dataset(field)
        rfield = list(d)[0]
        self.assertNumpyMayShareMemory(list(field.values())[0].get_value(), list(rfield.values())[0].get_value())

        # We do want a deepcopy on non-field objects.
        rd = self.test_data.get_rd('cancm4_tas')
        dd = Dataset(rd)
        self.assertNotEqual(id(rd), id(list(dd)[0]))

    @attr('esmf')
    def test_init_esmf(self):
        from ocgis.regrid.base import get_esmf_field_from_ocgis_field

        original_field = self.get_field()
        dimensions = original_field.data_variables[0].dimensions
        efield = get_esmf_field_from_ocgis_field(original_field)
        dd = Dataset(efield, esmf_field_dimensions=dimensions)
        ofield = list(dd)[0]
        self.assertIsInstance(ofield, Field)
        dimensioned = ofield.get_by_tag(TagName.DATA_VARIABLES)[0]
        self.assertTrue(np.may_share_memory(dimensioned.get_value(), efield.data))
        self.assertNumpyAll(dimensioned.get_value(), efield.data)

    @attr('data')
    def test_from_query(self):
        rd = self.test_data.get_rd('cancm4_tas')
        qs = 'uri={0}'.format(rd.uri)
        qi = QueryInterface(qs)
        d = Dataset.from_query(qi)
        self.assertEqual(list(d)[0].uri, rd.uri)
        qs += '&variable=tas&field_name=foobar'
        qi = QueryInterface(qs)
        d = Dataset.from_query(qi)
        self.assertEqual(list(d)[0].field_name, 'foobar')

        # Test with multiple URIs.
        uri1 = self.test_data.get_uri('cancm4_tas')
        uri2 = self.test_data.get_uri('cancm4_rhsmax')
        qs = 'uri={0}|{1}'.format(uri1, uri2)
        qi = QueryInterface(qs)
        d = Dataset.from_query(qi)
        self.assertEqual([r.variable for r in d], ['tas', 'rhsmax'])

    @attr('data')
    def test_get_meta(self):
        rd = self.test_data.get_rd('cancm4_tas')
        dd = Dataset(rd)
        self.assertIsInstance(dd.get_meta(), list)

        # Test passing a field object.
        rd = self.test_data.get_rd('cancm4_tas', kwds={'field_name': 'wow'})
        dd = Dataset(rd.get())
        ret = dd.get_meta()
        self.assertEqual(ret, ['* dataset=', 'Field object with name: "wow"', ''])

    def test_iter(self):
        f1 = Field(name='field_one', uid=4)
        f2 = Field(name='field_two')

        init_value = [f1, f2]
        d = Dataset(init_value)

        for ctr, field in enumerate(d):
            self.assertNotEqual(id(init_value[ctr]), id(field))
            self.assertIsNotNone(field.uid)

        self.assertIsNone(f2.uid)


class TestGeom(TestBase):
    create_dir = False

    @staticmethod
    def get_geometry_dictionaries(uid='UGID'):
        coordinates = [('France', [2.8, 47.16]),
                       ('Germany', [10.5, 51.29]),
                       ('Italy', [12.2, 43.4])]
        geom = []
        for ugid, coordinate in enumerate(coordinates, start=1):
            point = Point(coordinate[1][0], coordinate[1][1])
            geom.append({'geom': point,
                         'properties': {uid: ugid, 'COUNTRY': coordinate[0]}})
        return geom

    def test_init(self):
        geom = make_poly((37.762, 38.222), (-102.281, -101.754))

        g = Geom(geom)
        self.assertEqual(type(g.value), tuple)
        self.assertIsInstance(g.value[0], Field)
        g.value = None
        self.assertEqual(None, g.value)

        g = Geom(None)
        self.assertEqual(g.value, None)
        self.assertEqual(str(g), 'geom=None')

        g = Geom('-120|40|-110|50')
        self.assertEqual(g.value[0].geom.get_value()[0].bounds, (-120.0, 40.0, -110.0, 50.0))
        self.assertEqual(str(g), 'geom=-120.0|40.0|-110.0|50.0')

        ocgis.env.DIR_GEOMCABINET = self.path_bin
        g = Geom('state_boundaries')
        self.assertEqual(str(g), 'geom="state_boundaries"')

        geoms = list(GeomCabinetIterator('state_boundaries'))
        g = Geom('state_boundaries')
        self.assertEqual(len(list(g.value)), len(geoms))

        sci = GeomCabinetIterator(key='state_boundaries')
        self.assertFalse(sci.as_field)
        g = Geom(sci)
        for _ in range(2):
            for ii, element in enumerate(g.value):
                self.assertIsInstance(element, Field)
            self.assertGreater(ii, 10)

        su = GeomSelectUid([1, 2, 3])
        g = Geom('state_boundaries', select_ugid=su)
        self.assertEqual(len(list(g.value)), 3)

        geoms = [{'geom': geom, 'properties': {'UGID': 1}}, {'geom': geom, 'properties': {'UGID': 2}}]
        Geom(geoms)

        bbox = [-120, 40, -110, 50]
        g = Geom(bbox)
        self.assertEqual(g.value[0].geom.get_value()[0].bounds, tuple(map(float, bbox)))

        sui = GeomUid('ID')
        g = Geom(bbox, geom_uid=sui)
        self.assertEqual(g.geom_uid, 'ID')
        g = Geom(bbox, geom_uid='ID')
        self.assertEqual(g.geom_uid, 'ID')

        # Tests for geom_select_sql_where ##############################################################################

        g = Geom('state_boundaries')
        self.assertIsNone(g.geom_select_sql_where)

        s = 'STATE_NAME in ("Wisconsin", "Vermont")'
        ws = [GeomSelectSqlWhere(s), s]
        for w in ws:
            g = Geom('state_boundaries', geom_select_sql_where=w)
            self.assertEqual(g.geom_select_sql_where, s)

        # Test passing a folder which is not allowed ###################################################################

        with self.assertRaises(DefinitionValidationError):
            Geom(tempfile.gettempdir())

    def test_geometry_dictionaries(self):
        """Test geometry dictionaries as input."""

        for crs in [None, WGS84(), CoordinateReferenceSystem(epsg=2136)]:
            geom = self.get_geometry_dictionaries()
            if crs is not None:
                for g in geom:
                    g['crs'] = crs
            g = Geom(geom)
            self.assertEqual(len(g.value), 3)
            for gdict, field in zip(geom, g.value):
                self.assertEqual(field.geom.geom_type, 'Point')
                if crs is None:
                    self.assertIsInstance(field.crs, env.DEFAULT_COORDSYS.__class__)
                else:
                    self.assertIsInstance(field.crs, crs.__class__)
                self.assertEqual(set([v.name for v in field.data_variables]), set(['UGID', 'COUNTRY']))
                self.assertEqual(field.geom.ugid.shape[0], 1)
                self.assertEqual(field['UGID'].get_value()[0], gdict['properties']['UGID'])
                self.assertEqual(field['COUNTRY'].get_value()[0], gdict['properties']['COUNTRY'])

    def test_init_field(self):
        """Test using a field as initial value."""

        field = Field.from_records(GeomCabinetIterator(key='state_boundaries'))
        self.assertIsInstance(field.crs, env.DEFAULT_COORDSYS.__class__)
        g = Geom(field)
        self.assertEqual(len(g.value), 51)
        for field in g.value:
            self.assertIsInstance(field, Field)
            self.assertEqual(field.geom.shape, (1,))

    def test_parse(self):
        keywords = dict(geom_uid=[None, 'ID'],
                        geom=[None, self.get_geometry_dictionaries(), self.get_geometry_dictionaries(uid='ID')])
        for k in self.iter_product_keywords(keywords):
            g = Geom(k.geom, geom_uid=k.geom_uid)
            ret = g.parse(k.geom)
            if k.geom is None:
                self.assertIsNone(ret)
            else:
                if k.geom_uid is None:
                    actual = constants.OCGIS_UNIQUE_GEOMETRY_IDENTIFIER
                else:
                    actual = k.geom_uid
                for element in ret:
                    self.assertEqual(element.geom.ugid.name, actual)

    @attr('data')
    def test_parse_string(self):
        keywords = dict(geom_uid=[None, 'ID'])
        for k in self.iter_product_keywords(keywords):
            g = Geom(geom_uid=k.geom_uid)
            ret = g.parse_string('state_boundaries')
            self.assertIsInstance(ret, GeomCabinetIterator)
            if k.geom_uid is None:
                actual = None
            else:
                actual = k.geom_uid
            self.assertEqual(ret.uid, actual)

        ################################################################################################################
        # tests for geom_select_sql_where

        s = "STATE_NAME in ('Wisconsin', 'Vermont')"
        g = Geom('state_boundaries', geom_select_sql_where=s)
        ret = g.parse_string('state_boundaries')
        self.assertEqual(len(ret), 2)

        ################################################################################################################

    @attr('data')
    def test_using_shp_path(self):
        # pass a path to a shapefile as opposed to a key
        path = GeomCabinet().get_shp_path('state_boundaries')
        ocgis.env.set_geomcabinet_path(None)
        # make sure there is path associated with the GeomCabinet
        with self.assertRaises(ValueError):
            list(GeomCabinet().keys())
        g = Geom(path)
        self.assertEqual(g._shp_key, path)
        self.assertEqual(len(list(g.value)), 51)

    @attr('data')
    def test_with_changing_select_uid(self):
        select_ugid = [16, 17]
        g = Geom('state_boundaries', select_ugid=select_ugid)
        self.assertEqual(len(list(g.value)), 2)
        select_ugid.append(22)
        self.assertEqual(len(list(g.value)), 3)

        g = Geom('state_boundaries')
        self.assertEqual(len(list(g.value)), 51)
        g.select_ugid = [16, 17]
        self.assertEqual(len(list(g.value)), 2)


class TestGeomSelectSqlWhere(TestBase):
    def test_init(self):
        g = GeomSelectSqlWhere()
        self.assertIsNone(g.value)

        s = 'STATE_NAME = "Vermont"'
        g = GeomSelectSqlWhere(s)
        self.assertEqual(s, g.value)


class TestGeomSelectUid(TestBase):
    def test_init(self):
        g = GeomSelectUid()
        self.assertIsNone(g.value)

        g = GeomSelectUid([3, 4, 5])
        self.assertEqual(g.value, (3, 4, 5))


class TestGeomUid(TestBase):
    def test_init(self):
        g = GeomUid()
        self.assertIsNone(g.value)

        with self.assertRaises(DefinitionValidationError):
            GeomUid(5)

        g = GeomUid('ID')
        self.assertEqual(g.value, 'ID')

    def test_get_meta(self):
        g = GeomUid('ID')
        self.assertTrue(len(g._get_meta_()) > 5)


class TestLevelRange(TestBase):
    create_dir = False

    def test_constructor(self):
        LevelRange()

    def test_normal_int(self):
        lr = LevelRange([5, 10])
        self.assertEqual(lr.value, (5, 10))

    def test_normal_float(self):
        value = [4.5, 6.5]
        lr = LevelRange(value)
        self.assertEqual(tuple(value), lr.value)

    def test_bad_length(self):
        with self.assertRaises(DefinitionValidationError):
            LevelRange([5, 6, 7, 8])

    def test_bad_ordination(self):
        with self.assertRaises(DefinitionValidationError):
            LevelRange([11, 10])


class TestOutputCRS(TestBase):
    create_dir = False

    def test_init(self):
        crs = OutputCRS('4326')
        self.assertEqual(crs.value, CoordinateReferenceSystem(epsg=4326))


class TestOutputFormat(TestBase):
    create_dir = False

    def test_init(self):
        of = OutputFormat('csv+')
        self.assertEqual(of.value, constants.OutputFormatName.CSV_SHAPEFILE)

        of2 = OutputFormat(of)
        self.assertEqual(of.value, of2.value)

        # Test "numpy" is accepted and converted to the OCGIS format.
        of = OutputFormat('numpy')
        self.assertEqual(of.value, constants.OutputFormatName.OCGIS)

    @attr('esmf')
    def test_init_esmpy(self):
        oo = OutputFormat(constants.OutputFormatName.ESMPY_GRID)
        self.assertEqual(oo.value, constants.OutputFormatName.ESMPY_GRID)

    def test_get_converter_class(self):
        of = OutputFormat(constants.OutputFormatName.OCGIS)
        self.assertEqual(of.get_converter_class(), NumpyConverter)


class TestOutputFormatOptions(TestBase):
    def test_init(self):
        self.assertIsNone(OutputFormatOptions().value)
        opts = {'data_model': 'foo'}
        ofo = OutputFormatOptions(opts)
        self.assertDictEqual(opts, ofo.value)


class TestRegridDestination(TestBase):
    @property
    def possible_datasets(self):
        # One dataset to be regridded.
        dataset1 = [self.get_rd(regrid_source=True)]

        # One dataset without regrid_source set to True.
        dataset2 = [self.get_rd()]

        # Two datasets to be regridded.
        dataset3 = [self.get_rd(regrid_source=True), self.get_rd(regrid_source=True, field_name='number2')]

        # Two datasets but only regrid one.
        dataset4 = [self.get_rd(regrid_source=True), self.get_rd(field_name='number2')]

        # Three datasets, two to be regridded and one as the destination.
        dataset5 = [self.get_rd(regrid_source=True), self.get_rd(field_name='number2', regrid_destination=True),
                    self.get_rd(regrid_source=True, field_name='number3')]

        # Three datasets, but with two set as the regrid destinations.
        dataset6 = [self.get_rd(regrid_destination=True), self.get_rd(field_name='number2', regrid_destination=True),
                    self.get_rd(regrid_source=True, field_name='number3')]

        datasets = [dataset1, dataset2, dataset3, dataset4, dataset5, dataset6]
        datasets = [Dataset(d) for d in datasets]
        datasets = {ii: d for ii, d in enumerate(datasets, start=1)}

        return datasets

    @property
    def possible_values(self):
        rd = self.get_rd()
        possible = [None, 'tas', rd, rd.get(), rd.get().grid]
        return possible

    def get_rd(self, **kwargs):
        rd = self.test_data.get_rd('cancm4_tas', kwds=kwargs)
        return rd

    @attr('data')
    def test_init(self):
        """Also tests get_meta."""

        # ctr = 0
        for key, dataset in self.possible_datasets.items():
            for poss in self.possible_values:

                # if poss != 'tas':
                #     continue
                # ctr += 1
                # print ctr
                # if ctr != 2: continue

                try:
                    regrid = RegridDestination(init_value=poss, dataset=dataset)
                except DefinitionValidationError:
                    # Only one request dataset may be the destination grid if the init_value is None.
                    if key == 6 and poss is None:
                        continue
                    # No datasets are set as a source but there is a destination.
                    elif key == 2 and poss is not None:
                        continue
                    else:
                        raise
                if poss == 'tas':
                    self.assertEqual(regrid.value.name, 'tas')
                else:
                    type_check = isinstance(regrid.value, (Field, Grid))
                    try:
                        self.assertTrue(type_check)
                    except AssertionError:
                        # If the destination is None then this is okay.
                        if poss is None:
                            pass
                        else:
                            raise
                self.assertIsInstance(regrid._get_meta_(), six.string_types)


class TestRegridOptions(TestBase):
    def test_init(self):
        ro = RegridOptions()
        self.assertDictEqual(ro.value, RegridOptions.default)

        with self.assertRaises(DefinitionValidationError):
            RegridOptions({'value_mask': 'foo'})

        with self.assertRaises(DefinitionValidationError):
            RegridOptions({'value_mask': np.array([5, 6, 7])})

        ro = RegridOptions({'regrid_method': True})
        self.assertDictEqual(ro.value, {'regrid_method': True, 'value_mask': None})

        ro = RegridOptions({'value_mask': np.array([True, False])})
        self.assertEqual(ro.value['regrid_method'], 'auto')
        self.assertNumpyAll(ro.value['value_mask'], np.array([True, False]))

        with self.assertRaises(DefinitionValidationError):
            RegridOptions({'foo': 5})

    def test_get_meta(self):
        ro = RegridOptions()
        ro._get_meta_()

        ro = RegridOptions({'value_mask': np.array([True])})
        self.assertTrue('numpy.ndarray' in ro._get_meta_())


class TestSlice(TestBase):
    create_dir = False

    def test(self):
        v = {'realization': 1, 'level': slice(0, 1), 'x': [4, 5]}
        s = Slice(v)
        self.assertEqual(s.value, v)
        # Test setting dictionary slice results in a deep copy.
        s.value['realization'] = 2
        self.assertNotEqual(v['realization'], 2)

        v = [None, -1, None, None, None]
        s = Slice(v)
        desired = {'y': slice(None, None, None), 'x': slice(None, None, None), 'level': slice(None, None, None),
                   'time': slice(-1, None, None), 'realization': slice(None, None, None)}
        self.assertEqual(s.value, desired)

        v = [None, [-2, -1], None, None, None]
        s = Slice(v)
        desired = {'y': slice(None, None, None), 'x': slice(None, None, None), 'level': slice(None, None, None),
                   'time': slice(-2, -1, None), 'realization': slice(None, None, None)}
        self.assertEqual(s.value, desired)


class TestSnippet(TestBase):
    create_dir = False

    def test_init(self):
        """Test exception when snippet used in parallel."""

        s = Snippet()
        self.assertFalse(s.value)

        s = Snippet(True)
        self.assertTrue(s.value)


class TestSpatialOperation(TestBase):
    def test_init(self):
        values = (None, 'clip', 'intersects')
        ast = ('intersects', 'clip', 'intersects')

        klass = SpatialOperation
        for v, a in zip(values, ast):
            obj = klass(v)
            self.assertEqual(obj.value, a)


class TestSpatialWrapping(TestBase):
    def test_init(self):
        sw = SpatialWrapping()
        self.assertIsNone(sw.value)
        self.assertIsNone(sw.as_enum)

        sw = SpatialWrapping('wrap')
        self.assertEqual(sw.as_enum, WrapAction.WRAP)


class TestTimeRange(TestBase):
    create_dir = False

    def test_init(self):
        TimeRange()

        value = '20000101|20000107'
        tr = TimeRange(value)
        self.assertEqual(tr.value, (datetime.datetime(2000, 1, 1, 0, 0), datetime.datetime(2000, 1, 7, 0, 0)))

        value = '20000101-233000|20000107-234530'
        tr = TimeRange(value)
        self.assertEqual(tr.value, (datetime.datetime(2000, 1, 1, 23, 30), datetime.datetime(2000, 1, 7, 23, 45, 30)))

    def test_range(self):
        dt = [datetime.datetime(2000, 1, 1), datetime.datetime(2001, 1, 1)]
        tr = TimeRange(dt)
        self.assertEqual(tr.value, tuple(dt))

    def test_bad_ordination(self):
        dt = [datetime.datetime(2000, 1, 1), datetime.datetime(1999, 1, 1)]
        with self.assertRaises(DefinitionValidationError):
            TimeRange(dt)

    def test_incorrect_number_of_values(self):
        dt = [datetime.datetime(2000, 1, 1), datetime.datetime(1999, 1, 1), datetime.datetime(1999, 1, 1)]
        with self.assertRaises(DefinitionValidationError):
            TimeRange(dt)


class TestTimeRegion(TestBase):
    create_dir = False

    def test_init(self):
        TimeRegion()

        tr = TimeRegion('month~2|6,year~2000|2005')
        self.assertEqual(tr.value, {'year': [2000, 2005], 'month': [2, 6]})

    def test_normal(self):
        value = {'month': [6, 7, 8], 'year': [4, 5, 6]}
        tr = TimeRegion(value)
        self.assertEqual(value, tr.value)

    def test_month_only(self):
        value = {'month': [6]}
        tr = TimeRegion(value)
        self.assertEqual(tr.value, {'month': [6], 'year': None})

    def test_parse_string(self):
        values = ['month~2|6,year~2000|2005', 'year~2000']
        actual = [{'year': [2000, 2005], 'month': [2, 6]}, {'year': [2000]}]
        for v, a in zip(values, actual):
            res = TimeRegion()._parse_string_(v)
            self.assertEqual(res, a)

    def test_year_only(self):
        value = {'year': [6]}
        tr = TimeRegion(value)
        self.assertEqual(tr.value, {'month': None, 'year': [6]})

    def test_both_none(self):
        value = {'year': None, 'month': None}
        tr = TimeRegion(value)
        self.assertEqual(tr.value, None)

    def test_bad_keys(self):
        value = {'mnth': [4]}
        with self.assertRaises(DefinitionValidationError):
            TimeRegion(value)


class TestTimeSubsetFunction(TestBase):
    def test_init(self):
        tsf = TimeSubsetFunction()
        self.assertEqual(tsf.__class__.__bases__, (AbstractParameter,))

        def _func_(value, bounds=None):
            return [1, 2, 3]

        tsf = TimeSubsetFunction(_func_)
        self.assertEqual(tsf.value, _func_)
