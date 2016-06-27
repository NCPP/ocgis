import pickle
import tempfile

from ocgis import env
from ocgis.api.parms.base import BooleanParameter, AbstractParameter
from ocgis.api.parms.definition import *
from ocgis.api.query import QueryInterface
from ocgis.calc.eval_function import MultivariateEvalFunction
from ocgis.calc.library.statistics import Mean
from ocgis.conv.numpy_ import NumpyConverter
from ocgis.exc import OcgWarning
from ocgis.interface.base.dimension.spatial import SpatialDimension, SpatialGeometryPointDimension
from ocgis.test.base import TestBase, attr
from ocgis.util.geom_cabinet import GeomCabinet
from ocgis.util.helpers import make_poly
from ocgis.util.itester import itr_products_keywords
from ocgis.util.units import get_units_object, get_are_units_equal


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
        self.assertEqual(slc.value, (slice(None), slice(0, 1), slice(0, 1), slice(0, 1), slice(0, 1)))

        slc = Slice([None, 0, None, [0, 1], [0, 100]])
        self.assertEqual(slc.value, (slice(None), slice(0, 1), slice(None), slice(0, 1), slice(0, 100)))

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
        so.value = 'NUMPY'
        self.assertEqual(so.value, 'numpy')

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

    def test_init_(self):
        K = Abstraction

        k = K()
        self.assertEqual(k.value, None)
        self.assertEqual(str(k), 'abstraction="None"')

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
        self.assertEqual(str(cc),
                         "calc=[{'meta_attrs': None, 'name': 'my_mean', 'func': 'mean', 'kwds': OrderedDict()}]")

        cc = Calc(None)
        self.assertEqual(str(cc), 'calc=None')

        calc = [{'func': 'mean', 'name': 'my_mean', 'kwds': {'a': np.zeros(1000)}}]
        cc = Calc(calc)
        self.assertEqual(str(cc),
                         "calc=[{'meta_attrs': None, 'name': 'my_mean', 'func': 'mean', 'kwds': OrderedDict([('a', <type 'numpy.ndarray'>)])}]")

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

        cc = ConformUnitsTo('kelvin')
        self.assertEqual(cc.value, 'kelvin')

        cc = ConformUnitsTo('not_a_unit')
        self.assertEqual(cc.value, 'not_a_unit')

        cc = ConformUnitsTo(get_units_object('celsius'))
        target = get_are_units_equal((cc.value, get_units_object('celsius')))
        self.assertTrue(target)

        cc = ConformUnitsTo('hPa')
        target = get_are_units_equal((cc.value, get_units_object('hPa')))
        self.assertTrue(target)


class TestHeaders(TestBase):
    create_dir = False

    def test_init(self):
        headers = ['did', 'value']
        for htype in [list, tuple]:
            hvalue = htype(headers)
            hh = Headers(hvalue)
            self.assertEqual(hh.value, tuple(constants.HEADERS_REQUIRED + ['value']))

        headers = ['foo']
        with self.assertRaises(DefinitionValidationError):
            Headers(headers)

        headers = []
        hh = Headers(headers)
        self.assertEqual(hh.value, tuple(constants.HEADERS_REQUIRED))


class TestMelted(TestBase):
    create_dir = False

    @attr('data')
    def test_init(self):
        rd = self.test_data.get_rd('cancm4_tas')
        dataset = Dataset(rd)
        mm = Melted(dataset=dataset, output_format=constants.OUTPUT_FORMAT_NUMPY)
        self.assertIsInstance(mm, BooleanParameter)
        self.assertFalse(mm.value)

        # test with multiple request dataset
        def _run_():
            env.SUPPRESS_WARNINGS = False
            rd2 = self.test_data.get_rd('cancm4_tasmax_2011')
            dataset = Dataset([rd, rd2])
            of = OutputFormat(constants.OUTPUT_FORMAT_SHAPEFILE)
            mm = Melted(dataset=dataset, output_format=of)
            self.assertTrue(mm.value)

        self.assertWarns(OcgWarning, _run_)


class TestDataset(TestBase):
    @attr('data')
    def test_init(self):
        rd = self.test_data.get_rd('cancm4_tas')
        dd = Dataset(rd)

        with open('/tmp/dd.pkl', 'w') as f:
            pickle.dump(dd, f)

        uri = '/a/bad/path'
        with self.assertRaises(ValueError):
            RequestDataset(uri, 'foo')

        # test it can be initialized from itself
        dd = Dataset(rd)
        dd2 = Dataset(dd)
        self.assertEqual(dd.value.first().uri, dd2.value.first().uri)

        # test with a dictionary
        v = {'uri': rd.uri, 'variable': rd.variable}
        dd = Dataset(v)
        self.assertEqual(dd.value[rd.variable].variable, rd.variable)

        # test with a list/tuple
        v2 = v.copy()
        v2['alias'] = 'tas2'
        dd = Dataset([v, v2])
        self.assertEqual(set(dd.value.keys()), set([v['variable'], v2['alias']]))
        dd = Dataset((v, v2))
        self.assertEqual(set(dd.value.keys()), set([v['variable'], v2['alias']]))

        # test with a request dataset
        dd = Dataset(rd)
        self.assertIsInstance(dd.value, RequestDatasetCollection)

        # test with a request dataset collection
        dd = Dataset(dd.value)
        self.assertIsInstance(dd.value, RequestDatasetCollection)

        # test with a bad type
        with self.assertRaises(DefinitionValidationError):
            Dataset(5)

        # test with field does not load anything
        field = self.test_data.get_rd('cancm4_tas').get()
        dd = Dataset(field)
        rfield = dd.value.first()
        self.assertIsNone(rfield.temporal._value)
        self.assertIsNone(rfield.spatial.grid._value)
        self.assertIsNone(rfield.spatial.grid.row._value)
        self.assertIsNone(rfield.spatial.grid.col._value)
        self.assertIsNone(rfield.variables.first()._value)

        # test with a Field object
        field = self.test_data.get_rd('cancm4_tas').get()[:, 0, :, :, :]
        field_value = field.variables.first().value
        dd = Dataset(field)
        self.assertIsInstance(dd.value, RequestDatasetCollection)
        rdc_value = dd.value.first().variables.first().value
        # do not do a deep copy on the field object...
        self.assertTrue(np.may_share_memory(field_value, rdc_value))

        # we do want a deepcopy on the request dataset object and request dataset collection
        rd = self.test_data.get_rd('cancm4_tas')
        dd = Dataset(rd)
        self.assertIsInstance(dd.value.first(), RequestDataset)
        self.assertNotEqual(id(rd), id(dd.value.first()))
        rdc = RequestDatasetCollection(target=[rd])
        dd = Dataset(rdc)
        self.assertNotEqual(id(rdc), id(dd.value))

        # test loading dataset directly from uri with some overloads
        reference_rd = self.test_data.get_rd('cancm4_tas')
        rd = RequestDataset(reference_rd.uri, reference_rd.variable)
        ds = Dataset(rd)
        self.assertEqual(ds.value, RequestDatasetCollection([rd]))
        dsa = {'uri': reference_rd.uri, 'variable': reference_rd.variable}
        Dataset(dsa)
        reference_rd2 = self.test_data.get_rd('narccap_crcm')
        dsb = [dsa, {'uri': reference_rd2.uri, 'variable': reference_rd2.variable, 'alias': 'knight'}]
        Dataset(dsb)

    @attr('esmf')
    def test_init_esmf(self):
        efield = self.get_esmf_field()
        dd = Dataset(efield)
        self.assertIsInstance(dd.value, RequestDatasetCollection)
        ofield = dd.value.first()
        self.assertIsInstance(ofield, Field)
        ofield_value = ofield.variables.first().value
        self.assertTrue(np.may_share_memory(ofield_value, efield.data))
        self.assertNumpyAll(ofield_value.data, efield.data)

    @attr('data')
    def test(self):
        env.DIR_DATA = ocgis.env.DIR_TEST_DATA
        reference_rd = self.test_data.get_rd('cancm4_tas')
        rd = RequestDataset(reference_rd.uri, reference_rd.variable)
        ds = Dataset(rd)
        self.assertEqual(ds.value, RequestDatasetCollection([rd]))

        dsa = {'uri': reference_rd.uri, 'variable': reference_rd.variable}
        Dataset(dsa)

        reference_rd2 = self.test_data.get_rd('narccap_crcm')
        dsb = [dsa, {'uri': reference_rd2.uri, 'variable': reference_rd2.variable, 'alias': 'knight'}]
        Dataset(dsb)

    @attr('data')
    def test_from_query(self):
        rd = self.test_data.get_rd('cancm4_tas')
        qs = 'uri={0}'.format(rd.uri)
        qi = QueryInterface(qs)
        d = Dataset.from_query(qi)
        self.assertEqual(d.value.first().uri, rd.uri)
        qs += '&variable=tas&alias=foobar'
        qi = QueryInterface(qs)
        d = Dataset.from_query(qi)
        self.assertEqual(d.value.first().alias, 'foobar')

        # Test with multiple URIs.
        uri1 = self.test_data.get_uri('cancm4_tas')
        uri2 = self.test_data.get_uri('cancm4_rhsmax')
        qs = 'uri={0}|{1}'.format(uri1, uri2)
        qi = QueryInterface(qs)
        d = Dataset.from_query(qi)
        self.assertEqual(d.value.keys(), ['tas', 'rhsmax'])

    @attr('data')
    def test_get_meta(self):
        # test with standard request dataset collection
        rd = self.test_data.get_rd('cancm4_tas')
        dd = Dataset(rd)
        self.assertIsInstance(dd.get_meta(), list)

        # test passing a field object
        dd = Dataset(rd.get())
        ret = dd.get_meta()
        self.assertEqual(ret, ['* dataset=', 'NcField(name=tas, ...)', ''])

    @attr('data')
    def test_validate(self):
        rd = self.test_data.get_rd('cancm4_tas')
        for iv in [rd, rd.get()]:
            dd = Dataset(iv)
            self.assertIsInstance(dd.value, RequestDatasetCollection)

        # test with no dimensioned variables
        path = self.get_netcdf_path_no_dimensioned_variables()
        keywords = dict(name=['aname', None])
        for k in self.iter_product_keywords(keywords):
            rd = RequestDataset(uri=path, name=k.name)
            if k.name is not None:
                self.assertEqual(rd.name, k.name)
            with self.assertRaises(DefinitionValidationError):
                Dataset(rd)


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
        self.assertIsInstance(g.value[0], SpatialDimension)
        g.value = None
        self.assertEqual(None, g.value)

        g = Geom(None)
        self.assertEqual(g.value, None)
        self.assertEqual(str(g), 'geom=None')

        g = Geom('-120|40|-110|50')
        self.assertEqual(g.value[0].geom.polygon.value[0, 0].bounds, (-120.0, 40.0, -110.0, 50.0))
        self.assertEqual(str(g), 'geom=-120.0|40.0|-110.0|50.0')

        ocgis.env.DIR_GEOMCABINET = self.path_bin
        g = Geom('state_boundaries')
        self.assertEqual(str(g), 'geom="state_boundaries"')

        geoms = list(GeomCabinetIterator('state_boundaries'))
        g = Geom('state_boundaries')
        self.assertEqual(len(list(g.value)), len(geoms))

        sci = GeomCabinetIterator(key='state_boundaries')
        self.assertFalse(sci.as_spatial_dimension)
        g = Geom(sci)
        for _ in range(2):
            for ii, element in enumerate(g.value):
                self.assertIsInstance(element, SpatialDimension)
            self.assertGreater(ii, 10)

        su = GeomSelectUid([1, 2, 3])
        g = Geom('state_boundaries', select_ugid=su)
        self.assertEqual(len(list(g.value)), 3)

        geoms = [{'geom': geom, 'properties': {'UGID': 1}}, {'geom': geom, 'properties': {'UGID': 2}}]
        Geom(geoms)

        bbox = [-120, 40, -110, 50]
        g = Geom(bbox)
        self.assertEqual(g.value[0].geom.polygon.value[0, 0].bounds, tuple(map(float, bbox)))

        sui = GeomUid('ID')
        g = Geom(bbox, geom_uid=sui)
        self.assertEqual(g.geom_uid, 'ID')
        g = Geom(bbox, geom_uid='ID')
        self.assertEqual(g.geom_uid, 'ID')

        # Tests for geom_select_sql_where.

        g = Geom('state_boundaries')
        self.assertIsNone(g.geom_select_sql_where)

        s = 'STATE_NAME in ("Wisconsin", "Vermont")'
        ws = [GeomSelectSqlWhere(s), s]
        for w in ws:
            g = Geom('state_boundaries', geom_select_sql_where=w)
            self.assertEqual(g.geom_select_sql_where, s)

        # Test passing a folder which is not allowed.
        with self.assertRaises(DefinitionValidationError):
            Geom(tempfile.gettempdir())

    def test_geometry_dictionaries(self):
        """Test geometry dictionaries as input."""

        for crs in [None, CFWGS84(), CoordinateReferenceSystem(epsg=2136)]:
            geom = self.get_geometry_dictionaries()
            if crs is not None:
                for g in geom:
                    g['crs'] = crs
            g = Geom(geom)
            self.assertEqual(len(g.value), 3)
            for gdict, sdim in zip(geom, g.value):
                self.assertIsInstance(sdim.geom.get_highest_order_abstraction(), SpatialGeometryPointDimension)
                if crs is None:
                    self.assertIsInstance(sdim.crs, CFWGS84)
                else:
                    self.assertIsInstance(sdim.crs, crs.__class__)
                self.assertEqual(set(sdim.properties.dtype.names), set(['UGID', 'COUNTRY']))
                self.assertEqual(sdim.properties.shape, (1,))
                self.assertEqual(sdim.properties['UGID'][0], gdict['properties']['UGID'])
                self.assertEqual(sdim.properties['COUNTRY'][0], gdict['properties']['COUNTRY'])

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
                    self.assertEqual(element.name_uid, actual)

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

        s = 'STATE_NAME in ("Wisconsin", "Vermont")'
        g = Geom('state_boundaries', geom_select_sql_where=s)
        ret = g.parse_string('state_boundaries')
        self.assertEqual(len(ret), 2)

        ################################################################################################################

    @attr('data')
    def test_spatial_dimension(self):
        """Test using a SpatialDimension as input value."""

        sdim = SpatialDimension.from_records(GeomCabinetIterator(key='state_boundaries'))
        self.assertIsInstance(sdim.crs, CFWGS84)
        g = Geom(sdim)
        self.assertEqual(len(g.value), 51)
        for sdim in g.value:
            self.assertIsInstance(sdim, SpatialDimension)
            self.assertEqual(sdim.shape, (1, 1))

    @attr('data')
    def test_using_shp_path(self):
        # pass a path to a shapefile as opposed to a key
        path = GeomCabinet().get_shp_path('state_boundaries')
        ocgis.env.set_geomcabinet_path(None)
        # make sure there is path associated with the GeomCabinet
        with self.assertRaises(ValueError):
            GeomCabinet().keys()
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
        of = OutputFormat(constants.OUTPUT_FORMAT_CSV_SHAPEFILE_OLD)
        self.assertEqual(of.value, constants.OUTPUT_FORMAT_CSV_SHAPEFILE)

        of2 = OutputFormat(of)
        self.assertEqual(of.value, of2.value)

    @attr('esmf')
    def test_init_esmpy(self):
        oo = OutputFormat(constants.OUTPUT_FORMAT_ESMPY_GRID)
        self.assertEqual(oo.value, constants.OUTPUT_FORMAT_ESMPY_GRID)

    def test_get_converter_class(self):
        of = OutputFormat(constants.OUTPUT_FORMAT_NUMPY)
        self.assertEqual(of.get_converter_class(), NumpyConverter)

    def test_valid(self):
        self.assertAsSetEqual(OutputFormat.valid, ['csv', 'csv-shp', 'geojson', 'meta-ocgis', 'nc', 'numpy', 'shp',
                                                   constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH, 'meta-json',
                                                   constants.OUTPUT_FORMAT_ESMPY_GRID])


class TestOutputFormatOptions(TestBase):
    def test_init(self):
        self.assertIsNone(OutputFormatOptions().value)
        opts = {'data_model': 'foo'}
        ofo = OutputFormatOptions(opts)
        self.assertDictEqual(opts, ofo.value)


class TestRegridDestination(TestBase):
    @property
    def possible_datasets(self):
        # one dataset to be regridded
        dataset1 = [self.get_rd(regrid_source=True)]

        # one dataset without regrid_source set to True
        dataset2 = [self.get_rd()]

        # two datasets to be regridded
        dataset3 = [self.get_rd(regrid_source=True), self.get_rd(regrid_source=True, alias='number2')]

        # two datasets but only regrid one
        dataset4 = [self.get_rd(regrid_source=True), self.get_rd(alias='number2')]

        # three datasets, two to be regridded and one as the destination
        dataset5 = [self.get_rd(regrid_source=True), self.get_rd(alias='number2', regrid_destination=True),
                    self.get_rd(regrid_source=True, alias='number3')]

        # three datasets, but with two set as the regrid destinations
        dataset6 = [self.get_rd(regrid_destination=True), self.get_rd(alias='number2', regrid_destination=True),
                    self.get_rd(regrid_source=True, alias='number3')]

        datasets = [dataset1, dataset2, dataset3, dataset4, dataset5, dataset6]
        datasets = [Dataset(d) for d in datasets]
        datasets = {ii: d for ii, d in enumerate(datasets, start=1)}

        return datasets

    @property
    def possible_values(self):
        rd = self.get_rd()
        possible = [None, 'tas', rd, rd.get(), rd.get().spatial]
        return possible

    def get_rd(self, **kwargs):
        rd = self.test_data.get_rd('cancm4_tas', kwds=kwargs)
        return rd

    @attr('data')
    def test_init(self):
        """Also tests get_meta."""

        ctr = 0
        for key, dataset in self.possible_datasets.iteritems():
            for poss in self.possible_values:

                # ctr += 1
                # print ctr
                # if ctr != 2: continue

                try:
                    regrid = RegridDestination(init_value=poss, dataset=dataset)
                except DefinitionValidationError:
                    # only one request dataset may be the destination grid if the init_value is None
                    if key == 6 and poss is None:
                        continue
                    # no datasets are set as a source but there is a destination
                    elif key == 2 and poss is not None:
                        continue
                    else:
                        raise
                if poss == 'tas':
                    self.assertEqual(regrid.value.name, 'tas')
                else:
                    type_check = [isinstance(regrid.value, k) for k in Field, SpatialDimension]
                    try:
                        self.assertTrue(any(type_check))
                    except AssertionError:
                        # if the destination is none then this is okay
                        if poss is None:
                            pass
                        else:
                            raise
                self.assertIsInstance(regrid._get_meta_(), basestring)


class TestRegridOptions(TestBase):
    def test_init(self):
        ro = RegridOptions()
        self.assertDictEqual(ro.value, RegridOptions.default)

        with self.assertRaises(DefinitionValidationError):
            RegridOptions({'with_corners': 'bad'})

        with self.assertRaises(DefinitionValidationError):
            RegridOptions({'value_mask': 'foo'})

        with self.assertRaises(DefinitionValidationError):
            RegridOptions({'value_mask': np.array([5, 6, 7])})

        ro = RegridOptions({'with_corners': True})
        self.assertDictEqual(ro.value, {'with_corners': True, 'value_mask': None})

        ro = RegridOptions({'value_mask': np.array([True, False])})
        self.assertEqual(ro.value['with_corners'], 'auto')
        self.assertNumpyAll(ro.value['value_mask'], np.array([True, False]))

        with self.assertRaises(DefinitionValidationError):
            RegridOptions({'foo': 5})

    def test_get_meta(self):
        ro = RegridOptions()
        ro._get_meta_()

        ro = RegridOptions({'value_mask': np.array([True])})
        self.assertTrue('numpy.ndarray' in ro._get_meta_())


class TestSlice(TestBase):
    def test(self):
        v = [None, -1, None, None, None]
        s = Slice(v)
        self.assertEqual(s.value[1], slice(-1, None))

        v = [None, [-2, -1], None, None, None]
        s = Slice(v)
        self.assertEqual(s.value[1], slice(-2, -1))


class TestSpatialOperation(TestBase):
    def test_init(self):
        values = (None, 'clip', 'intersects')
        ast = ('intersects', 'clip', 'intersects')

        klass = SpatialOperation
        for v, a in zip(values, ast):
            obj = klass(v)
            self.assertEqual(obj.value, a)


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
