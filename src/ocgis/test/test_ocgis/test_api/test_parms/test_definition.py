import unittest
from cfunits import Units
from ocgis.api.parms.definition import *
from ocgis.interface.base.dimension.spatial import SpatialDimension, SpatialGeometryPointDimension
from ocgis.util.helpers import make_poly
import pickle
import tempfile
from ocgis.test.base import TestBase
from ocgis.calc.library.statistics import Mean
from ocgis.util.itester import itr_products_keywords
from ocgis.util.shp_cabinet import ShpCabinet
import numpy as np
from ocgis.calc.eval_function import MultivariateEvalFunction


class Test(TestBase):

    def test_callback(self):
        c = Callback()
        self.assertEqual(c.value,None)

        with self.assertRaises(DefinitionValidationError):
            Callback('foo')

        def callback(percent,message):
            pass

        c = Callback(callback)
        self.assertEqual(callback,c.value)

    def test_optimizations(self):
        o = Optimizations()
        self.assertEqual(o.value,None)
        with self.assertRaises(DefinitionValidationError):
            Optimizations({})
        with self.assertRaises(DefinitionValidationError):
            Optimizations({'foo':'foo'})
        o = Optimizations({'tgds':{'tas':'TemporalGroupDimension'}})
        self.assertEqual(o.value,{'tgds':{'tas':'TemporalGroupDimension'}})

    def test_optimizations_deepcopy(self):
        ## we should not deepcopy optimizations
        arr = np.array([1,2,3,4])
        value = {'tgds':{'tas':arr}}
        o = Optimizations(value)
        self.assertTrue(np.may_share_memory(o.value['tgds']['tas'],arr))

    def test_add_auxiliary_files(self):
        for val in [True,False]:
            p = AddAuxiliaryFiles(val)
            self.assertEqual(p.value,val)
        p = AddAuxiliaryFiles()
        self.assertEqual(p.value,True)

    def test_dir_output(self):
        ## raise an exception if the directory does not exist
        do = '/does/not/exist'
        with self.assertRaises(DefinitionValidationError):
            DirOutput(do)

        ## make sure directory name does not change case
        do = 'Some'
        new_dir = os.path.join(tempfile.gettempdir(),do)
        os.mkdir(new_dir)
        try:
            dd = DirOutput(new_dir)
            self.assertEqual(new_dir,dd.value)
        finally:
            os.rmdir(new_dir)

    def test_slice(self):
        slc = Slice(None)
        self.assertEqual(slc.value,None)

        slc = Slice([None,0,0,0,0])
        self.assertEqual(slc.value,(slice(None),slice(0,1),slice(0, 1),slice(0, 1),slice(0, 1)))

        slc = Slice([None,0,None,[0,1],[0,100]])
        self.assertEqual(slc.value,(slice(None),slice(0,1),slice(None),slice(0,1),slice(0,100)))

        with self.assertRaises(DefinitionValidationError):
            slc.value = 4
        with self.assertRaises(DefinitionValidationError):
            slc.value = [None,None]

    def test_snippet(self):
        self.assertFalse(Snippet().value)
        for ii in ['t','TRUE','tRue',1,'1',' 1 ']:
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
        self.assertEqual(so.value,'intersects')
        with self.assertRaises(DefinitionValidationError):
            so.value = 'clips'
        so.value = 'clip'

    def test_output_format(self):
        so = OutputFormat('csv')
        self.assertEqual(so.value,'csv')
        so.value = 'NUMPY'
        self.assertEqual(so.value,'numpy')

    def test_select_ugid(self):
        so = SelectUgid()
        self.assertEqual(so.value,None)
        with self.assertRaises(DefinitionValidationError):
            so.value = 98.5
        so.value = 'none'
        self.assertEqual(so.value,None)
        with self.assertRaises(DefinitionValidationError):
            so.value = 1
        so = SelectUgid('10')
        self.assertEqual(so.value,(10,))
        with self.assertRaises(DefinitionValidationError):
            so.value = ('1|1|2')
        with self.assertRaises(DefinitionValidationError):
            so.value = '22.5'
        so = SelectUgid('22|23|24')
        self.assertEqual(so.value,(22,23,24))
        with self.assertRaises(DefinitionValidationError):
            so.value = '22|23.5|24'

    def test_prefix(self):
        pp = Prefix()
        self.assertEqual(pp.value,'ocgis_output')
        pp.value = ' Old__man '
        self.assertEqual(pp.value,'Old__man')

    def test_calc_grouping(self):
        cg = CalcGrouping(['day','month'])
        self.assertEqual(cg.value,('day','month'))
        with self.assertRaises(DefinitionValidationError):
            cg.value = ['d','foo']

    def test_calc_grouping_all(self):
        cg = CalcGrouping('all')
        self.assertEqual(cg.value,'all')

    def test_calc_grouping_seasonal_aggregation(self):
        cg = CalcGrouping([[1,2,3],[4,5,6]])
        self.assertEqual(cg.value,([1,2,3],[4,5,6]))

        ## element groups must be composed of unique elements
        with self.assertRaises(DefinitionValidationError):
            CalcGrouping([[1,2,3],[4,4,6]])

        ## element groups must have an empty intersection
        with self.assertRaises(DefinitionValidationError):
            CalcGrouping([[1,2,3],[1,4,6]])

        ## months must be between 1 and 12
        with self.assertRaises(DefinitionValidationError):
            CalcGrouping([[1,2,3],[4,5,66]])

    def test_calc_grouping_seasonal_aggregation_with_year(self):
        cg = CalcGrouping([[1,2,3],[4,5,6],'year'])
        self.assertEqual(cg.value,([1,2,3],[4,5,6],'year'))

    def test_calc_grouping_seasonal_aggregation_with_unique(self):
        cg = CalcGrouping([[1,2,3],[4,5,6],'unique'])
        self.assertEqual(cg.value,([1,2,3],[4,5,6],'unique'))

    def test_calc_grouping_seasonal_aggregation_with_bad_flag(self):
        with self.assertRaises(DefinitionValidationError):
            CalcGrouping([[1,2,3],[4,5,6],'foo'])
        with self.assertRaises(DefinitionValidationError):
            CalcGrouping([[1,2,3],[4,5,6],'fod'])

    def test_dataset(self):
        rd = self.test_data.get_rd('cancm4_tas')
        dd = Dataset(rd)

        with open('/tmp/dd.pkl','w') as f:
            pickle.dump(dd,f)

        uri = '/a/bad/path'
        with self.assertRaises(ValueError):
            rd = RequestDataset(uri,'foo')


class TestCalc(TestBase):
    _create_dir = False

    def test_meta_attrs(self):
        """Test various forms for meta_attrs in the calculation definition dictionary."""

        kwds = dict(
            meta_attr=[None, {}, {'something_else': 'is_here with us'}],
            calc=[{'func': 'mean', 'name': 'my_mean'}, 'foo=tas+4', {'func': 'foo=tas+4', 'meta_attrs': {'something': 'special'}}],
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

    def test_init(self):
        calc = [{'func':'mean','name':'my_mean'}]
        cc = Calc(calc)
        eq = [{'ref':Mean,'name':'my_mean','func':'mean','kwds':{}, 'meta_attrs': None}]

        self.assertEqual(cc.value,eq)
        cc.value = 'mean~my_mean'
        self.assertEqual(cc.value,eq)
        cc.value = 'mean~my_mean|max~my_max|between~between5_10!lower~5!upper~10'
        with self.assertRaises(NotImplementedError):
            self.assertEqual(cc.get_url_string(),'mean~my_mean|max~my_max|between~between5_10!lower~5.0!upper~10.0')

    def test_bad_key(self):
        calc = [{'func':'bad_mean','name':'my_mean'}]
        with self.assertRaises(DefinitionValidationError):
            Calc(calc)

    def test_str(self):
        calc = [{'func': 'mean', 'name': 'my_mean'}]
        cc = Calc(calc)
        self.assertEqual(str(cc), "calc=[{'meta_attrs': None, 'name': 'my_mean', 'func': 'mean', 'kwds': OrderedDict()}]")

        cc = Calc(None)
        self.assertEqual(str(cc), 'calc=None')

    def test_get_meta(self):
        for poss in Calc._possible:
            cc = Calc(poss)
            cc.get_meta()

    def test_eval_underscores_in_variable_names(self):
        value = 'tas_4=tasmin_2+tasmin'
        self.assertEqual(Calc(value).value,[{'func':value,'ref':MultivariateEvalFunction, 'meta_attrs': None, 'name': None, 'kwds': OrderedDict()}])

    def test_eval_string(self):
        value = [
                 'es=tas+4',
                 ['es=tas+4']
                 ]
        actual = [{'func':'es=tas+4','ref':EvalFunction, 'meta_attrs': None, 'name': None, 'kwds': OrderedDict()}]
        for v in value:
            cc = Calc(v)
            self.assertEqual(cc.value,actual)

    def test_eval_string_multivariate(self):
        value = [
                 'es=exp(tas)+tasmax+log(4)',
                 ['es=exp(tas)+tasmax+log(4)']
                 ]
        actual = [{'func':'es=exp(tas)+tasmax+log(4)','ref':MultivariateEvalFunction, 'meta_attrs': None, 'name': None, 'kwds': OrderedDict()}]
        for v in value:
            cc = Calc(v)
            self.assertEqual(cc.value,actual)

    def test_eval_string_number_after_variable_alias(self):
        value = 'tas2=tas1+tas2'
        self.assertTrue(EvalFunction.is_multivariate(value))

        value = 'tas2=tas1+tas1'
        self.assertFalse(EvalFunction.is_multivariate(value))

    def test_eval_string_malformed(self):
        with self.assertRaises(DefinitionValidationError):
            Calc('estas+4')

    def test(self):
        calc = [{'func':'mean','name':'my_mean'}]
        cc = Calc(calc)
        eq = [{'ref':Mean,'name':'my_mean','func':'mean','kwds':{},'meta_attrs': None}]

        self.assertEqual(cc.value,eq)
        cc.value = 'mean~my_mean'
        self.assertEqual(cc.value,eq)
        cc.value = 'mean~my_mean|max~my_max|between~between5_10!lower~5!upper~10'
        with self.assertRaises(NotImplementedError):
            self.assertEqual(cc.get_url_string(),'mean~my_mean|max~my_max|between~between5_10!lower~5.0!upper~10.0')

    def test_bad_key(self):
        calc = [{'func':'bad_mean','name':'my_mean'}]
        with self.assertRaises(DefinitionValidationError):
            Calc(calc)


class TestConformUnitsTo(TestBase):
    _create_dir = False

    def test_constructor(self):
        cc = ConformUnitsTo()
        self.assertEqual(cc.value, None)

        cc = ConformUnitsTo('kelvin')
        self.assertEqual(cc.value, 'kelvin')

        cc = ConformUnitsTo('not_a_unit')
        self.assertEqual(cc.value, 'not_a_unit')

        cc = ConformUnitsTo(Units('celsius'))
        self.assertTrue(cc.value.equals(Units('celsius')))


class TestGeom(TestBase):
    _create_dir = False

    def test_init(self):
        geom = make_poly((37.762,38.222),(-102.281,-101.754))

        g = Geom(geom)
        self.assertEqual(type(g.value), tuple)
        self.assertIsInstance(g.value[0], SpatialDimension)
        g.value = None
        self.assertEqual(None,g.value)

        g = Geom(None)
        self.assertEqual(g.value,None)
        self.assertEqual(str(g),'geom=None')

        g = Geom('-120|40|-110|50')
        self.assertEqual(g.value[0].geom.polygon.value[0, 0].bounds,(-120.0, 40.0, -110.0, 50.0))
        self.assertEqual(str(g),'geom=-120.0|40.0|-110.0|50.0')

        g = Geom('state_boundaries')
        self.assertEqual(str(g),'geom="state_boundaries"')

        geoms = list(ShpCabinetIterator('state_boundaries'))
        g = Geom('state_boundaries')
        self.assertEqual(len(list(g.value)),len(geoms))

        sci = ShpCabinetIterator(key='state_boundaries')
        self.assertFalse(sci.as_spatial_dimension)
        g = Geom(sci)
        for _ in range(2):
            for ii, element in enumerate(g.value):
                self.assertIsInstance(element, SpatialDimension)
            self.assertGreater(ii, 10)

        su = SelectUgid([1,2,3])
        g = Geom('state_boundaries',select_ugid=su)
        self.assertEqual(len(list(g.value)),3)

        geoms = [{'geom':geom,'properties':{'UGID':1}},{'geom':geom,'properties':{'UGID':2}}]
        g = Geom(geoms)

        bbox = [-120,40,-110,50]
        g = Geom(bbox)
        self.assertEqual(g.value[0].geom.polygon.value[0, 0].bounds,tuple(map(float,bbox)))

    def test_spatial_dimension(self):
        """Test using a SpatialDimension as input value."""

        sdim = SpatialDimension.from_records(ShpCabinetIterator(key='state_boundaries'))
        self.assertIsInstance(sdim.crs, CFWGS84)
        g = Geom(sdim)
        self.assertEqual(len(g.value), 51)
        for sdim in g.value:
            self.assertIsInstance(sdim, SpatialDimension)
            self.assertEqual(sdim.shape, (1, 1))

    def test_using_shp_path(self):
        ## pass a path to a shapefile as opposed to a key
        path = ShpCabinet().get_shp_path('state_boundaries')
        ocgis.env.DIR_SHPCABINET = None
        ## make sure there is path associated with the ShpCabinet
        with self.assertRaises(ValueError):
            ShpCabinet().keys()
        g = Geom(path)
        self.assertEqual(g._shp_key,path)
        self.assertEqual(len(list(g.value)),51)

    def test_with_changing_select_ugid(self):
        select_ugid = [16,17]
        g = Geom('state_boundaries',select_ugid=select_ugid)
        self.assertEqual(len(list(g.value)),2)
        select_ugid.append(22)
        self.assertEqual(len(list(g.value)),3)

        g = Geom('state_boundaries')
        self.assertEqual(len(list(g.value)),51)
        g.select_ugid = [16,17]
        self.assertEqual(len(list(g.value)),2)

    @staticmethod
    def get_geometry_dictionaries():
        coordinates = [('France', [2.8, 47.16]),
                       ('Germany', [10.5, 51.29]),
                       ('Italy', [12.2, 43.4])]
        geom = []
        for ugid, coordinate in enumerate(coordinates, start=1):
            point = Point(coordinate[1][0], coordinate[1][1])
            geom.append({'geom': point,
                         'properties': {'UGID': ugid, 'COUNTRY': coordinate[0]}})
        return geom

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


class TestRegridDestination(TestBase):

    @property
    def possible_values(self):
        rd = self.get_rd()
        possible = [None, 'tas', rd, rd.get(), rd.get().spatial]
        return possible

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

    def get_rd(self, **kwargs):
        rd = self.test_data.get_rd('cancm4_tas', kwds=kwargs)
        return rd

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
        self.assertEqual(ro.value['with_corners'], 'choose')
        self.assertNumpyAll(ro.value['value_mask'], np.array([True, False]))

        with self.assertRaises(DefinitionValidationError):
            RegridOptions({'foo': 5})

    def test_get_meta(self):
        ro = RegridOptions()
        ro._get_meta_()

        ro = RegridOptions({'value_mask': np.array([True])})
        self.assertTrue('numpy.ndarray' in ro._get_meta_())

class TestLevelRange(TestBase):
    _create_dir = False

    def test_constructor(self):
        LevelRange()

    def test_normal_int(self):
        lr = LevelRange([5,10])
        self.assertEqual(lr.value,(5,10))

    def test_normal_float(self):
        value = [4.5,6.5]
        lr = LevelRange(value)
        self.assertEqual(tuple(value),lr.value)

    def test_bad_length(self):
        with self.assertRaises(DefinitionValidationError):
            LevelRange([5,6,7,8])

    def test_bad_ordination(self):
        with self.assertRaises(DefinitionValidationError):
            LevelRange([11,10])


class TestTimeRange(TestBase):
    _create_dir = False

    def test_constructor(self):
        TimeRange()

    def test_range(self):
        dt = [datetime.datetime(2000,1,1),datetime.datetime(2001,1,1)]
        tr = TimeRange(dt)
        self.assertEqual(tr.value,tuple(dt))

    def test_bad_ordination(self):
        dt = [datetime.datetime(2000,1,1),datetime.datetime(1999,1,1)]
        with self.assertRaises(DefinitionValidationError):
            TimeRange(dt)

    def test_incorrect_number_of_values(self):
        dt = [datetime.datetime(2000,1,1),datetime.datetime(1999,1,1),datetime.datetime(1999,1,1)]
        with self.assertRaises(DefinitionValidationError):
            TimeRange(dt)


class TestTimeRegion(TestBase):
    _create_dir = False

    def test_constructor(self):
        TimeRegion()

    def test_normal(self):
        value = {'month':[6,7,8],'year':[4,5,6]}
        tr = TimeRegion(value)
        self.assertEqual(value,tr.value)

    def test_month_only(self):
        value = {'month':[6]}
        tr = TimeRegion(value)
        self.assertEqual(tr.value,{'month':[6],'year':None})

    def test_year_only(self):
        value = {'year':[6]}
        tr = TimeRegion(value)
        self.assertEqual(tr.value,{'month':None,'year':[6]})

    def test_both_none(self):
        value = {'year':None,'month':None}
        tr = TimeRegion(value)
        self.assertEqual(tr.value,None)

    def test_bad_keys(self):
        value = {'mnth':[4]}
        with self.assertRaises(DefinitionValidationError):
            TimeRegion(value)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
