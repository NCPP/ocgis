from copy import deepcopy
from csv import DictReader
import os
import shutil
import itertools
import numpy as np

import fiona
from shapely import wkt

from ocgis.api.request.base import RequestDataset, RequestDatasetCollection
import ocgis
from ocgis.exc import DefinitionValidationError, ExtentError
from ocgis.interface.base.crs import CFWGS84, CoordinateReferenceSystem
from ocgis.util.spatial.fiona_maker import FionaMaker
from ocgis import OcgOperations, env, constants
from ocgis.test.base import TestBase, attr
from ocgis.test.test_simple.make_test_data import SimpleNc, SimpleNcNoBounds, SimpleNcNoLevel
from ocgis.test.test_simple.test_simple import TestSimpleBase


class TestCombinatorial(TestBase):

    def iter_dataset(self):
        for as_request_dataset in [True, False]:
            for k in self.test_data.iterkeys():
                kwds = {}
                if k == 'cmip3_extraction':
                    dimension_map = {'R': 'projection', 'T': 'time', 'Y': 'latitude', 'X': 'longitude'}
                    kwds['dimension_map'] = dimension_map
                rd = self.test_data.get_rd(k, kwds=kwds)
                if as_request_dataset:
                    yield k, rd
                else:
                    yield k, rd.get()

    @attr('slow')
    def test(self):
        import logbook

        log = logbook.Logger(name='combos', level=logbook.INFO)

        for key, dataset in self.iter_dataset():

            # if key != 'qed_2013_TNn_annual_min': continue

            # these datasets have only one time element
            if key in ('qed_2013_TNn_annual_min',
                       'qed_2013_TasMin_seasonal_max_of_seasonal_means',
                       'qed_2013_climatology_Tas_annual_max_of_annual_means',
                       'qed_2013_maurer02v2_median_txxmmedm_january_1971-2000',
                       'qed_2013_maurer02v2_median_txxmmedm_february_1971-2000',
                       'qed_2013_maurer02v2_median_txxmmedm_march_1971-2000',
                       'snippet_maurer_dtr',
                       'snippet_seasonalbias'):
                slc = None
            else:
                slc = [None, [10, 20], None, None, None]

            # this has different data types on the bounds for the coordinate variables. they currently get casted by the
            # software.
            if key == 'maurer_bcca_1991':
                check_types = False
            else:
                check_types = True

            log.debug('processing: {0} ({1})'.format(key, dataset.__class__.__name__))
            ops = OcgOperations(dataset=dataset, output_format='nc', prefix='nc1', slice=slc)
            try:
                log.debug('initial write...')
                ret1 = ops.execute()
            except ValueError:
                # realization dimensions may not be written to netCDF yet
                if key == 'cmip3_extraction':
                    continue
                else:
                    raise
            else:
                try:
                    ops2 = OcgOperations(dataset={'uri': ret1}, output_format='nc', prefix='nc2')
                    log.debug('second write...')
                    ret2 = ops2.execute()
                    log.debug('comparing...')
                    self.assertNcEqual(ret1, ret2, ignore_attributes={'global': ['history']}, check_types=check_types)
                finally:
                    for path in [ret1, ret2]:
                        folder = os.path.split(path)[0]
                        shutil.rmtree(folder)
        log.debug('success')


class TestProjectionCombinations(TestSimpleBase):
    base_value = np.array([[1.0, 1.0, 2.0, 2.0],
                           [1.0, 1.0, 2.0, 2.0],
                           [3.0, 3.0, 4.0, 4.0],
                           [3.0, 3.0, 4.0, 4.0]])
    nc_factory = SimpleNc
    fn = 'test_simple_spatial_01.nc'

    @attr('slow')
    def test_calc_sample_size(self):
        rd1 = self.get_dataset()
        rd1['alias'] = 'var1'
        rd2 = self.get_dataset()
        rd2['alias'] = 'var2'

        dataset = [
                   # RequestDatasetCollection([rd1]),
                   RequestDatasetCollection([rd1,rd2])
                   ]
        calc_sample_size = [
                            True,
                            # False
                            ]
        calc = [
                [{'func':'mean','name':'mean'},{'func':'max','name':'max'}],
                # [{'func':'ln','name':'ln'}],
                # None,
                # [{'func':'divide','name':'divide','kwds':{'arr1':'var1','arr2':'var2'}}]
                ]
        calc_grouping = [
                         # None,
                         ['month'],
                         # ['month','year']
                         ]
        output_format = ['numpy']

        for ii,tup in enumerate(itertools.product(dataset,calc_sample_size,calc,calc_grouping,output_format)):
            kwds = dict(zip(['dataset','calc_sample_size','calc','calc_grouping','output_format'],tup))
            kwds['prefix'] = str(ii)

            try:
                ops = OcgOperations(**kwds)
            except DefinitionValidationError:
                if kwds['calc'] is not None:
                    ## set functions require a temporal grouping otherwise the calculation
                    ## is meaningless
                    if kwds['calc'][0]['func'] == 'mean' and kwds['calc_grouping'] is None:
                        continue
                    ## multivariate functions may not implemented with sample size = True
                    elif kwds['calc_sample_size'] and kwds['calc'][0]['func'] == 'divide':
                        continue
                    ## multivariate functions require the correct number of variables
                    elif kwds['calc'][0]['func'] == 'divide' and len(kwds['dataset']) == 1:
                        continue
                    ## only one request dataset may be written to netCDF at this time
                    elif kwds['output_format'] == 'nc' and len(kwds['dataset']) == 2:
                        continue
                    else:
                        raise
                ## only one request dataset may be written to netCDF at this time
                elif kwds['output_format'] == 'nc' and len(ops.dataset) == 2:
                    continue
                else:
                    raise

            ret = ops.execute()

            if kwds['output_format'] == 'nc':
                if kwds['calc_sample_size'] and kwds['calc_grouping']:
                    if kwds['calc'] is not None and kwds['calc'][0]['func'] == 'mean':
                        with self.nc_scope(ret) as ds:
                            self.assertEqual(sum([v.startswith('n_') for v in ds.variables.keys()]),2)
                            self.assertEqual(ds.variables['n_max'][:].mean(),30.5)

            if kwds['output_format'] == 'csv':
                if kwds['calc'] is not None and kwds['calc'][0]['func'] == 'mean':
                    with open(ret,'r') as f:
                        reader = DictReader(f)
                        alias_set = set([row['CALC_ALIAS'] for row in reader])
                        if len(kwds['dataset']) == 1:
                            if kwds['calc_sample_size']:
                                self.assertEqual(alias_set,set(['max','n_max','n_mean','mean']))
                            else:
                                self.assertEqual(alias_set,set(['max','mean']))
                        else:
                            if kwds['calc_sample_size']:
                                self.assertEqual(alias_set,set(['max_var1','n_max_var1','n_mean_var1','mean_var1',
                                                                'max_var2','n_max_var2','n_mean_var2','mean_var2']))
                            else:
                                self.assertEqual(alias_set,set(['max_var1','mean_var1',
                                                                'max_var2','mean_var2']))

    @attr('slow')
    def test_combinatorial_projection_with_geometries(self):

        # self.get_ret(kwds={'output_format':'shp','prefix':'as_polygon'})
        # self.get_ret(kwds={'output_format':'shp','prefix':'as_point','abstraction':'point'})

        features = [
            {'NAME': 'a', 'wkt': 'POLYGON((-105.020430 40.073118,-105.810753 39.327957,-105.660215 38.831183,-104.907527 38.763441,-104.004301 38.816129,-103.643011 39.802151,-103.643011 39.802151,-103.643011 39.802151,-103.643011 39.802151,-103.959140 40.118280,-103.959140 40.118280,-103.959140 40.118280,-103.959140 40.118280,-104.327957 40.201075,-104.327957 40.201075,-105.020430 40.073118))'},
            {'NAME': 'b', 'wkt': 'POLYGON((-102.212903 39.004301,-102.905376 38.906452,-103.311828 37.694624,-103.326882 37.295699,-103.898925 37.220430,-103.846237 36.746237,-102.619355 37.107527,-102.634409 37.724731,-101.874194 37.882796,-102.212903 39.004301))'},
            {'NAME': 'c', 'wkt': 'POLYGON((-105.336559 37.175269,-104.945161 37.303226,-104.726882 37.175269,-104.696774 36.844086,-105.043011 36.693548,-105.283871 36.640860,-105.336559 37.175269))'},
            {'NAME': 'd', 'wkt': 'POLYGON((-102.318280 39.741935,-103.650538 39.779570,-103.620430 39.448387,-103.349462 39.433333,-103.078495 39.606452,-102.325806 39.613978,-102.325806 39.613978,-102.333333 39.741935,-102.318280 39.741935))'},
        ]

        for filename in ['polygon', 'point']:
            if filename == 'point':
                geometry = 'Point'
                to_write = deepcopy(features)
                for feature in to_write:
                    geom = wkt.loads(feature['wkt'])
                    feature['wkt'] = geom.centroid.wkt
            else:
                to_write = features
                geometry = 'Polygon'

            path = os.path.join(self.current_dir_output, 'ab_{0}.shp'.format(filename))
            with FionaMaker(path, geometry=geometry) as fm:
                fm.write(to_write)

        no_bounds_nc = SimpleNcNoBounds()
        no_bounds_nc.write()
        no_bounds_uri = os.path.join(env.DIR_OUTPUT, no_bounds_nc.filename)

        no_level_nc = SimpleNcNoLevel()
        no_level_nc.write()
        no_level_uri = os.path.join(env.DIR_OUTPUT, no_level_nc.filename)

        ocgis.env.DIR_SHPCABINET = self.current_dir_output
        # ocgis.env.DEBUG = True
        #        ocgis.env.VERBOSE = True

        aggregate = [
            False,
            True
        ]
        spatial_operation = [
            'intersects',
            'clip'
        ]
        epsg = [
            2163,
            4326,
            None
        ]
        output_format = [
            constants.OUTPUT_FORMAT_NETCDF,
            constants.OUTPUT_FORMAT_SHAPEFILE,
            constants.OUTPUT_FORMAT_CSV_SHAPEFILE
        ]
        abstraction = [
            'polygon',
            'point',
            None
        ]
        dataset = [
            self.get_dataset(),
            {'uri': no_bounds_uri, 'variable': 'foo'},
            {'uri': no_level_uri, 'variable': 'foo'}
        ]
        geom = [
            'ab_polygon',
            'ab_point'
        ]
        calc = [
            None,
            [{'func': 'mean', 'name': 'my_mean'}]
        ]
        calc_grouping = ['month']

        args = (aggregate, spatial_operation, epsg, output_format, abstraction, geom, calc, dataset)
        for ii, tup in enumerate(itertools.product(*args)):
            a, s, e, o, ab, g, c, d = tup

            if os.path.split(d['uri'])[1] == 'test_simple_spatial_no_bounds_01.nc':
                unbounded = True
            else:
                unbounded = False

            if o == constants.OUTPUT_FORMAT_NETCDF and e == 4326:
                output_crs = CFWGS84()
            else:
                output_crs = CoordinateReferenceSystem(epsg=e) if e is not None else None

            kwds = dict(aggregate=a, spatial_operation=s, output_format=o, output_crs=output_crs, geom=g,
                        abstraction=ab, dataset=d, prefix=str(ii), calc=c, calc_grouping=calc_grouping)

            try:
                ops = OcgOperations(**kwds)
                ret = ops.execute()
            except DefinitionValidationError:
                if o == constants.OUTPUT_FORMAT_NETCDF:
                    if e not in [4326, None]:
                        continue
                    if s == 'clip':
                        continue
                else:
                    raise
            except ExtentError:
                if unbounded or ab == 'point':
                    continue
                else:
                    raise

            if o == constants.OUTPUT_FORMAT_SHAPEFILE:
                ugid_path = os.path.join(self.current_dir_output, ops.prefix, ops.prefix + '_ugid.shp')
            else:
                ugid_path = os.path.join(self.current_dir_output, ops.prefix, constants.OUTPUT_FORMAT_SHAPEFILE,
                                         ops.prefix + '_ugid.shp')

            if o != constants.OUTPUT_FORMAT_NETCDF:
                with fiona.open(ugid_path, 'r') as f:
                    if e:
                        second = output_crs
                    else:
                        second = CoordinateReferenceSystem(epsg=4326)
                    self.assertEqual(CoordinateReferenceSystem(value=f.meta['crs']), second)

            if o == constants.OUTPUT_FORMAT_SHAPEFILE:
                with fiona.open(ret, 'r') as f:
                    if a and ab == 'point':
                        second = 'MultiPoint'
                    elif ab is None:
                        field = RequestDataset(uri=d['uri'], variable='foo').get()
                        second = field.spatial.geom.get_highest_order_abstraction().geom_type
                    else:
                        second = ab.title()

                    if second in ['Polygon', 'MultiPolygon']:
                        second = ['Polygon', 'MultiPolygon']
                    elif second in ['Point', 'MultiPoint']:
                        second = ['Point', 'MultiPoint']

                    self.assertTrue(f.meta['schema']['geometry'] in second)
