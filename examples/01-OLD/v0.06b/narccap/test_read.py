import os.path
import shutil
import tempfile
import unittest

import fiona
import netCDF4 as nc
from fiona import crs
from ocgis.interface.projection import get_projection, PolarStereographic, \
    NarccapObliqueMercator, RotatedPole
from ocgis.interface.shp import ShpDataset
from shapely.geometry.geo import mapping
from shapely.geometry.point import Point

import ocgis
from ocgis.util.helpers import iter_array


class NarccapTestBase(unittest.TestCase):

    def setUp(self):
        ocgis.env.DIR_DATA = '/usr/local/climate_data/narccap'
        self.polar_stereographic = ocgis.RequestDataset(uri='pr_CRCM_ccsm_1981010103.nc',
                                                        variable='pr')
        self.oblique_mercator = ocgis.RequestDataset(uri='pr_RCM3_gfdl_1981010103.nc',
                                                     variable='pr')
        self.rotated_pole = ocgis.RequestDataset(uri='pr_HRM3_gfdl_1981010103.nc',
                                                 variable='pr')
        self.ecp2 = ocgis.RequestDataset(uri='pr_ECP2_ncep_1981010103.nc',
                                         variable='pr')

    def tearDown(self):
        ocgis.env.reset()


class Test(NarccapTestBase):

    def work(self):
        ocgis.env.WRITE_TO_REFERENCE_PROJECTION = True
        ocgis.env.VERBOSE = True
        rd = self.oblique_mercator
        ops = ocgis.OcgOperations(dataset=rd, snippet=True, output_format='shp')
        ret = ops.execute()
        import ipdb;
        ipdb.set_trace()

    def write_RCM3(self):
        rd = self.oblique_mercator
        ds = nc.Dataset(rd.uri)
        path = os.path.join(tempfile.mkdtemp(prefix='RCM3'), 'RCM3.shp')
        crs = fiona.crs.from_epsg(4326)
        driver = 'ESRI Shapefile'
        schema = {'geometry': 'Point',
                  'properties': {}}
        #        path = os.path.join(tempfile.mkdtemp(prefix='RCM3'),'RCM3.shp')
        #        polygon = Polygon(coordinates)
        #            feature = {'id':feature_idx,'properties':{},'geometry':mapping(polygon)}
        #            f.write(feature)

        #    with fiona.open(out_path,'w',driver=driver,crs=crs,schema=schema) as f:
        try:
            lats = ds.variables['lat'][:]
            lons = ds.variables['lon'][:] - 360
            n = lons.shape[0] * lons.shape[1]
            print n
            with fiona.open(path, 'w', driver=driver, crs=crs, schema=schema) as f:
                for ctr, (ii, jj) in enumerate(iter_array(lats, use_mask=False)):
                    if ctr % 100 == 0:
                        print ctr, n
                    point = Point(lons[ii, jj], lats[ii, jj])
                    feature = {'properties': {}, 'geometry': mapping(point)}
                    f.write(feature)
            import ipdb;
            ipdb.set_trace()
        finally:
            ds.close()

    def test_ECP2(self):
        ops = ocgis.OcgOperations(dataset=self.ecp2, output_format='shp', snippet=True)
        ret = ops.execute()

    def test_polar_stereographic(self):
        proj = get_projection(nc.Dataset(self.polar_stereographic.uri, 'r'))
        self.assertEqual(type(proj), PolarStereographic)

        ocgis.env.OVERWRITE = True
        ops = ocgis.OcgOperations(dataset=self.polar_stereographic, output_format='shp',
                                  snippet=True)
        ret = ops.execute()
        print(ret)

    def test_oblique_mercator(self):
        proj = get_projection(nc.Dataset(self.oblique_mercator.uri, 'r'))
        self.assertEqual(type(proj), NarccapObliqueMercator)

    #        ocgis.env.OVERWRITE = True
    #        prefix = 'oblique_mercator'
    #        ops = ocgis.OcgOperations(dataset=self.oblique_mercator,output_format='shp',
    #                                  snippet=True,dir_output='/tmp',prefix=prefix)
    #        ret = ops.execute()

    def test_rotated_pole(self):
        ocgis.env.OVERWRITE = True
        proj = get_projection(nc.Dataset(self.rotated_pole.uri, 'r'))
        self.assertEqual(type(proj), RotatedPole)

        #        ops = ocgis.OcgOperations(dataset=self.rotated_pole,snippet=True,output_format='shp')
        #        ret = ops.execute()

        #        ops = ocgis.OcgOperations(dataset=self.rotated_pole,snippet=True,output_format='shp',
        #                                  geom=[-97.74278,30.26694])
        #        ret = ops.execute()

        ops = ocgis.OcgOperations(dataset=self.rotated_pole, snippet=True, output_format='shp',
                                  geom='state_boundaries', agg_selection=True)
        ret = ops.execute()

    def test_subset(self):
        rds = [
            self.polar_stereographic,
            self.oblique_mercator
        ]
        geom = ShpDataset('state_boundaries')
        dir_output = tempfile.mkdtemp()
        print(dir_output)
        for rd in rds:
            ops = ocgis.OcgOperations(dataset=rd, geom=geom, agg_selection=True,
                                      snippet=True, output_format='shp', dir_output=dir_output,
                                      prefix=os.path.split(rd.uri)[1])
            ret = ops.execute()
        shutil.rmtree(dir_output)
