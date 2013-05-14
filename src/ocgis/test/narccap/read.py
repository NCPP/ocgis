import unittest
import ocgis
from ocgis.interface.projection import get_projection, PolarStereographic,\
    NarccapObliqueMercator, RotatedPole
import netCDF4 as nc
from ocgis.interface.shp import ShpDataset
import tempfile
import os.path
import shutil


class NarccapTestBase(unittest.TestCase):
    
    def setUp(self):
        ocgis.env.DIR_DATA = '/usr/local/climate_data/narccap'
        self.polar_stereographic = ocgis.RequestDataset(uri='pr_CRCM_ccsm_1981010103.nc',
                                                        variable='pr')
        self.oblique_mercator = ocgis.RequestDataset(uri='pr_RCM3_gfdl_1981010103.nc',
                                                     variable='pr')
        self.rotated_pole = ocgis.RequestDataset(uri='pr_HRM3_gfdl_1981010103.nc',
                                                 variable='pr')
        
    def tearDown(self):
        ocgis.env.reset()
        
        
class Test(NarccapTestBase):
    
    def test_polar_stereographic(self):
        proj = get_projection(nc.Dataset(self.polar_stereographic.uri,'r'))
        self.assertEqual(type(proj),PolarStereographic)
        
        ocgis.env.OVERWRITE = True
        ops = ocgis.OcgOperations(dataset=self.polar_stereographic,output_format='shp',
                                  snippet=True)
        ret = ops.execute()
        print(ret)

    def test_oblique_mercator(self):
        proj = get_projection(nc.Dataset(self.oblique_mercator.uri,'r'))
        self.assertEqual(type(proj),NarccapObliqueMercator)
        
#        ocgis.env.OVERWRITE = True
#        prefix = 'oblique_mercator'
#        ops = ocgis.OcgOperations(dataset=self.oblique_mercator,output_format='shp',
#                                  snippet=True,dir_output='/tmp',prefix=prefix)
#        ret = ops.execute()

    def test_rotated_pole(self):
        ocgis.env.OVERWRITE = True
        proj = get_projection(nc.Dataset(self.rotated_pole.uri,'r'))
        self.assertEqual(type(proj),RotatedPole)
        
#        ops = ocgis.OcgOperations(dataset=self.rotated_pole,snippet=True,output_format='shp')
#        ret = ops.execute()
        
        ops = ocgis.OcgOperations(dataset=self.rotated_pole,snippet=True,output_format='shp',
                                  geom=[-97.74278,30.26694])
        ret = ops.execute()
        import ipdb;ipdb.set_trace()

    def test_subset(self):
        rds = [
               self.polar_stereographic,
               self.oblique_mercator
               ]
        geom = ShpDataset('state_boundaries')
        dir_output = tempfile.mkdtemp()
        print(dir_output)
        for rd in rds:
            ops = ocgis.OcgOperations(dataset=rd,geom=geom,agg_selection=True,
                   snippet=True,output_format='shp',dir_output=dir_output,
                   prefix=os.path.split(rd.uri)[1])
            ret = ops.execute()
        import ipdb;ipdb.set_trace()
        shutil.rmtree(dir_output)