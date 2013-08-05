import unittest
from ocgis.test.base import TestBase
import numpy as np
import netCDF4 as nc
from ocgis.util.helpers import make_poly
from ocgis.interface.projection import WGS84
import datetime
from ocgis.util.shp_cabinet import ShpCabinet
from ocgis.interface.shp import ShpDataset
from shapely.geometry.point import Point
from ocgis.interface.nc.dimension import NcRowDimension, NcColumnDimension,\
    NcGridDimension, NcSpatialDimension, NcPolygonDimension, NcPointDimension
from ocgis.interface.nc.dataset import NcDataset
from shapely.geometry.multipolygon import MultiPolygon
from ocgis.interface.geometry import GeometryDataset
from ocgis import env
import os.path


class TestNcDataset(TestBase):
    
    def test_row_dimension(self):
        value = np.arange(30,40,step=0.5)
        value = np.flipud(value).copy()
        bounds = np.empty((value.shape[0],2))
        bounds[:,0] = value + 0.25
        bounds[:,1] = value - 0.25
        
        ri = NcRowDimension(value=value)
        self.assertTrue(np.all(value == ri.value))
        self.assertEqual(ri.bounds,None)
        sri = ri.subset(35,38)
        self.assertEqual(len(sri.value),len(sri.uid))
        self.assertTrue(np.all(sri.value >= 35))
        self.assertTrue(np.all(sri.value <= 38))
        self.assertEqual(id(ri.value.base),id(sri.value.base))
        
        ri = NcRowDimension(value=value,bounds=bounds)
        self.assertTrue(np.all(bounds == ri.bounds))
        sri = ri.subset(30.80,38.70)
        self.assertTrue(np.all(sri.value >= 30.8))
        self.assertTrue(np.all(sri.value <= 38.7))
        
        with self.assertRaises(ValueError):
            NcRowDimension(bounds=bounds)
        
        ri = NcRowDimension(value=value)
        self.assertEqual(ri.extent,(value.min(),value.max()))
        
        ri = NcRowDimension(value=value,bounds=bounds)
        self.assertEqual(ri.extent,(bounds.min(),bounds.max()))
        self.assertTrue(np.all(ri.uid == np.arange(1,21)))
        self.assertEqual(ri.resolution,0.5)
        
    def test_spatial_dimension(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ds = nc.Dataset(rd.uri,'r')
        row_data = ds.variables['lat'][:]
        row_bounds = ds.variables['lat_bnds'][:]
        col_data = ds.variables['lon'][:]
        col_bounds = ds.variables['lon_bnds'][:]
        
        rd = NcRowDimension(value=row_data,bounds=row_bounds)
        cd = NcColumnDimension(value=col_data,bounds=col_bounds)
        gd = NcGridDimension(row=rd,column=cd)
        self.assertEqual(gd.resolution,2.8009135133922354)
        
        sd = NcSpatialDimension(row=rd,column=cd)
        
        sgd = gd.subset()
        self.assertEqual(sgd.shape,(rd.shape[0],cd.shape[0]))
        poly = make_poly((-62,59),(87,244))
        sgd = gd.subset(polygon=poly)
        self.assertEqual(sgd.uid.shape,(sgd.row.shape[0],sgd.column.shape[0]))
        self.assertTrue(sum(sgd.shape) < sum(gd.shape))
        lgd = gd[0:5,0:5]
        self.assertEqual(lgd.shape,(5,5))
        
        vd = NcPolygonDimension(gd)
        
        self.assertEqual(vd.geom.shape,vd.grid.shape)
        ivd = vd.intersects(poly)
        self.assertTrue(sum(ivd.geom.shape) < sum(vd.geom.shape))
        self.assertEqual(ivd.weights.max(),1.0)
        
        cvd = vd.clip(poly)
        self.assertEqual(ivd.shape,cvd.shape)
        self.assertFalse(ivd.weights.sum() == cvd.weights.sum())
        ds.close()
        
    def test_load(self):
        rd = self.test_data.get_rd('cancm4_tas')
        gi = NcDataset(request_dataset=rd)
        
        spatial = gi.spatial
        self.assertTrue(spatial.is_360)
        self.assertEqual(spatial.grid.shape,(64,128))
        self.assertTrue(isinstance(spatial.projection,WGS84))
        
        level = gi.level
        self.assertEqual(level,None)
        
        temporal = gi.temporal
        self.assertEqual(str(temporal.extent),'(datetime.datetime(2001, 1, 1, 0, 0), datetime.datetime(2011, 1, 1, 0, 0))')
        res = temporal.resolution
        self.assertAlmostEqual(int(res),1)
        trng = [datetime.datetime(2001,1,1),datetime.datetime(2001,12,31,23,59)]
        new_ds = gi.get_subset(temporal=trng)
        self.assertEqual(new_ds.temporal.shape[0],365)
        self.assertEqual(new_ds.value.shape,(365,1,64,128))
        
    def test_initial_subset(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ods = NcDataset(request_dataset=rd)
        ods._load_slice.update({'T':slice(0,1)})
        tp = ods.temporal
        to_check = [tp.value,tp.uid,tp.real_idx]
        for tc in to_check: self.assertEqual(tc.shape,(1,))
        self.assertEqual(tp.bounds.shape,(1,2))
        self.assertEqual(ods.value.shape,(1,1,64,128))
        
    def test_aggregate(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ods = NcDataset(request_dataset=rd)
        sd = ShpDataset('state_boundaries')
        sd.spatial.unwrap_geoms()
        utah = sd[23].spatial.geom[0]
        ods = ods.get_subset(spatial_operation='clip',igeom=utah)
        ods.aggregate()
        self.assertEqual(ods.value.shape,(3650,1,1,1))
        self.assertEqual(ods.raw_value.shape,(3650,1,3,3))
        self.assertNotEqual(ods.spatial.vector.raw_weights.shape,
                            ods.spatial.vector.weights.shape)
        self.assertEqual(ods.spatial.vector.uid[0],1)

    def test_abstraction_point(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ods = NcDataset(request_dataset=rd,abstraction='point')
        with self.assertRaises(AttributeError):
            ods.spatial.shape
        self.assertTrue(isinstance(ods.spatial.vector,NcPointDimension))
        geom = ods.spatial.vector.geom
        self.assertEqual(geom.shape,(64,128))
        self.assertTrue(isinstance(geom[0,0],Point))
        weights = ods.spatial.vector.weights
        self.assertEqual(weights.max(),1.)
        poly = make_poly((-62,59),(87,244))
        nods = ods.get_subset(spatial_operation='intersects',igeom=poly)
        with self.assertRaises(AttributeError):
            nods.spatial.shape
        self.assertNotEqual(ods.spatial.vector.shape,nods.spatial.vector.shape)
        nods.aggregate()
        self.assertEqual(nods.spatial.vector.geom.shape,(1,1))
        self.assertTrue(isinstance(nods.spatial.vector.geom[0,0],Point))
        
    def test_masking(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ods = NcDataset(request_dataset=rd)
        
        nods = ods[0:31,0:5,0:5]
        geoms = nods.spatial.vector.geom
        igeom = [geoms[0,2]] + list(geoms[1,:])
        igeom = MultiPolygon(igeom)
        self.assertFalse(np.any(nods.value.mask))
        
        sods = ods.get_subset(spatial_operation='intersects',igeom=igeom)
        geom_mask = sods.spatial.vector.geom.mask
        value_mask = sods.value.mask[0,:,:,:]
        self.assertTrue(np.all(geom_mask == value_mask))
        
    def test_temporal_group(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ods = NcDataset(request_dataset=rd)
        grouping = ['month']
        ods.temporal.set_grouping(grouping)
        self.assertEqual(ods.temporal.group.value.shape[0],12)
        
        for row in ods.temporal.group.get_iter():
            pass
        
    def test_slice(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ods = NcDataset(request_dataset=rd)
        
        sods = ods[:,0:5,0:5]
        self.assertEqual(sods.value.shape,(3650,1,5,5))
        self.assertEqual(sods.spatial.vector.geom.shape,(5,5))
        
        sods = ods[0,0:5,0:5]
        self.assertEqual(sods.value.shape,(1,1,5,5))
        
        sods = ods[0,0,0]
        self.assertEqual(sods.value.shape,(1,1,1,1))
        
        grouping = ['month']
        ods.temporal.set_grouping(grouping)
        sods = ods[0:31,0:5,0:5]
        self.assertEqual(sods.temporal.group,None)
        sods.temporal.set_grouping(grouping)
        self.assertEqual(sods.temporal.group.value.shape[0],1)
        
    def test_project(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ods = NcDataset(request_dataset=rd)
        dest = WGS84()
        with self.assertRaises(NotImplementedError):
            ods.project(dest)
            
            
class TestIteration(TestBase):
    
    def test_get_iter(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ods = NcDataset(request_dataset=rd)[0:31,0:5,0:5]
            
        itr = ods.get_iter_value()
        self.assertEqual(len(list(itr)),31*5*5)


class TestShpDataset(TestBase):
    
    def test_load(self):
        sds = ShpDataset('state_boundaries')
        geom = sds.spatial.geom
        self.assertEqual(geom.shape,(51,))
        self.assertEqual(sds.temporal,None)
        self.assertEqual(sds.level,None)
        
        ssds = sds[0]
        self.assertEqual(ssds.spatial.uid,1)
        
    def test_filter(self):
        sds = ShpDataset('state_boundaries',select_ugid=[20])
        self.assertTrue(sds.spatial.geom.shape,(1,))
        
    def test_write(self):
        sds = ShpDataset('state_boundaries')
        path = os.path.join(env.DIR_OUTPUT,'foo.shp')
        sds.write(path)
        
    def test_point_load(self):
        sds = ShpDataset('qed_city_centroids')
        self.assertIsInstance(sds.spatial.geom[0],Point)
        self.assertEqual(sds.spatial.geom.shape[0],3)
        
        
class TestGeometryDataset(TestBase):
    
    @property
    def polygon(self):
        return(make_poly((-62,59),(87,244)))
    
    @property
    def point(self):
        return(Point(1,2))
    
    def test_load(self):
        gds = GeometryDataset(uid=[1],geom=[self.polygon])
        
    def test_iter(self):
        gds = GeometryDataset(uid=[1,2],geom=[self.point,self.point])
        for row in gds:
            self.assertEqual(row.spatial.geom.shape,(1,))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()