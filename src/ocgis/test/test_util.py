from ocgis.test.base import TestBase
from ocgis.interface.shp import ShpDataset
from ocgis.util.spatial.wrap import unwrap_geoms, wrap_geoms
import numpy as np


class TestSpatialWrap(TestBase):
    axes = [-10.0,-5.0,0.0,5.0,10]

    def test_unwrap(self):
        sd = ShpDataset('state_boundaries')
        geoms = sd.spatial.geom
        for axis in self.axes:
            for new_geom in unwrap_geoms(geoms.flat,axis=axis):
                bounds = np.array(new_geom.bounds)
                self.assertFalse(np.any(bounds < axis))
                
    def test_wrap(self):
        sd = ShpDataset('state_boundaries')
        geoms = sd.spatial.geom
        for axis in self.axes:
            unwrapped = np.array(list(unwrap_geoms(geoms.flat,axis=axis)))
            for idx,new_geom in enumerate(wrap_geoms(unwrapped.flat,axis=axis)):
                self.assertFalse(unwrapped[idx].equals(new_geom))
                self.assertTrue(geoms[idx].almost_equals(new_geom))
