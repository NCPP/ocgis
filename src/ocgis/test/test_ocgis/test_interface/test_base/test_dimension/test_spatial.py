import unittest
import numpy as np
from ocgis.interface.base.dimension.spatial import SpatialDimension,\
    SpatialGeometryDimension, SpatialGeometryPolygonDimension,\
    SpatialGridDimension, SpatialGeometryPointDimension
from ocgis.util.helpers import iter_array, make_poly, get_interpolated_bounds,\
    get_date_list
import fiona
from fiona.crs import from_epsg
from shapely.geometry import shape, mapping
from shapely.geometry.point import Point
from ocgis.exc import EmptySubsetError, ImproperPolygonBoundsError
from ocgis.test.base import TestBase
from ocgis.interface.base.crs import CoordinateReferenceSystem
from ocgis.interface.base.dimension.base import VectorDimension
from ocgis.test.test_simple.test_simple import ToTest
import datetime


class TestSpatialBase(TestBase):
    
    def get_2d_state_boundaries(self):
        geoms = []
        build = True
        with fiona.open('/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/bin/shp/state_boundaries/state_boundaries.shp','r') as source:
            for ii,row in enumerate(source):
                if build:
                    nrows = len(source)
                    dtype = []
                    for k,v in source.schema['properties'].iteritems():
                        if v.startswith('str'):
                            v = str('|S{0}'.format(v.split(':')[1]))
                        else:
                            v = getattr(np,v.split(':')[0])
                        dtype.append((str(k),v))
                    fill = np.empty(nrows,dtype=dtype)
                    ref_names = fill.dtype.names
                    build = False
                fill[ii] = tuple([row['properties'][n] for n in ref_names])
                geoms.append(shape(row['geometry']))
        geoms = np.atleast_2d(geoms)
        return(geoms,fill)
    
    def get_2d_state_boundaries_sdim(self):
        geoms,attrs = self.get_2d_state_boundaries()
        poly = SpatialGeometryPolygonDimension(value=geoms)
        geom = SpatialGeometryDimension(polygon=poly)
        sdim = SpatialDimension(geom=geom,attrs=attrs)
        return(sdim)
    
    def get_col(self,bounds=True):
        value = [-100.,-99.,-98.,-97.]
        if bounds:
            bounds = [[v-0.5,v+0.5] for v in value]
        else:
            bounds = None
        row = VectorDimension(value=value,bounds=bounds,name='col')
        return(row)
    
    def get_row(self,bounds=True):
        value = [40.,39.,38.]
        if bounds:
            bounds = [[v+0.5,v-0.5] for v in value]
        else:
            bounds = None
        row = VectorDimension(value=value,bounds=bounds,name='row')
        return(row)
    
    def get_sdim(self,bounds=True):
        row = self.get_row(bounds=bounds)
        col = self.get_col(bounds=bounds)
        sdim = SpatialDimension(row=row,col=col)
        return(sdim)
    
    def write_sdim(self):
        sdim = self.get_sdim(bounds=True)
        crs = from_epsg(4326)
        schema = {'geometry':'Polygon','properties':{'UID':'int:8'}}
        with fiona.open('/tmp/test.shp','w',driver='ESRI Shapefile',crs=crs,schema=schema) as sink:
            for ii,poly in enumerate(sdim.geom.polygon.value.flat):
                row = {'geometry':mapping(poly),
                       'properties':{'UID':int(sdim.geom.uid.flatten()[ii])}}
                sink.write(row)


class TestSpatialDimension(TestSpatialBase):
    
    def test_get_interpolated_bounds(self):
        
        sdim = self.get_sdim(bounds=False)
        test_sdim = self.get_sdim(bounds=True)
        
        row_bounds = get_interpolated_bounds(sdim.grid.row.value)
        col_bounds = get_interpolated_bounds(sdim.grid.col.value)
        
        self.assertNumpyAll(row_bounds,test_sdim.grid.row.bounds)
        self.assertNumpyAll(col_bounds,test_sdim.grid.col.bounds)
        
        across_180 = np.array([-180,-90,0,90,180],dtype=float)
        bounds_180 = get_interpolated_bounds(across_180)
        self.assertEqual(bounds_180.tostring(),'\x00\x00\x00\x00\x00 l\xc0\x00\x00\x00\x00\x00\xe0`\xc0\x00\x00\x00\x00\x00\xe0`\xc0\x00\x00\x00\x00\x00\x80F\xc0\x00\x00\x00\x00\x00\x80F\xc0\x00\x00\x00\x00\x00\x80F@\x00\x00\x00\x00\x00\x80F@\x00\x00\x00\x00\x00\xe0`@\x00\x00\x00\x00\x00\xe0`@\x00\x00\x00\x00\x00 l@')
        
        dates = get_date_list(datetime.datetime(2000,1,31),datetime.datetime(2002,12,31),1)
        with self.assertRaises(NotImplementedError):
            get_interpolated_bounds(np.array(dates))
        
        with self.assertRaises(ValueError):    
            get_interpolated_bounds(np.array([0],dtype=float))
            
        just_two = get_interpolated_bounds(np.array([50,75],dtype=float))
        self.assertEqual(just_two.tostring(),'\x00\x00\x00\x00\x00\xc0B@\x00\x00\x00\x00\x00@O@\x00\x00\x00\x00\x00@O@\x00\x00\x00\x00\x00\xe0U@')
        
        just_two_reversed = get_interpolated_bounds(np.array([75,50],dtype=float))
        self.assertEqual(just_two_reversed.tostring(),'\x00\x00\x00\x00\x00\xe0U@\x00\x00\x00\x00\x00@O@\x00\x00\x00\x00\x00@O@\x00\x00\x00\x00\x00\xc0B@')

        zero_origin = get_interpolated_bounds(np.array([0,50,100],dtype=float))
        self.assertEqual(zero_origin.tostring(),'\x00\x00\x00\x00\x00\x009\xc0\x00\x00\x00\x00\x00\x009@\x00\x00\x00\x00\x00\x009@\x00\x00\x00\x00\x00\xc0R@\x00\x00\x00\x00\x00\xc0R@\x00\x00\x00\x00\x00@_@')
                
    def test_get_clip(self):
        sdim = self.get_sdim(bounds=True)
        poly = make_poly((37.75,38.25),(-100.25,-99.75))
        ret = sdim.get_clip(poly)
        
        self.assertEqual(ret.uid,np.array([[9]]))
        self.assertTrue(poly.almost_equals(ret.geom.polygon.value[0,0]))
        
        self.assertNumpyAll(ret.geom.point.value.shape,ret.geom.polygon.shape)
        ref_pt = ret.geom.point.value[0,0]
        ref_poly = ret.geom.polygon.value[0,0]
        self.assertTrue(ref_poly.intersects(ref_pt))
        
    def test_get_geom_iter(self):
        sdim = self.get_sdim(bounds=True)
        tt = list(sdim.get_geom_iter())
        ttt = list(tt[4])
        ttt[2] = ttt[2].bounds
        self.assertEqual(ttt,[1, 0, (-100.5, 38.5, -99.5, 39.5),5])
        
        sdim = self.get_sdim(bounds=False)
        tt = list(sdim.get_geom_iter(target='point'))
        ttt = list(tt[4])
        ttt[2] = [ttt[2].x,ttt[2].y]
        self.assertEqual(ttt,[1, 0, [-100.0, 39.0],5])
        
        sdim = self.get_sdim(bounds=False)
        self.assertEqual(sdim.abstraction,'polygon')
        with self.assertRaises(ImproperPolygonBoundsError):
            list(sdim.get_geom_iter(target='polygon'))
        
    def test_get_intersects_polygon_small(self):
        for b in [True,False]:
            sdim = self.get_sdim(bounds=b)
            poly = make_poly((37.75,38.25),(-100.25,-99.75))
            ret = sdim.get_intersects(poly)
            to_test = np.ma.array([[[38]],[[-100]]],mask=False)
            self.assertNumpyAll(ret.grid.value,to_test)
            self.assertNumpyAll(ret.uid,np.ma.array([[9]]))
            self.assertEqual(ret.shape,(1,1))
            to_test = ret.geom.point.value.compressed()[0]
            self.assertTrue(to_test.almost_equals(Point(-100,38)))
            if b is False:
                with self.assertRaises(ImproperPolygonBoundsError):
                    ret.geom.polygon
            else:
                to_test = ret.geom.polygon.value.compressed()[0].bounds
                self.assertEqual((-100.5,37.5,-99.5,38.5),to_test)
                
    def test_get_intersects_polygon_no_point_overlap(self):
        for b in [True,False]:
            sdim = self.get_sdim(bounds=b)
            poly = make_poly((39.25,39.75),(-97.75,-97.25))
            if b is False:
                with self.assertRaises(EmptySubsetError):
                    sdim.get_intersects(poly)
            else:
                ret = sdim.get_intersects(poly)
                self.assertEqual(ret.shape,(2,2))

    def test_get_intersects_polygon_all(self):
        for b in [True,False]:
            sdim = self.get_sdim(bounds=b)
            poly = make_poly((37,41),(-101,-96))
            ret = sdim.get_intersects(poly)
            self.assertNumpyAll(sdim.grid.value,ret.grid.value)
            self.assertNumpyAll(sdim.grid.value.mask[0,:,:],sdim.geom.point.value.mask)
            self.assertEqual(ret.shape,(3,4))
            
    def test_get_intersects_polygon_empty(self):
        for b in [True,False]:
            sdim = self.get_sdim(bounds=b)
            poly = make_poly((1000,1001),(-1000,-1001))
            with self.assertRaises(EmptySubsetError):
                sdim.get_intersects(poly)
    
    def test_state_boundaries_weights(self):
        geoms,attrs = self.get_2d_state_boundaries()
        poly = SpatialGeometryPolygonDimension(value=geoms)
        geom = SpatialGeometryDimension(polygon=poly)
        sdim = SpatialDimension(geom=geom)
        ref = sdim.weights
        self.assertEqual(ref[0,50],1.0)
        self.assertAlmostEqual(sdim.weights.mean(),0.07744121084026262)
    
    def test_geom_mask_by_polygon(self):
        geoms,properties = self.get_2d_state_boundaries()
        spdim = SpatialGeometryPolygonDimension(value=geoms)
        ref = spdim.value.mask
        self.assertEqual(ref.shape,(1,51))
        self.assertFalse(ref.any())
        select = properties['STATE_ABBR'] == 'NE'
        subset_polygon = geoms[:,select][0,0]
        
        msked = spdim.get_intersects_masked(subset_polygon)

        self.assertEqual(msked.value.mask.sum(),50)
        self.assertTrue(msked.value.compressed()[0].almost_equals(subset_polygon))
        
        with self.assertRaises(NotImplementedError):
            msked = spdim.get_intersects_masked(subset_polygon.centroid)
            self.assertTrue(msked.value.compressed()[0].almost_equals(subset_polygon))
        
        with self.assertRaises(EmptySubsetError):
            spdim.get_intersects_masked(Point(1000,1000).buffer(1))
            
    def test_update_crs(self):
        geoms,properties = self.get_2d_state_boundaries()
        crs = CoordinateReferenceSystem(epsg=4326)
        spdim = SpatialGeometryPolygonDimension(value=geoms)
        sdim = SpatialDimension(geom=SpatialGeometryDimension(polygon=spdim),properties=properties,
                                crs=crs)
        to_crs = CoordinateReferenceSystem(epsg=2163)
        sdim.update_crs(to_crs)
        
    def test_update_crs_with_grid(self):
        for b in [True,False]:
            sdim = self.get_sdim(bounds=b)
            orig = sdim.grid.value.copy()
            sdim.crs = CoordinateReferenceSystem(epsg=4326)
            to_crs = CoordinateReferenceSystem(epsg=2163)
            sdim.update_crs(to_crs)
            self.assertNumpyNotAll(sdim.grid.value,orig)
            self.assertEqual(sdim.grid.row,None)

    def test_grid_value(self):
        for b in [True,False]:
            row = self.get_row(bounds=b)
            col = self.get_col(bounds=b)
            sdim = SpatialDimension(row=row,col=col)
            col_test,row_test = np.meshgrid(col.value,row.value)
            self.assertNumpyAll(sdim.grid.value[0].data,row_test)
            self.assertNumpyAll(sdim.grid.value[1].data,col_test)
            self.assertFalse(sdim.grid.value.mask.any())
            try:
                ret = sdim.get_grid_bounds()
                self.assertEqual(ret.shape,(3,4,4))
                self.assertFalse(ret.mask.any())
            except ImproperPolygonBoundsError:
                if b is False:
                    pass
                else:
                    raise
                
    def test_grid_slice_all(self):
        sdim = self.get_sdim(bounds=True)
        slc = sdim[:]
        self.assertNumpyAll(sdim.grid.value,slc.grid.value)
    
    def test_grid_slice_1d(self):
        sdim = self.get_sdim(bounds=True)
        sdim_slc = sdim.grid[0,:]
        self.assertEqual(sdim_slc.value.shape,(2,1,4))
        self.assertNumpyAll(sdim_slc.value,np.ma.array([[[40,40,40,40]],[[-100,-99,-98,-97]]],mask=False))
        self.assertEqual(sdim_slc.row.value[0],40)
        self.assertNumpyAll(sdim_slc.col.value,np.array([-100,-99,-98,-97]))
    
    def test_grid_slice_2d(self):
        sdim = self.get_sdim(bounds=True)
        sdim_slc = sdim.grid[0,1]
        self.assertNumpyAll(sdim_slc.value,np.ma.array([[[40]],[[-99]]],mask=False))
        self.assertNumpyAll(sdim_slc.row.bounds,np.array([[40.5,39.5]]))
        self.assertEqual(sdim_slc.col.value[0],-99)
    
    def test_grid_slice_2d_range(self):
        sdim = self.get_sdim(bounds=True)
        sdim_slc = sdim.grid[1:3,0:3]
        self.assertNumpyAll(sdim_slc.value,np.ma.array([[[39,39,39],[38,38,38]],[[-100,-99,-98],[-100,-99,-98]]],mask=False))
        self.assertNumpyAll(sdim_slc.row.value,np.array([39,38]))
        
    def test_geom_point(self):
        sdim = self.get_sdim(bounds=True)
        with self.assertRaises(AttributeError):
            sdim.geom.value
        pt = sdim.geom.point.value
        fill = np.ma.array(np.zeros((2,3,4)),mask=False)
        for idx_row,idx_col in iter_array(pt):
            fill[0,idx_row,idx_col] = pt[idx_row,idx_col].y
            fill[1,idx_row,idx_col] = pt[idx_row,idx_col].x
        self.assertNumpyAll(fill,sdim.grid.value)
        
    def test_geom_polygon_no_bounds(self):
        sdim = self.get_sdim(bounds=False)
        with self.assertRaises(ImproperPolygonBoundsError):
            sdim.geom.polygon.value
            
    def test_geom_polygon_bounds(self):
        sdim = self.get_sdim(bounds=True)
        poly = sdim.geom.polygon.value
        fill = np.ma.array(np.zeros((2,3,4)),mask=False)
        for idx_row,idx_col in iter_array(poly):
            fill[0,idx_row,idx_col] = poly[idx_row,idx_col].centroid.y
            fill[1,idx_row,idx_col] = poly[idx_row,idx_col].centroid.x
        self.assertNumpyAll(fill,sdim.grid.value)   
        
    def test_grid_shape(self):
        sdim = self.get_sdim()
        shp = sdim.grid.shape
        self.assertEqual(shp,(3,4))
        
    def test_empty(self):
        with self.assertRaises(ValueError):
            SpatialDimension()
            
    def test_geoms_only(self):
        geoms = []
        with fiona.open('/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/bin/shp/state_boundaries/state_boundaries.shp','r') as source:
            for row in source:
                geoms.append(shape(row['geometry']))
        geoms = np.atleast_2d(geoms)
        poly_dim = SpatialGeometryPolygonDimension(value=geoms)
        sg_dim = SpatialGeometryDimension(polygon=poly_dim)
        sdim = SpatialDimension(geom=sg_dim)
        self.assertEqual(sdim.shape,(1,51))
        
    def test_slicing(self):
        sdim = self.get_sdim(bounds=True)
        self.assertEqual(sdim.shape,(3,4))
        self.assertEqual(sdim._geom,None)
        self.assertEqual(sdim.geom.point.shape,(3,4))
        self.assertEqual(sdim.geom.polygon.shape,(3,4))
        self.assertEqual(sdim.grid.shape,(3,4))
        with self.assertRaises(IndexError):
            sdim[0]
        sdim_slc = sdim[0,1]
        self.assertEqual(sdim_slc.shape,(1,1))
        self.assertEqual(sdim_slc.uid,np.array([[2]],dtype=np.int32))
        self.assertNumpyAll(sdim_slc.grid.value,np.ma.array([[[40]],[[-99]]],mask=False))
        self.assertNotEqual(sdim_slc,None)
        to_test = sdim_slc.geom.point.value[0,0].y,sdim_slc.geom.point.value[0,0].x
        self.assertEqual((40.0,-99.0),(to_test))
        to_test = sdim_slc.geom.polygon.value[0,0].centroid.y,sdim_slc.geom.polygon.value[0,0].centroid.x
        self.assertEqual((40.0,-99.0),(to_test))
        
        refs = [sdim_slc.geom.point.value,sdim_slc.geom.polygon.value]
        for ref in refs:
            self.assertIsInstance(ref,np.ma.MaskedArray)
        
        sdim_all = sdim[:,:]
        self.assertNumpyAll(sdim_all.grid.value,sdim.grid.value)
        
    def test_slicing_1d_none(self):
        sdim = self.get_sdim(bounds=True)
        sdim_slc = sdim[1,:]
        self.assertEqual(sdim_slc.shape,(1,4))
        
    def test_point_as_value(self):
        pt = Point(100.0,10.0)
        pt2 = Point(200.0,20.0)
        with self.assertRaises(ValueError):
            SpatialGeometryPointDimension(value=Point(100.0,10.0))
        with self.assertRaises(ValueError):
            SpatialGeometryPointDimension(value=[pt,pt])
        
        pts = np.array([[pt,pt2]],dtype=object)
        g = SpatialGeometryPointDimension(value=pts)
        self.assertEqual(g.value.mask.any(),False)
        self.assertNumpyAll(g.uid,np.ma.array([[1,2]]))
        
        sgdim = SpatialGeometryDimension(point=g)
        sdim = SpatialDimension(geom=sgdim)
        self.assertEqual(sdim.shape,(1,2))
        self.assertNumpyAll(sdim.uid,np.ma.array([[1,2]]))
        sdim_slc = sdim[:,1]
        self.assertEqual(sdim_slc.shape,(1,1))
        self.assertTrue(sdim_slc.geom.point.value[0,0].almost_equals(pt2))
        
    def test_grid_get_subset_bbox(self):
        for b in [True,False]:
            sdim = self.get_sdim(bounds=b)
            bg = sdim.grid.get_subset_bbox(-99,39,-98,39,closed=False)
            self.assertEqual(bg._value,None)
            self.assertEqual(bg.uid.shape,(1,2))
            self.assertNumpyAll(bg.uid,np.ma.array([[6,7]]))
            with self.assertRaises(EmptySubsetError):
                sdim.grid.get_subset_bbox(1000,1000,1001,10001)
                
            bg2 = sdim.grid.get_subset_bbox(-99999,1,1,1000)
            self.assertNumpyAll(bg2.value,sdim.grid.value)
            
    def test_weights(self):
        for b in [True,False]:
            sdim = self.get_sdim(bounds=b)
            ref = sdim.weights
            self.assertEqual(ref.mean(),1.0)
            self.assertFalse(ref.mask.any())
            
    def test_singletons(self):
        row = VectorDimension(value=10,name='row')
        col = VectorDimension(value=100,name='col')
        grid = SpatialGridDimension(row=row,col=col,name='grid')
        self.assertNumpyAll(grid.value,np.ma.array([[[10]],[[100]]],mask=False))
        sdim = SpatialDimension(grid=grid)
        to_test = sdim.geom.point.value[0,0].y,sdim.geom.point.value[0,0].x
        self.assertEqual((10.0,100.0),(to_test))


class TestSpatialGridDimension(TestSpatialBase):
    
    def test_grid_without_row_and_column(self):
        row = np.arange(39,42.5,0.5)
        col = np.arange(-104,-95,0.5)
        x,y = np.meshgrid(col,row)
        value = np.zeros([2]+list(x.shape))
        value = np.ma.array(value,mask=False)
        value[0,:,:] = y
        value[1,:,:] = x
        minx,miny,maxx,maxy = x.min(),y.min(),x.max(),y.max()
        grid = SpatialGridDimension(value=value)
        sub = grid.get_subset_bbox(minx,miny,maxx,maxy,closed=False)
        self.assertNumpyAll(sub.value,value)
    
    def test_load_from_source_grid_slicing(self):
        row = VectorDimension(src_idx=[10,20,30,40],name='row',data='foo')
        self.assertEqual(row.name,'row')
        col = VectorDimension(src_idx=[100,200,300],name='col',data='foo')
        grid = SpatialGridDimension(row=row,col=col,name='grid')
        self.assertEqual(grid.shape,(4,3))
        grid_slc = grid[1,2]
        self.assertEqual(grid_slc.shape,(1,1))
        with self.assertRaises(NotImplementedError):
            grid_slc.value
        with self.assertRaises(NotImplementedError):
            grid_slc.row.bounds
        self.assertNumpyAll(grid_slc.row._src_idx,np.array([20]))
        self.assertNumpyAll(grid_slc.col._src_idx,np.array([300]))
        self.assertEqual(grid_slc.row.name,'row')
        self.assertEqual(grid_slc.uid,np.array([[6]],dtype=np.int32))
        
    def test_singletons(self):
        row = VectorDimension(value=10,name='row')
        col = VectorDimension(value=100,name='col')
        grid = SpatialGridDimension(row=row,col=col,name='grid')
        self.assertNumpyAll(grid.value,np.ma.array([[[10]],[[100]]],mask=False))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
