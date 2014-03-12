import unittest
from ocgis.test.base import TestBase
from ocgis.api.request.nc import NcRequestDataset
import netCDF4 as nc
from ocgis.interface.base.crs import WGS84, CFWGS84, CFLambertConformal
import numpy as np
from datetime import datetime as dt
from ocgis.interface.base.dimension.spatial import SpatialGeometryPolygonDimension,\
    SpatialGeometryDimension, SpatialDimension
import fiona
from shapely.geometry.geo import shape
from ocgis.exc import EmptySubsetError, ImproperPolygonBoundsError,\
    DimensionNotFound
import datetime
from unittest.case import SkipTest
import ocgis
from ocgis.test.test_simple.test_simple import nc_scope, ToTest


class TestNcRequestDataset(TestBase):
    
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
        sdim = SpatialDimension(geom=geom,properties=attrs,crs=WGS84())
        return(sdim)
    
    def test_load_dtype_on_dimensions(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        with nc_scope(rd.uri) as ds:
            test_dtype_temporal = ds.variables['time'].dtype
            test_dtype_value = ds.variables['tas'].dtype
        self.assertEqual(field.temporal.dtype,test_dtype_temporal)
        self.assertEqual(field.variables['tas'].dtype,test_dtype_value)
        self.assertEqual(field.temporal.dtype,np.float64)

    def test_load(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')
        rd = NcRequestDataset(variable=ref_test['variable'],uri=uri)
        field = rd.get()
        ds = nc.Dataset(uri,'r')
        
        self.assertEqual(field.level,None)
        self.assertEqual(field.spatial.crs,WGS84())
        
        tv = field.temporal.value
        test_tv = ds.variables['time'][:]
        self.assertNumpyAll(tv,test_tv)
        self.assertNumpyAll(field.temporal.bounds,ds.variables['time_bnds'][:])
        
        tdt = field.temporal.value_datetime
        self.assertEqual(tdt[4],dt(2001,1,5,12))
        self.assertNumpyAll(field.temporal.bounds_datetime[1001],[dt(2003,9,29),dt(2003,9,30)])
        
        rv = field.temporal.value_datetime[100]
        rb = field.temporal.bounds_datetime[100]
        self.assertTrue(all([rv > rb[0],rv < rb[1]]))
        
        self.assertEqual(field.temporal.extent_datetime,(datetime.datetime(2001,1,1),datetime.datetime(2011,1,1)))
        
        ds.close()
        
    def test_multifile_load(self):
        uri = self.test_data.get_uri('narccap_pr_wrfg_ncep')
        rd = NcRequestDataset(uri,'pr')
        field = rd.get()
        self.assertEqual(field.temporal.extent_datetime,(datetime.datetime(1981, 1, 1, 0, 0), datetime.datetime(1991, 1, 1, 0, 0)))
        self.assertAlmostEqual(field.temporal.resolution,0.125)
        
    def test_load_dtype_fill_value(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        ## dtype and fill_value should be read from metadata. when accessed they
        ## should not load the value.
        self.assertEqual(field.variables['tas'].dtype,np.float32)
        self.assertEqual(field.variables['tas'].fill_value,np.float32(1e20))
        self.assertEqual(field.variables['tas']._value,None)
        
    def test_load_datetime_slicing(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')
        rd = NcRequestDataset(variable=ref_test['variable'],uri=uri)
        field = rd.get()
        
        field.temporal.value_datetime
        field.temporal.bounds_datetime
        
        slced = field[:,239,:,:,:]
        self.assertEqual(slced.temporal.value_datetime,np.array([dt(2001,8,28,12)]))
        self.assertNumpyAll(slced.temporal.bounds_datetime,np.array([dt(2001,8,28),dt(2001,8,29)]))
    
    def test_load_value_datetime_after_slicing(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')
        rd = NcRequestDataset(variable=ref_test['variable'],uri=uri)
        field = rd.get()
        slced = field[:,10:130,:,4:7,100:37]
        self.assertEqual(slced.temporal.value_datetime.shape,(120,))
    
    def test_load_bounds_datetime_after_slicing(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')
        rd = NcRequestDataset(variable=ref_test['variable'],uri=uri)
        field = rd.get()
        slced = field[:,10:130,:,4:7,100:37]
        self.assertEqual(slced.temporal.bounds_datetime.shape,(120,2))
        
    def test_load_slice(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')
        rd = NcRequestDataset(variable=ref_test['variable'],uri=uri)
        field = rd.get()
        ds = nc.Dataset(uri,'r')
        
        slced = field[:,56:345,:,:,:]
        self.assertNumpyAll(slced.temporal.value,ds.variables['time'][56:345])
        self.assertNumpyAll(slced.temporal.bounds,ds.variables['time_bnds'][56:345,:])
        to_test = ds.variables['tas'][56:345,:,:]
        to_test = np.ma.array(to_test.reshape(1,289,1,64,128),mask=False)
        self.assertNumpyAll(slced.variables['tas'].value,to_test)
        
        slced = field[:,2898,:,5,101]
        to_test = ds.variables['tas'][2898,5,101]
        to_test = np.ma.array(to_test.reshape(1,1,1,1,1),mask=False)
        with self.assertRaises(AttributeError):
            slced.variables['tas']._field._value
        self.assertNumpyAll(slced.variables['tas'].value,to_test)
        
        ds.close()
        
    def test_load_time_range(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')
        rd = NcRequestDataset(variable=ref_test['variable'],uri=uri,time_range=[dt(2005,2,15),dt(2007,4,18)])
        field = rd.get()
        self.assertEqual(field.temporal.value_datetime[0],dt(2005, 2, 15, 12, 0))
        self.assertEqual(field.temporal.value_datetime[-1],dt(2007, 4, 18, 12, 0))
        self.assertEqual(field.shape,(1,793,1,64,128))
        
    def test_load_time_region(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')
        ds = nc.Dataset(uri,'r')
        rd = NcRequestDataset(variable=ref_test['variable'],uri=uri,time_region={'month':[8]})
        field = rd.get()
        
        self.assertEqual(field.shape,(1,310,1,64,128))
        
        var = ds.variables['time']
        real_temporal = nc.num2date(var[:],var.units,var.calendar)
        select = [True if x.month == 8 else False for x in real_temporal]
        indices = np.arange(0,var.shape[0])[np.array(select)]
        self.assertNumpyAll(indices,field.temporal._src_idx)
        self.assertNumpyAll(field.temporal.value_datetime,real_temporal[indices])
        self.assertNumpyAll(field.variables['tas'].value.data.squeeze(),ds.variables['tas'][indices,:,:])

        bounds_temporal = nc.num2date(ds.variables['time_bnds'][indices,:],var.units,var.calendar)
        self.assertNumpyAll(bounds_temporal,field.temporal.bounds_datetime)
        
        ds.close()
        
    def test_load_time_region_with_years(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')
        ds = nc.Dataset(uri,'r')
        rd = NcRequestDataset(variable=ref_test['variable'],uri=uri,time_region={'month':[8],'year':[2008,2010]})
        field = rd.get()
        
        self.assertEqual(field.shape,(1,62,1,64,128))

        var = ds.variables['time']
        real_temporal = nc.num2date(var[:],var.units,var.calendar)
        select = [True if x.month == 8 and x.year in [2008,2010] else False for x in real_temporal]
        indices = np.arange(0,var.shape[0])[np.array(select)]
        self.assertNumpyAll(indices,field.temporal._src_idx)
        self.assertNumpyAll(field.temporal.value_datetime,real_temporal[indices])
        self.assertNumpyAll(field.variables['tas'].value.data.squeeze(),ds.variables['tas'][indices,:,:])

        bounds_temporal = nc.num2date(ds.variables['time_bnds'][indices,:],var.units,var.calendar)
        self.assertNumpyAll(bounds_temporal,field.temporal.bounds_datetime)
        
        ds.close()
        
    def test_load_geometry_subset(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')
        rd = NcRequestDataset(variable=ref_test['variable'],uri=uri,alias='foo')
        field = rd.get()
        
        states = self.get_2d_state_boundaries_sdim()
        ca = states[:,states.properties['STATE_NAME'] == 'California']
        self.assertTrue(ca.properties['STATE_NAME'] == 'California')
        ca.crs.unwrap(ca)
        ca = ca.geom.polygon.value[0,0]
        ca_sub = field.get_intersects(ca)
        self.assertEqual(ca_sub.shape,(1, 3650, 1, 5, 4))
        self.assertTrue(ca_sub.variables['foo'].value.mask.any())
        
        ca_sub = field.get_intersects(ca.envelope)
        self.assertEqual(ca_sub.shape,(1, 3650, 1, 5, 4))
        self.assertFalse(ca_sub.variables['foo'].value.mask.any())
        
        rd = NcRequestDataset(variable=ref_test['variable'],uri=uri,alias='foo',time_region={'year':[2007]})
        field = rd.get()
        ca_sub = field.get_intersects(ca)
        self.assertEqual(ca_sub.shape,(1, 365, 1, 5, 4))
        self.assertEqual(set([2007]),set([d.year for d in ca_sub.temporal.value_datetime]))
        
    def test_load_time_region_slicing(self):
        ref_test = self.test_data['cancm4_tas']
        uri = self.test_data.get_uri('cancm4_tas')
        
        rd = NcRequestDataset(variable=ref_test['variable'],uri=uri,alias='foo',
                              time_region={'month':[1,10],'year':[2011,2013]})
        with self.assertRaises(EmptySubsetError):
            rd.get()
            
        rd = NcRequestDataset(variable=ref_test['variable'],uri=uri,alias='foo',
                              time_region={'month':[1,10],'year':[2005,2007]})
        field = rd.get()
        sub = field[:,:,:,50,75]
        self.assertEqual(sub.shape,(1,124,1,1,1))
        self.assertEqual(sub.variables['foo'].value.shape,(1,124,1,1,1))
        
        field = rd.get()
        sub = field[:,:,:,50,75:77]
        sub2 = field[:,:,:,0,1]
        self.assertEqual(sub2.shape,(1, 124, 1, 1, 1))
        
    def test_load_remote(self):
        raise(SkipTest("server IO errors"))
        uri = 'http://cida.usgs.gov/thredds/dodsC/maurer/maurer_brekke_w_meta.ncml'
        variable = 'sresa1b_bccr-bcm2-0_1_Tavg'
        rd = NcRequestDataset(uri,variable,time_region={'month':[1,10],'year':[2011,2013]})
        field = rd.get()
        field.variables['sresa1b_bccr-bcm2-0_1_Tavg'].value
        values = field[:,:,:,50,75]
        to_test = values.variables['sresa1b_bccr-bcm2-0_1_Tavg'].value.compressed()
        
        ds = nc.Dataset('http://cida.usgs.gov/thredds/dodsC/maurer/maurer_brekke_w_meta.ncml','r')
        try:
            values = ds.variables['sresa1b_bccr-bcm2-0_1_Tavg'][:,50,75]
            times = nc.num2date(ds.variables['time'][:],ds.variables['time'].units,ds.variables['time'].calendar)
            select = np.array([True if time in list(field.temporal.value_datetime) else False for time in times])
            sel_values = values[select,:,:]
            self.assertNumpyAll(to_test,sel_values)
        finally:
            ds.close()
            
    def test_load_with_projection(self):
        uri = self.test_data.get_uri('narccap_wrfg')
        rd = NcRequestDataset(uri,'pr')
        field = rd.get()
        self.assertIsInstance(field.spatial.crs,CFLambertConformal)
        field.spatial.update_crs(CFWGS84())
        self.assertIsInstance(field.spatial.crs,CFWGS84)
        self.assertEqual(field.spatial.grid.row,None)
        self.assertAlmostEqual(field.spatial.grid.value.mean(),-26.269666952512416)
        field.spatial.crs.unwrap(field.spatial)
        self.assertAlmostEqual(field.spatial.grid.value.mean(),153.73033304748759)
        with self.assertRaises(ImproperPolygonBoundsError):
            field.spatial.geom.polygon
        self.assertAlmostEqual(field.spatial.geom.point.value[0,100].x,278.52630062012787)
        self.assertAlmostEqual(field.spatial.geom.point.value[0,100].y,21.4615681252577)
        
    def test_load_projection_axes(self):
        uri = self.test_data.get_uri('cmip3_extraction')
        variable = 'Tavg'
        rd = NcRequestDataset(uri,variable)
        with self.assertRaises(DimensionNotFound):
            rd.get()
        rd = NcRequestDataset(uri,variable,dimension_map={'R':'projection','T':'time','X':'longitude','Y':'latitude'})
        field = rd.get()
        self.assertEqual(field.shape,(36, 1800, 1, 7, 12))
        self.assertEqual(field.temporal.value_datetime[0],datetime.datetime(1950, 1, 16, 0, 0))
        self.assertEqual(field.temporal.value_datetime[-1],datetime.datetime(2099, 12, 15, 0, 0))
        self.assertEqual(field.level,None)
        self.assertNumpyAll(field.realization.value,np.arange(1,37))
        
        ds = nc.Dataset(uri,'r')
        to_test = ds.variables['Tavg']
        self.assertNumpyAll(to_test[:],field.variables['Tavg'].value.squeeze().data)
        ds.close()
        
    def test_load_projection_axes_slicing(self):
        uri = self.test_data.get_uri('cmip3_extraction')
        variable = 'Tavg'
        rd = NcRequestDataset(uri,variable,dimension_map={'R':'projection','T':'time','X':'longitude','Y':'latitude'})
        field = rd.get()
        sub = field[15,:,:,:,:]
        self.assertEqual(sub.shape,(1,1800,1,7,12))
        
        ds = nc.Dataset(uri,'r')
        to_test = ds.variables['Tavg']
        self.assertNumpyAll(to_test[15,:,:,:],sub.variables[variable].value.squeeze().data)
        ds.close()
        
    def test_load_climatology_bounds(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd,output_format='nc',geom='state_boundaries',
                                  select_ugid=[27],calc=[{'func':'mean','name':'mean'}],
                                  calc_grouping=['month'])
        ret = ops.execute()
        rd = NcRequestDataset(uri=ret,variable='mean_tas')
        field = rd.get()
        self.assertNotEqual(field.temporal.bounds,None)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
