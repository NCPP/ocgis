import unittest
from datetime import datetime
import numpy as np
import netCDF4 as nc
from ocgis.api.operations import OcgOperations
from ocgis.api.iocg.interpreter_ocg import OcgInterpreter
from ocgis.util.shp_cabinet import ShpCabinet
from shapely.geometry.polygon import Polygon


class Test360(unittest.TestCase):

    def test_get_data(self):
        path = self.make_data()
        
        dataset = {'uri':path,'variable':'foo'}
        output_format = 'nc'
        geom = self.nebraska
#        geom = None
        
#        geom[0]['geom'] = self.transform_to_360(geom[0]['geom'])
#        sc = ShpCabinet()
#        sc.write(geom,'/tmp/transformed_ne.shp')
        
        ops = OcgOperations(dataset=dataset,output_format=output_format,geom=geom)
        
        ret = OcgInterpreter(ops).execute()
        
        import ipdb;ipdb.set_trace()
        
    @property
    def nebraska(self):
        sc = ShpCabinet()
        geom_dict = sc.get_geom_dict('state_boundaries',{'id':[16]})
        return(geom_dict)
        
    def transform_to_360(self,polygon):
        
        def _transform_lon_(ctup):
            lon = ctup[0]
            if lon < 180:
                lon += 360
            return([lon,ctup[1]])
        
        transformed = map(_transform_lon_,polygon.exterior.coords)
        new_polygon = Polygon(transformed)
        return(new_polygon)
        
    def make_bounds(self,arr,res):
        shift = float(res)/2
        lower = arr - shift
        upper = arr + shift
        bounds = np.hstack((lower.reshape(-1,1),upper.reshape(-1,1)))
        return(bounds)
    
    def make_variable(self,varname,arr,dimensions):
        var = self.ds.createVariable(varname,arr.dtype,dimensions=dimensions)
        var[:] = arr
        return(var)

    def make_data(self):
        path = '/tmp/test360 {0}.nc'.format(datetime.now())
        
        lat_values = np.arange(-85,95,10,dtype=float)
        latb_values = self.make_bounds(lat_values,10)
        lon_values = np.arange(5,365,10,dtype=float)
        lonb_values = self.make_bounds(lon_values,10)
        
        calendar = 'standard'
        units = 'days since 0000-01-01'
        time_values = [datetime(2000,m,15) for m in range(1,13)]
        time_values = nc.date2num(time_values,units,calendar=calendar)
        
        level_values = np.array([100,200])
        
        values = np.empty((len(time_values),len(level_values),len(lat_values),len(lon_values)))
        col_values = np.arange(0,len(lon_values))
        for ii in range(0,len(lat_values)):
            values[:,:,ii,:] = col_values
        values = np.ma.array(values,mask=False,fill_value=1e20)
        
        self.ds = nc.Dataset(path,'w')
        ds = self.ds
        
        d_lat = ds.createDimension('d_lat',size=len(lat_values))
        d_lon = ds.createDimension('d_lon',size=len(lon_values))
        d_bnds = ds.createDimension('d_bnds',size=2)
        d_level = ds.createDimension('d_level',size=len(level_values))
        d_time = ds.createDimension('d_time',size=len(time_values))
        
        self.make_variable('lat',lat_values,'d_lat')
        self.make_variable('lon',lon_values,'d_lon')
        self.make_variable('lat_bnds',latb_values,('d_lat','d_bnds'))
        self.make_variable('lon_bnds',lonb_values,('d_lon','d_bnds'))
        v_time = self.make_variable('time',time_values,'d_time')
        v_time.calendar = calendar
        v_time.units = units
        self.make_variable('level',level_values,'d_level')
        self.make_variable('foo',values,('d_time','d_level','d_lat','d_lon'))
        
        self.ds.close()
        
        return(path)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()