import os
from datetime import datetime
from tempfile import mkstemp

import netCDF4 as nc
import numpy as np
from shapely.geometry.polygon import Polygon

import ocgis
from ocgis import env
from ocgis.api.interpreter import OcgInterpreter
from ocgis.api.operations import OcgOperations
from ocgis.test.base import TestBase, attr
from ocgis.util.geom_cabinet import GeomCabinetIterator


class NcSpatial(object):

    def __init__(self, resolution, lat_bnds, lon_bnds):
        self.resolution = resolution
        self.lat_bnds = lat_bnds
        self.lon_bnds = lon_bnds

    @property
    def shift(self):
        return self.resolution / 2.0

    @property
    def lat_values(self):
        return self.get_centroids(self.lat_bnds)

    @property
    def lon_values(self):
        return self.get_centroids(self.lon_bnds)

    @property
    def latb_values(self):
        return self.make_bounds(self.lat_values)

    @property
    def lonb_values(self):
        return self.make_bounds(self.lon_values)

    def get_centroids(self, bounds):
        return np.arange(bounds[0] + self.shift, bounds[1] + self.shift, self.resolution, dtype=float)

    def make_bounds(self, arr):
        lower = arr - self.shift
        upper = arr + self.shift
        bounds = np.hstack((lower.reshape(-1, 1), upper.reshape(-1, 1)))
        return bounds


@attr('simple')
class Test360(TestBase):

    def test_high_res(self):
        ocgis.env.OVERWRITE = True
        nc_spatial = NcSpatial(0.5, (-90.0, 90.0), (0.0, 360.0))
        f, out_path = mkstemp(dir=env.DIR_OUTPUT, suffix='.nc')
        os.close(f)
        path = self.make_data(nc_spatial, path=out_path)

        dataset = {'uri': path, 'variable': 'foo'}
        output_format = 'nc'
        snippet = True
        geom = self.nebraska

        for s_abstraction in ['point', 'polygon']:
            interface = {'s_abstraction': s_abstraction}
            ops = OcgOperations(dataset=dataset, output_format=output_format, geom=geom, snippet=snippet,
                                abstraction=s_abstraction)
            OcgInterpreter(ops).execute()

    def test_low_res(self):
        """Test data is appropriate wrapped with a 360 dataset at low resolution."""

        ocgis.env.OVERWRITE = True
        nc_spatial = NcSpatial(5.0, (-90.0, 90.0), (0.0, 360.0))
        f, out_path = mkstemp(dir=env.DIR_OUTPUT, suffix='.nc')
        os.close(f)
        path = self.make_data(nc_spatial, path=out_path)

        dataset = {'uri': path, 'variable': 'foo'}
        output_format = 'shp'
        geom = self.nebraska

        for s_abstraction in ['point', 'polygon']:
            ops = OcgOperations(dataset=dataset,
                                output_format=output_format,
                                geom=geom,
                                abstraction=s_abstraction)
            OcgInterpreter(ops).execute()

    @property
    def nebraska(self):
        path = os.path.join(self.path_bin, 'shp', 'state_boundaries', 'state_boundaries.shp')
        sci = GeomCabinetIterator(path=path, select_uid=[16])
        geom = list(sci)
        return geom

    def transform_to_360(self, polygon):

        def _transform_lon_(ctup):
            lon = ctup[0]
            if lon < 180:
                lon += 360
            return [lon, ctup[1]]

        transformed = map(_transform_lon_, polygon.exterior.coords)
        new_polygon = Polygon(transformed)
        return new_polygon

    def make_variable(self, varname, arr, dimensions, bounds=None):
        var = self.ds.createVariable(varname, arr.dtype, dimensions=dimensions)
        var[:] = arr

        if bounds is not None:
            var.bounds = bounds

        return var

    def make_data(self, nc_spatial, path=None):
        if path is None:
            path = os.path.join(env.DIR_OUTPUT, 'test360 {0}.nc'.format(datetime.now()))

        calendar = 'standard'
        units = 'days since 0001-01-01'
        time_values = [datetime(2000, m, 15) for m in range(1, 13)]
        time_values = nc.date2num(time_values, units, calendar=calendar)

        level_values = np.array([100, 200])

        values = np.empty((len(time_values), len(level_values), len(nc_spatial.lat_values), len(nc_spatial.lon_values)))
        col_values = np.arange(0, len(nc_spatial.lon_values))
        for ii in range(0, len(nc_spatial.lat_values)):
            values[:, :, ii, :] = col_values
        values = np.ma.array(values, mask=False, fill_value=1e20)

        self.ds = nc.Dataset(path, 'w')
        ds = self.ds

        ds.createDimension('d_lat', size=len(nc_spatial.lat_values))
        ds.createDimension('d_lon', size=len(nc_spatial.lon_values))
        ds.createDimension('d_bnds', size=2)
        ds.createDimension('d_level', size=len(level_values))
        ds.createDimension('d_time', size=len(time_values))

        self.make_variable('lat', nc_spatial.lat_values, 'd_lat', bounds='lat_bnds')
        self.make_variable('lon', nc_spatial.lon_values, 'd_lon', bounds='lon_bnds')
        self.make_variable('lat_bnds', nc_spatial.latb_values, ('d_lat', 'd_bnds'))
        self.make_variable('lon_bnds', nc_spatial.lonb_values, ('d_lon', 'd_bnds'))

        v_time = self.make_variable('time', time_values, 'd_time')
        v_time.calendar = calendar
        v_time.units = units

        self.make_variable('level', level_values, 'd_level')

        self.make_variable('foo', values, ('d_time', 'd_level', 'd_lat', 'd_lon'))

        self.ds.close()

        return path
