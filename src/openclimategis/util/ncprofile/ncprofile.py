import netCDF4 as nc
import os
import db
import sys
from shapely.geometry.point import Point
from geoalchemy.base import WKTSpatialElement
from shapely.geometry.polygon import Polygon


CLASS = dict(
             row=['lat','latitude'],
             col=['lon','longitude'],
             bounds=['bnds','bounds'],
             time=['time'],
             level=['lev','level','lvl']
             )


class NcProfiler(object):
    
    def __init__(self,uri):
        self.uri = uri
        self.dataset = nc.Dataset(uri,'r')
        pass
        
    def load(self):
        try:
            s = db.Session()
            dataset = self._dataset_(s)
            self._variables_(dataset,s)
            self._spatial_(s,dataset)
            s.commit()
        finally:
            s.close()
        
    def _dataset_(self,s):
        obj = db.Dataset()
        obj.code = os.path.split(self.uri)[1]
        obj.uri = self.uri
        s.add(obj)
        for attr in self.dataset.ncattrs():
            value = getattr(self.dataset,attr)
            a = db.AttributeDataset()
            a.dataset = obj
            a.code = attr
            a.value = str(value)
            s.add(a)
        return(obj)
    
    def _variables_(self,dataset,s):
        for key,value in self.dataset.variables.iteritems():
            obj = db.Variable()
            obj.dataset = dataset
            obj.code = key
#            obj.dimensions = str(value.dimensions)
            obj.ndim = value.ndim
#            obj.shape = str(value.shape)

            ## TIME DIMENSION -------------------------------------------------
            
            if key in CLASS['time']:
                vec = nc.num2date(value[:],value.units,value.calendar)
                dimobj = db.Dimension()
                dimobj.variable = obj
                dimobj.code = key
                dimobj.index_name = 'time'
                s.add(dimobj)
                for ii in xrange(len(vec)):
                    idx = db.IndexTime()
                    idx.dataset = dataset
                    idx.index = ii
                    idx.value = vec[ii]
                    bounds = nc.num2date(self.dataset.variables['time_bnds'][ii,:],
                                         value.units,
                                         value.calendar)
                    idx.lower = bounds[0]
                    idx.upper = bounds[1]
                    s.add(idx)
            
#            ## construct the dimensions
#            for dim,sh in zip(value.dimensions,value.shape):
#                d = db.Dimension()
#                d.variable = obj
#                d.dim_name = dim
#                d.size = sh
#                s.add(d)
            
#            ## classify the variable
#            obj.category = 'variable' ## the default
#            for key1,value1 in CLASS.iteritems():
#                if key1 == 'bounds':
#                    for v in value1:
#                        if v in key:
#                            obj.category = key1
#                            break
#                else:
#                    if key in value1:
#                        obj.category = key1
#                        break
            
            s.add(obj)
            for ncattr in value.ncattrs():
                a = db.AttributeVariable()
                a.variable = obj
                a.code = ncattr
                a.value = str(getattr(value,ncattr))
                s.add(a)
#            import ipdb;ipdb.set_trace()

    def _spatial_(self,s,dataset):
        x = self.dataset.variables['lon'][:]
        y = self.dataset.variables['lat'][:]
        x_bnds = self.dataset.variables['lon_bnds'][:]
        y_bnds = self.dataset.variables['lat_bnds'][:]
        for ii in xrange(len(x)):
            for jj in xrange(len(y)):
                obj = db.IndexSpatial()
                obj.row = ii
                obj.col = jj
                obj.dataset = dataset
                pt = Point(x[ii],y[jj])
                obj.centroid = WKTSpatialElement(str(pt))
                poly = Polygon(((x_bnds[ii,0],y_bnds[jj,0]),
                               (x_bnds[ii,0],y_bnds[jj,1]),
                               (x_bnds[ii,1],y_bnds[jj,1]),
                               (x_bnds[ii,1],y_bnds[jj,0])))
                obj.geom = WKTSpatialElement(str(poly))
                s.add(obj)
                pass

        
    def __del__(self):
        try:
            self.dataset.close()
        except:
            pass
        

if __name__ == '__main__':
#    uri = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data/maurer/bccr_bcm2_0.1.sresa1b.monthly.Prcp.1950.nc'
#    uri = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data/wcrp_cmip3/pcmdi.ipcc4.bccr_bcm2_0.1pctto2x.run1.monthly.cl_A1_1.nc'
#    ncp = NcProfiler(uri)
#    ncp.load()
#    import ipdb;ipdb.set_trace()
    d = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data'
    for root,dirs,files in os.walk(d):
        for f in files:
            if f.endswith('.nc'):
                uri = os.path.join(root,f)
                print(uri)
                ncp = NcProfiler(uri)
                ncp.load()
                sys.exit()



#import netCDF4 as nc
#import copy
#from netCDF4_utils import OrderedDict
#import inspect
#
#
#VARMAP = dict(
#              time=['time'],
#              row=['latitude','lat'],
#              row_bounds=['bounds_latitude','lat_bnds'],
#              col=['longitude','lon'],
#              col_bounds = ['bounds_longitude','lon_bnds'],
#              missing=['missing_value']
#              )
#
#
#class Meta(object):
#    
#    def __init__(self,name):
#        self.name = name
#    
#    def __repr__(self):
#        lines = ['"{0}" metadata\n'.format(self.name)]
#        base = '    {0} = {1}\n'
#        for key,value in self.__dict__.iteritems():
#            try:
#                if key in ['name']:
#                    continue
#                if type(value) not in [str,float,int]:
#                    value = '<truncated>'
#                if isinstance(getattr(self,key),Meta):
##                    import ipdb;ipdb.set_trace()
#                    value = self.__class__
#            except:
#                import ipdb;ipdb.set_trace()
#            lines.append(base.format(key,value))
#        return(''.join(lines))
#
#
#class NamedVariable(object):
#    
#    def __init__(self,choices):
#        self.choices = choices
#        
#    def get(self,variables):
#        for ii,value in enumerate(self.choices):
#            try:
#                if isinstance(variables,OrderedDict):
#                    ret = variables[value]
#                else:
#                    ret = getattr(variables,value)
#                break
#            except KeyError:
#                if ii+2 > len(self.choices):
#                    msg = ('no variable located for choices {0}. '
#                           'valid choices are: {1}')\
#                           .format(self.choices,variables.keys())
#                    raise KeyError(msg)
#                else:
#                    continue
#        return(value,ret)
#    
#
#class NcProfiler(object):
#    
#    def __init__(self,uri):
#        self.uri = uri
#        self.dataset = nc.Dataset(uri,'r')
#        
#        import ipdb;ipdb.set_trace()
#        
#        self.time = Meta('time')
#        self.row = Meta('row')
#        self.row.bounds = Meta('row bounds')
#        self.col = Meta('col')
#        self.col.bounds = Meta('col bounds')
#        self.variable = Meta('variable')
#        self.variable.missing = Meta('missing value')
#        
#        self._remove_registry = []
#        self._profile_()
#
#    def _profile_(self):
#        "Collect variables of interest for OpenClimateGIS"
#        
#        ## TIME ----------------------------------------------------------------
#        
#        name,t = NamedVariable(VARMAP['time']).get(self.dataset.variables)
#        self._remove_registry.append(name)
#        self.time.variable = name
#        self.time.units = t.units
#        self.time.calendar = t.calendar
#        self.time.vec = nc.num2date(t[:],t.units,calendar=t.calendar)
#        
#        ## SPATIAL -------------------------------------------------------------
#        
#        self._make_spatial_(self.row,'row','row_bounds')
#        self._make_spatial_(self.col,'col','col_bounds')
#        
#        ## VARIABLE ------------------------------------------------------------
#        
#        ## first, copy and then remove variable to leave only the remaining
#        ## variable.
#        vars = copy.copy(self.dataset.variables)
#        for v in self._remove_registry: vars.pop(v)
#        ## only one variable should remain
#        if len(vars) != 1:
#            msg = ('only one variable should remain in the netcdf variable '
#                   'dictionary after removing time & spatial variables. '
#                   'remain variables are: {0}').format(vars.keys())
#            raise ValueError(msg)
#        ## pull information on the variable from the netcdf
#        self.variable.variable = vars.keys()[0]
#        var = self.dataset.variables[self.variable.variable]
#        self.variable.units = var.units
#        self.variable.vec = var[:]
#        name,value = NamedVariable(VARMAP['missing']).get(var)
#        self.variable.missing.variable = name
#        self.variable.missing.value = float(value)
#        
#        import ipdb;ipdb.set_trace()
#        
#    def __del__(self):
#        try:
#            self.dataset.close()
#        except:
#            pass
#    
#    def _make_spatial_(self,var,name,bounds_name):
#        name,vals =  NamedVariable(VARMAP[name]).get(self.dataset.variables)
#        var.variable = name
#        var.vec = vals[:]
#        
#        name,bnds = NamedVariable(VARMAP[bounds_name]).get(self.dataset.variables)
#        var.bounds.vec = bnds[:]
#        var.bounds.variable = name
#        
#        self._remove_registry += [var.variable,var.bounds.variable]
##        
##        self.row.variable = name
##        self.row.vec = vals[:]
##        name,bnds = NamedVariable(VARMAP[bounds_name]).get(self.dataset.variables)
##        self.row.bounds.vec = bnds[:]
##        self.row.bounds.variable = name
#        
#        
#if __name__ == '__main__':
##    uri = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data/maurer/bccr_bcm2_0.1.sresa1b.monthly.Prcp.1950.nc'
#    uri = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data/wcrp_cmip3/pcmdi.ipcc4.bccr_bcm2_0.1pctto2x.run1.monthly.cl_A1_1.nc'
#    ncp = NcProfiler(uri)
#    import ipdb;ipdb.set_trace()