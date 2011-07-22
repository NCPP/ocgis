import os
import netCDF4
import warnings
from shapely.geometry.point import Point
from django.contrib.gis.geos.geometry import GEOSGeometry
from django.db import transaction
import re
from climatedata import models
import exc
import numpy as np
from django.contrib.gis.geos.polygon import Polygon

from ipdb import set_trace as tr


def import_examples():
    pass


class NcDatasetImporter(object):
    """
    Import a netCDF4 Dataset object into the database.
    """
    
    def __init__(self,uri,climatemodel_id,scenario_id,name=None):
        self.uri = uri
        self.name = name or os.path.split(self.uri)[1]
        self.dataset = netCDF4.Dataset(uri,'r')
        self.climatemodel = models.ClimateModel.objects.filter(pk=climatemodel_id)[0]
        self.scenario = models.Scenario.objects.filter(pk=scenario_id)[0]
        
        self._set_name_('time_name',['time'])
        keys = self.dataset.variables[self.time_name].ncattrs()
        self._set_name_('time_units_name',['units'],keys)
        self._set_name_('calendar_name',['calendar'],keys)
        
        self._set_name_('rowbnds_name',['lat_bounds','latitude_bounds','lat_bnds','latitude_bnds','bounds_latitude'])
        self._set_name_('colbnds_name',['lon_bounds','longitude_bounds','lon_bnds','longitude_bnds','bounds_longitude'])
    
        self.time_units = getattr(self.dataset.variables[self.time_name],self.time_units_name)
        self.calendar = getattr(self.dataset.variables[self.time_name],self.calendar_name)
    
    def __del__(self):
        try:
            self.dataset.close()
        except:
            pass
    
    def load(self):
        attrs = dict(climatemodel=self.climatemodel,
                     scenario=self.scenario,
                     name=self.name,
                     uri=self.uri,
                     rowbnds_name=self.rowbnds_name,
                     colbnds_name=self.colbnds_name,
                     time_name=self.time_name,
                     time_units=self.time_units,
                     calendar=self.calendar,
                     spatial_extent=self._spatial_extent_()
                     )
        attrs.update(self._temporal_fields_())
        
        dataset = models.Dataset(**attrs)
            
        ## get global attributes
        for attr in self.dataset.ncattrs():
            attrv = models.AttributeDataset(dataset=dataset,
                                            code=attr,
                                            value=str(getattr(self.dataset,attr)))
            attrv.save()
            
        ## loop for each variable
        for var in self.dataset.variables.keys():
            v = self.dataset.variables[var]
            variable_obj = models.Variable(dataset=dataset,
                                           code=var,
                                           ndim=len(v.dimensions))
            variable_obj.save()
            for attr in v.ncattrs():
                attrv = models.AttributeVariable(variable=variable_obj,
                                                 code=attr,
                                                 value=str(getattr(v,attr)))
                attrv.save()
            for dim in v.dimensions:
                dim_obj = models.Dimension(variable=variable_obj,
                                           code=dim,
                                           position=v.dimensions.index(dim))
                dim_obj.save()

    
    def _set_name_(self,target,options,keys=None):
        "Search naming options for target variables."
        
        ret = None
        if not keys:
            keys = self.dataset.variables.keys()
        for key in keys:
            if key in options:
                ret = key
                break
        setattr(self,target,ret)
        if not ret:
            warnings.warn('variable "{0}" not found in {1}. setting to "None" and no load is attempted.'.format(target,self.dataset.variables.keys()))

    def _temporal_fields_(self):
        timevec = netCDF4.netcdftime.num2date(self.dataset.variables[self.time_name][:],
                                              self.time_units,
                                              self.calendar)
        temporal_min = min(timevec)
        temporal_max = max(timevec)
        
        start = 0
        target = 1
        diffs = []
        
        while True:
            try:
                s = timevec[start]
                t = timevec[target]
                diffs.append((t-s).days)
                start += 1
                target += 1
            except IndexError:
                break
        
        temporal_interval_days = float(np.mean(diffs))
        
        return(dict(temporal_min=temporal_min,
                    temporal_max=temporal_max,
                    temporal_interval_days=temporal_interval_days))
        
    def _spatial_extent_(self):
        min_x = float(self.dataset.variables[self.colbnds_name][:].min())
        min_y = float(self.dataset.variables[self.rowbnds_name][:].min())
        max_x = float(self.dataset.variables[self.colbnds_name][:].max())
        max_y = float(self.dataset.variables[self.rowbnds_name][:].max())
#        import ipdb;ipdb.set_trace()
        p = Polygon(((min_x,min_y),(max_x,min_y),(max_x,max_y),(min_x,max_y),(min_x,min_y)),srid=4326)
        return(p)
        
        
if __name__ == '__main__':
    climatemodel = ClimateModel.objects.filter(pk=1)[0]
    scenario = Scenario.objects.filter(pk=1)[0]
    uri = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data/maurer/bccr_bcm2_0.1.sresa1b.monthly.Prcp.1950.nc'
    ncd = NcDatasetImporter(uri,climatemodel,scenario)
    ncd.load()
#@transaction.commit_manually()
#def load_dataset(uri,climatemodel,scenario):
#    
#    dataset,created = models.Dataset.objects.get_or_create(
#        climatemodel=climatemodel,
#        scenario=scenario,
#        uri=uri)
#    ## do not want to load duplicate datasets!
#    if created is False:
#        raise(exc.DatasetExists(uri))
    
    
    


#@transaction.commit_on_success()
#def load_climatemodel(archive,uris,**kwds):
#    ncm = NcModelImporter(uris,**kwds)
#    ncm.load(archive)
#
#
#class NcModelImporter(object):
#    """
#    Load a climate model with URIs pointing to model output datasets.
#    
#    code -- short name of the climate model. i.e. 'bccr_bcm2.0'.
#    uris -- list of URIs pointing to model output datasets. each URI should return
#      a Dataset instance when used by netCDF4.Dataset().
#    """
#    
#    def __init__(self,uris,**kwds):
#        self.uris = uris
#        self.name = kwds.get('name')
#        self.code = kwds.get('code')
#        self.url = kwds.get('url')
#        self.scenario_regex = kwds.get('scenario_regex') or '.*\..*\..*\.(.*)\.'
#        
#    def load(self,archive):
#        """
#        Load the model into the database using a foreign Archive object.
#        
#        archive -- Archive database object.
#        """
#        
#        kwds = dict(archive=archive,
#                    name=self.name,
#                    code=self.code,
#                    url=self.url)
#        cm,created = ClimateModel.objects.get_or_create(**kwds)
##        cm = db.ClimateModel(code=self.code,archive=archive)
##        s.add(cm)
#        first = True
#        for uri in self.uris:
#            if first:
#                spatial = True
#                first = False
#            else:
#                spatial = False
#            scenario_code = re.search(self.scenario_regex,uri).group(1)
#            scenario,created = Scenario.objects.get_or_create(code=scenario_code)
#            ndp = NcDatasetImporter(uri,cm,scenario)
#            ndp.load(spatial=spatial)
##        s.commit()
#        
#
#class NcDatasetImporter(object):
#    """
#    Import a netCDF4 Dataset object into the database.
#    """
#    
#    def __init__(self,uri,climatemodel,scenario):
#        self.uri = uri
#        self.dataset = netCDF4.Dataset(uri,'r')
#        self.climatemodel = climatemodel
#        self.scenario = scenario
#        
#        self._set_name_('time',['time'])
#        self._set_name_('time_bnds',['time_bnds','time_bounds'])
#        self._set_name_('row',['lat','latitude'])
#        self._set_name_('col',['lon','longitude'])
#        self._set_name_('row_bnds',['lat_bounds','latitude_bounds','lat_bnds','latitude_bnds','bounds_latitude'])
#        self._set_name_('col_bnds',['lon_bounds','longitude_bounds','lon_bnds','longitude_bnds','bounds_longitude'])
#    
#    def _set_name_(self,target,options):
#        "Search naming options for target variables."
#        
#        ret = None
#        for key in self.dataset.variables.keys():
#            if key in options:
#                ret = key
#                break
#        setattr(self,target,ret)
#        if not ret:
#            warnings.warn('variable "{0}" not found in {1}. setting to "None" and no load is attempted.'.format(target,self.dataset.variables.keys()))
#        
#    def load(self,spatial=True,temporal=True):
#        print('loading '+self.uri+'...')
#        print('loading dataset...')
#        dataset = self._dataset_(self.climatemodel,self.scenario)
#        print('loading variables...')
#        self._variables_(dataset)
#        if spatial:
#            print('loading spatial grid...')
#            self._spatial_(self.climatemodel)
#        if temporal:
#            print('loading temporal grid...')
#            self._temporal_(dataset)
##        s.commit()
#        print('success.')
#        
#    def _dataset_(self,climatemodel,scenario):
#        obj = Dataset()
#        obj.climatemodel = climatemodel
#        obj.scenario = scenario
#        obj.name = os.path.split(self.uri)[1]
#        obj.uri = self.uri
#        obj.save()
##        s.add(obj)
#        for attr in self.dataset.ncattrs():
#            value = getattr(self.dataset,attr)
#            a = AttributeDataset()
#            a.dataset = obj
#            a.name = attr
#            a.value = str(value)
#            a.save()
##            s.add(a)
##        s.commit()
#        return(obj)
#    
#    def _temporal_(self,dataset):
#        value = self.dataset.variables[self.time]
#        vec = netCDF4.num2date(value[:],value.units,value.calendar)
#        for ii in xrange(len(vec)):
#            idx = IndexTime()
#            idx.dataset = dataset
#            idx.index = ii
#            idx.value = vec[ii]
#            if self.time_bnds:
#                bounds = netCDF4.num2date(self.dataset.variables[self.time_bnds][ii,:],
#                                     value.units,
#                                     value.calendar)
#                idx.lower = bounds[0]
#                idx.upper = bounds[1]
##            s.add(idx)
#            idx.save()
#    
#    def _variables_(self,dataset):
#        for key,value in self.dataset.variables.iteritems():
##            q = s.query(db.Variable).filter(db.Variable.climatemodel==climatemodel)\
##                                    .filter(db.Variable.code==key)\
##                                    .filter(db.Variable.ndim==value.ndim)
##            try:
##                obj = q.one()
##            except NoResultFound:
#            obj = Variable()
#            obj.dataset = dataset
#            obj.name = key
#    #            obj.dimensions = str(value.dimensions)
#            obj.ndim = value.ndim
#            obj.save()
##            obj.shape = str(value.shape)
#
#            ## TIME DIMENSION -------------------------------------------------
#            
#            ## construct the dimensions
#            for dim,sh in zip(value.dimensions,value.shape):
##                d = db.Dimension()
#                d = Dimension()
#                d.variable = obj
#                d.name = dim
#                d.size = sh
#                d.position = value.dimensions.index(dim)
##                s.add(d)
#                d.save()
#            
##            ## classify the variable
##            obj.category = 'variable' ## the default
##            for key1,value1 in CLASS.iteritems():
##                if key1 == 'bounds':
##                    for v in value1:
##                        if v in key:
##                            obj.category = key1
##                            break
##                else:
##                    if key in value1:
##                        obj.category = key1
##                        break
#            
##            s.add(obj)
#            for ncattr in value.ncattrs():
#                a = AttributeVariable()
#                a.variable = obj
#                a.name = ncattr
#                a.value = str(getattr(value,ncattr))
##                s.add(a)
#                a.save()
##            import ipdb;ipdb.set_trace()
#
##    @transaction.commit_on_success()
#    def _spatial_(self,climatemodel):
#        col = self.dataset.variables[self.col][:]
#        row = self.dataset.variables[self.row][:]
#        col_bnds = self.dataset.variables[self.col_bnds][:]
#        row_bnds = self.dataset.variables[self.row_bnds][:]
##        geoms = []
#        total = len(col)*len(row)
#        ctr = 0
#        for ii in xrange(len(col)):
#            for jj in xrange(len(row)):
#                if ctr%2000 == 0:
#                    print('  {0} of {1}'.format(ctr,total))
#                ctr += 1
##                obj = db.IndexSpatial()
#                obj = IndexSpatial()
#                obj.row = jj
#                obj.col = ii
#                obj.climatemodel = climatemodel
#                pt = Point(col[ii],row[jj])
#                obj.centroid = GEOSGeometry(str(pt))
#                poly = Polygon(((col_bnds[ii,0],row_bnds[jj,0]),
#                               (col_bnds[ii,0],row_bnds[jj,1]),
#                               (col_bnds[ii,1],row_bnds[jj,1]),
#                               (col_bnds[ii,1],row_bnds[jj,0])))
#                obj.geom = GEOSGeometry(str(poly))
#                obj.save()
##                geoms.append(obj)
##                s.add(obj)
##        climatemodel.indexspatial = geoms
##        print('  committing...')
##        s.commit()
##                pass
#
#        
#    def __del__(self):
#        try:
#            self.dataset.close()
#        except:
#            pass