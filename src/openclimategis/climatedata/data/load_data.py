import os
import netCDF4
import warnings
from climatedata import models
import numpy as np
from django.contrib.gis.geos.polygon import Polygon


class NcDatasetImporter(object):
    """
    Import a netCDF4 Dataset object into the database.
    """
    
    def __init__(self,uri,climatemodel_id,scenario_id,archive_id,name=None):
        self.uri = uri
        self.name = name or os.path.split(self.uri)[1]
        self.dataset = netCDF4.Dataset(uri,'r')
        self.climatemodel = models.ClimateModel.objects.filter(pk=climatemodel_id)[0]
        self.scenario = models.Scenario.objects.filter(pk=scenario_id)[0]
        self.archive = models.Archive.objects.filter(pk=archive_id)[0]
        
        self._set_name_('time_name',['time'])
        keys = self.dataset.variables[self.time_name].ncattrs()
        self._set_name_('time_units_name',['units'],keys)
        self._set_name_('calendar_name',['calendar'],keys)
        
        self._set_name_('rowbnds_name',['lat_bounds','latitude_bounds','lat_bnds','latitude_bnds','bounds_latitude'])
        self._set_name_('colbnds_name',['lon_bounds','longitude_bounds','lon_bnds','longitude_bnds','bounds_longitude'])
        
        self._set_name_('level_name',['level','levels','lvl','lvls'])
        
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
                     archive=self.archive,
                     name=self.name,
                     uri=self.uri,
                     rowbnds_name=self.rowbnds_name,
                     colbnds_name=self.colbnds_name,
                     time_name=self.time_name,
                     time_units=self.time_units,
                     calendar=self.calendar,
                     spatial_extent=self._spatial_extent_(),
                     level_name=self.level_name
                     )
        attrs.update(self._temporal_fields_())
        
        dataset = models.Dataset(**attrs)
        dataset.save()
            
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
        p = Polygon(((min_x,min_y),(max_x,min_y),(max_x,max_y),(min_x,max_y),(min_x,min_y)),srid=4326)
        return(p)
        
        
if __name__ == '__main__':
    ## dropdb openclimategis_sql
    ## createdb -T postgis-1.5.2-template openclimategis_sql
    ## python manage.py syncdb
    
    ## python manage.py dumpdata climatedata > /home/bkoziol/git/OpenClimateGIS/src/openclimategis/climatedata/fixtures/luca_fixtures.json
    
    organization = models.Organization(pk=1,name='NOAA',code='noaa',country='USA')
    scenario = models.Scenario(pk=1,name='sresa1b',code='sresa1b')
    archive = models.Archive(pk=1,organization=organization,name='maurer',code='maurer')
    cm = models.ClimateModel(pk=1,name='bccr_bcm2.0',code='bccr_bcm2.0')
    ac = models.ArchiveCollection(archive=archive,climatemodel=cm)
    
    for obj in [organization,scenario,archive,cm,ac]:
        obj.save()
    
    uri = 'http://hydra.fsl.noaa.gov/thredds/dodsC/oc_gis_downscaling.bccr_bcm2.sresa1b.Prcp.Prcp.1.aggregation.1'
    nc = NcDatasetImporter(uri,1,1,1)
    nc.load()