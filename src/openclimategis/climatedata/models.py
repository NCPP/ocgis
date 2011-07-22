from django.contrib.gis.db import models


class AbstractGeoManager(models.Model):
    objects = models.GeoManager()

    class Meta:
        abstract = True


class Organization(AbstractGeoManager):
    '''Models an organization (that has a climate model)
    
    Example: National Center for Atmospheric Research (ncar) 
    '''
    name         = models.CharField(max_length=50)
    code         = models.CharField(max_length=25,unique=True)
    country      = models.CharField(max_length=50)
    url          = models.URLField(
        verify_exists=False, 
        max_length=200,
        null=True,
    )
    
    def __unicode__(self):
        return "{name} ({code})".format(name=self.name, code=self.code)
    
    
class Scenario(AbstractGeoManager):
    '''A climate model simulation experiment
    
    Example: 2xCO2 equilibrium experiment (2xCO2)
    Reference: http://www-pcmdi.llnl.gov/ipcc/standard_output.html#Experiments
    '''
    name    = models.CharField(max_length=50)
    code    = models.CharField(max_length=10,unique=True)
    
    def __unicode__(self):
        return "{name} ({code})".format(name=self.name, code=self.code)


class Archive(AbstractGeoManager):
    '''Models an climate model data archive
    
    Example: Coupled Model Intercomparison Project (CMIP3) 
    '''
    organization = models.ForeignKey(Organization)
#    climatemodel = models.ManyToManyField("ClimateModel")
    name         = models.CharField(max_length=50)
    code         = models.CharField(max_length=25,unique=True)
    url          = models.URLField(
        verify_exists=False, 
        max_length=200,
        null=True,
    )
    
    def __unicode__(self):
        return "{name} ({code})".format(name=self.name, code=self.code)
    
    
class ClimateModel(AbstractGeoManager):
    '''A climate model
    
    Example: Community Climate System Model, version 3.0 (CCSM3)
    Reference: http://www-pcmdi.llnl.gov/ipcc/model_documentation/ipcc_model_documentation.php
    '''
#    archive = models.ForeignKey(Archive)
    archive = models.ManyToManyField(Archive)
    name         = models.CharField(max_length=50)
    code         = models.CharField(max_length=25,unique=True)
#    organization = models.ForeignKey(Organization)
    url          = models.URLField(
        verify_exists=False,
        max_length=200,
        null=True,
    )
    
    def __unicode__(self):
        return "{name} ({code})".format(name=self.name, code=self.code)
    
    
class Dataset(AbstractGeoManager):
    '''Models a netCDF4 Dataset object. Also a NC file anywhere.'''
    
    climatemodel = models.ForeignKey(ClimateModel)
    scenario = models.ForeignKey(Scenario)
    name = models.TextField()
    uri = models.TextField()
    rowbnds_name = models.CharField(max_length=50)
    colbnds_name = models.CharField(max_length=50)
    time_name = models.CharField(max_length=50)
    time_units = models.CharField(max_length=100)
    calendar = models.CharField(max_length=50)
    spatial_extent = models.PolygonField(srid=4326)
    temporal_min = models.DateTimeField()
    temporal_max = models.DateTimeField()
    temporal_interval_days = models.FloatField()
    
    class Meta():
        unique_together = ('climatemodel','scenario','uri')
    
    def __unicode__(self):
        return 'Scenario: {scenario} ({uri})'.format(scenario=self.scenario,name=self.uri)


class Variable(AbstractGeoManager):
    '''Models an climate model variable
    
    Example: air_temperature (tas)
    
    Ref: http://www-pcmdi.llnl.gov/ipcc/standard_output.html#Highest_priority_output
    '''
    dataset = models.ForeignKey(Dataset)
    code         = models.CharField(max_length=25)
    ndim = models.IntegerField()
#    name         = models.CharField(max_length=50)
#    code         = models.CharField(max_length=25)
#    units        = models.CharField(max_length=25)
#    description  = models.CharField(max_length=1000, null=True)
    
    class Meta():
        unique_together = ('dataset','code')
    
    def __unicode__(self):
        return "{name} ({code})".format(name=self.name, code=self.code)
    
    
class AttributeVariable(AbstractGeoManager):
    
    variable = models.ForeignKey(Variable)
    code = models.CharField(max_length=50)
    value = models.TextField()
    
    class Meta():
        unique_together = ('variable','code')

class AttributeDataset(AbstractGeoManager):
    
    dataset = models.ForeignKey(Dataset)
    code = models.CharField(max_length=100)
    value = models.TextField()
    
    class Meta():
        unique_together = ('dataset','code')

class Dimension(AbstractGeoManager):
    
    variable = models.ForeignKey(Variable)
    code = models.CharField(max_length=100)
    position = models.IntegerField()
    
    class Meta():
        unique_together = ('variable','code')
        
#    size = models.IntegerField()
    
    
#class IndexTime(AbstractGeoManager):
#    
#    dataset = models.ForeignKey(Dataset)
#    index = models.IntegerField()
#    lower = models.DateField(null=True)
#    value = models.DateField()
#    upper = models.DateField(null=True)
#    
#    
#class IndexSpatial(AbstractGeoManager):
#    
#    climatemodel = models.ForeignKey(ClimateModel)
#    row = models.IntegerField()
#    col = models.IntegerField()
#    geom = models.PolygonField(srid=4326)
#    centroid = models.PointField(srid=4326)


#class Experiment(AbstractGeoManager):
#    '''A climate model simulation experiment
#    
#    Example: 2xCO2 equilibrium experiment (2xCO2)
#    Reference: http://www-pcmdi.llnl.gov/ipcc/standard_output.html#Experiments
#    '''
#    name    = models.CharField(max_length=50)
#    code    = models.CharField(max_length=10)
#    
#    def __unicode__(self):
#        return "{name} ({code})".format(name=self.name, code=self.code)


#class SpatialGrid(AbstractGeoManager):
#    '''A climate model spatial grid (collection of grid cells)'''
#    boundary_geom = models.PolygonField(srid=4326)
#    native_srid   = models.IntegerField()
#    description   = models.TextField()
#    
#    def __unicode__(self):
#        return "{description}".format(
#            description=self.description,
#        )
#
#
#class SpatialGridCell(AbstractGeoManager):
#    '''A climate model spatial grid cell'''
#    grid_spatial = models.ForeignKey(SpatialGrid)
#    row          = models.IntegerField()
#    col          = models.IntegerField()
#    geom         = models.PolygonField(srid=4326)


#class TemporalUnit(AbstractGeoManager):
#    '''A unit of time
#    
#    For example: hours since 1800-01-01 00:00:00
#    
#    Ref: http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.5/cf-conventions.html#time-coordinate
#    '''
#    time_unit = models.CharField(max_length=7)
#    reference = models.DateTimeField()
#    
#    def __unicode__(self):
#        return "{time_unit} since {time_reference}".format(
#            time_unit=self.time_unit,
#            time_reference=self.reference
#        )
#
#
#class Calendar(AbstractGeoManager):
#    '''A calendar used by time references
#    
#    Ref: http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.5/cf-conventions.html#calendar
#    '''
#    name          = models.CharField(max_length=20)
#    description   = models.TextField()
#    
#    def __unicode__(self):
#        return "{name} ({description})".format(
#            name=self.name,
#            description=self.description
#        )

#class TemporalGrid(AbstractGeoManager):
#    '''A climate model temporal grid (collection of grid cells)'''
#    date_min      = models.DateField()
#    date_max      = models.DateField()
#    temporal_unit = models.ForeignKey(TemporalUnit)
#    calendar      = models.ForeignKey(Calendar)
#    description   = models.TextField()
#
#    def __unicode__(self):
#        return "{date_min} to {date_max} (unit: {unit}; calendar: {calendar})".format(
#            date_min=self.date_min,
#            date_max=self.date_max,
#            unit=self.temporal_unit,
#            calendar=self.calendar.name,
#        )
#
#
#class TemporalGridCell(AbstractGeoManager):
#    '''A climate model temporal grid cell (time interval)'''
#    grid_temporal = models.ForeignKey(TemporalGrid)
#    index         = models.IntegerField()
#    date_min      = models.DateField(
#        help_text='the minimum date for the time interval'
#    )
#    date_ref      = models.DateField(
#        help_text='the reference date for the time interval',
#    )
#    date_max      = models.DateField(
#        help_text='the maximum date for the time interval',
#    )


#class Prediction(AbstractGeoManager):
#    '''Models of a climate prediction datafile'''
#    archive       = models.ForeignKey(Archive)
#    climate_model = models.ForeignKey(ClimateModel)
#    experiment    = models.ForeignKey(Experiment)
#    variable      = models.ForeignKey(Variable)
#    run           = models.IntegerField(
#        help_text='a run number, which may indicating different initial conditions',
#    )
#    url           = models.URLField(
#        verify_exists=False,
#        help_text='URL for accessing the dataset',
#    )
#    grid_spatial  = models.ForeignKey(SpatialGrid)
#    grid_temporal = models.ForeignKey(TemporalGrid)
#    spacing_temporal = models.CharField(
#        max_length=1,
#        choices=(
#            ('D', 'Daily'),
#            ('M', 'Monthly'),
#        )
#    )
#    description   = models.TextField()
#    
#    def __unicode__(self):
#        return "{url} ({description})".format(
#            url=self.url,
#            description=self.description,
#        )