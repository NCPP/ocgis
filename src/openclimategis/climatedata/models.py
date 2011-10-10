from django.contrib.gis.db import models


class AbstractGeoManager(models.Model):
    objects = models.GeoManager()
    
    class Meta:
        abstract = True


class NetcdfDataset(AbstractGeoManager):
    '''Models a netCDF4 Dataset object. Also a NC file anywhere.'''
    uri = models.TextField(unique=True)
    rowbnds_name = models.CharField(max_length=50)
    colbnds_name = models.CharField(max_length=50)
    time_name = models.CharField(max_length=50)
    time_units = models.CharField(max_length=100)
    calendar = models.CharField(max_length=50)
    spatial_extent = models.PolygonField(srid=4326)
    temporal_min = models.DateTimeField()
    temporal_max = models.DateTimeField()
    temporal_interval_days = models.FloatField()
    
    def __unicode__(self):
        return '{uri}'.format(uri=self.uri)


class NetcdfDatasetAttribute(AbstractGeoManager):
    '''stores key value pairs for NetCDF dataset global attributes'''
    netcdf_dataset  = models.ForeignKey(NetcdfDataset)
    key             = models.CharField(max_length=50)
    value           = models.TextField()
    
    class Meta():
        unique_together = ('netcdf_dataset','key')


class NetcdfDimension(AbstractGeoManager):
    '''Models the dimension of a NetCDF dataset'''
    netcdf_dataset = models.ForeignKey(NetcdfDataset)
    name           = models.CharField(max_length=50)
    size           = models.IntegerField()
    
    class Meta():
        unique_together = ('netcdf_dataset','name')
    
    def __unicode__(self):
        return '{name} (size={size})'.format(name=self.name,size=self.size)

class NetcdfDimensionAttribute(AbstractGeoManager):
    '''stores key value pairs for NetCDF dimension attributes'''
    netcdf_dimension = models.ForeignKey(NetcdfDimension)
    key              = models.CharField(max_length=50)
    value            = models.TextField()
    
    class Meta():
        unique_together = ('netcdf_dimension','key')


class NetcdfVariable(AbstractGeoManager):
    '''Models an abstract NetCDF dataset variable
    '''
    netcdf_dataset = models.ForeignKey(NetcdfDataset)
    code           = models.CharField(max_length=255)
    ndim           = models.IntegerField()
    
    class Meta():
        unique_together = ('netcdf_dataset','code')
    
    def __unicode__(self):
        return "{code} (dimensions={ndim})".format(code=self.code, ndim=self.ndim)


class NetcdfVariableAttribute(AbstractGeoManager):
    '''stores key value pairs for NetCDF variable attributes'''
    netcdf_variable = models.ForeignKey(NetcdfVariable)
    key             = models.CharField(max_length=50)
    value           = models.TextField()
    
    class Meta():
        unique_together = ('netcdf_variable','key')


class NetcdfVariableDimension(AbstractGeoManager):
    '''Models the dimension of a NetCDF Variable'''
    netcdf_variable  = models.ForeignKey(NetcdfVariable)
    netcdf_dimension = models.ForeignKey(NetcdfDimension)
    position = models.IntegerField()
    
    class Meta():
        unique_together = ('netcdf_variable','netcdf_dimension')
    
    def __unicode__(self):
        return "{var} {dim} ({position})".format(
            var=self.variable, 
            dim=self.code,
            position=self.position,
        )


class Organization(AbstractGeoManager):
    '''Models an organization (that has a climate model)
    
    Example: National Center for Atmospheric Research (ncar) 
    '''
    name         = models.CharField(max_length=255)
    code         = models.CharField(max_length=25, unique=True)
    country      = models.CharField(max_length=255)
    url          = models.URLField(
        verify_exists=False, 
        max_length=200,
        null=True,
    )
    
    def __unicode__(self):
        return "{name} ({code})".format(name=self.name, code=self.code)


class Scenario(AbstractGeoManager):
    '''A climate model emissions scenario
    
    Example: 2xCO2 equilibrium experiment (2xCO2)
    Reference: http://www-pcmdi.llnl.gov/ipcc/standard_output.html#Experiments
    '''
    name         = models.CharField(max_length=50)
    code         = models.CharField(max_length=10, unique=True)
    urlslug      = models.CharField(max_length=10, unique=True)
    description  = models.TextField(null=True)
    
    class Meta():
        verbose_name = "climate emissions scenario"
    
    def __unicode__(self):
        return "{code}".format(code=self.code)


class Archive(AbstractGeoManager):
    '''Models an climate model data archive
    
    Example: Coupled Model Intercomparison Project (CMIP3) 
    '''
    name         = models.CharField(max_length=255, unique=True)
    code         = models.CharField(max_length=25, unique=True)
    urlslug      = models.CharField(max_length=25, unique=True)
    url          = models.URLField(
        verify_exists=False, 
        max_length=200,
        null=True,
        unique=True,
    )
    
    class Meta():
        verbose_name = "climate simulation archive"
    
    def __unicode__(self):
        return "{code}".format(code=self.code)


class ClimateModel(AbstractGeoManager):
    '''A climate model
    
    Example: Community Climate System Model, version 3.0 (CCSM3)
    Reference: http://www-pcmdi.llnl.gov/ipcc/model_documentation/ipcc_model_documentation.php
    '''
    name         = models.CharField(max_length=50)
    code         = models.CharField(max_length=25, unique=True)
    urlslug      = models.CharField(max_length=25, unique=True)
    organization = models.ForeignKey(Organization)
    url          = models.URLField(
        verify_exists=False,
        max_length=200,
        null=True,
    )
    comments     = models.TextField(null=False, blank=True)
    
    class Meta():
        verbose_name = "climate model"
    
    def __unicode__(self):
        return "{code}".format(code=self.code)


class Variable(AbstractGeoManager):
    '''Models an climate model variable
    
    Example: air_temperature (tas)
    
    Ref: http://www-pcmdi.llnl.gov/ipcc/standard_output.html#Highest_priority_output
    '''
    code         = models.CharField(max_length=25, unique=True)
    urlslug      = models.CharField(max_length=25, unique=True)
    name         = models.CharField(max_length=50)
    units        = models.CharField(max_length=25)
    ndim         = models.IntegerField()
    description  = models.TextField(null=True)
    
    class Meta():
        verbose_name = "climate simulation variable"
    
    def __unicode__(self):
        return "{name} ({code})".format(name=self.name, code=self.code)


class SimulationOutput(AbstractGeoManager):
    '''Models climate model output datasets'''
    
    archive         = models.ForeignKey(Archive)
    scenario        = models.ForeignKey(Scenario)
    climate_model   = models.ForeignKey(ClimateModel)
    variable        = models.ForeignKey(Variable)
    run             = models.IntegerField()
    netcdf_variable = models.ForeignKey(NetcdfVariable)
    
    class Meta():
        unique_together = ('archive','scenario','climate_model','variable','run')
        verbose_name = "climate simulation output"
    
    
    def __unicode__(self):
        return '{archive}:{scenario}:{model}:{variable}'.format(
            archive=self.archive,
            scenario=self.scenario,
            model=self.climate_model,
            variable=self.netcdf_variable,
        )
