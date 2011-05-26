from django.contrib.gis.db import models

class Organization(models.Model):
    '''Models an organization (that has a climate model)
    
    Example: National Center for Atmospheric Research (ncar) 
    '''
    name         = models.CharField(max_length=50)
    code         = models.CharField(max_length=25)
    country      = models.CharField(max_length=50)
    url          = models.URLField(
        verify_exists=False, 
        max_length=200,
        null=True,
    )
    objects = models.GeoManager()
    
    def __unicode__(self):
        return "{name} ({code})".format(name=self.name, code=self.code)


class Archive(models.Model):
    '''Models an climate model data archive
    
    Example: Coupled Model Intercomparison Project (CMIP3) 
    '''
    name         = models.CharField(max_length=50)
    code         = models.CharField(max_length=25)
    url          = models.URLField(
        verify_exists=False, 
        max_length=200,
        null=True,
    )
    objects = models.GeoManager()
    
    def __unicode__(self):
        return "{name} ({code})".format(name=self.name, code=self.code)


class Variable(models.Model):
    '''Models an climate model variable
    
    Example: air_temperature (tas)
    
    Ref: http://www-pcmdi.llnl.gov/ipcc/standard_output.html#Highest_priority_output
    '''
    name         = models.CharField(max_length=50)
    code         = models.CharField(max_length=25)
    units        = models.CharField(max_length=25)
    description  = models.CharField(max_length=1000, null=True)
    objects = models.GeoManager()
    
    def __unicode__(self):
        return "{name} ({code})".format(name=self.name, code=self.code)


class ClimateModel(models.Model):
    '''A climate model
    
    Example: Community Climate System Model, version 3.0 (CCSM3)
    Reference: http://www-pcmdi.llnl.gov/ipcc/model_documentation/ipcc_model_documentation.php
    '''
    name         = models.CharField(max_length=50)
    code         = models.CharField(max_length=25)
    organization = models.ForeignKey(Organization)
    url          = models.URLField(
                        verify_exists=False,
                        max_length=200,
                        null=True,
                    )
    objects      = models.GeoManager()
    
    def __unicode__(self):
        return "{name} ({code})".format(name=self.name, code=self.code)


class Experiment(models.Model):
    '''A climate model simulation experiment
    
    Example: 2xCO2 equilibrium experiment (2xCO2)
    Reference: http://www-pcmdi.llnl.gov/ipcc/standard_output.html#Experiments
    '''
    name    = models.CharField(max_length=50)
    code    = models.CharField(max_length=10)
    objects = models.GeoManager()
    
    def __unicode__(self):
        return "{name} ({code})".format(name=self.name, code=self.code)

class Frequency(models.Model):
    '''Temporal frequency of the data
    
    Example: monthly, daily
    '''
    code = models.CharField(max_length=2)
    name = models.CharField(max_length=50)
    objects = models.GeoManager()
    
    def __unicode__(self):
        return "{name} ({code})".format(name=self.name, code=self.code)


class Grid(models.Model):
    '''A climate model grid (collection of grid cells)'''
    boundary_geom = models.PolygonField(srid=4326)
    native_srid   = models.IntegerField()
    description   = models.TextField()
    objects       = models.GeoManager()


class GridCell(models.Model):
    '''A climate model grid cell'''
    grid    = models.ForeignKey(Grid)
    row     = models.IntegerField()
    col     = models.IntegerField()
    geom    = models.PolygonField(srid=4326)
    objects = models.GeoManager()


class Prediction(models.Model):
    '''Models of a climate prediction datafile'''
    climate_model = models.ForeignKey(ClimateModel)
    experiment    = models.ForeignKey(Experiment)
    run           = models.IntegerField(
                    help_text='a run number, which may indicating different initial conditions',
                    )
    min_date      = models.DateTimeField()
    max_date      = models.DateTimeField()
    frequency     = models.ForeignKey(Frequency)
    url           = models.URLField()
    grid          = models.ForeignKey(Grid)
    description   = models.TextField()
    objects       = models.GeoManager()

