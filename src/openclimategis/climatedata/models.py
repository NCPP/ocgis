from django.contrib.gis.db import models

class Organization(models.Model):
    '''Models an organization (that has a climate model)
    
    Example: National Center for Atmospheric Research (ncar) 
    '''
    name         = models.CharField(max_length=50)
    code         = models.CharField(max_length=10)
    objects = models.GeoManager()


class ClimateModel(models.Model):
    '''A climate model
    
    Example: Community Climate System Model, version 3.0 (CCSM3)
    Reference: http://www-pcmdi.llnl.gov/ipcc/model_documentation/ipcc_model_documentation.php
    '''
    name         = models.CharField(max_length=50)
    code         = models.CharField(max_length=10)
    organization = models.ForeignKey(Organization)
    objects = models.GeoManager()


class Experiment(models.Model):
    '''A climate model simulation experiment
    
    Example: 2xCO2 equilibrium experiment (2xCO2)
    Reference: http://www-pcmdi.llnl.gov/ipcc/standard_output.html#Experiments
    '''
    name    = models.CharField(max_length=50)
    code         = models.CharField(max_length=10)
    objects = models.GeoManager()


class Realization(models.Model):
    '''A climate model simulation run (realization)
    '''
    climate_model = models.ForeignKey(ClimateModel)
    experiment    = models.ForeignKey(Experiment)
    run_number    = models.IntegerField()
    start_datetime= models.DateTimeField()
    objects = models.GeoManager()


class Frequency(models.Model):
    '''Temporal frequency of the data
    
    Example: monthly, daily
    '''
    code = models.CharField(max_length=2)
    name = models.CharField(max_length=50)
    objects = models.GeoManager()


class Grid(models.Model):
    '''A climate model grid'''
    pass
    objects = models.GeoManager()


class GridCell(models.Model):
    '''A climate model grid cell'''
    grid    = models.ForeignKey(Grid)
    geom    = models.PolygonField(srid=4326)
    objects = models.GeoManager()


class Prediction(models.Model):
    '''Models of a climate prediction datafile'''
    climatemodel = models.ForeignKey(ClimateModel)
    experiment   = models.ForeignKey(Experiment)
    realization  = models.ForeignKey(Realization)
    frequency    = models.ForeignKey(Frequency)
    url          = models.URLField()
    min_date     = models.DateTimeField()
    max_date     = models.DateTimeField()
    grid         = models.ForeignKey(Grid)
    #name     = models.CharField(max_length=50)
    #metadata = models.CharField(max_length=16)
    #format   = models.CharField(max_length=16)
    #datatype = models.CharField(max_length=16)
    #size     = models.IntegerField()
    objects = models.GeoManager()

