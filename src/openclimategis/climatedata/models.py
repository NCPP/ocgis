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


class SpatialGrid(models.Model):
    '''A climate model spatial grid (collection of grid cells)'''
    boundary_geom = models.PolygonField(srid=4326)
    native_srid   = models.IntegerField()
    description   = models.TextField()
    objects       = models.GeoManager()


class SpatialGridCell(models.Model):
    '''A climate model spatial grid cell'''
    grid_spatial = models.ForeignKey(SpatialGrid)
    row          = models.IntegerField()
    col          = models.IntegerField()
    geom         = models.PolygonField(srid=4326)
    objects      = models.GeoManager()


class TemporalUnit(models.Model):
    '''A unit of time
    
    For example: hours since 1800-01-01 00:00:00 -6:00
    
    Ref: http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.5/cf-conventions.html#time-coordinate
    '''
    time_unit = models.CharField(max_length=7)
    reference = models.DateField()
    objects       = models.GeoManager()
    
    def __unicode__(self):
        return "{time_unit} since {time_reference}".format(
            time_unit=self.time_unit,
            time_reference=self.reference
        )


class Calendar(models.Model):
    '''A calendar used by time references
    
    For example: hours since 1800-01-01 00:00:00 -6:00
    
    Ref: http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.5/cf-conventions.html#calendar
    '''
    name = models.CharField(
        max_length=20,
        choices=(
            ('gregorian',
                'Mixed Gregorian/Julian calendar as defined by Udunits.'), 
            ('standard',
                'Mixed Gregorian/Julian calendar as defined by Udunits.'), 
            ('proleptic_gregorian',
                'A Gregorian calendar extended to dates before 1582-10-15.'
                ' That is, a year is a leap year if either'
                ' (i) it is divisible by 4 but not by 100 or'
                ' (ii) it is divisible by 400.'),
            ('noleap',
                'Gregorian calendar without leap years,'
                ' i.e., all years are 365 days long.'), 
            ('365_day', 'Gregorian calendar without leap years,'
                ' i.e., all years are 365 days long.'), 
            ('all_leap', 'Gregorian calendar with every year being a leap year,'
                ' i.e., all years are 366 days long.'),
            ('366_day', 'Gregorian calendar with every year being a leap year,'
                ' i.e., all years are 366 days long.'),
            ('360_day', 
                'All years are 360 days divided into 30 day months.'), 
            ('julian', 
                'Julian calendar.'), 
            ('none', 
                'no calendar')
        )
    )
    objects       = models.GeoManager()


class TemporalGrid(models.Model):
    '''A climate model temporal grid (collection of grid cells)'''
    date_min      = models.DateField()
    date_max      = models.DateField()
    temporal_unit = models.ForeignKey(TemporalUnit)
    calendar      = models.ForeignKey(Calendar)
    description   = models.TextField()
    objects       = models.GeoManager()


class TemporalGridCell(models.Model):
    '''A climate model temporal grid cell (time interval)'''
    grid_temporal = models.ForeignKey(TemporalGrid)
    index         = models.IntegerField()
    date_min      = models.DateField(
        help_text='the minimum date for the time interval'
    )
    date_ref      = models.DateField(
        help_text='the reference date for the time interval',
    )
    date_max      = models.DateField(
        help_text='the maximum date for the time interval',
    )
    objects       = models.GeoManager()


class Prediction(models.Model):
    '''Models of a climate prediction datafile'''
    climate_model = models.ForeignKey(ClimateModel)
    experiment    = models.ForeignKey(Experiment)
    run           = models.IntegerField(
        help_text='a run number, which may indicating different initial conditions',
    )
    url           = models.URLField(
        verify_exists=False,
        help_text='URL for accessing the dataset',
    )
    grid_spatial  = models.ForeignKey(SpatialGrid)
    grid_temporal = models.ForeignKey(TemporalGrid)
    spacing_temporal = models.CharField(
        max_length=1,
        choices=(
            ('D', 'Daily'),
            ('M', 'Monthly'),
        )
    )
    description   = models.TextField()
    objects       = models.GeoManager()

