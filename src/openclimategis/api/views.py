from django import forms
from climatedata import models
from django.http import HttpResponseRedirect
from django.shortcuts import render_to_response
from django.core.exceptions import ValidationError
from django.core.context_processors import csrf
from django.template.context import RequestContext
from shapely import wkt
import pdb
from util.helpers import reverse_wkt


CHOICES_AGGREGATE = [('true','TRUE'),
                     ('false','FALSE')]
CHOICES_SOP = [('intersects','Intersects'),
               ('clip','Clip')]
CHOICES_EXT = [('shz','Zipped Shapefile'),
               ('csv','Comma Separated Value'),
               ('kcsv','Linked Comma Separated Value'),
               ('geojson','GeoJSON Text File')]


#def get_choices(model,code_field='urlslug',desc_field='name',distinct=False,select=True):
#    qs = model.objects.order_by(desc_field).values(code_field,desc_field).all()
#    if distinct:
#        qs = qs.distinct()
#    if select:
#        choices = [('_select_','Please Make a Selection...')]
#    else:
#        choices = []
#    for row in qs:
#        ctup = (row[code_field],row[desc_field])
#        choices.append(ctup)
#    return(choices)
#
#def validate_choice(value):
#    if value == '_select_':
#        raise ValidationError('Please make a selection.')
#
#class OcgChoiceField(forms.ChoiceField):
#    default_validators = [validate_choice]


def get_SpatialQueryForm(simulation_output):
    ## the dataset object contains the test values
    dataset = simulation_output.netcdf_variable.netcdf_dataset
    
    ## validators and custom field classes for the dynamic form ----------------
    
    def _validate_drange_lower(value):
        if value > dataset.temporal_max.date():
            raise(ValidationError('Lower date range exceeds maximum dataset date.'))
    
    def _validate_drange_upper(value):
        if value < dataset.temporal_min.date():
            raise(ValidationError('Upper date range less than minimum dataset date.'))
        
    class OcgWktField(forms.CharField):
    
        def clean(self,value):
            ## check the geometry is valid
            try:
                geom = wkt.loads(value)
            except:
                raise(ValidationError('Unable to parse WKT.'))
            ## check that spatial operations will return data
            ogeom = wkt.loads(dataset.spatial_extent.wkt)
            if not geom.intersects(ogeom):
                raise(ValidationError('Input geometry will return an empty intersection.'))
            ## convert WKT to a format acceptable for the URL
            return(reverse_wkt(value))
        
    ## -------------------------------------------------------------------------
    
    class SpatialQueryForm(forms.Form):
        
        drange_lower = forms.DateField(required=True,
                                       initial='1/1/2000',
                                       label='Lower Date Range',
                                       validators=[_validate_drange_lower])
        drange_upper = forms.DateField(required=True,
                                       initial='3/1/2000',
                                       label='Upper Date Range',
                                       validators=[_validate_drange_upper])
        wkt_extent = OcgWktField(required=True,
                                 label='WKT Extent',
                                 initial='POLYGON ((-104 39, -95 39, -95 44, -104 44, -104 39))')
        aggregate = forms.ChoiceField(choices=CHOICES_AGGREGATE,
                                      initial='true')
        spatial_op = forms.ChoiceField(choices=CHOICES_SOP,
                                       initial='intersects',
                                       label='Spatial Operation')
        extension = forms.ChoiceField(choices=CHOICES_EXT)
        
        def clean(self):
            if self.is_valid():
                ## test that dates are not switched or equal
                if self.cleaned_data['drange_lower'] >= self.cleaned_data['drange_upper']:
                    raise(ValidationError('Date range values equal or switched.'))
            return(self.cleaned_data)
        
    return(SpatialQueryForm)


def display_spatial_query(request):
    ## get the dynamically generated form class
    SpatialQueryForm = get_SpatialQueryForm(request.ocg.simulation_output)
    ## process the request
    if request.method == 'POST': # If the form has been submitted...
        form = SpatialQueryForm(request.POST) # A form bound to the POST data
        if form.is_valid(): # All validation rules pass
            ## merge keyword arguments for url string
            form.cleaned_data.update(dict(archive=request.ocg.archive.urlslug,
                                          climate_model=request.ocg.climate_model.urlslug,
                                          run=request.ocg.run,
                                          variable=request.ocg.variable.urlslug,
                                          scenario=request.ocg.scenario.urlslug))
            ## fill in the URL string
#            pdb.set_trace()
            url = ('/api'
                   '/archive/{archive}/model'
                   '/{climate_model}/scenario/{scenario}'
                   '/run/{run}'
                   '/temporal/{drange_lower}+{drange_upper}'
                   '/spatial/{spatial_op}+{wkt_extent}'
                   '/aggregate/{aggregate}'
                   '/variable/{variable}.{extension}').format(**form.cleaned_data)
#            pdb.set_trace()
            return HttpResponseRedirect(url) # Redirect after POST
    else:
        form = SpatialQueryForm() # An unbound form
        
    return render_to_response('query.html',
                              {'form': form,},
                              context_instance=RequestContext(request))