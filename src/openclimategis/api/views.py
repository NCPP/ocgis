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


def get_choices(model,code_field='urlslug',desc_field='name',distinct=False,select=True):
    qs = model.objects.order_by(desc_field).values(code_field,desc_field).all()
    if distinct:
        qs = qs.distinct()
    if select:
        choices = [('_select_','Please Make a Selection...')]
    else:
        choices = []
    for row in qs:
        ctup = (row[code_field],row[desc_field])
        choices.append(ctup)
    return(choices)

def validate_choice(value):
    if value == '_select_':
        raise ValidationError('Please make a selection.')


class OcgChoiceField(forms.ChoiceField):
    default_validators = [validate_choice]
    
    
class OcgWktField(forms.CharField):
    
    def clean(self,value):
        try:
            wkt.loads(value)
            return(reverse_wkt(value))
        except:
            raise ValidationError('Unable to parse WKT.')


class SpatialQueryForm(forms.Form):
    archive = OcgChoiceField(
                choices=get_choices(models.Archive))
    climate_model = OcgChoiceField(
                      choices=get_choices(models.ClimateModel),
                      label='Climate Model')
    scenario = OcgChoiceField(
                 choices=get_choices(models.Scenario))
    run = OcgChoiceField(
            choices=get_choices(models.SimulationOutput,'run','run',True,False),
            initial=1)
    drange_lower = forms.DateField(required=True,label='Lower Date Range')
    drange_upper = forms.DateField(required=True,label='Upper Date Range')
    wkt_extent = OcgWktField(required=True,label='WKT Extent')
    aggregate = forms.ChoiceField(choices=CHOICES_AGGREGATE,
                                  initial='true')
    spatial_op = forms.ChoiceField(choices=CHOICES_SOP,
                                   initial='intersects',
                                   label='Spatial Operation')
    variable = OcgChoiceField(choices=get_choices(models.Variable,
                                                  desc_field='name'))
    extension = forms.ChoiceField(choices=CHOICES_EXT)


def display_spatial_query(request):
    if request.method == 'POST': # If the form has been submitted...
        form = SpatialQueryForm(request.POST) # A form bound to the POST data
        if form.is_valid(): # All validation rules pass
            # Process the data in form.cleaned_data
            ## fill in the URL string
            url = ('/api'
                   '/archive/{archive}/model'
                   '/{climate_model}/scenario/{scenario}'
                   '/run/{run}'
                   '/temporal/{drange_lower}+{drange_upper}'
                   '/spatial/{spatial_op}+{wkt_extent}'
                   '/aggregate/{aggregate}'
                   '/variable/{variable}.{extension}').format(**form.cleaned_data)
            pdb.set_trace()
            return HttpResponseRedirect(url) # Redirect after POST
    else:
        form = SpatialQueryForm() # An unbound form
        
    return render_to_response('query.html',
                              {'form': form,},
                              context_instance=RequestContext(request))