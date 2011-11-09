from django import forms
#from climatedata import models
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render_to_response
from django.core.exceptions import ValidationError
#from django.core.context_processors import csrf
from django.template.context import RequestContext
from shapely import wkt
from util.helpers import reverse_wkt, get_temp_path
import pdb
import os
import zipfile
from util.ncconv.experimental.helpers import get_wkt_from_shp
from climatedata.models import UserGeometryData
from django.contrib.gis.geos.geometry import GEOSGeometry
from django.contrib.gis.geos.collections import MultiPolygon


CHOICES_AGGREGATE = [
    ('true','TRUE'),
    ('false','FALSE'),
]
CHOICES_SOP = [
    ('intersects','Intersects'),
    ('clip','Clip'),
]
CHOICES_EXT = [
    ('geojson','GeoJSON Text File'),
    ('csv','Comma Separated Value'),
    ('kcsv','Linked Comma Separated Value (zipped)'),
    ('shz','ESRI Shapefile (zipped)'),
    ('kml','Keyhole Markup Language'),
    ('kmz','Keyhole Markup Language (zipped)'),
]

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
        
        drange_lower = forms.DateField(
            required=True,
            initial='1/1/2000',
            label='Lower Date Range',
            validators=[_validate_drange_lower],
        )
        drange_upper = forms.DateField(
            required=True,
            initial='3/1/2000',
            label='Upper Date Range',
            validators=[_validate_drange_upper],
        )
        wkt_extent = OcgWktField(
            required=True,
            label='WKT Extent',
            widget=forms.Textarea(attrs={'cols': 80, 'rows': 10}),
            initial='POLYGON ((-104 39, -95 39, -95 44, -104 44, -104 39))',
        )
        aggregate = forms.ChoiceField(
            choices=CHOICES_AGGREGATE,
            initial='true',
        )
        spatial_op = forms.ChoiceField(
            choices=CHOICES_SOP,
            initial='intersects',
            label='Spatial Operation',
        )
        extension = forms.ChoiceField(
            choices=CHOICES_EXT,
            label='Format'
        )
        
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
            url = ('/api'
                   '/archive/{archive}/model'
                   '/{climate_model}/scenario/{scenario}'
                   '/run/{run}'
                   '/temporal/{drange_lower}+{drange_upper}'
                   '/spatial/{spatial_op}+{wkt_extent}'
                   '/aggregate/{aggregate}'
                   '/variable/{variable}.{extension}').format(**form.cleaned_data)
            return HttpResponseRedirect(url) # Redirect after POST
    else:
        form = SpatialQueryForm() # An unbound form
        
    return render_to_response('query.html',
                              {'form': form, 'request': request},
                              context_instance=RequestContext(request))

## SHAPEFILE UPLOAD ------------------------------------------------------------

def validate_zipfile(value):
    if not os.path.splitext(value.name)[1] == '.zip':
        raise(ValidationError("File extension not '.zip'"))

class UploadShapefileForm(forms.Form):
#    uid = forms.CharField(max_length=50,min_length=1,initial='foo',label='UID')
    objectid = forms.IntegerField(label='ObjectID',initial=1)
    file = forms.FileField(label='Zipped Shapefile',
                           validators=[validate_zipfile])
    
    
def display_shpupload(request):
    if request.method == 'POST':
        form = UploadShapefileForm(request.POST,request.FILES)
        if form.is_valid():
            wkt = handle_uploaded_shapefile(request.FILES['file'],
                                            form.cleaned_data['objectid'])
            geom = GEOSGeometry(wkt,srid=4326)
            obj = UserGeometryData(geom=geom)
            obj.save()
            return(HttpResponse('Your geometry ID is: {0}'.format(obj.pk)))
    else:
        form = UploadShapefileForm()
    return(render_to_response('shpupload.html', {'form': form}))

def handle_uploaded_shapefile(file,objectid):
    path = get_temp_path(nest=True,suffix='.zip')
    dir = os.path.split(path)[0]
    ## write the data to file
    with open(path,'wb+') as dest:
        for chunk in file.chunks():
            dest.write(chunk)
    ## unzip the file
    zip = zipfile.ZipFile(path,'r')
    try:
        zip.extractall(os.path.split(path)[0])
    finally:
        zip.close()
    ## get the shapefile path
    for f in os.listdir(dir):
        if f.endswith('.shp'):
            break
    ## extract the wkt
    wkt = get_wkt_from_shp(os.path.join(dir,f),objectid)
    return(wkt)