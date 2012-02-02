from django.conf.urls.defaults import *
from piston.resource import Resource
import api.handlers as handlers


#helloworld_handler = Resource(handlers.HelloWorldHandler)
api_handler = Resource(handlers.ApiHandler)
archive_handler = Resource(handlers.ArchiveHandler)
climatemodel_handler = Resource(handlers.ClimateModelHandler)
scenario_handler = Resource(handlers.ScenarioHandler)
variable_handler = Resource(handlers.VariableHandler)
simulationoutput_handler = Resource(handlers.SimulationOutputHandler)
usergeometrydata_handler = Resource(handlers.AoiHandler)
spatial_handler = Resource(handlers.SpatialHandler)
query_handler = Resource(handlers.QueryHandler)
aoiupload_handler = Resource(handlers.AoiUploadHandler)
metacontent_handler = Resource(handlers.MetacontentHandler)

## REGEX VARIABLES -------------------------------------------------------------

nonspatial_formats = '|'.join([
    'html',
    'json',
])
spatial_formats = '|'.join([
    'html',
    'json',
    'kml',
    'kmz',
    'shz',
    'lshz',
    'csv',
    'kcsv',
    'nc'
])

re_archive = 'archive/(?P<archive>.*)'
re_model = 'model/(?P<model>.*)'
re_scenario = 'scenario/(?P<scenario>.*)'
re_run = 'run/(?P<run>.*)'
re_temporal = 'temporal/(?P<temporal>.*)'
re_spatial = 'spatial/(?P<operation>intersects|clip)\+(?P<aoi>.*)'
re_aggregate = 'aggregate/(?P<aggregate>true|false)'
re_variable = 'variable/(?P<variable>.*)'
#re_format = '\.(?P<emitter_format>.*)'
re_spatial_format = '\.(?P<emitter_format>{0})'.format(spatial_formats)
re_nonspatial_format = '\.(?P<emitter_format>{0})'.format(nonspatial_formats)
re_urlslug = '(?P<urlslug>[\.\(\)A-Za-z0-9_-]+)'
re_code = '(?P<code>[\.\(\)A-Za-z0-9_-]+)'
re_id = '(?P<id>\d+)'

urlpatterns = patterns('',
                       
#(r'^helloworld/$|^helloworld/(?P<model_name>[^/]+)',
# helloworld_handler,
# {'emitter_format':'helloworld'}),

## TEST URLS ONLY !!!!!!! ------------------------------------------------------

## FULL URL FOR VARIABLE QUERY
## /test/archive/cmip3/
##       model/ncar_ccsm3_0/
##       scenario/1pctto2x/
##       temporal/(1997-07-16|<from>+<to>)/
##       spatial/<operation(intersects|clip)>+(<wkt>|<aoi id>)/
##       aggregate/(true|false)/
##       variable/<variable name>.<format>

## URL FOR RETURNING MODEL GRID
## /test/archive/cmip3/
##       model/ncar_ccsm3_0/
##       scenario/1pctto2x/
##       spatial/<operation(intersects|clip)>+(<wkt>|<aoi id>)/
##       grid.<format>

### FUNCTION DEFINITION JSON ---------------------------------------------------

    (r'^functions.json','api.views.get_function_json'),
    
### AOI DEFINITION JSON

    (r'^aois.json','api.views.get_aois_json'),

### METACONTENT ----------------------------------------------------------------

    url(r'^{re_archive}/{re_model}/{re_scenario}/{re_run}/{re_temporal}/{re_spatial}/'
         '{re_aggregate}/{re_variable}\.meta'.format(
         re_archive=re_archive,
         re_model=re_model,
         re_scenario=re_scenario,
         re_temporal=re_temporal,
         re_spatial=re_spatial,
         re_aggregate=re_aggregate,
         re_variable=re_variable,
         re_run=re_run),
       metacontent_handler,
       {'emitter_format':'meta'}
     ),

## SPATIAL QUERY ---------------------------------------------------------------

((r'^{re_archive}/{re_model}/{re_scenario}/{re_run}/{re_temporal}/{re_spatial}/'
   '{re_aggregate}/{re_variable}{re_format}'.format(
    re_archive=re_archive,
    re_model=re_model,
    re_scenario=re_scenario,
    re_temporal=re_temporal,
    re_spatial=re_spatial,
    re_aggregate=re_aggregate,
    re_variable=re_variable,
    re_run=re_run,
    re_format=re_spatial_format,
    )),
   spatial_handler
 ),

#(r'^test/shz/($|(?P<spatial_op>intersect|intersection)_(?P<model>dissolve|grid)\.(?P<emitter_format>))',
# spatial_handler,
# {'emitter_format':'shz'}),

    url(
        r'^$|^.html$', 
        api_handler, 
        {'emitter_format':'html', 'template_name':'api.html'},
        name='api_list',
    ),

### ARCHIVES --------------------------------------------------------------------
    # collection of climate model archive resources
    #Note: the following combines url configurations, but defaults to json 
    # as the emitter format.
    #r'^archives(?:\.(?P<emitter_format>.*))?$'.format(re_format=re_format),
    url(
        (r'^archives{re_format}$').format(
            re_format=re_nonspatial_format,
        ),
        archive_handler, 
        {
            'template_name':'Archive.html',
            'is_collection':True,
        },
        name='archive_list',
    ),
    # use HTML for when the output format is not specified
    url(
        r'^archives/?$',
        archive_handler, 
        {
            'emitter_format':'html',
            'template_name':'Archive.html',
            'is_collection':True,
        },
        name='archive_list',
    ),
    # a single climate model archive resource
    url(
        (r'^archives/{re_urlslug}{re_format}$').format(
            re_urlslug=re_urlslug,
            re_format=re_nonspatial_format,
        ),
        archive_handler,
        {
            'template_name':'Archive.html',
            'is_collection':False,
        },
        name='archive_single',
    ),
    # use HTML for when the output format is not specified
    url(
        r'^archives/{re_urlslug}/?$'.format(
            re_urlslug=re_urlslug,
        ),
        archive_handler,
        {
            'emitter_format':'html',
            'template_name':'Archive.html',
            'is_collection':False,
        },
        name='archive_single',
    ),

### CLIMATE MODELS --------------------------------------------------------------
    # collection of climate model resources
    url(
        (r'^models{re_format}$').format(
            re_format=re_nonspatial_format,
        ),
        climatemodel_handler, 
        {
            'template_name':'ClimateModel.html',
            'is_collection':True,
        },
        name='climatemodel_list',
    ),
    # use HTML for when the output format is not specified
    url(
        r'^models/?$',
        climatemodel_handler, 
        {
            'emitter_format':'html',
            'template_name':'ClimateModel.html',
            'is_collection':True,
        },
        name='climatemodel_list',
    ),
    # a single climate model resource
    url(
        (r'^models/{re_urlslug}{re_format}$').format(
            re_urlslug=re_urlslug,
            re_format=re_nonspatial_format,
        ),
        climatemodel_handler,
        {
            'template_name':'ClimateModel.html',
            'is_collection':False,
        },
        name='climatemodel_single',
    ),
    # use HTML for when the output format is not specified
    url(
        r'^models/{re_urlslug}/?$'.format(
            re_urlslug=re_urlslug,
        ),
        climatemodel_handler,
        {
            'emitter_format':'html',
            'template_name':'ClimateModel.html',
            'is_collection':False,
        },
        name='climatemodel_single',
    ),

### EMISSIONS SCENARIOS -------------------------------------------------------

    # collection of emissions scenario resources
    url(
        (r'^scenarios{re_format}$').format(
            re_format=re_nonspatial_format,
        ),
        scenario_handler, 
        {
            'template_name':'Scenario.html',
            'is_collection':True,
        },
        name='scenario_list',
    ),
    # use HTML for when the output format is not specified
    url(
        r'^scenarios/?$',
        scenario_handler, 
        {
            'emitter_format':'html',
            'template_name':'Scenario.html',
            'is_collection':True,
        },
        name='scenario_list',
    ),
    # a single emissions resource
    url(
        (r'^scenarios/{re_urlslug}{re_format}$').format(
            re_urlslug=re_urlslug,
            re_format=re_nonspatial_format,
        ),
        scenario_handler,
        {
            'template_name':'Scenario.html',
            'is_collection':False,
        },
        name='scenario_single',
    ),
    # use HTML for when the output format is not specified
    url(
        r'^scenarios/{re_urlslug}/?$'.format(
            re_urlslug=re_urlslug,
        ),
        scenario_handler,
        {
            'emitter_format':'html',
            'template_name':'Scenario.html',
            'is_collection':False,
        },
        name='scenario_single',
    ),

### VARIABLES -----------------------------------------------------------------
    # collection of output variable resources
    url(
        (r'^variables{re_format}$').format(
            re_format=re_nonspatial_format,
        ),
        variable_handler, 
        {
            'template_name':'Variable.html',
            'is_collection':True,
        },
        name='variable_list',
    ),
    # use HTML for when the output format is not specified
    url(
        r'^variables/?$',
        variable_handler, 
        {
            'emitter_format':'html',
            'template_name':'Variable.html',
            'is_collection':True,
        },
        name='variable_list',
    ),
    # a single output variable resource
    url(
        (r'^variables/{re_urlslug}{re_format}$').format(
            re_urlslug=re_urlslug,
            re_format=re_nonspatial_format,
        ),
        variable_handler,
        {
            'template_name':'Variable.html',
            'is_collection':False,
        },
        name='variable_single',
    ),
    # use HTML for when the output format is not specified
    url(
        r'^variables/{re_urlslug}/?$'.format(
            re_urlslug=re_urlslug,
        ),
        variable_handler,
        {
            'emitter_format':'html',
            'template_name':'Variable.html',
            'is_collection':False,
        },
        name='variable_single',
    ),
    
### SIMULATION OUTPUT ---------------------------------------------------------

    # collection of simulation output resources
    url(
        (r'^simulations{re_format}$').format(
            re_format=re_nonspatial_format,
        ),
        simulationoutput_handler, 
        {
            'template_name':'SimulationOutput.html',
            'is_collection':True,
        },
        name='simulation_list',
    ),
    # use HTML for when the output format is not specified
    url(
        r'^simulations/?$',
        simulationoutput_handler, 
        {
            'emitter_format':'html',
            'template_name':'SimulationOutput.html',
            'is_collection':True,
        },
        name='simulation_list',
    ),
    # a single simulation output resource
    url(
        (r'^simulations/{re_id}{re_format}$').format(
            re_id=re_id,
            re_format=re_nonspatial_format,
        ),
        simulationoutput_handler,
        {
            'template_name':'SimulationOutput.html',
            'is_collection':False,
        },
        name='simulation_single',
    ),
    # use HTML for when the output format is not specified
    url(
        r'^simulations/{re_id}/?$'.format(
            re_id=re_id,
        ),
        simulationoutput_handler,
        {
            'emitter_format':'html',
            'template_name':'SimulationOutput.html',
            'is_collection':False,
        },
        name='simulation_single',
    ),

### AREA OF INTEREST (USER GEOMETRY DATA) ------------------------------------

    # collection of AOI resources
    url(
        (r'^aois{re_format}$').format(
            re_format=re_spatial_format,
        ),
        usergeometrydata_handler, 
        {
            'template_name':'aoi.html',
            'is_collection':True,
        },
        name='aoi_list',
    ),
    # use HTML for when the output format is not specified
    url(
        r'^aois/?$',
        usergeometrydata_handler, 
        {
            'emitter_format':'html',
            'template_name':'aoi.html',
            'is_collection':True,
        },
        name='aoi_list',
    ),
    # a single area-of-interest resource
    url(
        (r'^aois/{re_code}{re_format}$').format(
            re_code=re_code,
            re_format=re_spatial_format,
        ),
        usergeometrydata_handler,
        {
            'template_name':'aoi.html',
            'is_collection':False,
        },
        name='aoi_single',
    ),
    # use HTML for when the output format is not specified
    url(
        r'^aois/{re_code}/?$'.format(
            re_code=re_code,
        ),
        usergeometrydata_handler,
        {
            'emitter_format':'html',
            'template_name':'aoi.html',
            'is_collection':False,
        },
        name='aoi_single',
    ),
    
### QUERY BUILDER -------------------------------------------------------------
#    url(
#        r'^query/',
#        'api.views.display_spatial_query'
#    ),
    url(( # example: archive/usgs-cida-maurer/model/ccsm3/scenario/sres-a1b/variable/tas/run/2/query.html
        r'^{re_archive}' + \
         '/{re_model}' + \
         '/{re_scenario}' + \
         '/{re_variable}' + \
         '/{re_run}' + \
         '/query{re_format}'
         ).format(
            re_archive=re_archive,
            re_model=re_model,
            re_scenario=re_scenario,
            re_variable=re_variable,
            re_run=re_run,
            re_format=re_spatial_format,
        ),
        query_handler,
        {
            'emitter_format':'html',
            'template_name':'query.html',
        },
        name='query_form',
    ),

    # Query Builder web application
    url( # example: query/builder
        r'^query/builder',
        'api.views.display_query_builder_app'
    ),
    
### SHAPEFILE UPLOAD ----------------------------------------------------------

    url(( # example: aoi_upload.html
        r'^aoi_upload.html'),
        aoiupload_handler,
        {
            'emitter_format':'html',
            'template_name':'aoi_upload.html',
        },
        name='aoi_upload_form',
    ),
    
)
#
##print archive_regex + '.json'
