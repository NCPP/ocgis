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
spatial_handler = Resource(handlers.SpatialHandler)

## REGEX VARIABLES -------------------------------------------------------------

nonspatial_formats = '|'.join([
    'html',
    'json',
    #'csv',
    #'kcsv',
])
spatial_formats = '|'.join([
    'html',
    'json',
    'kml',
    'kmz',
    'shz',
])

re_archive = 'archive/(?P<archive>.*)'
re_model = 'model/(?P<model>.*)'
re_scenario = 'scenario/(?P<scenario>.*)'
re_run = 'run/(?P<run>.*)'
re_temporal = 'temporal/(?P<temporal>.*)'
re_spatial = 'spatial/(?P<operation>intersects|clip)\+(?P<aoi>.*)'
re_aggregate = 'aggregate/(?P<aggregate>true|false)'
re_variable = 'variable/(?P<variable>.*)\.(?P<emitter_format>.*)'
re_urlslug = '(?P<urlslug>[\.\(\)A-Za-z0-9_-]+)'
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

((r'^test/{re_archive}/{re_model}/{re_scenario}/{re_run}/{re_temporal}/{re_spatial}/'
   '{re_aggregate}/{re_variable}'.format(
    re_archive=re_archive,
    re_model=re_model,
    re_scenario=re_scenario,
    re_temporal=re_temporal,
    re_spatial=re_spatial,
    re_aggregate=re_aggregate,
    re_variable=re_variable,
    re_run=re_run)),
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
    # collection of climate model archives
    url(
        r'^archives/?$|^archives\.html$',
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
        r'^archives/{re_urlslug}.(?P<emitter_format>{formats})$'.format(
            formats=nonspatial_formats,
            re_urlslug=re_urlslug,
        ),
        archive_handler,
        {
            'template_name':'Archive.html',
            'is_collection':False,
        },
        name='archive_single',
    ),
    url(
        r'^archives/{re_urlslug}$'.format(
            re_urlslug=re_urlslug,
        ),
        archive_handler,
        {
            'emitter_format':'html',
            'template_name':'Archive.html',
            'is_collection':False,
        },
        name='archive_single_default',
    ),

### CLIMATE MODELS --------------------------------------------------------------
    # collection of climate models
    url(
        r'^models/?$|^models\.html$',
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
        r'^models/{re_urlslug}.(?P<emitter_format>{formats})$'.format(
            formats=nonspatial_formats,
            re_urlslug=re_urlslug,
        ),
        climatemodel_handler,
        {
            'template_name':'ClimateModel.html',
            'is_collection':False,
        },
        name='climatemodel_single',
    ),
    url(
        r'^models/{re_urlslug}$'.format(
            re_urlslug=re_urlslug,
        ),
        climatemodel_handler,
        {
            'emitter_format':'html',
            'template_name':'ClimateModel.html',
            'is_collection':False,
        },
        name='climatemodel_single_default',
    ),

### EMISSIONS SCENARIOS -------------------------------------------------------
    # collection of emissions scenarios
    url(
        r'^scenarios/?$|^scenarios\.html$',
        scenario_handler, 
        {
            'emitter_format':'html',
            'template_name':'Scenario.html',
            'is_collection':True,
        },
        name='scenario_list',
    ),
    # a single emissions scenario resource
    url(
        r'^scenarios/{re_urlslug}.(?P<emitter_format>{formats})$'.format(
            formats=nonspatial_formats,
            re_urlslug=re_urlslug,
        ),
        scenario_handler,
        {
            'template_name':'Scenario.html',
            'is_collection':False,
        },
        name='scenario_single',
    ),
    url(
        r'^scenarios/{re_urlslug}$'.format(
            re_urlslug=re_urlslug,
        ),
        scenario_handler,
        {
            'emitter_format':'html',
            'template_name':'Scenario.html',
            'is_collection':False,
        },
        name='scenario_single_default',
    ),

### VARIABLES -----------------------------------------------------------------
    # collection of output variables
    url(
        r'^variables/?$|^variables\.html$',
        variable_handler, 
        {
            'emitter_format':'html',
            'template_name':'Variable.html',
            'is_collection':True,
        },
        name='variable_list',
    ),
    # a single emissions scenario resource
    url(
        r'^variables/{re_urlslug}.(?P<emitter_format>{formats})$'.format(
            formats=nonspatial_formats,
            re_urlslug=re_urlslug,
        ),
        variable_handler,
        {
            'template_name':'Variable.html',
            'is_collection':False,
        },
        name='variable_single',
    ),
    url(
        r'^variables/{re_urlslug}$'.format(
            re_urlslug=re_urlslug,
        ),
        variable_handler,
        {
            'emitter_format':'html',
            'template_name':'Variable.html',
            'is_collection':False,
        },
        name='variable_single_default',
    ),
    
### SIMULATION OUTPUT ---------------------------------------------------------
    # collection of simulation outputs
    url(
        r'^simulations/?$|^simulations\.html$',
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
        r'^simulations/{re_id}.(?P<emitter_format>{formats})$'.format(
            re_id=re_id,
            formats=nonspatial_formats,
        ),
        simulationoutput_handler,
        {
            'emitter_format':'html',
            'template_name':'SimulationOutput.html',
            'is_collection':False,
        },
        name='simulation_single',
    ),
    url(
        r'^simulations/{re_id}$'.format(
            re_id=re_id,
        ),
        simulationoutput_handler,
        {
            'emitter_format':'html',
            'template_name':'SimulationOutput.html',
            'is_collection':False,
        },
        name='simulation_single_default',
    ),
)
#
##print archive_regex + '.json'