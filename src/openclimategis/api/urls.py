from django.conf.urls.defaults import *
from piston.resource import Resource
import api.handlers as handlers


#helloworld_handler = Resource(handlers.HelloWorldHandler)
#archive_handler = Resource(handlers.ArchiveHandler)
climatemodel_handler = Resource(handlers.ClimateModelHandler)
#experiment_handler = Resource(handlers.ExperimentHandler)
#variable_handler = Resource(handlers.VariableHandler)
spatial_handler = Resource(handlers.SpatialHandler)

## REGEX VARIABLES -------------------------------------------------------------

re_archive = 'archive/(?P<archive>.*)'
re_model = 'model/(?P<model>.*)'
re_scenario = 'scenario/(?P<scenario>.*)'
re_run = 'run/(?P<run>.*)'
re_temporal = 'temporal/(?P<temporal>.*)'
re_spatial = 'spatial/(?P<operation>intersects|clip)\+(?P<aoi>.*)'
re_aggregate = 'aggregate/(?P<aggregate>true|false)'
re_variable = 'variable/(?P<variable>.*)\.(?P<emitter_format>.*)'


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

### ARCHIVES --------------------------------------------------------------------
#
#(r'^archives/$|archives\.html|^archives/(?P<code>[^/]+)/$',
# archive_handler,
# {'emitter_format':'html'}),
#
#(r'^archives/(?P<code>.*)\.html',
# archive_handler,
# {'emitter_format':'html'}),
#
#(r'^archives.json|^archives/(?P<code>.*)\.json',
# archive_handler,
# {'emitter_format':'json'}),
#
### CLIMATE MODELS --------------------------------------------------------------

(r'^models.html$', climatemodel_handler, {'emitter_format':'html'}),
url(
    r'^model/(?P<code>.*).html$',
    climatemodel_handler,
    {'emitter_format':'html'},
    name='single_climatemodel',
),

#(r'^models/$|models\.html|^models/(?P<code>[^/]+)/$',
# climatemodel_handler,
# {'emitter_format':'html'}),
#
#(r'^models/(?P<code>.*)\.html',
# climatemodel_handler,
# {'emitter_format':'html'}),
#
#(r'^models.json|^models/(?P<code>.*)\.json',
# climatemodel_handler,
# {'emitter_format':'json'}),

### EXPERIMENTS -----------------------------------------------------------------
#
#(r'^experiments/$|experiments\.html|^experiments/(?P<code>[^/]+)/$',
# experiment_handler,
# {'emitter_format':'html'}),
#
#(r'^experiments/(?P<code>.*)\.html',
# experiment_handler,
# {'emitter_format':'html'}),
#
#(r'^experiments.json|^experiments/(?P<code>.*)\.json',
# experiment_handler,
# {'emitter_format':'json'}),
#
### VARIABLES -----------------------------------------------------------------
#
#(r'^variables/$|variables\.html|^variables/(?P<code>[^/]+)/$',
# variable_handler,
# {'emitter_format':'html'}),
#
#(r'^variables/(?P<code>.*)\.html',
# variable_handler,
# {'emitter_format':'html'}),
#
#(r'^variables.json|^variables/(?P<code>.*)\.json',
# variable_handler,
# {'emitter_format':'json'}),
#
)
#
##print archive_regex + '.json'