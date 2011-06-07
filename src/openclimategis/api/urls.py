from django.conf.urls.defaults import *
from piston.resource import Resource
import api.handlers as handlers


#helloworld_handler = Resource(handlers.HelloWorldHandler)
archive_handler = Resource(handlers.ArchiveHandler)
climatemodel_handler = Resource(handlers.ClimateModelHandler)
experiment_handler = Resource(handlers.ExperimentHandler)
variable_handler = Resource(handlers.VariableHandler)
spatial_handler = Resource(handlers.SpatialHandler)

#archive_regex = r'^archives/$|^archives/(?P<code>[^/]+)'

urlpatterns = patterns('',
                       
#(r'^helloworld/$|^helloworld/(?P<model_name>[^/]+)',
# helloworld_handler,
# {'emitter_format':'helloworld'}),

## TEST URLS ONLY !!!!!!! ------------------------------------------------------

(r'^test/shz/$',
 spatial_handler,
 {'emitter_format':'shz'}),

## ARCHIVES --------------------------------------------------------------------

(r'^archives/$|archives\.html|^archives/(?P<code>[^/]+)/$',
 archive_handler,
 {'emitter_format':'html'}),

(r'^archives/(?P<code>.*)\.html',
 archive_handler,
 {'emitter_format':'html'}),

(r'^archives.json|^archives/(?P<code>.*)\.json',
 archive_handler,
 {'emitter_format':'json'}),

## CLIMATE MODELS --------------------------------------------------------------

(r'^models/$|models\.html|^models/(?P<code>[^/]+)/$',
 climatemodel_handler,
 {'emitter_format':'html'}),

(r'^models/(?P<code>.*)\.html',
 climatemodel_handler,
 {'emitter_format':'html'}),

(r'^models.json|^models/(?P<code>.*)\.json',
 climatemodel_handler,
 {'emitter_format':'json'}),

## EXPERIMENTS -----------------------------------------------------------------

(r'^experiments/$|experiments\.html|^experiments/(?P<code>[^/]+)/$',
 experiment_handler,
 {'emitter_format':'html'}),

(r'^experiments/(?P<code>.*)\.html',
 experiment_handler,
 {'emitter_format':'html'}),

(r'^experiments.json|^experiments/(?P<code>.*)\.json',
 experiment_handler,
 {'emitter_format':'json'}),

## VARIABLES -----------------------------------------------------------------

(r'^variables/$|variables\.html|^variables/(?P<code>[^/]+)/$',
 variable_handler,
 {'emitter_format':'html'}),

(r'^variables/(?P<code>.*)\.html',
 variable_handler,
 {'emitter_format':'html'}),

(r'^variables.json|^variables/(?P<code>.*)\.json',
 variable_handler,
 {'emitter_format':'json'}),

                       )

#print archive_regex + '.json'