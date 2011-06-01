from django.conf.urls.defaults import *
from piston.resource import Resource
from api.handlers import HelloWorldHandler, IntersectsHandler


helloworld_handler = Resource(HelloWorldHandler)
intersects_handler = Resource(IntersectsHandler)

urlpatterns = patterns('',
                       (r'^helloworld/$|^helloworld/(?P<model_name>[^/]+)',
                        helloworld_handler,
                        {'emitter_format':'helloworld'})
                       )

urlpatterns += patterns('',
                        (r'^shz/$',intersects_handler,{'emitter_format':'shz'})
                        )

r'^archive/(?P<archive>[^/]+)/model/(?P<model>[^/]+)/grid/'