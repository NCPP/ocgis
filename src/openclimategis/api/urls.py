from django.conf.urls.defaults import *
from piston.resource import Resource
from api.handlers import HelloWorldHandler


helloworld_handler = Resource(HelloWorldHandler)

urlpatterns = patterns('',
                       (r'^helloworld/$|^helloworld/(?P<model_name>[^/]+)',
                        helloworld_handler,
                        {'emitter_format':'helloworld'})
                       )