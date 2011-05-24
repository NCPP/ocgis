from piston.emitters import Emitter
from django.http import HttpResponse


class OpenClimateEmitter(Emitter):
    """
    Superclass for all OpenClimateGIS emitters.
    """
    def render(self,request):
        raise NotImplementedError
    

class HelloWorldEmitter(OpenClimateEmitter):
    
    def render(self,request):
        names = [n['name'] for n in self.construct()]
        msg = 'Hello, World!! The climate model names are:<br><br>{0}'.format('<br>'.join(names))
        return HttpResponse(msg)

   
class ShapefileEmitter(OpenClimateEmitter):
    """
    Emits zipped shapefile (.shz)
    """
    
    def render(self,request):
        pass
    
    
class KmlEmitter(OpenClimateEmitter):
    """
    Emits raw KML (.kml)
    """

    def render(self,request):
        pass
    
    
class KmzEmitter(KmlEmitter):
    """
    Subclass of KmlEmitter. Emits KML in a zipped format (.kmz)
    """
    
    def render(self,request):
        kml = super(KmzEmitter,self).render()
        
        
class GeoJsonEmitter(OpenClimateEmitter):
    """
    JSON format for geospatial data (.json)
    """
    
    def render(self,request):
        pass
    

Emitter.register('helloworld',HelloWorldEmitter,'text/html; charset=utf-8')
