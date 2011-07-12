from piston.emitters import Emitter
from django.http import HttpResponse
from util.shapes.views.export import ShpResponder
from util.toshp import OpenClimateShp
from util.helpers import get_temp_path
from django.core import serializers
import geojson
from django.shortcuts import render_to_response


class OpenClimateEmitter(Emitter):
    """
    Superclass for all OpenClimateGIS emitters.
    """
    def render(self,request):
        raise NotImplementedError
    
class GeometryEmitter(OpenClimateEmitter):
    
    def construct(self):
        return self.data
    
#    def construct(self):
#        import ipdb;ipdb.set_trace()
    

class HelloWorldEmitter(OpenClimateEmitter):
    
    def render(self,request):
        names = [n['name'] for n in self.construct()]
        msg = 'Hello, World!! The climate model names are:<br><br>{0}'.format('<br>'.join(names))
        return HttpResponse(msg)
    
    
class HtmlEmitter(OpenClimateEmitter):
    
    def render(self,request):
#        import pdb;pdb.set_trace()
        return render_to_response('archives.html',self.construct())
#        return HttpResponse(str(self.construct()))

   
class ShapefileEmitter(GeometryEmitter):
    """
    Emits zipped shapefile (.shz)
    """
    
    def render(self,request):
        dl = self.construct()
        path = get_temp_path(suffix='.shp')
#        import ipdb;ipdb.set_trace()
#        try:
        shp = OpenClimateShp(path,dl)
#        except:
#            import ipdb;ipdb.set_trace()
        shp.write()
        return shp.zip_response()
    
    
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
        
        
class GeoJsonEmitter(GeometryEmitter):
    """
    JSON format for geospatial data (.json)
    """
    
    def render(self,request):
        ## return the data from a spatial handler
        data = self.construct()
        ## this list holds the features to add to a feature collection
        features = []
        ## loop for each row/feature in the returned data
        for row in data:
            ## construct the geometry
            geom = geojson.MultiPolygon(row.pop('geom').coords)
            ## pull the unique identifier
            id = row.pop('id')
            ## change the timestamp to a string
            row['timestamp'] = str(row['timestamp'])
            ## construct the feature
#            import ipdb;ipdb.set_trace()
            feature = geojson.Feature(id=id,geometry=geom,properties=row)
            ## append the feature to the collection
            features.append(feature)
        ## make the feature collection
        fc = geojson.FeatureCollection(features)
        
        return(geojson.dumps(fc))
    
#Emitter.register('helloworld',HelloWorldEmitter,'text/html; charset=utf-8')
Emitter.register('html',HtmlEmitter,'text/html; charset=utf-8')
Emitter.register('shz',ShapefileEmitter,'application/zip; charset=utf-8')
#Emitter.unregister('json')
Emitter.register('geojson',GeoJsonEmitter,'application/geojson; charset=utf-8')