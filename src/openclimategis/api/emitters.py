from django.template.context import RequestContext
from django.shortcuts import render_to_response
from django.http import HttpResponse
from piston.emitters import Emitter
from util.toshp import OpenClimateShp
from util.helpers import get_temp_path
import pdb
from api.views import display_spatial_query
from util.ncconv.converters import as_geojson, as_tabular, as_keyTabular
import zipfile
import io
import os
import tempfile
from util.ncconv.converters import as_kml

import logging
logger = logging.getLogger(__name__)


class OpenClimateEmitter(Emitter):
    """
    Superclass for all OpenClimateGIS emitters.
    """
    
    def render(self,request):
        raise NotImplementedError


class IdentityEmitter(OpenClimateEmitter):
    """
    The standard Django Piston emitter does unnecessary computations when an
    emitter is searching for the raw data from its associated handler.
    """
    
    def construct(self):
        return self.data


class HelloWorldEmitter(OpenClimateEmitter):
    
    def render(self,request):
        names = [n['name'] for n in self.construct()]
        msg = 'Hello, World!! The climate model names are:<br><br>{0}'.format('<br>'.join(names))
        return HttpResponse(msg)


class HTMLEmitter(Emitter):
    """Emits an HTML representation 
    """
    def render(self,request):
        
        logger.info("starting HTMLEmitter.render()...")
        
        c = RequestContext(request)
        
        template_name = request.url_args.get('template_name')
        is_collection = request.url_args.get('is_collection')
        
        ## return data from the construct method of the resource's handler
        try:
            data = self.construct()
            logger.debug("len(data) = {0}".format(len(data)))
        except:
            data = []
            logger.debug("data is None!")
        
        ## form the basis dictionary for the template data
        dictionary = {'data': data, 'is_collection': is_collection}
        
        ## if we need the query form generate and pass accordingly
        if template_name == 'query.html':
            response = display_spatial_query(request)
        else:
            response = render_to_response(
                template_name=template_name, 
                dictionary=dictionary,
                context_instance=c,
            )
        
        logger.info("...ending HTMLEmitter.render()")
        
        return(response)
Emitter.register('html', HTMLEmitter, 'text/html; charset=utf-8')


class ShapefileEmitter(IdentityEmitter):
    """
    Emits zipped shapefile (.shz)
    """
    
    def render(self,request):
        logger.info("starting ShapefileEmitter.render()...")
        
        elements = self.construct()
        logger.debug("elements = {0}".format(elements))
        cfvar = request.ocg.simulation_output.variable.code
        path = str(os.path.join(tempfile.gettempdir(),cfvar+'.shp'))
        shp = OpenClimateShp(path,elements)
        paths = shp.write()
        response = shp.zip_response()
        for path in paths: os.remove(path)
        return(response)
        
        logger.info("...ending ShapefileEmitter.render()")


class KmlEmitter(IdentityEmitter):
    """
    Emits raw KML (.kml)
    """
    def render(self,request):
        from lxml import etree
        
        logger.info("starting KmlEmitter.render()...")
        
        ## return the elements
        elements = self.construct()
        ## conversion
        kml_doc = as_kml(elements, request=request)
        # return a string representation of the KML document
        return(etree.tostring(kml_doc, pretty_print=True))
        
        logger.info("...ending KmlEmitter.render()")
Emitter.register('kml',KmlEmitter,'application/vnd.google-earth.kml+xml')


class KmzEmitter(KmlEmitter):
    """
    Subclass of KmlEmitter. Emits KML in a zipped format (.kmz)
    """
    
    def render(self,request):
        logger.info("starting KmzEmitter.render()...")
        kml = super(KmzEmitter,self).render(request)
        iobuffer = io.BytesIO()
        zf = zipfile.ZipFile(
            iobuffer, 
            mode='w',
            compression=zipfile.ZIP_DEFLATED, 
        )
        try:
            zf.writestr('doc.kml',kml)
        finally:
            zf.close()
        iobuffer.flush()
        zip_stream = iobuffer.getvalue()
        iobuffer.close()
        logger.info("...ending KmzEmitter.render()")
        return(zip_stream)
Emitter.register('kmz',KmzEmitter,'application/vnd.google-earth.kmz')


class GeoJsonEmitter(IdentityEmitter):
    """
    JSON format for geospatial data (.json)
    """
    
    def render(self,request):
        ## return the elements
        elements = self.construct()
        ## conversion
        conv = as_geojson(elements)
        return(conv)


class CsvEmitter(IdentityEmitter):
    """
    Tabular CSV format. (.csv)
    """
    
    def render(self,request):
        elements = self.construct()
        var = request.ocg.simulation_output.netcdf_variable.code
        conv = as_tabular(elements,var)
        return(conv)


class CsvKeyEmitter(IdentityEmitter):
    """
    Tabular CSV format reduced to relational tables. (.csv)
    """
    
    def render(self,request):
        elements = self.construct()
        var = request.ocg.simulation_output.netcdf_variable.code
        cfvar = request.ocg.simulation_output.variable.code
        path = os.path.join(tempfile.gettempdir(),cfvar)
        paths = as_keyTabular(elements,var,path=path)
        buffer = io.BytesIO()
        zip = zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED)
        for path in paths:
            zip.write(path,arcname=os.path.split(path)[1])
        zip.close()
        buffer.flush()
        zip_stream = buffer.getvalue()
        buffer.close()
        
        ## remove the files from disk
        for path in paths: os.remove(path)
        
        # Stick it all in a django HttpResponse
        response = HttpResponse()
        response['Content-Disposition'] = 'attachment; filename={0}.kcsv.zip'.\
          format(cfvar)
        response['Content-length'] = str(len(zip_stream))
        response['Content-Type'] = 'application/zip'
        response.write(zip_stream)
        
        return(response)


#Emitter.register('helloworld',HelloWorldEmitter,'text/html; charset=utf-8')
Emitter.register('shz',ShapefileEmitter,'application/zip; charset=utf-8')
#Emitter.unregister('json')
Emitter.register('geojson',GeoJsonEmitter,'text/plain; charset=utf-8')
Emitter.register('csv',CsvEmitter,'text/csv; charset=utf-8')
Emitter.register('kcsv',CsvKeyEmitter,'application/zip; charset=utf-8')
