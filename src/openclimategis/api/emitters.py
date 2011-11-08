from django.template.context import RequestContext
from django.shortcuts import render_to_response
from django.http import HttpResponse
from piston.emitters import Emitter
from api.views import display_spatial_query
import zipfile
import io
from util.ncconv.experimental import ocg_converter

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
        
        ## if we need the query form generate and pass accordingly
        if template_name == 'query.html':
            response = display_spatial_query(request)
        else:
            response = render_to_response(
                template_name=template_name, 
                dictionary={'data':data, 'is_collection':is_collection},
                context_instance=c,
            )
        
        logger.info("...ending HTMLEmitter.render()")
        
        return(response)
Emitter.register('html', HTMLEmitter, 'text/html; charset=utf-8')


class SubOcgDataEmitter(IdentityEmitter):
    __converter__ = None
    __file_ext__ = ''
    __kwds__ = {}
    
    def render(self,request):
        logger.info("starting {0}.render()...".format(self.__converter__.__name__))
        sub = self.construct()
        logger.debug("n geometries = {0}".format(len(sub.geometry)))
        cfvar = request.ocg.simulation_output.variable.code
        shp = self.__converter__(sub,cfvar+self.__file_ext__,**self.__kwds__)
        logger.info("...ending {0}.render()...".format(self.__converter__.__name__))
        return(shp.response())


class ShapefileEmitter(SubOcgDataEmitter):
    """
    Emits zipped shapefile (.shz)
    """
    __converter__ = ocg_converter.ShpConverter
    __file_ext__ = '.shp'


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


class GeoJsonEmitter(SubOcgDataEmitter):
    """
    JSON format for geospatial data (.json)
    """
    __converter__ = ocg_converter.GeojsonConverter
    __file_ext__ = '.json'


class CsvEmitter(SubOcgDataEmitter):
    """
    Tabular CSV format. (.csv)
    """
    __converter__ = ocg_converter.CsvConverter
    __file_ext__ = '.csv'
    __kwds__ = dict(as_wkt=False,
                    as_wkb=False,
                    add_area=True,
                    area_srid=3005,
                    to_disk=False)
    
    
class LinkedCsvEmitter(SubOcgDataEmitter):
    __converter__ = ocg_converter.LinkedCsvConverter
    __file_ext__ = ''


#Emitter.register('helloworld',HelloWorldEmitter,'text/html; charset=utf-8')
Emitter.register('shz',ShapefileEmitter,'application/zip; charset=utf-8')
#Emitter.unregister('json')
Emitter.register('geojson',GeoJsonEmitter,'text/plain; charset=utf-8')
Emitter.register('csv',CsvEmitter,'text/csv; charset=utf-8')
Emitter.register('kcsv',LinkedCsvEmitter,'application/zip; charset=utf-8')
