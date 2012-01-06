from django.template.context import RequestContext
from django.shortcuts import render_to_response
from django.http import HttpResponse
from piston.emitters import Emitter
from api.views import display_spatial_query, display_aoi_uploader
from util.ncconv.experimental import ocg_converter
from util.ncconv.experimental.ocg_stat import OcgStat
from django.conf import settings
import time

import logging
logger = logging.getLogger(__name__)


class OpenClimateEmitter(Emitter):
    """
    Superclass for all OpenClimateGIS emitters.
    """
    
    def render(self,request):
        return(self._render_(request))
    
    def _render_(self,request):
        raise(NotImplementedError)


class IdentityEmitter(OpenClimateEmitter):
    """
    The standard Django Piston emitter does unnecessary computations when an
    emitter is searching for the raw data from its associated handler.
    """
    
    def construct(self):
        return self.data


class HelloWorldEmitter(OpenClimateEmitter):
    
    def _render_(self,request):
        names = [n['name'] for n in self.construct()]
        msg = 'Hello, World!! The climate model names are:<br><br>{0}'.format('<br>'.join(names))
        return HttpResponse(msg)


class HTMLEmitter(IdentityEmitter):
    """Emits an HTML representation 
    """
    def _render_(self,request):
        
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
        elif template_name == 'aoi_upload.html':
            response = display_aoi_uploader(request)
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
    
    def _render_(self,request):
        logger.info("starting {0}.render()...".format(self.__converter__.__name__))
#        import ipdb;ipdb.set_trace()
        self.sub = self.construct()
        self.db = self.get_db()
        self.request = request
        if request.ocg.query.use_stat:
            st = OcgStat(self.db,
                         self.sub,
                         request.ocg.query.grouping,
                         procs=settings.MAXPROCESSES)
            st.calculate_load(self.request.ocg.query.functions)
        self.cfvar = request.ocg.simulation_output.variable.code
        self.converter = self.get_converter()
        logger.info("...ending {0}.render()...".format(self.__converter__.__name__))
        return(self.get_response())
    
    def get_db(self):
        return(self.sub.to_db(procs=settings.MAXPROCESSES))
    
    def get_converter(self):
        return(self.__converter__(self.db,
                                  self.cfvar+self.__file_ext__,
                                  use_stat=self.request.ocg.query.use_stat))
        
    def get_response(self):
        return(self.converter.response())


class ZippedSubOcgDataEmitter(SubOcgDataEmitter):
    
    def _render_(self,request):
        base_response = super(ZippedSubOcgDataEmitter,self)._render_(request)
        response = HttpResponse()
        response['Content-Disposition'] = 'attachment; filename={0}.zip'.\
            format(request.ocg.simulation_output.variable.code)
        response['Content-length'] = str(len(base_response))
        response['Content-Type'] = 'application/zip'
        response.write(base_response)
        return(response)
    
    
class MetacontentEmitter(SubOcgDataEmitter):
    
    def _render_(self,request):
        request.ocg.end_time = time.time()
        converter = ocg_converter.MetacontentConverter(request)
        response = converter.response()
        return(response)


class SqliteEmitter(ZippedSubOcgDataEmitter):
    __converter__ = ocg_converter.SqliteConverter
    __file_ext__ = ''
    
    def get_db(self):
        return(self.sub.as_sqlite(to_disk=True))


class ShapefileEmitter(ZippedSubOcgDataEmitter):
    """
    Emits zipped shapefile (.shz)
    """
    __converter__ = ocg_converter.ShpConverter
    __file_ext__ = '.shp'
    
    
class LinkedShapefileEmitter(ZippedSubOcgDataEmitter):
    __converter__ = ocg_converter.LinkedShpConverter
    __file_ext__ = '.lshz'


class KmlEmitter(SubOcgDataEmitter):
    """
    Emits raw KML (.kml)
    """
    
    __converter__ = ocg_converter.KmlConverter
    __file_ext__ = '.kml'

    def _response_(self):
        return(self.converter())
    
    def get_response(self):
        return(self.converter.response(self.request))


class KmzEmitter(KmlEmitter):
    """
    Subclass of KmlEmitter. Emits KML in a zipped format (.kmz)
    """
    
    __converter__ = ocg_converter.KmzConverter
    __file_ext__ = '.kmz'
    
#    def _render_(self,request):
#        logger.info("starting KmzEmitter.render()...")
#        kml = super(KmzEmitter,self).render(request)
#        iobuffer = io.BytesIO()
#        zf = zipfile.ZipFile(
#            iobuffer, 
#            mode='w',
#            compression=zipfile.ZIP_DEFLATED, 
#        )
#        try:
#            zf.writestr('doc.kml',kml)
#        finally:
#            zf.close()
#        iobuffer.flush()
#        zip_stream = iobuffer.getvalue()
#        iobuffer.close()
#        logger.info("...ending KmzEmitter.render()")
#        return(zip_stream)


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
    
    
class LinkedCsvEmitter(ZippedSubOcgDataEmitter):
    __converter__ = ocg_converter.LinkedCsvConverter
    __file_ext__ = ''
    
    
class NcEmitter(SubOcgDataEmitter):
    __converter__ = ocg_converter.NcConverter
    __file_ext__ = '.nc'
    
    def get_response(self):
        from util.ncconv.experimental.ocg_dataset.dataset import OcgDataset
        ds = OcgDataset(self.request.ocg.dataset_uri,**self.request.ocg.ocg_opts)
        return(self.converter.response(self.sub,ds))


Emitter.register('shz',ShapefileEmitter,'application/zip; charset=utf-8')
Emitter.register('lshz',LinkedShapefileEmitter,'application/zip; charset=utf-8')
#Emitter.unregister('json')
Emitter.register('kml',KmlEmitter,'application/vnd.google-earth.kml+xml')
Emitter.register('kmz',KmzEmitter,'application/vnd.google-earth.kmz')
Emitter.register('geojson',GeoJsonEmitter,'text/plain; charset=utf-8')
Emitter.register('csv',CsvEmitter,'text/csv; charset=utf-8')
Emitter.register('kcsv',LinkedCsvEmitter,'application/zip; charset=utf-8')
Emitter.register('sqlite',SqliteEmitter,'application/zip; charset=utf-8')
Emitter.register('nc',NcEmitter,'application/x-netcdf; charset=utf-8')
Emitter.register('meta',MetacontentEmitter,'text/plain; charset=utf-8')