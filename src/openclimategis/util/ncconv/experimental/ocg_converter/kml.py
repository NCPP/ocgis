import zipfile
import io
from util.ncconv.experimental.ocg_converter.ocg_converter import OcgConverter


class KmlConverter(OcgConverter):
    '''Converts data to a KML string'''
    
    def _convert_(self,request):
        from pykml.factory import KML_ElementMaker as KML
        from lxml import etree
        
        ## create the database
        db = self.db
        
        meta = request.ocg
        if request.environ['SERVER_PORT']=='80':
            portstr = ''
        else:
            portstr = ':{port}'.format(port=request.environ['SERVER_PORT'])
        
        url='{protocol}://{server}{port}{path}'.format(
            protocol='http',
            port=portstr,
            server=request.environ['SERVER_NAME'],
            path=request.environ['PATH_INFO'],
        )
        description = (
            '<table border="1">'
              '<tbody>'
                '<tr><th>Archive</th><td>{archive}</td></tr>'
                '<tr><th>Emissions Scenario</th><td>{scenario}</td></tr>'
                '<tr><th>Climate Model</th><td>{model}</td></tr>'
                '<tr><th>Run</th><td>{run}</td></tr>'
                '<tr><th>Output Variable</th><td>{variable}</td></tr>'
                '<tr><th>Units</th><td>{units}</td></tr>'
                '<tr><th>Start Time</th><td>{start}</td></tr>'
                '<tr><th>End Time</th><td>{end}</td></tr>'
                '<tr>'
                  '<th>Request URL</th>'
                  '<td><a href="{url}">{url}</a></td>'
                '</tr>'
                '<tr>'
                  '<th>Other Available Formats</th>'
                  '<td>'
                    '<a href="{url}">KML</a> - Keyhole Markup Language<br/>'
                    '<a href="{url_kmz}">KMZ</a> - Keyhole Markup Language (zipped)<br/>'
                    '<a href="{url_shz}">Shapefile</a> - ESRI Shapefile<br/>'
                    '<a href="{url_csv}">CSV</a> - Comma Separated Values (text file)<br/>'
                    '<a href="{url_json}">JSON</a> - Javascript Object Notation'
                  '</td>'
                '</tr>'
              '</tbody>'
            '</table>'
        ).format(
            archive=meta.archive.name,
            scenario=meta.scenario,
            model=meta.climate_model,
            run=meta.run,
            variable=meta.variable,
            units=meta.variable.units,
            simout=meta.simulation_output.netcdf_variable,
            start=meta.temporal[0],
            end=meta.temporal[-1],
            operation=meta.operation,
            url=url,
            url_kmz=url.replace('.kml', '.kmz'),
            url_shz=url.replace('.kml', '.shz'),
            url_csv=url.replace('.kml', '.csv'),
            url_json=url.replace('.kml', '.geojson'),
        )
        
        doc = KML.kml(
          KML.Document(
            KML.name('Climate Simulation Output'),
            KML.open(1),
            KML.description(description),
            KML.snippet(
                '<i>Click for metadata!</i>',
                maxLines="2",
            ),
            KML.StyleMap(
              KML.Pair(
                KML.key('normal'),
                KML.styleUrl('#style-normal'),
              ),
              KML.Pair(
                KML.key('highlight'),
                KML.styleUrl('#style-highlight'),
              ),
              id="smap",
            ),
            KML.Style(
              KML.LineStyle(
                KML.color('ff0000ff'),
                KML.width('2'),
              ),
              KML.PolyStyle(
                KML.color('400000ff'),
              ),
              id="style-normal",
            ),
            KML.Style(
              KML.LineStyle(
                KML.color('ff00ff00'),
                KML.width('4'),
              ),
              KML.PolyStyle(
                KML.color('400000ff'),
              ),
              id="style-highlight",
            ),
            #Time Folders will be appended here
          ),
        )
        
        try:
            s = db.Session()
            for time in s.query(db.Time).all():
                # create a folder for the time
                timefld = KML.Folder(
#                    KML.Style(
#                      KML.ListStyle(
#                        KML.listItemType('checkHideChildren'),
#                        KML.bgColor('00ffffff'),
#                        KML.maxSnippetLines('2'),
#                      ),
#                    ),
                    KML.name(time.as_xml_date()),
                    # placemarks will be appended here
                )
                for val in time.value:
                    poly_desc = (
                        '<table border="1">'
                          '<tbody>'
                            '<tr><th>Variable</th><td>{variable}</td></tr>'
                            '<tr><th>Date/Time (UTC)</th><td>{time}</td></tr>'
                            '<tr><th>Value</th><td>{value:.{digits}f} {units}</td></tr>'
                          '</tbody>'
                        '</table>'
                    ).format(
                        variable=meta.variable.name,
                        time=val.time_ref.as_xml_date(),
                        value=val.value,
                        digits=3,
                        units=meta.variable.units,
                    )
                    
                    coords = val.geometry.as_kml_coords()
                    timefld.append(
                      KML.Placemark(
                        KML.name('Geometry'),
                        KML.description(poly_desc),
                        KML.styleUrl('#smap'),
                        KML.Polygon(
                          KML.tessellate('1'),
                          KML.outerBoundaryIs(
                            KML.LinearRing(
                              KML.coordinates(coords),
                            ),
                          ),
                        ),
                      )
                    )
                doc.Document.append(timefld)
            pass
        finally:
            s.close()
        
        # return the pretty print sting
        return(etree.tostring(doc, pretty_print=True))


class KmzConverter(KmlConverter):
    
    def _response_(self,payload):
        '''Get the KML response and zip it up'''
#        logger.info("starting KmzConverter._response_()...")
        #kml = super(KmzConverter,self)._response_(payload)
        
        iobuffer = io.BytesIO()
        zf = zipfile.ZipFile(
            iobuffer, 
            mode='w',
            compression=zipfile.ZIP_DEFLATED, 
        )
        try:
            zf.writestr('doc.kml',payload)
        finally:
            zf.close()
        iobuffer.flush()
        zip_stream = iobuffer.getvalue()
        iobuffer.close()
#        logger.info("...ending KmzConverter._response_()")
        return(zip_stream)