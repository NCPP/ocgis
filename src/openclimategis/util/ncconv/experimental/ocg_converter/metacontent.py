from util.ncconv.experimental.ocg_meta import metacontent
from util.ncconv.experimental.ocg_converter.ocg_converter import OcgConverter
import StringIO


class MetacontentConverter(OcgConverter):
    
    def __init__(self,request,payload_size=0):
        self.request = request
        request_duration = int(request.ocg.end_time - request.ocg.start_time)
        self.report = metacontent.Report(request_duration,payload_size)
        
    def _convert_(self):
        ## list of section to add.
        Sections = [
                    metacontent.SectionGeneratedUrl,
                    metacontent.SectionTemporalRange,
                    metacontent.SectionSpatial,
                    metacontent.SectionGrouping,
                    metacontent.SectionFunction,
                    metacontent.SectionAttributes
                    ]
        for Section in Sections:
            self.report.add_section(Section(self.request))
        return(self.report.format())
    
    def _response_(self,payload):
        buffer = StringIO.StringIO()
        for line in payload:
            buffer.write(line+'\n')
        content = buffer.getvalue()
        buffer.close()
        return(content)