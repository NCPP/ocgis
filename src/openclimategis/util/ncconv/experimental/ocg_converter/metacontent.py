from util.ncconv.experimental.ocg_meta import metacontent
from util.ncconv.experimental.ocg_converter.ocg_converter import OcgConverter
import StringIO
import time


class MetacontentConverter(OcgConverter):
    
    def __init__(self,request):
        self.request = request
    
    @property
    def request_duration(self):
        return(int(time.time() - self.request.ocg.start_time))
    
    def _convert_(self):
        report = metacontent.Report(self.request_duration)
        ## list of section to add.
        Sections = [
                    metacontent.SectionGeneratedUrl,
                    metacontent.SectionArchive,
                    metacontent.SectionScenario,
                    metacontent.SectionClimateModel,
                    metacontent.SectionVariable,
                    metacontent.SectionSimulationOutput,
                    metacontent.SectionTemporalRange,
                    metacontent.SectionSpatial,
                    metacontent.SectionGrouping,
                    metacontent.SectionFunction,
                    metacontent.SectionAttributes
                    ]
        for Section in Sections:
            report.add_section(Section(self.request))
        return(report.format())
    
    def _response_(self,payload):
        buffer = StringIO.StringIO()
        for line in payload:
            buffer.write(line+'\n')
        content = buffer.getvalue()
        buffer.close()
        return(content)