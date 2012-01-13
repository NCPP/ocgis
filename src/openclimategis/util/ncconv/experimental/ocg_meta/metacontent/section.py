from util.ncconv.experimental.ocg_stat import OcgStatFunction
class Section(object):
    _title = None
    
    def __init__(self,lines=[],title=None):
        self.lines = lines
        self.title = title or self._title
        
    @property
    def formatted_title(self):
        msg = '=== {0} ==='.format(self._title)
        return(msg)
    
    def add_line(self,line):
        self.lines.append(line)
    
    def format(self):
        frmt = [self.formatted_title] + self.lines
        return(frmt)
    

class RequestSection(Section):
    
    def __init__(self,request,**kwds):
        self.request = request
        super(RequestSection,self).__init__(**kwds)
        
    def format(self):
        frmt = [self.formatted_title] + self.get_lines()
        return(frmt)
        
    def get_lines(self):
        return(['(None)'])


class SectionGeneratedUrl(RequestSection):
    _title = "Generated URL"
    
    def get_lines(self):
        return([self.request.build_absolute_uri()])
    
    
class SectionTemporalRange(RequestSection):
    _title = 'Temporal Range (inclusive)'
    
    def get_lines(self):
        return(['Lower :: {0}'.format(self.request.ocg.temporal[0]),
                'Upper :: {0}'.format(self.request.ocg.temporal[1])])
        
        
class SectionSpatial(RequestSection):
    _title = 'Spatial Operations Performed'
    _descs = {
              'Intersect':'Grid cells overlapping or sharing a border with the AOI geometry are included.',
              'Clip':'Full geometric intersection of grid cell and AOI geometries.',
              'Aggregate=False':'Geometries are not merged.',
              'Aggregate=True':'Geometries merged and climate variable aggregated using area-weighted mean.'
              }
    
    def get_lines(self):
        spatial = self.request.ocg.operation.title()
        aggregate = 'Aggregate={0}'.format(self.request.ocg.aggregate)
        lines = [
                 '{0} :: {1}'.format(spatial,
                                     self._descs[spatial]),
                 '{0} :: {1}'.format(aggregate,
                                     self._descs[aggregate]),
                ]
        return(lines)
    
    
class SectionGrouping(RequestSection):
    _title = 'Temporal Grouping Method'
    
    def _extract_(self):
        grps = [str(a.title()) for a in self.request.ocg.query.grouping]
        return(['-'.join(grps)])
    
    def get_lines(self):
        try:
            lines = self._extract_()
        except:
            lines = super(SectionGrouping,self).get_lines()
        return(lines)
    
    
class SectionFunction(SectionGrouping):
    _title = 'Temporal Statistics Calculated'
    
    def _extract_(self):
        lines = []
        for ii,f in enumerate(self.request.ocg.query.functions):
            msg = '{0} :: {1}'
            ## always add count & ignore it if in the function dictionary
            if 'name' in f and f['name'].lower() == 'count':
                continue
            if ii == 0:
                name = 'COUNT'
                desc = 'Count of values in the series.'
            else:
                name = f['name'].upper()
                desc = f['desc']
            if 'args' in f:
                desc = desc.format(*f['args'])
            lines.append(msg.format(name,desc))
        return(lines)
    
    
class SectionAttributes(RequestSection):
    _title = 'Attribute Definitions (non-statistical)'
    _descs = {
              'OCGID':'Unique record identifier (OCG=OpenClimateGIS).',
              'GID':'Unique geometry identifier.',
              'TID':'Unique time identifier.',
              'LEVEL':('Level indicator from "1" to max level. With "1" '
                       'indicating level nearest the terrestrial surface. Level'
                       ' is included for all variables.'),
              'VALUE':"Requested variable's or aggregate statistic's value.",
              'TIME':'Record timestamp with same units as the dataset.',
              'DAY':'Day component of the timestamp.',
              'MONTH':'Month component of the timestamp.',
              'YEAR':'Year component of the timestamp.',
              'AREA_M2':'Area of geometry in square meters using SRID 3005 as area-preserving projection.'
              }
    
    def get_lines(self,use_stat=False):
        attrs = ['OCGID','GID','TID','TIME','LEVEL','VALUE','DAY','MONTH','YEAR']
        lines = ['{0} :: {1}'.format(attr,self._descs[attr]) 
                 for attr in attrs]
        return(lines)
    
    
class SectionLinks(RequestSection):
    _attr = None
    _filter_field = None
    
    def get_lines(self):
        qs = getattr(self.request.ocg,self._attr)
        metalist = qs.metadata_list(
            request=self.request,
            filter_field=self._filter_field,
        )
        return(metalist)
    
    
class SectionArchive(SectionLinks):
    _title = 'Climate Data Archive'
    _attr = 'archive'


class SectionScenario(SectionLinks):
    _title = 'Emissions Scenario'
    _attr = 'scenario'


class SectionClimateModel(SectionLinks):
    _title = 'Climate Model'
    _attr = 'climate_model'
    _filter_field = 'model'


class SectionVariable(SectionLinks):
    _title = 'Output Variable'
    _attr = 'variable'


class SectionSimulationOutput(SectionLinks):
    _title = 'Simulation Output'
    _attr = 'simulation_output'