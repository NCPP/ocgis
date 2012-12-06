import netCDF4 as nc
from ocgis.api.dataset.dataset import OcgDataset
from ocgis.interface.interface import DummyLevelInterface


class Inspect(object):
    
    def __init__(self,uri):
        self.uri = uri
        self.ds = OcgDataset(uri)
        
    def __repr__(self):
        msg = ''
        for line in self.get_report():
            msg += line + '\n'
        return(msg)
        
    @property
    def _t(self):
        return(self.ds.i.temporal)
    @property
    def _s(self):
        return(self.ds.i.spatial)
    @property
    def _l(self):
        return(self.ds.i.level)
        
    def get_temporal_report(self):
        start_date = self._t.time.value.min()
        end_date = self._t.time.value.max()
        res = int(self._t.get_approx_res_days())
        n = len(self._t.time.value)
        calendar = self._t.time.calendar.value
        units = self._t.time.units.value
        
        lines = []
        lines.append('       Start Date = {0}'.format(start_date))
        lines.append('         End Date = {0}'.format(end_date))
        lines.append('         Calendar = {0}'.format(calendar))
        lines.append('            Units = {0}'.format(units))
        lines.append('Resolution (Days) = {0}'.format(res))
        lines.append('            Count = {0}'.format(n))
        
        return(lines)
    
    def get_spatial_report(self):
        res = self._s.resolution
        extent = self._s.extent().bounds
        itype = self._s.__class__.__name__
        projection = self.ds.i.projection.__class__.__name__
        
        lines = []
        lines.append('Spatial Reference = {0}'.format(projection))
        lines.append('           Extent = {0}'.format(extent))
        lines.append('   Interface Type = {0}'.format(itype))
        lines.append('       Resolution = {0}'.format(res))
        lines.append('            Count = {0}'.format(self._s.gid.reshape(-1).shape[0]))
        
        return(lines)
    
    def get_level_report(self):
        if isinstance(self._l,DummyLevelInterface):
            lines = ['No level dimension found.']
        else:
            lines = []
            lines.append('Level Variable = {0}'.format(self._l.level.name))
            lines.append('         Count = {0}'.format(self._l.level.value.shape[0]))
        return(lines)
    
    def get_dump_report(self):
        ds = nc.Dataset(self.uri,'r')
        try:
            lines = ['++ GLOBAL ATTRIBUTES ++']
            template = '   - {0} :: {1}'
            for attr in ds.ncattrs():
                lines.append(template.format(attr,getattr(ds,attr)))
            
            lines += ['','++ DIMENSIONS ++']
            for key,value in ds.dimensions.iteritems():
                lines.append('   - {0} :: {1}'.format(key,len(value)))
                
            lines += ['','++ VARIABLES ++']
            for key,value in ds.variables.iteritems():
                lines.append('   + {0} :: Dimensions({1})'.format(key,map(str,value.dimensions)))
                for attr in value.ncattrs():
                    lines.append('      - {0} :: {1}'.format(attr,getattr(value,attr)))
#                import ipdb;ipdb.set_trace()
        finally:
            ds.close()
            
        return(lines)
    
    def get_report(self):
        mp = [
              {'=== Temporal =============':self.get_temporal_report},
              {'=== Spatial ==============':self.get_spatial_report},
              {'=== Level ================':self.get_level_report},
              {'=== Dump =================':self.get_dump_report}
              ]
        
        lines = ['','URI = {0}'.format(self.uri)]
        lines.append('')
        for dct in mp:
            for key,value in dct.iteritems():
                lines.append(key)
                lines.append('')
                for line in value():
                    lines.append(line)
            lines.append('')
        
        return(lines)