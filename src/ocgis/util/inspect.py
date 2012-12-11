import netCDF4 as nc
from ocgis.api.dataset.dataset import OcgDataset
from ocgis.interface.ncmeta import NcMetadata


class Inspect(object):
    
    def __init__(self,uri,variable=None,interface_overload={}):
        self.uri = uri
        self.variable = variable
        if self.variable is None:
            try:
                self.ds = None
                rootgrp = nc.Dataset(uri)
                self.meta = NcMetadata(rootgrp)
            finally:
                rootgrp.close()
        else:
            self.ds = OcgDataset({'uri':uri,'variable':variable},
                                 interface_overload=interface_overload)
            self.meta = self.ds.i._meta
        
    def __repr__(self):
        msg = ''
        if self.variable is None:
            lines = self.get_report_no_variable()
        else:
            lines = self.get_report()
        for line in lines:
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
        start_date = self._t.value.min()
        end_date = self._t.value.max()
        res = int(self._t.get_approx_res_days())
        n = len(self._t.value)
        calendar = self._t.calendar
        units = self._t.units
        
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
        projection = self.ds.i.spatial.projection.__class__.__name__
        
        lines = []
        lines.append('Spatial Reference = {0}'.format(projection))
        lines.append('           Extent = {0}'.format(extent))
        lines.append('   Interface Type = {0}'.format(itype))
        lines.append('       Resolution = {0}'.format(res))
        lines.append('            Count = {0}'.format(self._s.gid.reshape(-1).shape[0]))
        
        return(lines)
    
    def get_level_report(self):
        if self._l.is_dummy:
            lines = ['No level dimension found.']
        else:
            lines = []
            lines.append('Level Variable = {0}'.format(self._l.name))
            lines.append('         Count = {0}'.format(self._l.value.shape[0]))
        return(lines)
    
    def get_dump_report(self):
        return(self.meta._get_lines_())
    
    def get_report_no_variable(self):
        lines = ['','URI = {0}'.format(self.uri)]
        lines.append('VARIABLE = {0}'.format(self.variable))
        lines.append('')
        lines += self.get_dump_report()
        return(lines)
    
    def get_report(self):
        mp = [
              {'=== Temporal =============':self.get_temporal_report},
              {'=== Spatial ==============':self.get_spatial_report},
              {'=== Level ================':self.get_level_report},
              {'=== Dump =================':self.get_dump_report}
              ]
        
        lines = ['','URI = {0}'.format(self.uri)]
        lines.append('VARIABLE = {0}'.format(self.variable))
        lines.append('')
        for dct in mp:
            for key,value in dct.iteritems():
                lines.append(key)
                lines.append('')
                for line in value():
                    lines.append(line)
            lines.append('')
        
        return(lines)