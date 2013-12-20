import netCDF4 as nc
from ocgis.interface.metadata import NcMetadata
from ocgis.exc import ResolutionError
from collections import OrderedDict
import re
from warnings import warn
from ocgis.util.logging_ocgis import ocgis_lh


class Inspect(object):
    """
    Inspect a local or remote dataset returning a printout similar to `ncdump`_.
    
    >>> from ocgis import Inspect
    ...
    >>> # Just do an dataset attribute dump.
    >>> ip = Inspect('/my/local/dataset')
    >>> print(ip)
    ...
    >>> # Get variable-specific info.
    >>> ip = Inspect('/my/local/dataset',variable='tas')
    >>> print(ip)
    
    :param uri: Absolute path to data's location.
    :type uri: str
    :param variable: Specific variable to inspect.
    :type variable: str
    :param interface_overload: Overloads for autodiscover.
    :type interface_overload: dict
    
    .. _ncdump: http://www.unidata.ucar.edu/software/netcdf/docs/netcdf/ncdump.html
    """
    
    def __init__(self,uri=None,variable=None,interface_overload={},request_dataset=None):
        self.request_dataset = request_dataset
        if self.request_dataset is None:
            self.uri = uri
            self.variable = variable
            self.alias = None
            self.did = None
            if self.variable is None:
                try:
                    self.ds = None
                    rootgrp = nc.Dataset(uri)
                    self.meta = NcMetadata(rootgrp)
                finally:
                    rootgrp.close()
            else:
                from ocgis.api.request.base import RequestDataset
                kwds = {'uri':uri,'variable':variable}
                kwds.update(interface_overload)
                rd = RequestDataset(**kwds)
                self.ds = rd.get()
                self.meta = self.ds.meta
        else:
            self.uri = self.request_dataset.uri
            self.variable = self.request_dataset.variable
            self.ds = self.request_dataset.get()
            self.meta = self.ds.meta
            self.alias = self.request_dataset.alias
            self.did = self.request_dataset.did
        
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
        return(self.ds.temporal)
    @property
    def _s(self):
        return(self.ds.spatial)
    @property
    def _l(self):
        return(self.ds.level)
        
    def get_temporal_report(self):
            
        try:
            if self._t.format_time:
                res = int(self._t.resolution)
                try:
                    start_date,end_date = self._t.extent_datetime
                ## the times may not be formattable
                except ValueError as e:
                    if e.message == 'year is out of range':
                        start_date,end_date = self._t.extent
                    else:
                        ocgis_lh(exc=e,logger='inspect')
            else:
                res = 'NA (non-formatted times requested)'
                start_date,end_date = self._t.extent
        ## raised if the temporal dimension has a single value. possible with
        ## snippet or a small dataset...
        except ResolutionError:
            res,start_date,end_date = ['NA (singleton)']*3
            
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
        
        ## append information on bounds
        if self._t.bounds is not None:
            has_bounds = True
        else:
            has_bounds = False
        lines.append('       Has Bounds = {0}'.format(has_bounds))
        
        return(lines)
    
    def get_spatial_report(self):
        res = self._s.grid.resolution
        extent = self._s.grid.extent

        itype = self._s.geom.get_highest_order_abstraction().__class__.__name__
        projection = self.ds.spatial.crs
        
        lines = []
        lines.append('Spatial Reference = {0}'.format(projection.__class__.__name__))
        lines.append('     Proj4 String = {0}'.format(projection.sr.ExportToProj4()))
        lines.append('           Extent = {0}'.format(extent))
        lines.append('   Interface Type = {0}'.format(itype))
        lines.append('       Resolution = {0}'.format(res))
        lines.append('            Count = {0}'.format(self._s.grid.uid.reshape(-1).shape[0]))
        
        return(lines)
    
    def get_level_report(self):
        if self._l is None:
            lines = ['No level dimension found.']
        else:
            lines = []
            lines.append('Level Variable = {0}'.format(self._l.name))
            lines.append('         Count = {0}'.format(self._l.value.shape[0]))
            
            ## append information on bounds
            if self._l.bounds is not None:
                has_bounds = True
            else:
                has_bounds = False
            lines.append('    Has Bounds = {0}'.format(has_bounds))
            
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
        
        ## a variable target is required for this method
        if self.variable is None:
            raise(AttributeError('A "variable" target is required.'))
        
        mp = [
              {'=== Temporal =============':self.get_temporal_report},
              {'=== Spatial ==============':self.get_spatial_report},
              {'=== Level ================':self.get_level_report},
              {'=== Dump =================':self.get_dump_report}
              ]
        
        lines = ['','URI = {0}'.format(self.uri)]
        lines.append('VARIABLE = {0}'.format(self.variable))
        lines.append('ALIAS = {0}'.format(self.alias))
        lines.append('DID = {0}'.format(self.did))
        lines.append('')
        for dct in mp:
            for key,value in dct.iteritems():
                lines.append(key)
                lines.append('')
                for line in value():
                    lines.append(line)
            lines.append('')
        
        return(lines)
    
    def _as_dct_(self):
        ret = self.meta.copy()
        ## without a target variable, attempt to set start and end dates.
        if self.variable is None:
            ds = nc.Dataset(self.uri,'r')
            try:
                time = ds.variables['time']
                time_bounds = [time[0],time[-1]]
                time_bounds = nc.num2date(time_bounds,time.units,calendar=time.calendar)
                derived = {'Start Date':str(time_bounds[0]),'End Date':str(time_bounds[1])}
            except:
                warn('Time variable not found or improperly attributed. Setting "derived" key to None.')
                derived = None
            finally:
                ds.close()
        ## we can get derived values
        else:
            derived = OrderedDict()
            to_add = self.get_temporal_report() + self.get_spatial_report() + self.get_level_report()
            for row in to_add:
                try:
                    key,value = re.split(' = ',row,maxsplit=1)
                ## here to catch oddities of the returns
                except ValueError:
                    if row == 'No level dimension found.':
                        continue
                    else:
                        raise
                key = key.strip()
                derived.update({key:value})
        ret.update({'derived':derived})
        return(ret)
