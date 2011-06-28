import netCDF4 as nc


VARMAP = dict(
              time=['time'],
              row=['latitude'],
              row_bounds=['bounds_latitude'],
              col=['longitude'],
              col_bounds = ['bounds_longitude']
              )


class Meta(object):
    pass


class NamedVariable(object):
    
    def __init__(self,choices):
        self.choices = choices
        
    def get(self,variables):
        for ii,value in enumerate(self.choices):
            try:
                ret = variables[value]
                break
            except KeyError:
                if ii+2 > len(self.choices):
                    raise KeyError('no variable located for choices {0}'.format(self.choices))
                else:
                    continue
        return(value,ret)
    

class NcProfiler(object):
    
    def __init__(self,uri):
        self.uri = uri
        self.dataset = nc.Dataset(uri,'r')
        
        self.time = Meta()
        self.row = Meta()
        self.row.bounds = Meta()
        self.col = Meta()
        self.col.bounds = Meta()
        self.variable = Meta()
        
        self._remove_registry = []
        self._profile_()

    def _profile_(self):
        "Collect variables of interest for OpenClimateGIS"
        
        ## TIME ----------------------------------------------------------------
        
        name,t = NamedVariable(VARMAP['time']).get(self.dataset.variables)
        self._remove_registry.append(name)
        self.time.variable = name
        self.time.units = t.units
        self.time.calendar = t.calendar
        self.time.vec = nc.num2date(t[:],t.units,calendar=t.calendar)
        
        ## SPATIAL -------------------------------------------------------------
        
        self._make_spatial_(self.row,'row','row_bounds')
        self._make_spatial_(self.col,'col','col_bounds')
        
        import ipdb;ipdb.set_trace()
        
    def __del__(self):
        try:
            self.dataset.close()
        except:
            pass
    
    def _make_spatial_(self,var,name,bounds_name):
        name,vals =  NamedVariable(VARMAP[name]).get(self.dataset.variables)
        var.variable = name
        var.vec = vals[:]
        
        name,bnds = NamedVariable(VARMAP[bounds_name]).get(self.dataset.variables)
        var.bounds.vec = bnds[:]
        var.bounds.variable = name
        
        self._remove_registry += [var.variable,var.bounds.variable]
#        
#        self.row.variable = name
#        self.row.vec = vals[:]
#        name,bnds = NamedVariable(VARMAP[bounds_name]).get(self.dataset.variables)
#        self.row.bounds.vec = bnds[:]
#        self.row.bounds.variable = name
        
        
if __name__ == '__main__':
    uri = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data/maurer/bccr_bcm2_0.1.sresa1b.monthly.Prcp.1950.nc'
    ncp = NcProfiler(uri)
    import ipdb;ipdb.set_trace()