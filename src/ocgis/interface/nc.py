import base


class NcColumnDimension(base.AbstractColumnDimension):
    pass


class NcLevelDimension(base.AbstractLevelDimension):
    pass


class NcRowDimension(base.AbstractRowDimension):
    _name_id = None
    _name_long = None
    
    @property
    def extent(self):
        if self.bounds is None:
            ret = (self.value.min(),self.value.max())
        else:
            ret = (self.bounds.min(),self.bounds.max())
        return(ret)
    
    def _load_(self,subset_by=None):
        raise(NotImplementedError)


class NcTemporalDimension(base.AbstractTemporalDimension):
    pass


class NcSpatialDimension(base.AbstractSpatialDimension):
    pass


class NcGlobalInterface(base.AbstractGlobalInterface):
    _dtemporal = NcTemporalDimension
    _dlevel = NcLevelDimension
    _dspatial = NcSpatialDimension
#    _metdata_cls = NcMetadata