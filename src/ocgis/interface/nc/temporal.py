from ocgis.interface.base.dimension.temporal import TemporalDimension
from ocgis.interface.nc.dimension import NcVectorDimension


class NcTemporalDimension(TemporalDimension, NcVectorDimension):

    def __init__(self, *args, **kwargs):
        TemporalDimension.__init__(self, *args, **kwargs)
