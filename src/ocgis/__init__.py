from osgeo import ogr, osr

from ocgis.util.environment import env

from ocgis.api.collection import SpatialCollection
from ocgis.api.operations import OcgOperations
from ocgis.api.request.base import RequestDataset, RequestDatasetCollection
from ocgis.calc.library.register import FunctionRegistry
from ocgis.interface.base import crs
from ocgis.interface.base.crs import CoordinateReferenceSystem
from ocgis.interface.base.dimension.spatial import SpatialDimension
from ocgis.interface.base.field import Field
from ocgis.util.inspect import Inspect
from ocgis.util.shp_cabinet import ShpCabinet, ShpCabinetIterator
from ocgis.util.zipper import format_return
from ocgis.interface.base.dimension.temporal import TemporalDimension


__version__ = '1.0.1'
__release__ = '1.0.1-next'


# tell ogr/osr to raise exceptions
ogr.UseExceptions()
osr.UseExceptions()