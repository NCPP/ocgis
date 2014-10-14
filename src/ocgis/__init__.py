__VER__ = '1.0.0'
__RELEASE__ = '1.0.0-next'

from util.environment import env
from api.operations import OcgOperations
from util.shp_cabinet import ShpCabinet, ShpCabinetIterator
from util.inspect import Inspect
from api.request.base import RequestDataset, RequestDatasetCollection
from util.zipper import format_return
from interface.base import crs
from calc.library.register import FunctionRegistry
from api.collection import SpatialCollection
from interface.base import crs
from ocgis.interface.base.crs import CoordinateReferenceSystem
from osgeo import ogr, osr

# tell ogr/osr to raise exceptions
ogr.UseExceptions()
osr.UseExceptions()