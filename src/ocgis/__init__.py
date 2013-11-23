__VER__ = '0.06'
__RELEASE__ = '0.06b-dev'

from util.environment import env
from api.operations import OcgOperations
from util.shp_cabinet import ShpCabinet, ShpCabinetIterator
from util.inspect import Inspect
from api.request.base import RequestDataset, RequestDatasetCollection
from util.zipper import format_return
from interface.base import crs
from calc.library.register import FunctionRegistry

from osgeo import ogr
## tell ogr to raise exceptions
ogr.UseExceptions()