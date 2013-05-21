__VER__ = '0.05.1'
__RELEASE__ = '0.05.1b'

from util.environment import env
from api import OcgOperations
from util.shp_cabinet import ShpCabinet
from util.inspect import Inspect
from api.request import RequestDataset, RequestDatasetCollection

from osgeo import ogr
## tell ogr to raise exceptions
ogr.UseExceptions()