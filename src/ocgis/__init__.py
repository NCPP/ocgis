from ocgis.api.collection import SpatialCollection
from ocgis.api.operations import OcgOperations
from ocgis.api.request.base import RequestDataset, RequestDatasetCollection
from ocgis.calc.library.register import FunctionRegistry
from ocgis.interface.base import crs
from ocgis.interface.base.crs import CoordinateReferenceSystem
from ocgis.interface.base.dimension.base import VectorDimension
from ocgis.interface.base.dimension.spatial import SpatialDimension, SpatialGridDimension, SpatialGeometryDimension, \
    SpatialGeometryPolygonDimension, SpatialGeometryPointDimension
from ocgis.interface.base.dimension.temporal import TemporalDimension
from ocgis.interface.base.field import Field
from ocgis.interface.base.variable import Variable
from ocgis.util.environment import env
from ocgis.util.inspect import Inspect
from ocgis.util.shp_cabinet import ShpCabinet, ShpCabinetIterator
from ocgis.util.zipper import format_return

__version__ = '1.2.1'
__release__ = '1.2.1'
