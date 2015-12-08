from ocgis.util.environment import env
import osgeo
from osgeo import osr
from osgeo import ogr
from ocgis.interface.base.dimension.base import VectorDimension
from ocgis.interface.base.dimension.temporal import TemporalDimension
from ocgis.api.collection import SpatialCollection
from ocgis.calc.library.register import FunctionRegistry
from ocgis.interface.base import crs
from ocgis.interface.base.crs import CoordinateReferenceSystem
from ocgis.interface.base.dimension.spatial import SpatialDimension, SpatialGridDimension, SpatialGeometryDimension, \
    SpatialGeometryPolygonDimension, SpatialGeometryPointDimension
from ocgis.interface.base.field import Field
from ocgis.util.inspect import Inspect
from ocgis.util.geom_cabinet import GeomCabinet, GeomCabinetIterator, ShpCabinet, ShpCabinetIterator
from ocgis.util.zipper import format_return
from ocgis.interface.base.variable import Variable
from ocgis.api.request.base import RequestDataset, RequestDatasetCollection
from ocgis.api.operations import OcgOperations

__version__ = '1.2.0'
__release__ = '1.2.0.dev1'
