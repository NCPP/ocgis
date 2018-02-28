########################################################################################################################
# DO NOT CHANGE IMPORT ORDER!! #########################################################################################
########################################################################################################################

import osgeo
from osgeo import osr, ogr
from .vmachine.core import vm, OcgVM
from .environment import env
from .variable.base import SourcedVariable, Variable, VariableCollection, Dimension
from .driver.dimension_map import DimensionMap
from .calc.library.register import FunctionRegistry
from .spatial.grid import Grid, GridUnstruct
from .spatial.geom_cabinet import GeomCabinet, GeomCabinetIterator, ShpCabinet, ShpCabinetIterator
from .util.zipper import format_return
from .variable import crs
from .variable.crs import CoordinateReferenceSystem, CRS
from .collection.field import Field
from .collection.spatial import SpatialCollection
from .variable.temporal import TemporalVariable
from .variable.geom import GeometryVariable
from .spatial.geomc import PolygonGC, LineGC, PointGC
from .driver.request.core import RequestDataset
from .driver.request.multi_request import MultiRequestDataset
from .ops.core import OcgOperations

__version__ = '2.1.0.dev1'
__release__ = '2.1.0.dev1'
