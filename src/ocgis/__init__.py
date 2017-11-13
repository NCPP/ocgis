########################################################################################################################
# DO NOT CHANGE IMPORT ORDER!! #########################################################################################
########################################################################################################################

import osgeo
from osgeo import osr, ogr
from ocgis import constants
from ocgis.vmachine.core import vm, OcgVM
from ocgis.environment import env
from ocgis.variable.base import SourcedVariable, Variable, VariableCollection, Dimension
from ocgis.driver.dimension_map import DimensionMap
from ocgis.calc.library.register import FunctionRegistry
from ocgis.spatial.grid import Grid, GridUnstruct
from ocgis.spatial.geom_cabinet import GeomCabinet, GeomCabinetIterator, ShpCabinet, ShpCabinetIterator
from ocgis.util.zipper import format_return
from ocgis.variable import crs
from ocgis.variable.crs import CoordinateReferenceSystem, CRS
from ocgis.collection.field import Field
from ocgis.collection.spatial import SpatialCollection
from ocgis.variable.temporal import TemporalVariable
from ocgis.variable.geom import GeometryVariable
from ocgis.spatial.geomc import PolygonGC, LineGC, PointGC
from ocgis.driver.request.core import RequestDataset
from ocgis.driver.request.multi_request import MultiRequestDataset
from ocgis.ops.core import OcgOperations
from ocgis.spatial.grid_splitter import GridSplitter

__version__ = '2.1.0.dev1'
__release__ = '2.1.0.dev1'
