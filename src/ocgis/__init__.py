########################################################################################################################
# DO NOT CHANGE IMPORT ORDER!! #########################################################################################
########################################################################################################################

import osgeo
from osgeo import osr, ogr
from ocgis.environment import env
from ocgis.variable.base import SourcedVariable, Variable, VariableCollection, Dimension
from ocgis.driver.request.core import RequestDataset
from ocgis.driver.request.multi_request import MultiRequestDataset
from ocgis.ops.core import OcgOperations
from ocgis.calc.library.register import FunctionRegistry
from ocgis.spatial.grid import GridXY
from ocgis.spatial.geom_cabinet import GeomCabinet, GeomCabinetIterator, ShpCabinet, ShpCabinetIterator
from ocgis.util.zipper import format_return
from ocgis.variable import crs
from ocgis.variable.crs import CoordinateReferenceSystem
from ocgis.collection.field import OcgField
from ocgis.collection.spatial import SpatialCollection
from ocgis.variable.temporal import TemporalVariable

__version__ = '2.0.0.dev1'
__release__ = '2.0.0.dev1'
