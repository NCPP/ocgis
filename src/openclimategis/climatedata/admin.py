from django.contrib.gis import admin

from climatedata.models import NetcdfDataset
from climatedata.models import NetcdfDatasetAttribute
from climatedata.models import NetcdfDimension
from climatedata.models import NetcdfDimensionAttribute
from climatedata.models import NetcdfVariable
from climatedata.models import NetcdfVariableAttribute
from climatedata.models import NetcdfVariableDimension

from climatedata.models import Organization
from climatedata.models import Scenario
from climatedata.models import ClimateModel
from climatedata.models import Archive
from climatedata.models import Variable
from climatedata.models import SimulationOutput

admin.site.register(NetcdfDataset)
admin.site.register(NetcdfDatasetAttribute)
admin.site.register(NetcdfDimension)
admin.site.register(NetcdfDimensionAttribute)
admin.site.register(NetcdfVariable)
admin.site.register(NetcdfVariableAttribute)
admin.site.register(NetcdfVariableDimension)

admin.site.register(Organization)
admin.site.register(Scenario)
admin.site.register(ClimateModel)
admin.site.register(Archive)
admin.site.register(Variable)
admin.site.register(SimulationOutput)