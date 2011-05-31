from django.contrib import admin

from climatedata.models import Organization
from climatedata.models import Archive
from climatedata.models import Variable
from climatedata.models import ClimateModel
from climatedata.models import Experiment
from climatedata.models import SpatialGrid
from climatedata.models import SpatialGridCell
from climatedata.models import TemporalGrid
from climatedata.models import TemporalGridCell
from climatedata.models import Prediction

admin.site.register(Organization)
admin.site.register(Archive)
admin.site.register(Variable)
admin.site.register(ClimateModel)
admin.site.register(Experiment)
admin.site.register(SpatialGrid)
admin.site.register(SpatialGridCell)
admin.site.register(TemporalGrid)
admin.site.register(TemporalGridCell)
admin.site.register(Prediction)