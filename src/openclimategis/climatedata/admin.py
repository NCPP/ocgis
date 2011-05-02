from django.contrib import admin

from climatedata.models import Organization
from climatedata.models import Archive
from climatedata.models import Variable
from climatedata.models import ClimateModel
from climatedata.models import Experiment
from climatedata.models import Frequency

admin.site.register(Organization)
admin.site.register(Archive)
admin.site.register(Variable)
admin.site.register(ClimateModel)
admin.site.register(Experiment)
admin.site.register(Frequency)
