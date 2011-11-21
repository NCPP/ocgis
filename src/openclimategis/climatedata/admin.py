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
from climatedata.models import Variable
from climatedata.models import Archive
from climatedata.models import UserGeometryData
from climatedata.models import SimulationOutput

class NetcdfDatasetAdmin(admin.ModelAdmin):
    list_display = ('uri','calendar','temporal_min','temporal_max',)


class NetcdfDimensionAdmin(admin.ModelAdmin):
    list_display = ('name','size',)


class NetcdfVariableAdmin(admin.ModelAdmin):
    list_display = ('netcdf_dataset','code','ndim',)


class OrganizationAdmin(admin.ModelAdmin):
    list_display = ('name','code','country',)


class ScenarioAdmin(admin.ModelAdmin):
    #fields = ['code','name','description']
    list_display = ('name','code','urlslug',)


class ClimateModelAdmin(admin.ModelAdmin):
    list_display = ('name','code','urlslug',)


class VariableAdmin(admin.ModelAdmin):
    list_display = ('code','name','units','ndim',)


class ArchiveAdmin(admin.ModelAdmin):
    list_display = ('code','url','name',)


#class UserGeometryDataAdmin(admin.ModelAdmin):
#    list_display = ('code','desc',)


class SimulationOutputAdmin(admin.ModelAdmin):
    list_display = ('id','scenario','climate_model','variable','run','archive',)
    list_filter = ['archive','scenario','climate_model','variable',]
    search_fields = ['scenario','climate_model','variable',]
    ordering = ('archive','climate_model','variable','run',)

admin.site.register(NetcdfDataset, NetcdfDatasetAdmin)
admin.site.register(NetcdfDatasetAttribute)
admin.site.register(NetcdfDimension, NetcdfDimensionAdmin)
admin.site.register(NetcdfDimensionAttribute)
admin.site.register(NetcdfVariable, NetcdfVariableAdmin)
admin.site.register(NetcdfVariableAttribute)
admin.site.register(NetcdfVariableDimension)

admin.site.register(Organization, OrganizationAdmin)
admin.site.register(Scenario, ScenarioAdmin)
admin.site.register(ClimateModel, ClimateModelAdmin)
admin.site.register(Variable, VariableAdmin)
admin.site.register(Archive, ArchiveAdmin)
#admin.site.register(UserGeometryData, UserGeometryDataAdmin)
admin.site.register(SimulationOutput, SimulationOutputAdmin)