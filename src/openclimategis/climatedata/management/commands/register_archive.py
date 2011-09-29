import sys
import os
import types
import warnings
import netCDF4
import numpy as np
from django.core.management.base import LabelCommand
from django.contrib.gis.geos.polygon import Polygon
from climatedata import models
from climatedata.models import Archive
from climatedata.models import ClimateModel
from climatedata.models import Scenario
from climatedata.models import Variable

def build_netcdf_dictionary(netcdf_dataset_uri):
    '''creates a dictionary corresponding to a NetCDF file metadata'''
    try:
        remote_dataset = netCDF4.Dataset(netcdf_dataset_uri,'r')
        info = {'uri': netcdf_dataset_uri}
        for attr in [attr for attr in dir(remote_dataset) if attr[0]!='_']:
            attr_obj = getattr(remote_dataset,attr)
            if attr == 'dimensions':
                # store the NetCDF Dimension information
                dim_dict = {}
                for dim_name in attr_obj:
                    dim_obj = attr_obj[dim_name]
                    dim_dict[dim_name]= {
                        'size': len(dim_obj),
                        'isunlimited': dim_obj.isunlimited(),
                    }
                info['dimensions'] = dim_dict
            elif attr == 'variables':
                # store the NetCDF Variable information
                var_dict = {}
                for var_name in attr_obj:
                    var_obj = attr_obj[var_name]
                    var_info = {}
                    for attr in [attr for attr in dir(var_obj) if attr[0]!='_']:
                        if isinstance(getattr(var_obj,attr), types.BuiltinMethodType):
                            pass # ignore the build in methods
                        else:
                            var_info[attr] = getattr(var_obj,attr)
                    var_dict[var_name] = var_info
                info['variables'] = var_dict
            elif isinstance(attr_obj, types.BuiltinMethodType):
                pass
            else:
                # store the global attributes
                info[attr] = getattr(remote_dataset,attr)
            
        # extract data on the spatial extent
        bounds_longitude = remote_dataset.variables['bounds_longitude'][:]
        bounds_latitude = remote_dataset.variables['bounds_latitude'][:]
        info['spatial_extent'] = {
            'min_x': bounds_longitude.min(),
            'max_x': bounds_longitude.max(),
            'min_y': bounds_latitude.min(),
            'max_y': bounds_latitude.max(),
        }
        
        # extract data on the time periods
        info['time_vector'] = netCDF4.netcdftime.num2date(
                            remote_dataset.variables['time'][:],
                            remote_dataset.variables['time'].units,
                            remote_dataset.variables['time'].calendar,
                        )
    finally:
        remote_dataset.close()
    return info


def register_usgs_maurer_archive(archive_obj):
    '''populates database tables with data corresponding to the USGS CIDA
    maurer et. al archive
    '''
    
    # maps climate model names used by the USGS to model names used by the IPCC
    model_map = {
        'bccr-bcm2-0': 'BCCR-BCM2.0',
        'cccma-cgcm3-1': 'CGCM3.1(T47)',
        'cnrm-cm3': 'CNRM-CM3',
        'csiro-mk3-0': 'CSIRO-Mk3.0',
        'gfdl-cm2-0': 'GFDL-CM2.0',
        'gfdl-cm2-1': 'GFDL-CM2.1',
        'giss-model-e-r': 'GISS-ER',
        'inmcm3-0': 'INM-CM3.0',
        'ipsl-cm4': 'IPSL-CM4',
        'miroc3-2-medres': 'MIROC3.2(medres)',
        'miub-echo-g': 'ECHO-G',
        'mpi-echam5': 'ECHAM5/MPI-OM',
        'mri-cgcm2-3-2a': 'MRI-CGCM2.3.2',
        'ncar-ccsm3-0': 'CCSM3',
        'ncar-pcm1': 'PCM',
        'ukmo-hadcm3': 'UKMO-HadCM3',
    }
    scenario_map = {
        'sresa1b': 'SRES A1B',
        'sresa2': 'SRES A2',
        'sresb1': 'SRES B1',
    }
    variable_map = {
        'Prcp': 'pr',
        'Tavg': 'tas',
    }
    
    def parse_variable_name(var_name):
        '''Parses information contained in the NetCDF variable name
        
        Example: sresa1b_bccr-bcm2-0_1_Prcp
        '''
        scenario,model,run,variable = var_name.split('_')
        
        # remap to the IPCC standard codes
        model = model_map[model]
        scenario = scenario_map[scenario]
        variable = variable_map[variable]
        
        return {
            'scenario':scenario,
            'climate_model':model,
            'run':int(run),
            'sim_variable':variable,
        }
    
    def extract_simulation_parameters_from_variablename(archive_meta):
        '''Extract the simulation parameters from the NetCDF Variable name
        '''
        for var in archive_meta['variables']:
            try:
                archive_meta['variables'][var]['simulation_params'] = parse_variable_name(var)
            except:
                pass # ignore dimension variables
        return archive_meta
    
    def populate_simulation_output_table(archive_obj, archive_meta):
        '''Populates the SimulationOutput table
        
        This links simulation parameters (climate model, emissions scenario,
        run, climate variable) to a corresponding NetCDF Variable
        '''
        sys.stdout.write('Starting to populate the SimulationOutput table')
        netcdf_dataset_obj = models.NetcdfDataset.objects.get(
            uri=archive_meta['uri']
        )
        for var_name in archive_meta['variables']:
            sys.stdout.write('.')
            try:
                sim_params = archive_meta['variables'][var_name]['simulation_params']
                sim, created = models.SimulationOutput.objects.get_or_create(
                    archive = archive_obj,
                    scenario = models.Scenario.objects.get(
                        code=sim_params['scenario']
                    ),
                    climate_model = models.ClimateModel.objects.get(
                        code=sim_params['climate_model']
                    ),
                    variable = models.Variable.objects.get(
                        code=sim_params['sim_variable']
                    ),
                    run = sim_params['run'],
                    netcdf_variable = models.NetcdfVariable.objects.get(
                        netcdf_dataset=netcdf_dataset_obj,
                        code = var_name,
                    ),
                )
            except:
                pass
        sys.stdout.write('Done.\n')
    
    dataset_uris= [
        'http://cida.usgs.gov/qa/thredds/dodsC/maurer/monthly',
    ]
    for dataset_uri in dataset_uris:
        # create a dictionary of metadata for the remote NetCDF Dataset
        archive_meta = build_netcdf_dictionary(dataset_uri)
        
        # populate the NetCDF Metadata database tables
        nc = NcDatasetImporter(archive_meta)
        nc.load()

        # extract the simulation parameters which, for the USGS CIDA Maurer 
        # archive, are stored in the variable name
        archive_meta = extract_simulation_parameters_from_variablename(archive_meta)

        # populate the SimulationOutput database table
        populate_simulation_output_table(archive_obj, archive_meta)

# list information on the archives that can be registered
archives = {
    'http://cida.usgs.gov/qa/thredds/dodsC/maurer/monthly':
        {
            'name': '1/8 degree-CONUS Monthly Bias Corrected Spatially Downscaled Climate Projections by Maurer, Brekke, Pruitt, and Duffy',
            'code': 'cida.usgs.gov/maurer',
            'populate_netcdf_tables_method': register_usgs_maurer_archive,
        },
    'http://hydra.fsl.noaa.gov/thredds/esgcet/catalog.html':
        {
            'name': 'NOAA OpenClimateGIS Downscaling Archive',
            'code': 'oc_gis_downscaling',
            'populate_netcdf_tables_method': 'PLACEHOLDER',
#                'http://hydra.fsl.noaa.gov/thredds/dodsC/oc_gis_downscaling.bccr_bcm2.sresa1b.Prcp.Prcp.1.aggregation.1'
#                'http://hydra.fsl.noaa.gov/thredds/dodsC/oc_gis_downscaling.bccr_bcm2.sresa1b.Tavg.Tavg.1.aggregation.1'
#                'http://hydra.fsl.noaa.gov/thredds/dodsC/oc_gis_downscaling.cccma_cgcm3.sresa2.Prcp.Prcp.1.aggregation.1'
#                'http://hydra.fsl.noaa.gov/thredds/dodsC/oc_gis_downscaling.cccma_cgcm3.sresa2.Tavg.Tavg.1.aggregation.1'
        },
}


class ArchiveMissingException(Exception):
    '''Exception triggered when an invalid archive is requested'''
    def __init__(self, value):
        self.parameter = value
    def __str__(self):
        return repr(self.parameter)


class NcDatasetImporter(object):
    """
    Import a netCDF4 Dataset object into the database.
    """
    
    def __init__(self,archive_meta):
        self.archive_meta = archive_meta
        self._set_name_('time_name',['time'])
        keys = self.archive_meta['variables'][self.time_name].keys()
        self._set_name_('time_units_name',['units'],keys)
        self._set_name_('calendar_name',['calendar'],keys)
        
        self._set_name_('rowbnds_name',['lat_bounds','latitude_bounds','lat_bnds','latitude_bnds','bounds_latitude'])
        self._set_name_('colbnds_name',['lon_bounds','longitude_bounds','lon_bnds','longitude_bnds','bounds_longitude'])
        
        #self._set_name_('level_name',['level','levels','lvl','lvls'])
        
        self.time_units = self.archive_meta['variables'][self.time_name][self.time_units_name]
        self.calendar = self.archive_meta['variables'][self.time_name][self.calendar_name]
    
    def load(self):
        
        ## save a record for the NetCDF Dataset
        sys.stdout.write('...saving the NetCDF dataset record...')
        attrs = dict(
            uri=self.archive_meta['uri'],
            rowbnds_name=self.rowbnds_name,
            colbnds_name=self.colbnds_name,
            time_name=self.time_name,
            time_units=self.time_units,
            calendar=self.calendar,
            spatial_extent=self._spatial_extent_(),
            #level_name=self.level_name,
        )
        attrs.update(self._temporal_fields_())
        dataset, created = models.NetcdfDataset.objects.get_or_create(**attrs)
        sys.stdout.write(' Done!\n')
        
        ## save the global attributes for the NetCDF Dataset
        sys.stdout.write('...saving the NetCDF global attribute records...')
        for key,value in self.archive_meta.items():
            if isinstance(value, (basestring, int, tuple,)):
                attrv, created = models.NetcdfDatasetAttribute.objects.get_or_create(
                            netcdf_dataset=dataset,
                            key=key,
                            value=str(value),
                        )
        sys.stdout.write('Done!\n')
        
        ## save the dimensions of the netCDF Dataset
        sys.stdout.write('...saving the NetCDF dataset dimension records...')
        dimensions = {}
        for key,value in self.archive_meta['dimensions'].items():
            dim_obj, created = models.NetcdfDimension.objects.get_or_create(
                netcdf_dataset=dataset,
                name=key,
                size=value['size'],
            )
            # save the dimension so it can be later paired with variables
            dimensions[key]=dim_obj
            # save the dimension attributes
            for dim_key, dim_value in value.items():
                dim_attr_obj, created = models.NetcdfDimensionAttribute.objects.get_or_create(
                    netcdf_dimension=dim_obj,
                    key=dim_key,
                    value=dim_value,
                )
            
        sys.stdout.write('Done!\n')
            
        ## loop for each NetCDF variable
        sys.stdout.write('...saving the NetCDF dataset variable records...\n')
        for key,value in self.archive_meta['variables'].items():
            sys.stdout.write('{indent}variable={var}\n'.format(indent=' '*4,var=key))
            variable_obj, created = models.NetcdfVariable.objects.get_or_create(
                                           netcdf_dataset=dataset,
                                           code=key,
                                           ndim=len(value['dimensions'])
                                        )
            for dim in value['dimensions']:
                #sys.stdout.write('{indent}dimension={dim}\n'.format(indent=' '*8,dim=dim))
                var_dim_obj, created = models.NetcdfVariableDimension.objects.get_or_create(
                    netcdf_variable=variable_obj,
                    netcdf_dimension=dimensions[dim],
                    position=value['dimensions'].index(dim),
                )
            for var_attr_key,var_attr_value in value.items():
#                sys.stdout.write('{indent}attribute={attr}\n'.format(
#                                                indent=' '*8,
#                                                attr=var_attr_key
#                                            ))
                if isinstance(var_attr_value, (basestring, int, tuple, np.dtype, np.float32,)):
                    attrv, created = models.NetcdfVariableAttribute.objects.get_or_create(
                        netcdf_variable=variable_obj,
                        key=var_attr_key,
                        value=str(var_attr_value),
                    )
                else:
                    pass
            pass
    
    def _set_name_(self,target,options,keys=None):
        "Search naming options for target variables."
        
        ret = None
        if not keys:
            keys = self.archive_meta['variables'].keys()
        for key in keys:
            if key in options:
                ret = key
                break
        setattr(self,target,ret)
        if not ret:
            warnings.warn('variable "{0}" not found in {1}. setting to "None" and no load is attempted.'.format(target,self.remote_dataset.variables.keys()))

    def _temporal_fields_(self):
        timevec = self.archive_meta['time_vector']
        temporal_min = min(timevec)
        temporal_max = max(timevec)
        
        start = 0
        target = 1
        diffs = []
        
        while True:
            try:
                s = timevec[start]
                t = timevec[target]
                diffs.append((t-s).days)
                start += 1
                target += 1
            except IndexError:
                break
        
        temporal_interval_days = float(np.mean(diffs))
        
        return(dict(temporal_min=temporal_min,
                    temporal_max=temporal_max,
                    temporal_interval_days=temporal_interval_days))
        
    def _spatial_extent_(self):
        spatial_extent= self.archive_meta['spatial_extent']
        min_x = float(spatial_extent['min_x'])
        max_x = float(spatial_extent['max_x'])
        min_y = float(spatial_extent['min_y'])
        max_y = float(spatial_extent['max_y'])
        p = Polygon(
            ((min_x,min_y),
             (max_x,min_y),
             (max_x,max_y),
             (min_x,max_y),
             (min_x,min_y)),srid=4326
        )
        return(p)


class Command(LabelCommand):
    args = '[archive_url]'
    help = 'Registers a climate model archive'
    label = 'Climate model archive URL'
    
    def handle_label(self, archive_url, **options):
        try:
            try:
                info = archives[archive_url]
            except KeyError:
                raise ArchiveMissingException(archive_url)
            
            # create an archive
            archive_obj, created = Archive.objects.get_or_create(
                url=archive_url,
                name=info['name'],
                code=info['code'],
            )
            if created:
                self.stdout.write('Archive record was created.\n')
                info['populate_netcdf_tables_method'](archive_obj)
                self.stdout.write('Done!\n')
            else:
                self.stdout.write('Archive record already exists. '
                                  'Skipping the registration process.\n')
            
        except ArchiveMissingException, (instance):
            self.stdout.write(
                ('Archive URL ({url}) was not recognized!\n'
                'Recognized URLs are:\n    {valid_list}\n'
                ).format(
                        url=instance.parameter,
                        valid_list='\n    '.join(archives.keys())
                    )
            )
        except:
            import sys
            self.stdout.write("Unexpected error:", sys.exc_info()[0])
            