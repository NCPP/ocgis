import ocgis
from ocgis.test.base import TestBase


class Test(TestBase):

    def test_writing_projections(self):
        ocgis.env.DIR_DATA = '/usr/local/climate_data/narccap'
        ocgis.env.WRITE_TO_REFERENCE_PROJECTION = True

        files = [
            #                 'pr_CRCM_ccsm_1981010103.nc',
            #                 'pr_ECP2_gfdl_1981010103.nc',
            #                 'pr_HRM3_gfdl_1981010103.nc',
            #                 'pr_MM5I_ccsm_1986010103.nc',
            'pr_RCM3_gfdl_1981010103.nc',
            #                 'pr_WRFG_ccsm_1981010103.nc'
        ]
        for file in files:
            prefix = file[0:7]
            rd = ocgis.RequestDataset(uri=file, variable='pr')
            ops = ocgis.OcgOperations(dataset=rd, output_format='shp',
                                      snippet=True, prefix=prefix)
            ret = ops.execute()

        import ipdb;
        ipdb.set_trace()
