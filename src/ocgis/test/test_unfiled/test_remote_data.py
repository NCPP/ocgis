import ocgis
from ocgis.test.base import TestBase, attr


class Test(TestBase):
    @attr('remote', 'slow')
    def test_geodataportal_prism(self):
        uri = 'http://cida.usgs.gov/thredds/dodsC/prism'
        for variable in ['tmx', 'tmn', 'ppt']:
            rd = ocgis.RequestDataset(uri, variable, t_calendar='standard')
            ops = ocgis.OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[25],
                                      snippet=True, output_format='numpy', aggregate=False,
                                      prefix=variable)
            ret = ops.execute()
            self.assertEqual(ret[25][variable].variables[variable].value.shape, (1, 1, 1, 227, 246))
