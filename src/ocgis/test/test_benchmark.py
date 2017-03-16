import ocgis
from ocgis.driver.request import RequestDataset
from ocgis.ops.core import OcgOperations
from ocgis.test.base import TestBase, attr


class Test(TestBase):
    @attr('benchmark')
    def test(self):
        # development laptop: 8 procs: 146 seconds
        ocgis.env.VERBOSE = True
        uri = '/home/benkoziol/l/data/bekozi-work/lisa-rensi-nwm/nwm.t00z.analysis_assim.terrain_rt.tm00.conus.nc_georeferenced.nc'
        # dimension_map = {'time': {'variable': 'time', 'names': ['time']},
        #                  'x': {'variable': 'x', 'names': ['x']},
        #                  'y': {'variable': 'y', 'names': ['y']},
        #                  'crs': {'variable': 'ProjectionCoordinateSystem'}}
        dimension_map = {'time': {'variable': 'time'},
                         'x': {'variable': 'x'},
                         'y': {'variable': 'y'},
                         'crs': {'variable': 'ProjectionCoordinateSystem'}}
        rd = RequestDataset(uri, dimension_map=dimension_map)
        # barrier_print(rd.dist.get_dimension('x').bounds_local)
        # barrier_print(rd.dist.get_dimension('y').bounds_local)
        # tkk
        field = rd.get()
        None

        ops = OcgOperations(dataset=rd, geom='state_boundaries', geom_select_uid=[16])
        ret = ops.execute()

        # path = self.get_temporary_file_path('foo.nc')

        # to_pop = get_variable_names(field.data_variables)

        # for tp in to_pop:
        #     field.remove_variable(tp)
        #
        # field.write(path)
        # tkk
        # pprint_dict(field.dimension_map)
        # print field.grid.extent
