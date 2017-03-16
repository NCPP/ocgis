import csv

import numpy as np

from ocgis import RequestDataset
from ocgis.collection.field import OcgField
from ocgis.constants import HeaderNames
from ocgis.driver.csv_ import DriverCSV
from ocgis.test.base import TestBase, attr
from ocgis.variable.base import Variable, VariableCollection
from ocgis.variable.temporal import TemporalVariable
from ocgis.vm.mpi import MPI_RANK, MPI_COMM, OcgMpi, variable_collection_scatter


class TestDriverCSV(TestBase):
    def assertCSVFilesEqual(self, path1, path2):
        with open(path1) as one:
            lines1 = one.readlines()
        with open(path2) as two:
            lines2 = two.readlines()
        self.assertEqual(lines1, lines2)

    def get_path_to_template_csv(self):
        headers = [HeaderNames.DATASET_IDENTIFER, 'ONE', 'two', 'THREE', 'x', 'y']
        record1 = [None, 1, 'number', 4.5, 10.3, 12.4]
        record2 = [None, 2, 'letter', 5.5, 11.3, 13.4]
        path = self.get_temporary_file_path('foo.csv')
        with open(path, 'w') as out:
            writer = csv.writer(out)
            for row in [headers, record1, record2]:
                writer.writerow(row)
        return path

    def test(self):
        path = self.get_path_to_template_csv()
        path_out = self.get_temporary_file_path('foo_out.csv')

        rd = RequestDataset(uri=path)
        vc = rd.get_variable_collection()

        for v in vc.values():
            self.assertIsNotNone(v.value)

        field = rd.get()
        self.assertIsInstance(field, OcgField)

        vc.write(path_out, driver=DriverCSV)

        self.assertCSVFilesEqual(path, path_out)

    def test_get_dump_report(self):
        path = self.get_path_to_template_csv()
        rd = RequestDataset(path)
        driver = DriverCSV(rd)
        dr = driver.get_dump_report()
        self.assertGreater(len(dr), 1)

    @attr('mpi')
    def test_system_parallel_write(self):
        if MPI_RANK == 0:
            in_path = self.get_path_to_template_csv()
            out_path = self.get_temporary_file_path('foo_out.csv')
        else:
            in_path, out_path = [None] * 2

        in_path = MPI_COMM.bcast(in_path)
        out_path = MPI_COMM.bcast(out_path)

        rd = RequestDataset(in_path)
        rd.metadata['dimensions'].values()[0]['dist'] = True
        vc = rd.get_variable_collection()
        vc.write(out_path, driver=DriverCSV)

        if MPI_RANK == 0:
            self.assertCSVFilesEqual(in_path, out_path)

        MPI_COMM.Barrier()

    @attr('mpi')
    def test_system_parallel_write_ndvariable(self):
        """Test a parallel CSV write with a n-dimensional variable."""

        ompi = OcgMpi()
        ompi.create_dimension('time', 3)
        ompi.create_dimension('extra', 2)
        ompi.create_dimension('x', 4)
        ompi.create_dimension('y', 7, dist=True)
        ompi.update_dimension_bounds()

        if MPI_RANK == 0:
            path = self.get_temporary_file_path('foo.csv')

            t = TemporalVariable(name='time', value=[1, 2, 3], dtype=float, dimensions='time')
            t.set_extrapolated_bounds('the_time_bounds', 'bounds')

            extra = Variable(name='extra', value=[7, 8], dimensions='extra')

            x = Variable(name='x', value=[9, 10, 11, 12], dimensions='x', dtype=float)
            x.set_extrapolated_bounds('x_bounds', 'bounds')

            # This will have the distributed dimension.
            y = Variable(name='y', value=[13, 14, 15, 16, 17, 18, 19], dimensions='y', dtype=float)
            y.set_extrapolated_bounds('y_bounds', 'bounds')

            data = Variable(name='data', value=np.random.rand(3, 2, 7, 4), dimensions=['time', 'extra', 'y', 'x'])

            vc = VariableCollection(variables=[t, extra, x, y, data])
        else:
            path, vc = [None] * 2

        path = MPI_COMM.bcast(path)
        vc, _ = variable_collection_scatter(vc, ompi)

        vc.write(path, iter_kwargs={'variable': 'data', 'followers': ['time', 'extra', 'y', 'x']}, driver=DriverCSV)

        if MPI_RANK == 0:
            desired = 169
            with open(path, 'r') as f:
                lines = f.readlines()
            self.assertEqual(len(lines), desired)
