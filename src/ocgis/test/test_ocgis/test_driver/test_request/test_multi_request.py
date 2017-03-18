from ocgis import RequestDataset, MultiRequestDataset, OcgOperations
from ocgis.base import get_variable_names
from ocgis.test.base import TestBase


class TestMultiRequestDataset(TestBase):
    f_variable_names = ('foo1', 'foo2')

    def get_multirequestdataset(self):
        request_datasets = []
        for ctr, vn in enumerate(self.f_variable_names):
            path = self.get_temporary_file_path('{}.nc'.format(vn))
            field = self.get_field(variable_name=vn)
            value = field[vn].get_value()
            value[:] = value + ctr
            field.write(path)
            rd = RequestDataset(path)
            request_datasets.append(rd)

        mrd = MultiRequestDataset(request_datasets)
        return mrd

    def test_init(self):
        assert self.get_multirequestdataset()

    def test_system_through_operations(self):
        mrd = self.get_multirequestdataset()
        ops = OcgOperations(dataset=mrd)
        ret = ops.execute()
        field = ret.get_element()
        actual = get_variable_names(field.data_variables)
        self.assertEqual(actual, self.f_variable_names)

        mrd = self.get_multirequestdataset()
        ops = OcgOperations(dataset=mrd, output_format='nc')
        ret = ops.execute()
        actual_field = RequestDataset(ret).get()
        actual = get_variable_names(actual_field.data_variables)
        self.assertEqual(actual, self.f_variable_names)

        actual_diff = actual_field.data_variables[1].get_value() - actual_field.data_variables[0].get_value()
        self.assertAlmostEqual(actual_diff.mean(), 1.0)

    def test_get(self):
        mrd = self.get_multirequestdataset()
        mfield = mrd.get()
        actual = get_variable_names(mfield.data_variables)
        self.assertEqual(actual, self.f_variable_names)
