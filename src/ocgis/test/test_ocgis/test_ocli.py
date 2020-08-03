import os
import sys
from unittest import SkipTest

import mock
import numpy as np
from click.testing import CliRunner

import ocgis
from ocgis import RequestDataset, Variable, Grid, vm
from ocgis import env
from ocgis.constants import DecompositionType
from ocgis.ocli import ocli
from ocgis.test.base import TestBase, attr, create_gridxy_global, create_exact_field
from ocgis.util.addict import Dict
from ocgis.variable.crs import Spherical


@attr('cli')
class TestChunkedRWG(TestBase):
    """Tests for target ``chunked-rwg``."""

    def fixture_flags_good(self):
        poss = Dict()

        source = self.get_temporary_file_path('source.nc')
        with open(source, 'w') as f:
            f.write('foo')

        destination = self.get_temporary_file_path('destination.nc')
        with open(destination, 'w') as f:
            f.write('foo')

        poss.source = [source]
        poss.destination = [destination]
        poss.nchunks_dst = ['1,1', '1', '2,2', '2', '__exclude__']
        poss.esmf_src_type = ['__exclude__', 'GRIDSPEC']
        poss.esmf_dst_type = ['__exclude__', 'GRIDSPEC']
        poss.src_resolution = ['__exclude__', '1.0']
        poss.dst_resolution = ['__exclude__', '2.0']
        poss.buffer_distance = ['__exclude__', '3.0']
        poss.wd = ['__exclude__', self.get_temporary_file_path('wd')]
        poss.persist = ['__exclude__', '__include__']
        poss.no_merge = ['__exclude__', '__include__']
        poss.spatial_subset = ['__exclude__', '__include__']
        poss.not_eager = ['__exclude__', '__include__']
        poss.ignore_degenerate = ['__exclude__', '__include__']
        poss.weightfilemode = ['__exclude__', 'basic', 'withaux']

        return poss

    def test_init(self):
        runner = CliRunner()
        result = runner.invoke(ocli)
        self.assertEqual(result.exit_code, 0)

    @attr('mpi', 'esmf')
    def test_system_chunked_versus_global(self):
        """Test weight files are equivalent using the chunked versus global weight generation and SMM approach."""
        if ocgis.vm.size not in [1, 4]:
            raise SkipTest('ocgis.vm.size not in [1, 4]')

        import ESMF

        # Do not put units on bounds variables.
        env.CLOBBER_UNITS_ON_BOUNDS = False

        # Create source and destination files. -------------------------------------------------------------------------
        src_grid = create_gridxy_global(resolution=15, dist_dimname='x')
        dst_grid = create_gridxy_global(resolution=12, dist_dimname='x')

        src_field = create_exact_field(src_grid, 'foo', crs=Spherical(), dtype=np.float64)
        dst_field = create_exact_field(dst_grid, 'foo', crs=Spherical(), dtype=np.float64)

        if ocgis.vm.rank == 0:
            source = self.get_temporary_file_path('source.nc')
        else:
            source = None
        source = ocgis.vm.bcast(source)
        src_field.write(source)
        if ocgis.vm.rank == 0:
            destination = self.get_temporary_file_path('destination.nc')
        else:
            destination = None
        destination = ocgis.vm.bcast(destination)
        dst_field['foo'].v()[:] = -9999
        dst_field.write(destination)
        # --------------------------------------------------------------------------------------------------------------

        # Directory for output grid chunks.
        wd = os.path.join(self.current_dir_output, 'chunks')
        # Path to the merged weight file.
        weight = self.get_temporary_file_path('merged_weights.nc')

        # Generate the source and destination chunks and a merged weight file.
        runner = CliRunner()
        cli_args = ['chunked-rwg', '--source', source, '--destination', destination, '--nchunks_dst', '2,3', '--wd',
                    wd, '--weight', weight, '--persist']
        result = runner.invoke(ocli, args=cli_args, catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(len(os.listdir(wd)) > 3)

        # Also apply the sparse matrix
        runner2 = CliRunner()
        cli_args = ['chunked-smm', '--wd', wd, '--insert_weighted', '--destination', destination]
        result = runner2.invoke(ocli, args=cli_args, catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)

        # Create a standard ESMF weights file from the original grid files.
        esmf_weights_path = self.get_temporary_file_path('esmf_desired_weights.nc')

        # Generate weights using ESMF command line interface.
        # cmd = ['ESMF_RegridWeightGen', '-s', source, '--src_type', 'GRIDSPEC', '-d', destination, '--dst_type',
        #        'GRIDSPEC', '-w', esmf_weights_path, '--method', 'conserve', '--no-log']
        # subprocess.check_call(cmd)

        # Create a weights file using the ESMF Python interface.
        srcgrid = ESMF.Grid(filename=source, filetype=ESMF.FileFormat.GRIDSPEC, add_corner_stagger=True)
        dstgrid = ESMF.Grid(filename=destination, filetype=ESMF.FileFormat.GRIDSPEC, add_corner_stagger=True)
        srcfield = ESMF.Field(grid=srcgrid, typekind=ESMF.TypeKind.R8)
        srcfield.data[:] = np.swapaxes(np.squeeze(src_field['foo'].v()), 0, 1)
        dstfield = ESMF.Field(grid=dstgrid, typekind=ESMF.TypeKind.R8)
        _ = ESMF.Regrid(srcfield=srcfield, dstfield=dstfield, filename=esmf_weights_path,
                        regrid_method=ESMF.RegridMethod.CONSERVE)

        if ocgis.vm.rank == 0:
            # Assert the weight files are equivalent using chunked versus global creation.
            self.assertWeightFilesEquivalent(esmf_weights_path, weight)

        actual_dst = RequestDataset(uri=destination, decomp_type=DecompositionType.ESMF).create_field()['foo'].v()
        actual_dst = np.swapaxes(np.squeeze(actual_dst), 0, 1)
        desired_dst = dstfield.data
        self.assertNumpyAllClose(actual_dst, desired_dst)

    def test_system_merged_weight_file_in_working_directory(self):
        """Test merged weight file may not be created inside the chunking working directory."""

        flags = self.fixture_flags_good()

        source = flags['source'][0]
        destination = flags['destination'][0]
        wd = os.path.join(self.current_dir_output, 'chunks')
        weight = os.path.join(wd, 'weights.nc')

        runner = CliRunner()
        cli_args = ['chunked-rwg', '--source', source, '--destination', destination, '--wd', wd, '--weight', weight]
        with self.assertRaises(ValueError):
            _ = runner.invoke(ocli, args=cli_args, catch_exceptions=False)

    @mock.patch('ocgis.ocli._write_spatial_subset_')
    @mock.patch('os.makedirs')
    @mock.patch('shutil.rmtree')
    @mock.patch('tempfile.mkdtemp')
    @mock.patch('ocgis.ocli.GridChunker')
    @mock.patch('ocgis.ocli.RequestDataset')
    @attr('mpi', 'slow', 'no-3.5')
    def test_system_mock_combinations(self, mRequestDataset, mGridChunker, m_mkdtemp, m_rmtree, m_makedirs,
                                      m_write_spatial_subset):
        if sys.version_info.major == 3 and sys.version_info.minor == 5:
            raise SkipTest('undefined behavior with Python 3.5')

        if ocgis.vm.size not in [1, 2]:
            raise SkipTest('ocgis.vm.size not in [1, 2]')

        poss_weight = {'filename': self.get_temporary_file_path('weights.nc')}

        m_mkdtemp.return_value = 'mkdtemp return value'

        poss = self.fixture_flags_good()
        for ctr, k in enumerate(self.iter_product_keywords(poss, as_namedtuple=False), start=1):
            # print(k)
            new_poss = {}
            for k2, v2 in k.items():
                if v2 != '__exclude__':
                    new_poss[k2] = v2
            cli_args = ['chunked-rwg']
            for k2, v2 in new_poss.items():
                cli_args.append('--{}'.format(k2))
                if v2 != '__include__':
                    cli_args.append(v2)

            # Add the output weight filename if requested.
            if 'no_merge' not in new_poss or 'spatial_subset' in new_poss:
                weight = poss_weight['filename']
                new_poss['weight'] = weight
                cli_args.extend(['--weight', weight])

            runner = CliRunner()
            result = runner.invoke(ocli, args=cli_args, catch_exceptions=False)
            self.assertEqual(result.exit_code, 0)

            mGridChunker.assert_called_once()
            instance = mGridChunker.return_value
            call_args = mGridChunker.call_args

            if k['wd'] == '__exclude__' and 'spatial_subset' not in new_poss:
                actual = call_args[1]['paths']['wd']
                try:
                    self.assertEqual(actual, m_mkdtemp.return_value)
                except AssertionError:
                    self.assertTrue(k['nchunks_dst'][0], '1')

            if 'no_merge' not in new_poss and 'spatial_subset' not in new_poss and vm.rank == 0:
                try:
                    instance.create_merged_weight_file.assert_called_once_with(new_poss['weight'])
                except AssertionError:
                    self.assertTrue(k['nchunks_dst'][0], '1')
            else:
                instance.create_merged_weight_file.assert_not_called()
            if new_poss.get('nchunks_dst') is not None and 'spatial_subset' not in new_poss:
                try:
                    instance.write_chunks.assert_called_once()
                except AssertionError:
                    self.assertTrue(k['nchunks_dst'][0], '1')

            if k['nchunks_dst'] == '1,1':
                self.assertEqual(call_args[1]['nchunks_dst'], (1, 1))
            elif k['nchunks_dst'] == '1':
                self.assertEqual(call_args[1]['nchunks_dst'], (1,))

            self.assertIn(call_args[1]["filemode"].lower(), ["withaux", "basic"])

            actual = call_args[1]['eager']
            if k['not_eager'] == '__include__':
                self.assertFalse(actual)
            else:
                self.assertTrue(actual)

            self.assertEqual(mRequestDataset.call_count, 2)

            if 'merge' not in new_poss:
                if 'wd' not in new_poss:
                    if ocgis.vm.rank == 0:
                        try:
                            m_mkdtemp.assert_called_once()
                        except AssertionError:
                            self.assertTrue(k['nchunks_dst'][0], '1')
                    else:
                        m_mkdtemp.assert_not_called()
                else:
                    if ocgis.vm.rank == 0:
                        try:
                            m_makedirs.assert_called_once()
                        except AssertionError:
                            self.assertTrue(k['nchunks_dst'][0], '1')
                    else:
                        m_makedirs.assert_not_called()
            else:
                m_mkdtemp.assert_not_called()
                m_makedirs.assert_not_called()

            if 'persist' not in new_poss:
                if ocgis.vm.rank == 0:
                    m_rmtree.assert_called_once()
                else:
                    m_rmtree.assert_not_called()
            else:
                m_rmtree.assert_not_called()

            # Test ESMF weight writing is called directly with a spatial subset.
            if 'spatial_subset' in new_poss:
                m_write_spatial_subset.assert_called_once()
            else:
                m_write_spatial_subset.assert_not_called()

            mocks = [mRequestDataset, mGridChunker, m_mkdtemp, m_rmtree, m_makedirs, m_write_spatial_subset]
            for m in mocks:
                m.reset_mock()

    @attr('esmf')
    def test_chunked_rwg_spatial_subset(self):
        env.CLOBBER_UNITS_ON_BOUNDS = False

        src_grid = create_gridxy_global(crs=Spherical())
        src_field = create_exact_field(src_grid, 'foo')

        xvar = Variable(name='x', value=[-90., -80.], dimensions='xdim')
        yvar = Variable(name='y', value=[40., 50.], dimensions='ydim')
        dst_grid = Grid(x=xvar, y=yvar, crs=Spherical())

        if ocgis.vm.rank == 0:
            source = self.get_temporary_file_path('source.nc')
        else:
            source = None
        source = ocgis.vm.bcast(source)
        src_field.write(source)

        if ocgis.vm.rank == 0:
            destination = self.get_temporary_file_path('destination.nc')
        else:
            destination = None
        destination = ocgis.vm.bcast(destination)
        dst_grid.parent.write(destination)

        wd = os.path.join(self.current_dir_output, 'chunks')
        weight = os.path.join(self.current_dir_output, 'weights.nc')
        spatial_subset = os.path.join(self.current_dir_output, 'spatial_subset.nc')

        runner = CliRunner()
        cli_args = ['chunked-rwg', '--source', source, '--destination', destination, '--wd', wd, '--spatial_subset',
                    '--spatial_subset_path', spatial_subset, '--weight', weight, '--esmf_regrid_method', 'BILINEAR',
                    '--persist']
        result = runner.invoke(ocli, args=cli_args, catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)

        actual = RequestDataset(uri=spatial_subset).create_field()
        actual_ymean = actual.grid.get_value_stacked()[0].mean()
        actual_xmean = actual.grid.get_value_stacked()[1].mean()
        self.assertEqual(actual_ymean, 45.)
        self.assertEqual(actual_xmean, -85.)
        self.assertEqual(actual.grid.shape, (14, 14))

        self.assertTrue(os.path.exists(weight))
        actual = RequestDataset(weight, driver='netcdf').create_field()
        self.assertIn('history', actual.attrs)
