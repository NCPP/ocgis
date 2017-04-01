import csv
from collections import OrderedDict

import six

from ocgis.constants import MPIWriteMode, KeywordArguments, DriverKeys
from ocgis.driver.base import driver_scope, AbstractTabularDriver
from ocgis.vm.mpi import barrier_ranks


class DriverCSV(AbstractTabularDriver):
    extensions = ('.*\.csv',)
    key = DriverKeys.CSV
    output_formats = 'all'
    common_extension = 'csv'

    def get_variable_value(self, variable):
        # For CSV files, it makes sense to load all variables from source simultaneously.
        if variable.parent is None:
            to_load = [variable]
        else:
            to_load = list(variable.parent.values())

        with driver_scope(self) as f:
            reader = csv.DictReader(f)
            bounds_local = variable.dimensions[0].bounds_local
            for idx, row in enumerate(reader):
                if idx < bounds_local[0]:
                    continue
                else:
                    if idx >= bounds_local[1]:
                        break
                for tl in to_load:
                    if not tl.has_allocated_value:
                        tl.allocate_value()
                    tl.get_value()[idx - bounds_local[0]] = row[tl.name]
        return variable.get_value()

    def _get_metadata_main_(self):
        with driver_scope(self) as f:
            meta = {}
            # Get variable names assuming headers are always on the first row.
            reader = csv.reader(f)
            variable_names = six.next(reader)

            # Fill in variable and dimension metadata.
            meta['variables'] = OrderedDict()
            meta['dimensions'] = OrderedDict()
            for varname in variable_names:
                meta['variables'][varname] = {'name': varname, 'dtype': object, 'dimensions': ('n_records',)}
            meta['dimensions']['n_records'] = {'name': 'n_records', 'size': sum(1 for _ in f)}
        return meta

    @classmethod
    def _write_variable_collection_main_(cls, vc, opened_or_path, comm, rank, size, write_mode, ranks_to_write,
                                         **kwargs):
        iter_kwargs = kwargs.pop(KeywordArguments.ITER_KWARGS, {})

        if vc.is_empty:
            fieldnames = None
        else:
            fieldnames = list(six.next(vc.iter(**iter_kwargs))[1].keys())

        if rank == ranks_to_write[0] and write_mode != MPIWriteMode.FILL:
            with driver_scope(cls, opened_or_path, mode='w') as opened:
                writer = csv.DictWriter(opened, fieldnames)
                writer.writeheader()

        if write_mode != MPIWriteMode.TEMPLATE:
            for current_rank_write in ranks_to_write:
                if rank == current_rank_write:
                    with driver_scope(cls, opened_or_path, mode='a') as opened:
                        writer = csv.DictWriter(opened, fieldnames)
                        for _, record in vc.iter(**iter_kwargs):
                            writer.writerow(record)
                barrier_ranks(ranks_to_write, comm=comm)

    def _init_variable_from_source_main_(self, *args, **kwargs):
        pass
