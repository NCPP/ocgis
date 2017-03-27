import logging
import os
from csv import excel

from ocgis import Variable
from ocgis import constants
from ocgis.collection.field import OcgField
from ocgis.constants import KeywordArguments, HeaderNames, DriverKeys
from ocgis.conv.base import AbstractTabularConverter
from ocgis.driver.csv_ import DriverCSV
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.vm.mpi import MPI_RANK, MPI_COMM


class OcgDialect(excel):
    lineterminator = '\n'


class CsvConverter(AbstractTabularConverter):
    _ext = 'csv'

    def _write_coll_(self, f, coll, add_geom_uid=True):
        write_mode = f[KeywordArguments.WRITE_MODE]
        path = f[KeywordArguments.PATH]

        iter_kwargs = {'melted': self.melted}

        for field, container in coll.iter_fields(yield_container=True):
            if container.geom is not None:
                repeater = [(self.geom_uid, container.geom.ugid.value[0])]
            else:
                repeater = None

            if add_geom_uid and field.geom is not None and field.geom.ugid is None:
                global_ugid = field.geom.create_ugid_global(HeaderNames.ID_GEOMETRY)
                if field.grid is not None:
                    archetype = field.grid.archetype
                    if hasattr(archetype, '_request_dataset'):
                        if global_ugid.repeat_record is None:
                            repeat_record = []
                        else:
                            repeat_record = global_ugid.repeat_record
                        repeat_record.append((HeaderNames.DATASET_IDENTIFER, archetype._request_dataset.uid))
                        global_ugid.repeat_record = repeat_record

            iter_kwargs[KeywordArguments.REPEATERS] = repeater
            iter_kwargs[KeywordArguments.DRIVER] = DriverCSV
            field.write(path, write_mode=write_mode, driver=DriverCSV, iter_kwargs=iter_kwargs)


class CsvShapefileConverter(CsvConverter):
    _add_ugeom = True

    def __init__(self, *args, **kwargs):
        CsvConverter.__init__(self, *args, **kwargs)
        if self.ops is None:
            raise ValueError('The argument "ops" may not be "None".')

    def _write_coll_(self, f, coll, add_geom_uid=True):
        # Load the geometries. The geometry identifier is needed for the data write.
        for field, container in coll.iter_fields(yield_container=True):
            field.set_abstraction_geom(create_ugid=True)

        # Write the output CSV file.
        super(CsvShapefileConverter, self)._write_coll_(f, coll, add_geom_uid=add_geom_uid)

        # The output geometry identifier shapefile path.
        if MPI_RANK == 0:
            fiona_path = os.path.join(self._get_or_create_shp_folder_(), self.prefix + '_gid.shp')
        else:
            fiona_path = None
        fiona_path = MPI_COMM.bcast(fiona_path)

        if self.ops.aggregate:
            ocgis_lh('creating a UGID-GID shapefile is not necessary for aggregated data. use UGID shapefile.',
                     'conv.csv-shp',
                     logging.WARN)
        else:
            # Write the geometries for each container/field combination.
            for field, container in coll.iter_fields(yield_container=True):

                if not field.is_empty:
                    # The container may be empty. Only add the unique geometry identifier if the container has an
                    # associated geometry.
                    if container.geom is not None:
                        ugid_var = Variable(name=container.geom.ugid.name, dimensions=field.geom.dimensions,
                                            dtype=constants.DEFAULT_NP_INT)
                        ugid_var.value[:] = container.geom.ugid.value[0]

                    # Extract the variable components of the geometry file.
                    geom = field.geom.copy()
                    geom = geom.extract()
                    if field.crs is not None:
                        crs = field.crs.copy()
                        crs = crs.extract()
                    else:
                        crs = None

                    # If the dataset geometry identifier is not present, create it.
                    gid = field[HeaderNames.ID_GEOMETRY].copy()
                    gid = gid.extract()

                    # Construct the field to write.
                    field_to_write = OcgField(geom=geom, crs=crs, uid=field.uid)
                    if container.geom is not None:
                        field_to_write.add_variable(ugid_var, is_data=True)
                    field_to_write.add_variable(gid, is_data=True)

                    # Maintain the field/dataset unique identifier if there is one.
                    if field.uid is not None:
                        if gid.repeat_record is None:
                            rr = []
                        else:
                            rr = list(gid.repeat_record)
                        rr.append((HeaderNames.DATASET_IDENTIFER, field.uid))
                        gid.repeat_record = rr

                else:
                    field_to_write = OcgField(is_empty=True)

                # Write the field.
                field_to_write.write(fiona_path, write_mode=f[KeywordArguments.WRITE_MODE], driver=DriverKeys.VECTOR)
