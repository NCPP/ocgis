import numpy as np

from ocgis import constants
from ocgis.interface.base.variable import Variable, VariableCollection
from ocgis.interface.base.field import Field
from ocgis.api.request.driver.base import AbstractDriver


class DriverVector(AbstractDriver):
    extensions = ('.*\.shp',)
    key = 'vector'
    output_formats = [constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH,
                      constants.OUTPUT_FORMAT_SHAPEFILE]

    def close(self, obj):
        pass

    def get_crs(self):
        from ocgis import CoordinateReferenceSystem

        return CoordinateReferenceSystem(self.rd.source_metadata['crs'])

    def get_dimensioned_variables(self):
        return self.rd.source_metadata['schema']['properties'].keys()

    def get_source_metadata(self):
        try:
            data = self.open()
            return data.sc.get_meta(path=self.rd.uri)
        finally:
            self.close(data)

    def inspect(self):
        lines = self._inspect_get_lines_()
        for line in lines:
            print line

    def open(self):
        from ocgis import ShpCabinetIterator

        return ShpCabinetIterator(path=self.rd.uri)

    def _get_field_(self, format_time=None):
        # todo: option to pass select_ugid
        # todo: option for time dimension and time subsetting
        # todo: remove format_time option - there for compatibility with the netCDF driver
        from ocgis import SpatialDimension

        ds = self.open()
        try:
            records = list(ds)
            sdim = SpatialDimension.from_records(records, crs=self.get_crs())
            # do not load the properties - they are transformed to variables in the case of the values put into fields
            sdim.properties = None
            vc = VariableCollection()
            for xx in self.rd:
                value = np.array([yy['properties'][xx['variable']] for yy in records]).reshape(1, 1, 1, 1, -1)
                var = Variable(name=xx['variable'], alias=xx['alias'], units=xx['units'], conform_units_to=xx['units'],
                               value=value)
                vc.add_variable(var, assign_new_uid=True)
            field = Field(spatial=sdim, variables=vc, name=self.rd.name)
            return field
        finally:
            self.close(ds)

    def _inspect_get_lines_(self):
        """
        :returns: A sequence of strings suitable for printing or writing to file.
        :rtype: [str, ...]
        """

        from ocgis import CoordinateReferenceSystem

        meta = self.rd.source_metadata
        try:
            ds = self.open()
            n = len(ds)
        finally:
            self.close(ds)
        lines = []
        lines.append('')
        lines.append('URI = {0}'.format(self.rd.uri))
        lines.append('')
        lines.append('Geometry Type: {0}'.format(meta['schema']['geometry']))
        lines.append('Geometry Count: {0}'.format(n))
        lines.append('CRS: {0}'.format(CoordinateReferenceSystem(value=meta['crs']).value))
        lines.append('Properties:')
        for k, v in meta['schema']['properties'].iteritems():
            lines.append(' {0} {1}'.format(v, k))
        lines.append('')

        return lines
