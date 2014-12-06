import ESMF
from ocgis.conv.base import AbstractConverter
from ocgis.exc import DefinitionValidationError
from ocgis.regrid.base import get_esmf_grid_from_sdim


class ESMPyConverter(AbstractConverter):
    #todo: doc

    def __init__(self, *args, **kwargs):
        self.with_corners = kwargs.pop('with_corners', True)
        self.value_mask = kwargs.pop('value_mask', None)
        self.esmf_field_name = kwargs.pop('esmf_field_name', None)
        super(ESMPyConverter, self).__init__(*args, **kwargs)

    def __iter__(self):
        for coll in self.colls:
            yield coll

    @classmethod
    def validate_ops(cls, ops):
        msg = None
        if len(ops.dataset) > 1:
            msg = 'Only one requested dataset may be written for "esmpy" output.'
            target = 'dataset'
        elif ops.spatial_operation == 'clip':
            msg = 'Clip operations not allowed for "esmpy" output.'
            target = 'spatial_operation'
        elif ops.select_ugid is not None and not ops.agg_selection and len(ops.select_ugid) > 1:
            msg = 'Only one selection geometry allowed for "esmpy" output.'
            target = 'select_ugid'
        elif ops.aggregate:
            msg = 'No spatial aggregation for "esmpy" output.'
            target = 'aggregate'

        if msg is not None:
            raise DefinitionValidationError(target, msg)

    def write(self):
        #todo: doc

        for coll in self.colls:
            """:type coll: :class:`ocgis.api.collection.SpatialCollection`"""
            for row in coll.get_iter_melted():
                field = row['field']
                variable = row['variable']
                egrid = get_esmf_grid_from_sdim(field.spatial, with_corners=self.with_corners,
                                                value_mask=self.value_mask)
                esmf_field_name = self.esmf_field_name or variable.alias
                efield = ESMF.Field(egrid, esmf_field_name, ndbounds=field.shape[0:-2], mask_values=[0])
                efield[:] = variable.value

                return efield