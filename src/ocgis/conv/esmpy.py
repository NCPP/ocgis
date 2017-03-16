import ESMF

from ocgis.constants import TagNames
from ocgis.conv.base import AbstractCollectionConverter
from ocgis.exc import DefinitionValidationError


class ESMPyConverter(AbstractCollectionConverter):
    """
    Convert a spatial collection to an ESMF field object.

    .. note:: Accepts all parameters to :class:`~ocgis.conv.base.AbstractCollectionConverter`.

    :param with_corners: (``=True``) See :func:`~ocgis.regrid.base.get_esmf_grid`.
    :param value_mask: (``=None``) See :func:`~ocgis.regrid.base.get_esmf_grid`.
    :param esmf_field_name: (``=None``) Optional name for the returned ESMF field.
    :type esmf_field_name: str
    """

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
        if len(list(ops.dataset)) > 1:
            msg = 'Only one requested dataset may be written for "esmpy" output.'
            target = 'dataset'
        elif ops.spatial_operation == 'clip':
            msg = 'Clip operations not allowed for "esmpy" output.'
            target = 'spatial_operation'
        elif ops.geom_select_uid is not None and not ops.agg_selection and len(ops.geom_select_uid) > 1:
            msg = 'Only one selection geometry allowed for "esmpy" output.'
            target = 'select_ugid'
        elif ops.aggregate:
            msg = 'No spatial aggregation for "esmpy" output.'
            target = 'aggregate'

        if msg is not None:
            raise DefinitionValidationError(target, msg)

    def write(self):
        for coll in self.colls:
            from ocgis.regrid.base import get_esmf_grid

            for row in coll.iter_melted(tag=TagNames.DATA_VARIABLES):
                field = row['field']
                variable = row['variable']
                egrid = get_esmf_grid(field, with_corners=self.with_corners, value_mask=self.value_mask)

                esmf_field_name = self.esmf_field_name or variable.name

                # TODO: The undistributed dimensions should be handled more cleanly. This approach assumes well-shaped
                # data with the last two holding the spatial coordinates.
                if variable.ndim > 2:
                    ndbounds = variable.shape[0:-2]
                else:
                    ndbounds = None

                efield = ESMF.Field(egrid, name=esmf_field_name, ndbounds=ndbounds)
                efield.data[:] = variable.value

                return efield
