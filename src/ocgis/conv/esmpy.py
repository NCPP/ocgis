from ocgis.constants import TagName
from ocgis.conv.base import AbstractCollectionConverter
from ocgis.exc import DefinitionValidationError


class ESMPyConverter(AbstractCollectionConverter):
    """
    Convert a spatial collection to an ESMF field object.

    .. note:: Accepts all parameters to :class:`~ocgis.conv.base.AbstractCollectionConverter`.

    :param regrid_method: (``='auto'``) See :func:`~ocgis.regrid.base.create_esmf_grid`.
    :param value_mask: (``=None``) See :func:`~ocgis.regrid.base.create_esmf_grid`.
    :param esmf_field_name: (``=None``) Optional name for the returned ESMF field.
    :type esmf_field_name: str
    """

    def __init__(self, *args, **kwargs):
        self.regrid_method = kwargs.pop('regrid_method', True)
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
        from ocgis.regrid.base import get_esmf_field_from_ocgis_field

        for coll in self.colls:
            for row in coll.iter_melted(tag=TagName.DATA_VARIABLES):
                field = row['field']
                efield = get_esmf_field_from_ocgis_field(field,
                                                         esmf_field_name=self.esmf_field_name,
                                                         regrid_method=self.regrid_method,
                                                         value_mask=self.value_mask)

                return efield
