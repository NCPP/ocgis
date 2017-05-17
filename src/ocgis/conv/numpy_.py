from ocgis.conv.base import AbstractCollectionConverter


class NumpyConverter(AbstractCollectionConverter):
    _create_directory = False
    _allow_empty = True

    def write(self):
        for ctr_coll, coll in enumerate(self):

            if ctr_coll == 0:
                ret = coll.copy()
            else:
                # Make sure the counter is set for parallel run. Some ranks may have no attached fields.
                ctr = 0
                for ctr, (field, container) in enumerate(coll.iter_fields(yield_container=True)):
                    ret.add_field(field, container, force=True)
                # Only one field should be added per iteration.
                assert ctr == 0
        return ret
