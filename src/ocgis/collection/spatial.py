from collections import OrderedDict
from pprint import pformat

from ocgis import VariableCollection
from ocgis.collection.field import Field


class SpatialCollection(VariableCollection):
    """
    Spatial collections use a group hierarchy to nest fields in their representative geometries. First-level groups on
    a spatial collection are the container fields with a single geometry. The second-level groups or grandchildren
    (nested under the container geometries) are field associated with the parent container. These associations are
    typically defined using a spatial subset.
    
    Spatial collections are the `ocgis` output object type. It is possible to not provide a subset geometry when a
    spatial collection is created. In this case, the container geometry is ``None``, but the data is still nested.
    
    .. note:: Accepts all parameters to :class:`ocgis.VariableCollection`.
    """

    def __getitem__(self, item_or_slc):
        if isinstance(item_or_slc, int) or item_or_slc is None:
            ret = self.children[item_or_slc]
        else:
            ret = super(SpatialCollection, self).__getitem__(item_or_slc)
        return ret

    def __repr__(self):
        msg = '<{klass}(Containers :: {ids})>'.format(klass=self.__class__.__name__,
                                                      ids=pformat(list(self.children.keys())))
        return msg

    @property
    def archetype_field(self):
        for child in list(self.children.values()):
            for grandchild in list(child.children.values()):
                return grandchild

    @property
    def crs(self):
        for child in list(self.children.values()):
            return child.crs

    @property
    def geoms(self):
        ret = OrderedDict()
        for k, v in list(self.children.items()):
            if v.geom is not None:
                ret[k] = v.geom.get_value()[0]
        return ret

    @property
    def has_container_geometries(self):
        ret = False
        if len(self.children) > 0:
            if list(self.children.keys())[0] is not None:
                ret = True
        return ret

    @property
    def properties(self):
        ret = OrderedDict()
        for k, v in list(self.children.items()):
            ret[k] = OrderedDict()
            for variable in v.iter_data_variables():
                ret[k][variable.name] = variable.get_value()[0]
        return ret

    def add_field(self, field, container, force=False):
        # Assume a NoneType container if there is no geometry associated with the container.
        if container is not None and container.geom is not None:
            ugid = container.geom.ugid.get_value()[0]
            if ugid not in self.children:
                self.children[ugid] = container
            else:
                # We want to use the reference to the container in the collection.
                container = self.children[ugid]
            container.add_child(field, force=force)
        else:
            if None not in self.children:
                self.children[None] = Field()
            container = self.children[None]
            container.add_child(field, force=force)

    def get_element(self, field_name=None, variable_name=None, container_ugid=None):
        if container_ugid is None:
            for ret in list(self.children.values()):
                break
        else:
            ret = self.children[container_ugid]
        if field_name is None:
            for ret in list(ret.children.values()):
                break
        else:
            ret = ret.children[field_name]
        if variable_name is not None:
            ret = ret[variable_name]
        return ret

    def iter_fields(self, yield_container=False):
        for ugid, container in list(self.children.items()):
            for field in list(container.children.values()):
                if yield_container:
                    yld = field, container
                else:
                    yld = field
                yield yld

    def iter_melted(self, tag=None):
        for ugid, container in list(self.children.items()):
            for field_name, field in list(container.children.items()):
                if tag is not None:
                    variables_to_itr = field.get_by_tag(tag)
                else:
                    variables_to_itr = list(field.values())
                for variable in variables_to_itr:
                    yield dict(ugid=ugid, field_name=field_name, field=field, variable_name=variable.name,
                               variable=variable)
