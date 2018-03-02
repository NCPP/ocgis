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
    
    Spatial collections are the ``'ocgis'`` output format. It is possible to not provide a subset geometry when a
    spatial collection is created. In this case, the container geometry is ``None``, but the data is still nested.
    
    .. note:: Accepts all parameters to :class:`~ocgis.VariableCollection`.
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
        """
        Return an archetype field from the spatial collection. This is first field encountered during field iteration.
        
        :rtype: :class:`~ocgis.Field`
        """

        for child in list(self.children.values()):
            for grandchild in list(child.children.values()):
                return grandchild

    @property
    def crs(self):
        """
        Return the spatial collection's coordinate system. This is the coordinate system of the first encountered field
        in iteration.
        
        :rtype: :class:`~ocgis.variable.crs.AbstractCRS`
        """
        for child in list(self.children.values()):
            return child.crs

    @property
    def geoms(self):
        """
        Reformat container geometries into a dictionary. Keys are the child geometries unique identifiers. The values 
        are Shapely geometries.
        
        :rtype: :class:`~collections.OrderedDict` 
        """
        ret = OrderedDict()
        for k, v in list(self.children.items()):
            if v.geom is not None:
                ret[k] = v.geom.get_value()[0]
        return ret

    @property
    def has_container_geometries(self):
        """
        Return ``True`` if there are container geometries.
        
        :rtype: bool 
        """
        ret = False
        if len(self.children) > 0:
            if list(self.children.keys())[0] is not None:
                ret = True
        return ret

    @property
    def properties(self):
        """
        Reformat container geometry values into a properties dictionary.
        
        :rtype: :class:`~collections.OrderedDict` 
        """
        ret = OrderedDict()
        for k, v in list(self.children.items()):
            ret[k] = OrderedDict()
            if v.has_data_variables:
                for variable in v.iter_data_variables():
                    ret[k][variable.name] = variable.get_value()[0]
        return ret

    def add_field(self, field, container, force=False):
        """
        Add a field to the spatial collection.
        
        :param field: The field to add.
        :type field: :class:`~ocgis.Field`
        :param container: The container geometry. A ``None`` value is allowed.
        :type container: :class:`~ocgis.Field` | ``None``
        :param bool force: If ``True``, clobber any field names in the spatial collection. 
        :return: 
        """
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
        """
        Get a field or variable from the spatial collection.
        
        :param str field_name: The field name to get from the collection.
        :param str variable_name: The variable name to get from the collection. If ``None``, a field will be returned.
        :param container_ugid: The container unique identifier. If ``None``, the first container will be used.
        :rtype: :class:`~ocgis.Field` | :class:`~ocgis.Variable`
        """

        children = self.children
        if container_ugid is None:
            for ret in list(children.values()):
                break
        else:
            try:
                ret = children[container_ugid]
            except TypeError:
                # Try to extract the unique identifier from the field.
                try:
                    container_ugid = container_ugid.geom.ugid.get_value()[0]
                except AttributeError:
                    # It may be a NoneType geometry object.
                    container_ugid = None
                ret = children[container_ugid]
        if field_name is None:
            for ret in list(ret.children.values()):
                break
        else:
            ret = ret.children[field_name]
        if variable_name is not None:
            ret = ret[variable_name]
        return ret

    def iter_fields(self, yield_container=False):
        """Iterate field objects in the collection."""

        for ugid, container in list(self.children.items()):
            for field in list(container.children.values()):
                if yield_container:
                    yld = field, container
                else:
                    yld = field
                yield yld

    def iter_melted(self, tag=None):
        """Iterate a melted dictionary containing all spatial collection elements."""

        for ugid, container in list(self.children.items()):
            for field_name, field in list(container.children.items()):
                if tag is not None:
                    variables_to_itr = field.get_by_tag(tag)
                else:
                    variables_to_itr = list(field.values())
                for variable in variables_to_itr:
                    yield dict(ugid=ugid, field_name=field_name, field=field, variable_name=variable.name,
                               variable=variable)
