import abc

import six

from ocgis.base import AbstractInterfaceObject


@six.add_metaclass(abc.ABCMeta)
class AbstractRequestObject(AbstractInterfaceObject):
    pass
