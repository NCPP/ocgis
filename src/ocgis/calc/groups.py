import abc

import six

from ocgis.base import AbstractOcgisObject
from ocgis.util.helpers import itersubclasses


@six.add_metaclass(abc.ABCMeta)
class OcgFunctionGroup(AbstractOcgisObject):
    @abc.abstractproperty
    def name(self):
        str

    def __init__(self):
        from .base import AbstractFunction

        self.Children = []
        for sc in itersubclasses(AbstractFunction):
            if sc.Group == self.__class__:
                self.Children.append(sc)

    def format(self):
        children = [Child().format() for Child in self.Children]
        ret = dict(text=self.name, expanded=True, children=children)
        return ret


class MathematicalOperations(OcgFunctionGroup):
    name = 'Mathematical Operations'


class BasicStatistics(OcgFunctionGroup):
    name = 'Basic Statistics'


class Thresholds(OcgFunctionGroup):
    name = 'Thresholds'


class MultivariateStatistics(OcgFunctionGroup):
    name = 'Multivariate Statistics'


class Percentiles(OcgFunctionGroup):
    name = 'Percentiles'
