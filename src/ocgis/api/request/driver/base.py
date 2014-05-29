import abc


class AbstractDriver(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, rd):
        self.rd = rd

    def __eq__(self, other):
        return self.key == other.key

    @abc.abstractproperty
    def key(self):
        str

    @abc.abstractmethod
    def close(self, obj):
        pass

    @abc.abstractmethod
    def get_crs(self):
        return object

    @abc.abstractmethod
    def get_dimensioned_variables(self):
        return tuple(str, )

    @abc.abstractmethod
    def get_field(self, **kwargs):
        return object

    @abc.abstractmethod
    def get_source_metadata(self):
        return dict

    @abc.abstractmethod
    def open(self):
        return object