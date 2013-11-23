import abc


class NcDimension(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractproperty
    def axis(self): str