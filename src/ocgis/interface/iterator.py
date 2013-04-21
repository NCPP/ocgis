from abc import ABCMeta, abstractmethod


class AbstractMeltedIterator(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def iter_rows(self): pass