import os


class PolyElement(object):
    _possible = []
    
    def __init__(self,dataset):
        self.dataset = dataset