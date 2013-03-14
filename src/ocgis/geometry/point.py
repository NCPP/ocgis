from shapely.geometry.multipoint import MultiPoint



class MultiPoint(MultiPoint):
    
    def __init__(self,*args,**kwds):
        self.sr = kwds.get('sr')