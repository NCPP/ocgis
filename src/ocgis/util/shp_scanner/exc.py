class NoSubcategoryError(Exception):
    pass


class MalformedLabel(Exception):
    
    def __init__(self,label,key,reason='comma'):
        self.label = label
        self.reason = reason
        self.key = key
        
    def __str__(self):
        msg = 'A {0} is present in the label "{1}" contained in shapefile dataset with key "{2}"'.format(self.reason,self.label,self.key)
        return(msg)