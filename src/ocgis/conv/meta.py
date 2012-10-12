class MetaConverter(object):
    
    def __init__(self,desc):
        self.desc = desc
        
    def write(self):
        from ocgis.api.interp.definition import DEF_ARGS
        
        for Da in DEF_ARGS:
            obj = Da(self.desc.get(Da.name))
            print obj.name
            print obj.message()
            print ''