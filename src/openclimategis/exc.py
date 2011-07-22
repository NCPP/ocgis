class DatasetExists(Exception):
    
    def __init__(self,uri):
        self.uri = uri
        
    def __str__(self):
        msg = 'Dataset with URI={0} already exists.'.format(self.uri)
        return(msg)