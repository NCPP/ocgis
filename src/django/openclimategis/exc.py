class WsOcgException(Exception):
    pass


class QueryParmError(WsOcgException):
    
    def __init__(self,key):
        self.key = key
        
    def __str__(self):
        return(self._get_msg_().format(self.key))
    
    def _get_msg_(self):
        return('query parameter with key "{0}" requested but no match found.')
    
    
class ScalarError(QueryParmError):
    
    def _get_msg_(self):
        msg = 'scalar value requested for "{0}", but more than one element present in value list.'
        return(msg)
    
    
class NotNullableError(QueryParmError):
    
    def _get_msg_(self):
        msg = 'query parameter "{0}" is not nullable.'
        return(msg)