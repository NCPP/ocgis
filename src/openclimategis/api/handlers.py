from piston.handler import BaseHandler
from climatedata.models import ClimateModel
from emitters import *
from piston.utils import rc


class OpenClimateHandler(BaseHandler):
    pass
    
    
class HelloWorldHandler(OpenClimateHandler):
    allowed_methods = ('GET',)
    model = ClimateModel
    
    def read(self,request,model_name=None):
        if model_name != None:
            query = self.model.objects.filter(name=str(model_name))
            if len(query) == 0:
                return rc.NOT_FOUND
        else:
            query = self.model.objects.all()
        return query