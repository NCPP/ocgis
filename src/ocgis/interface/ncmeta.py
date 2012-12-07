from ordereddict import OrderedDict


class NcMetadata(OrderedDict):
    
    def __init__(self,rootgrp):
        super(NcMetadata,self).__init__()
        self._parse_(rootgrp)
        
    def _parse_(self,rootgrp):
        ## get global metadata
        dataset = OrderedDict()
        for attr in rootgrp.ncattrs():
            dataset.update({attr:getattr(rootgrp,attr)})
        self.update({'dataset':dataset})
        
        ## get variables
        variables = OrderedDict()
        for key,value in rootgrp.variables.iteritems():
            subvar = OrderedDict()
            for attr in value.ncattrs():
                if attr.startswith('_'): continue
                subvar.update({attr:getattr(value,attr)})
            variables.update({key:{'dimensions':value.dimensions,
                                   'attrs':subvar}})
        self.update({'variables':variables})