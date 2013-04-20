from abc import ABCMeta, abstractmethod
from collections import OrderedDict


class AbstractMetadata(OrderedDict):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def _get_lines_(self): pass
    
    @abstractmethod
    def _parse_(self): pass
    
    
class NcMetadata(AbstractMetadata):
    
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
                                   'attrs':subvar,
                                   'dtype':str(value.dtype)}})
        self.update({'variables':variables})
        
        ## get dimensions
        dimensions = OrderedDict()
        for key,value in rootgrp.dimensions.iteritems():
            subdim = {key:{'len':len(value),'isunlimited':value.isunlimited()}}
            dimensions.update(subdim)
        self.update({'dimensions':dimensions})
        
    def _get_lines_(self):
        lines = ['dimensions:']
        template = '    {0} = {1} ;{2}'
        for key,value in self['dimensions'].iteritems():
            if value['isunlimited']:
                one = 'ISUNLIMITED'
                two = ' // {0} currently'.format(value['len'])
            else:
                one = value['len']
                two = ''
            lines.append(template.format(key,one,two))
        
        lines.append('')
        lines.append('variables:')
        var_template = '    {0} {1}({2}) ;'
        attr_template = '      {0}:{1} = "{2}" ;'
        for key,value in self['variables'].iteritems():
            dims = [str(d) for d in value['dimensions']]
            dims = ', '.join(dims)
            lines.append(var_template.format(value['dtype'],key,dims))
            for key2,value2 in value['attrs'].iteritems():
                lines.append(attr_template.format(key,key2,value2))
                
        lines.append('')
        lines.append('// global attributes:')
        template = '    :{0} = {1} ;'
        for key,value in self['dataset'].iteritems():
            lines.append(template.format(key,value))
        
        return(lines)