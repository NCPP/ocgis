import datetime


class SlugError(Exception):
    
    def __init__(self,slug):
        self.slug = slug
        
    def __str__(self):
        msg = 'Cannot parse URL argument "{0}"'.format(self.slug.url_arg)
        return(msg)


class OcgSlug(object):
    
    def __init__(self,code,url_arg=None,possible=None,str_lower=True):
        self.code = code
        
        if url_arg is None and possible is not None:
            self.url_arg = possible.get(self.code)
        else:
            self.url_arg = url_arg
        
        if str_lower is True and self.url_arg is not None:
            self.url_arg = str(self.url_arg).lower()
        
        self.value = self.get()
        
    def __repr__(self):
        msg = '{0}={1}'.format(self.code,self.value)
        return(msg)
    
    def get(self):
        if self.url_arg is None:
            ret = None
        else:
            ret = self._get_()
        return(ret)
    
    def _get_(self):
        return(self.url_arg)
    
    def _exception_(self):
        raise(SlugError(self))
    
    
class BooleanSlug(OcgSlug):
    
    def _get_(self):
        return(self._parse_(self.url_arg))
    
    def _parse_(self,str_rep):
        if str_rep in ('true','t','1'):
            ret = True
        elif str_rep in ('false','f','0'):
            ret = False
        else:
            self._exception_()
        return(ret)


class DjangoQuerySlug(OcgSlug):
    """
    DjangoQuerySlug(model,code,filter_kwds={},code_field='code',**kwds)
    """
    
    def __init__(self,*args,**kwds):
        args = list(args)
        self.model = args.pop(0)
        self._extract_kwd_(kwds,'filter_kwds',{})
        self._extract_kwd_(kwds,'one',False)
        self._extract_kwd_(kwds,'code_field','code')
        super(DjangoQuerySlug,self).__init__(*args,**kwds)
        
    def _get_(self):
#        filter.update(self.filter_kwds)
        qs = self.model.objects.filter(**self.filter_kwds)
        if self.one:
            if len(qs) > 1:
                raise ValueError('One record requested, {0} records returned.'.format(len(qs)))
            elif len(qs) == 0:
                raise ValueError('One record requested. None returned.')
            ret = qs[0]
        else:
            ret = qs
        return(ret)
    
    def _extract_kwd_(self,kwds,name,default=None):
        if name in kwds:
            setattr(self,name,kwds.pop(name))
        else:
            setattr(self,name,default)
            
class IExactQuerySlug(DjangoQuerySlug):
    """
    IExactQuerySlug(model,code,filter_kwds={},code_field='code',**kwds)
    """
    
#    def __init__(self,*args,**kwds):
#        args = list(args)
#        self.iexact_target = args.pop(0)
#        super(IExactQuerySlug,self).__init__(*args,**kwds)
    
    def _get_(self):
        self.filter_kwds.update({self.code_field+'__iexact':self.url_arg})
        ret = super(IExactQuerySlug,self)._get_()
        return(ret)
        

class TemporalSlug(OcgSlug):
    """
    >>> value = '2007-1-12+2007-3-4'
    >>> t = TemporalSlug('temporal',url_arg=None,possible=dict(temporal=value))
    >>> t.value
    [datetime.datetime(2007, 1, 12, 0, 0), datetime.datetime(2007, 3, 4, 0, 0)]
    >>> TemporalSlug('temporal',url_arg=value).value
    [datetime.datetime(2007, 1, 12, 0, 0), datetime.datetime(2007, 3, 4, 0, 0)]
    >>> t = TemporalSlug('temporal',possible=dict(foo=5))
    >>> assert(t.value is None)
    >>> print(t)
    temporal=None
    """
    
    def _get_(self):
        if '+' in self.url_arg:
            start,end = self.url_arg.split('+')
        else:
            start,end = (self.url_arg,self.url_arg)
        return(self._format_date_(start,end))
    
    @staticmethod
    def _format_date_(start,end):
        return([datetime.datetime.strptime(d,'%Y-%m-%d') for d in [start,end]])
    
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()