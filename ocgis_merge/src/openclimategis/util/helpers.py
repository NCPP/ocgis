import tempfile
import copy
import pdb
from shapely import wkt
import os


def get_temp_path(suffix='',name=None,nest=False,only_dir=False):
    """Return absolute path to a temporary file."""
    if nest:
        f = tempfile.NamedTemporaryFile()
        f.close()
        dir = os.path.join(tempfile.gettempdir(),f.name)
        os.mkdir(dir)
    else:
        dir = tempfile.gettempdir()
    if only_dir:
        ret = dir
    else:
        if name is not None:
            ret = os.path.join(dir,name)
        else:
            f = tempfile.NamedTemporaryFile(suffix=suffix,dir=dir)
            f.close()
            ret = f.name
    return(str(ret))

def parse_polygon_wkt(txt):
    """Parse URL polygon text into WKT. Text must be well-formed in that fully
    qualified WKT test must be returned by replacing '+' with ' '.
    
    >>> poly = 'polygon((30+10,10+20,20+40,40+40,30+10))'
    >>> wkt = parse_polygon_wkt(poly)
    >>> m = 'multipolygon(((30+10,10+20,20+40,40+40,30+10)),((30+10,10+20,20+40,40+40,30+10)))'
    >>> wkt = parse_polygon_wkt(m)
    """
    
    ## POLYGON ((30 10, 10 20, 20 40, 40 40, 30 10))
    ## POLYGON((30+10,10+20,20+40,40+40))
    ## MULTIPOLYGON (((30 10, 10 20, 20 40, 40 40, 30 10),(30 10, 10 20, 20 40, 40 40, 30 10)))
    
    txt = txt.upper().replace('+',' ')
    txt = txt.replace('POLYGON((','POLYGON ((')
    try:
        poly = wkt.loads(txt)
    except:
        raise(ValueError('Unable to parse WKT text.'))  
    return(poly.to_wkt())

def reverse_wkt(txt):
    """Turn Polygon WKT into URL text
    
    >>> txt = 'POLYGON ((30 10, 10 20, 20 40, 40 40, 30 10))'
    >>> reverse_wkt(txt)
    'polygon((30+10,10+20,20+40,40+40,30+10))'
    """
    
    ## POLYGON ((30 10, 10 20, 20 40, 40 40, 30 10))
    ## POLYGON((30+10,10+20,20+40,40+40))
    
    txt = txt.replace('POLYGON ((','POLYGON((')
    txt = txt.lower().replace(' ','+')
    
    return(txt)


def merge_dict(*args):
    """
    Merge two dictionaries with dict values as lists.
    
    >>> one = dict(a=[1,2,3],b=['one'])
    >>> two = dict(a=[4,5,6],b=['two'])
    >>> three = dict(a=[7,8,9],b=['three'])
    >>> merge_dict(one,two,three)
    {'a': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'b': ['one', 'two', 'three']}
    """
    
    def _gen():
        for ii in xrange(len(args)):
            try:
                yield(args[ii+1])
            except IndexError:
                raise StopIteration
    
    ## get the base dictionary
    b = copy.copy(args[0])
    ## loop through merging
    for t in _gen():
        for key,value in t.iteritems():
            b[key] += value
    return(b)

#def html_table(ld,order):
#    """
#    Generate an HTML table code from a nested dictionary. DOEST NOT generate
#    the <table> tag to allow table customization.
#    
#    >>> ld = [dict(month='January',savings='$100'),dict(month='April',savings ='$50')]
#    >>> order = (('month','Month'),('savings','Savings'))
#    >>> html_table(ld,order)
#    """
#    
#    ## contains the formated html code
#    dump = []
#    
#    tr = '<tr>{0}</tr>'
#    th = '<th>{0}</th>'
#    td = '<td>{0}</td>'
#    
#    ## set the headings
#    dump.append(tr.format(''.join([th.format(o[1]) for o in order])))
#    ## add the data
#    for l in ld:
#        dump.append(tr.format(''.join([td.format(l[o[0]]) for o in order])))
#    
#    return(''.join(dump))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    