import tempfile
import re


def get_temp_path(suffix=''):
    """Return absolute path to a temporary file."""
    f = tempfile.NamedTemporaryFile(suffix=suffix)
    f.close()
    return f.name

def parse_polygon_wkt(txt):
    """Parse URL polygon text into WKT"""
    
    ## POLYGON ((30 10, 10 20, 20 40, 40 40, 30 10))
    ## POLYGON((30+10,10+20,20+40,40+40))
    
    def _fc(c):
        c = c.replace('+',' ')
        c = c.replace('_','.')
        return(c)
    
    txt = txt.lower()
    
    coord = '.*[\+ ]{1}.*'
    exp = 'polygon\(\((?P<c1>{0}),(?P<c2>{0}),(?P<c3>{0}),(?P<c4>{0})\)\)'.format(coord)
    m = re.match(exp,txt)
    kwds = {'c1':_fc(m.group('c1')),
            'c2':_fc(m.group('c2')),
            'c3':_fc(m.group('c3')),
            'c4':_fc(m.group('c4'))
            }
    wkt = 'POLYGON (({c1},{c2},{c3},{c4},{c1}))'.format(**kwds)
    return(wkt)