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
    ## poly_30-10_10-20_20-40_40-40
    
    def _fc(c):
        return(c.replace('-',' '))
    
    txt = txt.lower()
    
    coord = '.*-.*'
    exp = 'poly_(?P<c1>{0})_(?P<c2>{0})_(?P<c3>{0})_(?P<c4>{0})'.format(coord)
    m = re.match(exp,txt)
    kwds = {'c1':_fc(m.group('c1')),
            'c2':_fc(m.group('c2')),
            'c3':_fc(m.group('c3')),
            'c4':_fc(m.group('c4'))
            }
    wkt = 'POLYGON (({c1},{c2},{c3},{c4},{c1}))'.format(**kwds)
    return(wkt)