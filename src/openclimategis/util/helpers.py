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
    
    txt = txt.lower()
    txt = txt.replace('+',' ')
    
    coords = re.match('.*\(\((.*)\)\)',txt).group(1)
    coords = coords.split(',')
    ## replicate last coordinate if it is not passed
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    coords = ', '.join(coords)
    
    wkt = 'POLYGON (({0}))'.format(str(coords))
        
    return(wkt)