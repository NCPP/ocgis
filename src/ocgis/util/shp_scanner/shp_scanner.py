import fiona
from ocgis.util.shp_cabinet import ShpCabinet, ShpCabinetIterator
from ConfigParser import SafeConfigParser, NoOptionError
from sqlalchemy.orm.exc import NoResultFound
import json
import labels
from ocgis.util.shp_scanner import db
import webbrowser
from ocgis.api.operations import OcgOperations
from ocgis.exc import ExtentError
from ocgis.api.request.base import RequestDataset


KEYS = {
        'state_boundaries':['US State Boundaries',labels.StateBoundaries],
        'us_counties':['US Counties',labels.UsCounties],
#        'world_countries':['World Countries',labels.WorldCountries],
        'climate_divisions':['NOAA Climate Divisions',labels.ClimateDivisions],
        'eco_level_III_us':['Ecoregions (Level 3)',labels.UsLevel3Ecoregions],
        'WBDHU8_June2013':['HUC 8 Boundaries',labels.Huc8Boundaries]
        }
METADATA_ATTRS = ['download_url','metadata_url','download_date','description','history']
OUT_JSON_PATH = '/tmp/ocgis_geometries.json'
#: if None, do not filter geometries. if the value is a RequestDataset, use it
#: for filtering geometries.
try:
    FILTER_GEOMETRIES = RequestDataset(uri='/usr/local/climate_data/maurer/2010-concatenated/Maurer02new_OBS_tasmax_daily.1971-2000.nc',
                                   variable='tasmax')
except ValueError:
    pass


def get_does_intersect(request_dataset,geom):
    '''
    :param :class:`ocgis.RequestDataset` request_dataset:
    :param shapely.geometry geom:
    '''
    ops = OcgOperations(dataset=request_dataset,geom=geom,snippet=True)
    try:
        ops.execute()
        ret = True
    except ExtentError:
        ret = False
    return(ret)

def get_select_ugids(request_dataset,shp_path):
    '''
    :param :class:`ocgis.RequestDataset` request_dataset:
    :param str shp_path: Path to the shapefile containing geometries to test.
    '''
    ugids = []
    for row in ShpCabinetIterator(path=shp_path):
        if get_does_intersect(request_dataset,row['geom']):
            ugids.append(row['properties']['UGID'])
    return(ugids)

def get_or_create(session,Model,**kwargs):
    try:
        obj = session.query(Model).filter_by(**kwargs).one()
    except NoResultFound:
        commit = kwargs.pop('commit',True)
        obj = Model(**kwargs)
        session.add(obj)
        if commit:
            session.commit()
    return(obj)

def build_database(keys=None,filter_request_dataset=None,verbose=False):
    keys = keys or KEYS
    db.metadata.create_all()
    session = db.Session()
    
    try:
        build_key(keys,session,filter_request_dataset=filter_request_dataset,verbose=verbose)
    finally:
        session.close()
    
    if verbose: print(db.db_path)
        
def build_key(keys,session,filter_request_dataset=None,verbose=False):
    filter_request_dataset = filter_request_dataset or FILTER_GEOMETRIES
    for key in keys:
        if verbose: print('building {0}'.format(key))
        ## build the category
        kwds = get_metadata(key)
        kwds['label'] = keys[key][0]
        kwds['key'] = key
        category = db.Category(**kwds)
    
        for row in keys[key][1](key,filter_geometries=filter_request_dataset):
            try:
                subcategory = get_or_create(session,db.Subcategory,label=row['subcategory_label'],category=category)
            except KeyError:
                subcategory = None
            geometry = db.Geometry(ugid=row['properties']['UGID'],envelope=row['envelope'],
                        label=row['geometry_label'],category=category,subcategory=subcategory)
            session.add(geometry)
        session.commit()
    
def write_json(path):
    session = db.Session()
    try:
        ret = {}
        for category in session.query(db.Category):
            ret[category.label] = {}
            ret[category.label]['geometries'] = {}
            r_geometries = ret[category.label]['geometries']
            ret[category.label]['key'] = category.key
            if len(category.subcategory) == 0:
                r_geometries[None] = {}
                for geometry in category.geometry:
                    r_geometries[None].update({geometry.label:geometry.ugid})
            else:
                for subcategory in category.subcategory:
                    r_geometries[subcategory.label] = {}
                    for geometry in subcategory.geometry:
                        r_geometries[subcategory.label].update({geometry.label:geometry.ugid})
        fmtd = json.dumps(ret)
        with open(path,'w') as f:
            f.write(fmtd)
#        webbrowser.open(path)
    finally:
        session.close()
    
def get_metadata(key):
    try:
        cfg_path = ShpCabinet().get_cfg_path(key)
        config = SafeConfigParser()
        config.read(cfg_path)
        if config.has_section('metadata'):
            ret = {}
            for k in METADATA_ATTRS:
                try:
                    v = config.get('metadata',k)
                except NoOptionError:
                    v = None
                ret.update({k:v})
        else:
            ret = dict.fromkeys(METADATA_ATTRS)
    except ValueError:
        ret = dict.fromkeys(METADATA_ATTRS)
    return(ret)
        
def main():
    build_database(verbose=True)
    write_json(OUT_JSON_PATH)
    

if __name__ == '__main__':
#    import doctest
#    doctest.testmod()
    main()
