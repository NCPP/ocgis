import fiona
from ocgis.util.shp_cabinet import ShpCabinet, ShpCabinetIterator
from ConfigParser import SafeConfigParser, NoOptionError
from sqlalchemy.orm.exc import NoResultFound
import json
import labels
from ocgis.util.shp_scanner import db
import webbrowser


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
#: if True, remove geometries overlapping the geometries returned by
#: get_filter_geometries.
FILTER_GEOMETRIES = True


def get_filter_geometries():
    '''
    >>> geoms = get_filter_geometries()
    >>> len(geoms)
    2
    '''
    state_names = ['Hawaii','Alaska']
    geoms = []
    for row in ShpCabinetIterator('state_boundaries'):
        if row['properties']['STATE_NAME'] in state_names:
            geoms.append(row['geom'])
    return(geoms)

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

def build_database():
    db.metadata.create_all()
    session = db.Session()
    
    try:
        for key in KEYS.keys():
            print('building: {0}'.format(key))
            build_key(key,session)
    finally:
        session.close()
    
    print(db.db_path)
        
def build_key(key,session):
    ## build the category
    kwds = get_metadata(key)
    kwds['label'] = KEYS[key][0]
    kwds['key'] = key
    category = db.Category(**kwds)
    
    if FILTER_GEOMETRIES:
        filter_geometries = get_filter_geometries()
    else:
        filter_geometries = None
    
    for row in KEYS[key][1](key,filter_geometries=filter_geometries):
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
        webbrowser.open(path)
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
    build_database()
    write_json(OUT_JSON_PATH)
    

if __name__ == '__main__':
#    import doctest
#    doctest.testmod()
    main()
