import os
import fiona
from ocgis.util.shp_cabinet import ShpCabinet
from ConfigParser import SafeConfigParser, NoOptionError
from sqlalchemy.orm.exc import NoResultFound
from collections import OrderedDict
import json
import labels
from ocgis.util.shp_scanner import db
import webbrowser
from sqlalchemy.exc import IntegrityError


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
    
    for row in KEYS[key][1](key):
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

#def build_huc_table(key,huc_attr):
#    sc = ShpCabinet()
#    ## get the HUC2 codes for reference
#    huc2_path = sc.get_shp_path('WBDHU2_June2013')
#    with fiona.open(huc2_path,'r') as source:
#        huc2 = {feature['properties']['HUC2']:feature['properties']['Name'] for feature in source}
#    
#    ## bring in the HUC8 features determining the category based on the the HUC2
#    ## codes.
#    session = Session()
#    huc_path = sc.get_shp_path(key)
#    with fiona.open(huc_path,'r') as source:
#        for feature in source:
#            code = feature['properties'][huc_attr]
#            name = feature['properties']['Name']
#            ugid = feature['properties']['UGID']
#            huc2_region = huc2[code[0:2]]
#            db_category = get_or_create(session,Category,value='{0} - {1}'.format(huc_attr,huc2_region),key=key)
#            db_geometry = Geometry(value=name,ugid=ugid,category=db_category,subcategory=None)
#            session.add(db_geometry)
#    session.commit()
#    session.close()

#def abuild_database():
#    try:
#        os.remove(db_path)
#    except:
#        pass
#    
#    metadata.create_all()
#    session = Session()
#    
#    sc = ShpCabinet()
#    
#    keys = [
##            'qed_tbw_basins',
##            'qed_tbw_watersheds',
#            'state_boundaries',
#            'us_counties',
#            'WBDHU8_June2013',
##            'qed_city_centroids',
#            'eco_level_III_us',
#            ]
#    
#    for key in keys:
#        print(key)
#        
#        ## special processing required for HUC tables
#        if key in ['WBDHU8_June2013']:
#            build_huc_table(key,'HUC8')
#            continue
#        
#        shp_path = sc.get_shp_path(key)
#        cfg_path = sc.get_cfg_path(key)
#        
#        config = SafeConfigParser()
#        config.read(cfg_path)
#        try:
#            ugid = config.get('mapping','ugid')
#        except:
#            ugid = 'UGID'
#        category = config.get('mapping','category')
#        try:
#            subcategory = config.get('mapping','subcategory')
#        except NoOptionError:
#            subcategory = None
#        name = config.get('mapping','name')
#        
#        db_category = Category(value=category,key=key)
#        session.add(db_category)
#        session.commit()
#        
#        with fiona.open(shp_path, 'r') as source:
#            n = len(source)
#            for ctr,feature in enumerate(source):
#                if ctr % 1000 == 0:
#                    print('{0} of {1}'.format(ctr,n))
#                
#                if subcategory is None:
#                    db_subcategory = None
#                else:
#                    value_subcategory = get_variant(subcategory,feature)
#                    db_subcategory = get_or_create(session,Subcategory,value=value_subcategory,category=db_category)
#                
#                value_name = get_variant(name,feature)
#                value_name_ugid = get_variant(ugid,feature)
#                if len(value_name) > 25:
#                    value_name = value_name[0:25]+'...'
#                db_name = Geometry(value=value_name,ugid=value_name_ugid,category=db_category,
#                               subcategory=db_subcategory)
#                session.add(db_name)
#            session.commit()
#    
#    session.close()
    

if __name__ == '__main__':
    main()

### first build the database
#build_database()
### fill the json dictionary
#to_dump = OrderedDict()
#session = Session()
#for row in session.query(Geometry):
#    try:
#        ref = to_dump[row.label_formatted]['geometries']
#    except KeyError:
#        to_dump[row.label_formatted] = {'key':row.category.key,'geometries':{}}
#        ref = to_dump[row.label_formatted]['geometries']
#    ref.update({row.value_formatted:row.ugid})
#    
#fmtd = json.dumps(to_dump)
#with open(json_path,'w') as f:
#    f.write(fmtd)
#
#category = 'US State Boundaries'
#geometry = ['Delaware']
#
#with open(json_path,'r') as f:
#    mapping = json.load(f)
#    
#geom = mapping[category]['key']
#select_ugid = [mapping[category]['geometries'][g] for g in geometry]

#webbrowser.open(json_path)
