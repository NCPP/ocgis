from sqlalchemy import create_engine
from sqlalchemy.schema import MetaData, Column, UniqueConstraint, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.types import Integer, String
from sqlalchemy.orm import relationship
import os
import fiona
from ocgis.util.shp_cabinet import ShpCabinet
from ConfigParser import SafeConfigParser, NoOptionError
from sqlalchemy.orm.exc import NoResultFound
from collections import OrderedDict
import json
from tempfile import mkstemp


#db_path = '/tmp/foo.sqlite'
fd,db_path = mkstemp(suffix='.sqlite')
json_path = '/tmp/ocgis_geometries.json'
connstr = 'sqlite:///{0}'.format(db_path)
engine = create_engine(connstr)
metadata = MetaData(bind=engine)
Base = declarative_base(metadata=metadata)
Session = sessionmaker(bind=engine)


class AbstractBase(object):
    value = Column(String,nullable=False)


class Category(AbstractBase,Base):
    __tablename__ = 'category'
    __table_args__ = (UniqueConstraint('value'),)
    cid = Column(Integer,primary_key=True)
    key = Column(String)
    
    
class Subcategory(AbstractBase,Base):
    __tablename__ = 'subcategory'
    __table_args__ = (UniqueConstraint('value'),)
    sid = Column(Integer,primary_key=True)
    cid = Column(Integer,ForeignKey(Category.cid))
    
    category = relationship(Category,backref="subcategory")
    
    
class Geometry(AbstractBase,Base):
    __tablename__ = 'geometry'
    __table_args__ = (UniqueConstraint('value','ugid'),)
    cid = Column(Integer,ForeignKey(Category.cid),nullable=False)
    nid = Column(Integer,primary_key=True)
    sid = Column(Integer,ForeignKey(Subcategory.sid),nullable=True)
    ugid = Column(Integer,nullable=False)
    
    category = relationship(Category,backref='geometry')
    subcategory = relationship(Subcategory,backref='geometry')
    
    @property
    def label(self):
        if self.subcategory is None:
            label = self.category.value
        else:
            label = self.subcategory.value
        return(label)
    
    @property
    def label_formatted(self):
        if self.category.key == 'us_counties':
            ret = '{0} Counties'.format(self.label)
        else:
            ret = self.label
        return(ret)

 
def get_variant(key,feature):
    ref = feature['properties']
    variants = [str,str.lower,str.upper,str.title]
    for ii,variant in enumerate(variants,start=1):
        try:
            ret = ref[variant(key)]
            break
        except KeyError:
            if ii == len(variants):
                raise
            else:
                continue
    return(ret)

def get_or_create(session,Model,**kwargs):
    try:
        obj = session.query(Model).filter_by(**kwargs).one()
    except NoResultFound:
        obj = Model(**kwargs)
        session.add(obj)
        session.commit()
    return(obj)

def build_huc_table(key,huc_attr):
    sc = ShpCabinet()
    ## get the HUC2 codes for reference
    huc2_path = sc.get_shp_path('WBDHU2_June2013')
    with fiona.open(huc2_path,'r') as source:
        huc2 = {feature['properties']['HUC2']:feature['properties']['Name'] for feature in source}
    
    ## bring in the HUC8 features determining the category based on the the HUC2
    ## codes.
    session = Session()
    huc_path = sc.get_shp_path(key)
    with fiona.open(huc_path,'r') as source:
        for feature in source:
            code = feature['properties'][huc_attr]
            name = feature['properties']['Name']
            ugid = feature['properties']['UGID']
            huc2_region = huc2[code[0:2]]
            db_category = get_or_create(session,Category,value='{0} - {1}'.format(huc_attr,huc2_region),key=key)
            db_geometry = Geometry(value=name,ugid=ugid,category=db_category,subcategory=None)
            session.add(db_geometry)
    session.commit()
    session.close()

def build_database():
    try:
        os.remove(db_path)
    except:
        pass
    
    metadata.create_all()
    session = Session()
    
    sc = ShpCabinet()
    
    keys = [
            'state_boundaries',
            'us_counties',
            'WBDHU8_June2013',
            'qed_city_centroids'
            ]
    
    for key in keys:
        print(key)
        
        ## special processing required for HUC tables
        if key in ['WBDHU8_June2013']:
            build_huc_table(key,'HUC8')
            continue
        
        shp_path = sc.get_shp_path(key)
        cfg_path = sc.get_cfg_path(key)
        
        config = SafeConfigParser()
        config.read(cfg_path)
        ugid = config.get('mapping','ugid')
        category = config.get('mapping','category')
        try:
            subcategory = config.get('mapping','subcategory')
        except NoOptionError:
            subcategory = None
        name = config.get('mapping','name')
        
        db_category = Category(value=category,key=key)
        session.add(db_category)
        session.commit()
        
        with fiona.open(shp_path, 'r') as source:
            n = len(source)
            for ctr,feature in enumerate(source):
                if ctr % 1000 == 0:
                    print('{0} of {1}'.format(ctr,n))
                
                if subcategory is None:
                    db_subcategory = None
                else:
                    value_subcategory = get_variant(subcategory,feature)
                    db_subcategory = get_or_create(session,Subcategory,value=value_subcategory,category=db_category)
                
                value_name = get_variant(name,feature)
                value_name_ugid = get_variant(ugid,feature)
                db_name = Geometry(value=value_name,ugid=value_name_ugid,category=db_category,
                               subcategory=db_subcategory)
                session.add(db_name)
            session.commit()
    
    session.close()

## first build the database
build_database()
## fill the json dictionary
to_dump = OrderedDict()
session = Session()
for row in session.query(Geometry):
    try:
        ref = to_dump[row.label_formatted]['geometries']
    except KeyError:
        to_dump[row.label_formatted] = {'key':row.category.key,'geometries':{}}
        ref = to_dump[row.label_formatted]['geometries']
    ref.update({row.value:row.ugid})
    
fmtd = json.dumps(to_dump)
with open(json_path,'w') as f:
    f.write(fmtd)
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
