from sqlalchemy import create_engine
from sqlalchemy.schema import MetaData, Column, UniqueConstraint, ForeignKey
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.types import Integer, Float, String
import copy
from sqlalchemy.orm import relationship
import os
import fiona
from ocgis.util.shp_cabinet import ShpCabinet
import ConfigParser
from ConfigParser import SafeConfigParser, NoOptionError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound
from collections import OrderedDict
import json
import webbrowser


db_path = '/tmp/foo.sqlite'
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

#def dct_get_or_create(key,dct,fill):
#    try:
#        dct[key] = fill
#    except KeyError:
#        dct[key] = fill
#        ret = dct[key]
#    return(ret)


def build():
    try:
        os.remove(db_path)
    except:
        pass
    
    metadata.create_all()
    session = Session()
    
    sc = ShpCabinet()
    
    keys = ['state_boundaries','us_counties']
    
    for key in keys:
        print(key)
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

#build()

json_path = '/tmp/ocgis_geometries.json'
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

category = 'US State Boundaries'
geometry = ['Delaware']

with open(json_path,'r') as f:
    mapping = json.load(f)
    
geom = mapping[category]['key']
select_ugid = [mapping[category]['geometries'][g] for g in geometry]

import ipdb;ipdb.set_trace()
#webbrowser.open(json_path)
