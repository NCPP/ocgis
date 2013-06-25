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
from ConfigParser import SafeConfigParser
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound


connstr = 'sqlite:////tmp/foo.sqlite'
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
    
    
class Name(AbstractBase,Base):
    __tablename__ = 'name'
    __table_args__ = (UniqueConstraint('value','ugid'),)
    nid = Column(Integer,primary_key=True)
    sid = Column(Integer,ForeignKey(Subcategory.sid),nullable=True)
    ugid = Column(Integer,nullable=False)
    
    subcategory = relationship(Subcategory,backref="name")

try:
    os.remove('/tmp/foo.sqlite')
except:
    pass

metadata.create_all()
session = Session()

sc = ShpCabinet()
shp_path = sc.get_shp_path('us_counties')
cfg_path = sc.get_cfg_path('us_counties')

config = SafeConfigParser()
config.read(cfg_path)
ugid = config.get('mapping','ugid')
category = config.get('mapping','category')
subcategory = config.get('mapping','subcategory')
name = config.get('mapping','name')

db_category = Category(value=category,key='us_counties')
session.add(db_category)
session.commit()

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

with fiona.open(shp_path, 'r') as source:
    n = len(source)
    for ctr,feature in enumerate(source):
        if ctr % 100 == 0:
            print('{0} of {1}'.format(ctr,n))
        value_subcategory = get_variant(subcategory,feature)
        db_subcategory = get_or_create(session,Subcategory,value=value_subcategory,category=db_category)
        
        value_name = get_variant(name,feature)
        value_name_ugid = get_variant(ugid,feature)
        db_name = Name(value=value_name,ugid=value_name_ugid,subcategory=db_subcategory)
        session.add(db_name)
    session.commit()
