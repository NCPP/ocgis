from sqlalchemy.schema import MetaData, Column, ForeignKey
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.types import Integer, Float, String, DateTime
from sqlalchemy.orm import relationship


## metadata will be bound to an engine when the database is created
metadata = MetaData()
Base = declarative_base(metadata=metadata)


class Geometry(Base):
    __tablename__ = 'geometry'
    gid = Column(Integer,primary_key=True)
    area_m2 = Column(Float,nullable=True)
    wkt = Column(String,nullable=True)
    wkb = Column(String,nullable=True)
    
#    def as_kml(self):
#        from pykml.factory import KML_ElementMaker as KML
#        pass
    
    def as_kml_coords(self):
        '''converts to a list of KML-formatted coordinate strings'''
        from django.contrib.gis.gdal import OGRGeometry
        from pykml.parser import fromstring
        return fromstring(OGRGeometry(self.wkt).kml).findall('.//coordinates')
    
    
class Time(Base):
    __tablename__ = 'time'
    tid = Column(Integer,primary_key=True)
    time = Column(DateTime,unique=True,nullable=False)
    day = Column(Integer,index=True)
    month = Column(Integer,index=True)
    year = Column(Integer,index=True)
    
    def as_xml_date(self):
        '''Return the time as a XML time formatted string (UTC time)'''
        return self.time.strftime('%Y-%m-%d')
    
    def as_xml_datetime(self):
        '''Return the time as a XML time formatted string (UTC time)'''
        return self.time.strftime('%Y-%m-%dT%H:%M:%SZ')


class AbstractValue(object):
#    ocgid = Column(Integer,primary_key=True)
#    
#    @declared_attr
#    def gid(self):
#        return(Column(Integer,ForeignKey(Geometry.gid)))
#    
#    level = Column(Integer,nullable=False)
    
    @declared_attr
    def geometry(self):
        return(relationship(Geometry))
    
    @property
    def wkt(self):
        return(self.geometry.wkt)
    @property
    def wkb(self):
        return(self.geometry.wkb)
    @property
    def area_m2(self):
        return(self.geometry.area_m2)


class Value(AbstractValue,Base):
    __tablename__ = 'value'
    ocgid = Column(Integer,primary_key=True)
    gid = Column(ForeignKey(Geometry.gid))
    tid = Column(ForeignKey(Time.tid))
    level = Column(Integer,nullable=False)
    value = Column(Float,nullable=False)
    
    time_ref = relationship(Time,backref="values")
    geometry_ref = relationship(Geometry,backref="values")

    @property
    def time(self):
        return(self.time_ref.time)


#class AbstractStat(AbstractValue):
#    __tablename__ = 'stat'
#    ocgid = Column(Integer,primary_key=True)
#    gid = Column(Integer,ForeignKey(Geometry.gid))
#    level = Column(Integer,nullable=False)