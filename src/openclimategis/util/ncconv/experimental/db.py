from sqlalchemy import create_engine
from sqlalchemy.schema import MetaData, Column, ForeignKey
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.types import Integer, Float, String, DateTime, Date
from sqlalchemy.orm import relationship


## metadata will be bound to an engine when the database is created
metadata = MetaData()
Base = declarative_base(metadata=metadata)


class Geometry(Base):
    __tablename__ = 'geometry'
    gid = Column(Integer,primary_key=True)
    area_m2 = Column(String,nullable=False)
    wkt = Column(String,unique=True,nullable=False)
    
#    def as_kml(self):
#        from pykml.factory import KML_ElementMaker as KML
#        pass
    
    def as_kml_coords(self):
        '''converts to a KML-formatted coordinate string'''
        from django.contrib.gis.gdal import OGRGeometry
        from pykml.parser import fromstring
        return fromstring(OGRGeometry(self.wkt).kml).find('.//coordinates').text
    
    
class Time(Base):
    __tablename__ = 'time'
    tid = Column(Integer,primary_key=True)
    time = Column(DateTime,unique=True,nullable=False,index=True)
    
    def as_xml_date(self):
        '''Return the time as a XML time formatted string (UTC time)'''
        return self.time.strftime('%Y-%m-%d')
    
    def as_xml_datetime(self):
        '''Return the time as a XML time formatted string (UTC time)'''
        return self.time.strftime('%Y-%m-%dT%H:%M:%SZ')

class Value(Base):
    __tablename__ = 'value'
    ocgid = Column(Integer,primary_key=True)
    gid = Column(ForeignKey(Geometry.gid))
    tid = Column(ForeignKey(Time.tid))
    level = Column(Integer,nullable=False)
    value = Column(Float,nullable=False)
    
    geometry = relationship(Geometry)
    time = relationship(Time,backref="value")
    
    def __repr__(self):
        msg = ['geometry={0}'.format(self.geometry.wkt[0:7])]
        msg.append('time={0}'.format(self.time.time))
        msg.append('level={0}'.format(self.level))
        msg.append('value={0}'.format(self.value))
        return(','.join(msg))
