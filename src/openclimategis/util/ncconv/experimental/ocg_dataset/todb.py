from util.ncconv.experimental import ploader as pl
from util.helpers import get_temp_path
from sqlalchemy.pool import NullPool
from util.ncconv.experimental.helpers import get_sr, get_area
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon


def sub_to_db(sub,
              add_area=True,
              area_srid=3005,
              wkt=True,
              wkb=False,
              as_multi=True,
              to_disk=False,
              procs=1):
    """
    Convert the object to a SQLite database. Returns the |db| module exposing
        the database ORM and additional SQLAlchemy objects. Note that |procs|
        greater than one results in the database being written to disk (if the
        desired database is SQLite).
    
    sub (SubOcgDataset) -- The object to convert to the database.  
    add_area=True -- Insert the geometric area.
    area_srid=3005 -- SRID to use for geometric transformation.
    wkt=True -- Insert the geomtry's WKT representation.
    wkb=False -- Insert the geometry's WKB representation.
    as_multi=True -- Convert geometries to shapely.MultiPolygon.
    to_disk=False -- Write the database to disk.
    procs=1 -- Number of processes to use when loading data.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm.session import sessionmaker
    from util.ncconv.experimental import db
    
    path = 'sqlite://'
    if to_disk or procs > 1:
        path = path + '/' + get_temp_path('.sqlite',nest=True)
        db.engine = create_engine(path,
                                  poolclass=NullPool)
    else:
        db.engine = create_engine(path,
#                                      connect_args={'check_same_thread':False},
#                                      poolclass=StaticPool
                                  )
    db.metadata.bind = db.engine
    db.Session = sessionmaker(bind=db.engine)
    db.metadata.create_all()

    print('  loading geometry...')
    ## spatial reference for area calculation
    sr = get_sr(4326)
    sr2 = get_sr(area_srid)

#        data = dict([[key,list()] for key in ['gid','wkt','wkb','area_m2']])
#        for dd in self.dim_data:
#            data['gid'].append(int(self.gid[dd]))
#            geom = self.geometry[dd]
#            if isinstance(geom,Polygon):
#                geom = MultiPolygon([geom])
#            if wkt:
#                wkt = str(geom.wkt)
#            else:
#                wkt = None
#            data['wkt'].append(wkt)
#            if wkb:
#                wkb = str(geom.wkb)
#            else:
#                wkb = None
#            data['wkb'].append(wkb)
#            data['area_m2'].append(get_area(geom,sr,sr2))
#        self.load_parallel(db.Geometry,data,procs)

    def f(idx,geometry=sub.geometry,gid=sub.gid,wkt=wkt,wkb=wkb,sr=sr,sr2=sr2,get_area=get_area):
        geom = geometry[idx]
        if isinstance(geom,Polygon):
            geom = MultiPolygon([geom])
        if wkt:
            wkt = str(geom.wkt)
        else:
            wkt = None
        if wkb:
            wkb = str(geom.wkb)
        else:
            wkb = None
        return(dict(gid=int(gid[idx]),
                    wkt=wkt,
                    wkb=wkb,
                    area_m2=get_area(geom,sr,sr2)))
    fkwds = dict(geometry=sub.geometry,gid=sub.gid,wkt=wkt,wkb=wkb,sr=sr,sr2=sr2,get_area=get_area)
    gen = pl.ParallelGenerator(db.Geometry,sub.dim_data,f,fkwds=fkwds,procs=procs)
    gen.load()

    print('  loading time...')
    ## load the time data
    data = dict([[key,list()] for key in ['tid','time','day','month','year']])
    for ii,dt in enumerate(sub.dim_time,start=1):
        data['tid'].append(ii)
        data['time'].append(sub.timevec[dt])
        data['day'].append(sub.timevec[dt].day)
        data['month'].append(sub.timevec[dt].month)
        data['year'].append(sub.timevec[dt].year)
    load_parallel(db.Time,data,procs)
        
    print('  loading value...')
    ## set up parallel loading data
    data = dict([key,list()] for key in ['gid','level','tid','value'])
    for ii,dt in enumerate(sub.dim_time,start=1):
        for dl in sub.dim_level:
            for dd in sub.dim_data:
                data['gid'].append(int(sub.gid[dd]))
                data['level'].append(int(sub.levelvec[dl]))
                data['tid'].append(ii)
                data['value'].append(float(sub.value[dt,dl,dd]))
    load_parallel(db.Value,data,procs)

    return(db)

def load_parallel(Model,data,procs):
    pmodel = pl.ParallelModel(Model,data)
    ploader = pl.ParallelLoader(procs=procs)
    ploader.load_model(pmodel)