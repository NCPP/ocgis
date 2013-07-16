import geojson
from util.helpers import get_temp_path
from util.toshp import OpenClimateShp
#from pykml.factory import KML_ElementMaker as KML
from osgeo import osr, ogr
import io
import csv
import os

def get_sr(srid):
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(srid)
    return(sr)

def as_geojson(elements):
    features = []
    for e in elements:
        e['properties']['timestamp'] = str(e['properties']['timestamp'])
        features.append(geojson.Feature(**e))
    fc = geojson.FeatureCollection(features)
    return(geojson.dumps(fc))

def as_shp(elements,path=None):
    if path is None:
        path = get_temp_path(suffix='.shp')
    ocs = OpenClimateShp(path,elements)
    ocs.write()
    return(path)

def as_tabular(elements,var,wkt=False,wkb=False,todisk=False,area_srid=3005,path=None):
    '''writes output in a tabular, CSV format geometry output is optional
    
    elements -- standard geojson-like data representation
    var -- name of the output variable
    wkt=False -- set to True to write wkt to text file
    wkb=False -- set to True to write wkb to text file
    todisk=False -- set to True to write output to disk as well. if no path
        name is specified in the |path| argument, then one is generated.
    path=None -- specify an output file name. under the default, no file is
        written.
    '''

    ## define spatial references
    sr = get_sr(4326)
    sr2 = get_sr(area_srid)
    
    ## this will hold the data to write
    data = []
        
    ## prepare column header
    header = ['id','timestamp',var]
    if 'level' in elements[0]['properties'].keys():
            header += ['level']
    header += ['area_m2']
    if wkb:
        header += ['wkb']

    if wkt:
        header += ['wkt']
    data.append(header)
    
    for ii,element in enumerate(elements):
        
        ## will hold data to append to global list
        subdata = []
        
        #convert area from degrees to m^2
        geo = ogr.CreateGeometryFromWkb(element['geometry'].wkb)
        geo.AssignSpatialReference(sr)
        geo.TransformTo(sr2)
        area = geo.GetArea()
        
        ## retrieve additional elements ----------------------------------------
        
        props = element['properties']
        geom = element['geometry']
        
        subdata.append(ii+1)
        subdata.append(props['timestamp'])
        subdata.append(props[var])
        subdata.append(area)
        if 'level' in props.keys(): subdata.append(props['level'])
        if wkb: subdata.append(geom.wkb)
        if wkt: subdata.append(geom.wkt)
        
        ## append to global data
        data.append(subdata)
        
    ## write to the buffer
    buffer = io.BytesIO()
    writer = csv.writer(buffer)
    writer.writerows(data)
    
    ## set-up disk writing
    if todisk:
        if path is None:
            path = get_temp_path(suffix='.txt')
        with open(path,'w') as f:
            f.write(buffer.getvalue())
            
    try:
        return(buffer.getvalue())
    finally:
        buffer.close()

def as_keyTabular(elements,var,wkt=False,wkb=False,path=None,area_srid=3005):
    '''writes output as tabular csv files, but uses foreign keys
on time and geometry to reduce file size'''

    if path is None:
        path = get_temp_path(suffix='')
#    prefix = os.path.splitext(path)[0]
#
#    if len(path)>4 and path[-4] == '.':
#        path = path[:-4]

    patht = path+"_time.csv"
    pathg = path+"_geometry.csv"
    pathd = path+"_data.csv"
    
    paths = [patht,pathg,pathd]

    #define spatial references for the projection
    sr = get_sr(4326)
    sr2 = get_sr(area_srid)
    data = {}

    #sort the data into dictionaries so common times and geometries can be identified
    for ii,element in enumerate(elements):

        #record new element ids (otherwise threads will produce copies of ids)
        element['id'] = ii + 1

        #get the time and geometry
        time = element['properties']['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        ewkt = element['geometry'].wkt

        if not (time in data):
            data[time] = {}

        #put the data into the dictionary
        if not (ewkt in data[time]):
            data[time][ewkt] = [element]
        else:
            data[time][ewkt].append(element)


    #get a unique set of geometry keys
    locs = []

    for key in data:
        locs.extend(data[key].keys())

    locs = set(locs)

    ft = open(patht,'w')
    fg = open(pathg,'w')
    fd = open(pathd,'w')
    
    ft.write(','.join(['tid','timestamp']))
    ft.write('\n')
    
    fgheader = ['gid','area_m2']
    
    if wkt:
        fgheader += ['wkt']
    if wkb:
        fgheader += ['wkb']

    fg.write(','.join(fgheader))
    fg.write('\n')
    
    fdheader = ['id','tid','gid',var]
    if 'level' in elements[0]['properties']:
        fdheader += ['level']
        
    fd.write(','.join(fdheader))
    fd.write('\n')

    #write the features to file
    for ii,time in enumerate(data.keys()):

        #write out id's and time values to the time file
        tdat = data[time]
        ft.write(repr(ii+1)+','+time+'\n')

        for jj,loc in enumerate(locs):
            if ii==0:

                #find the geometry area
                geo = ogr.CreateGeometryFromWkt(loc)
                geo.AssignSpatialReference(sr)
                geo.TransformTo(sr2)

                #write the id and area
                fg.write(repr(jj+1))
                fg.write(','+repr(geo.GetArea()))

                #write out optional geometry
                if wkt:
                    fg.write(','+loc)
                if wkb:
                    fg.write(','+repr(ogr.CreateGeometryFromWkt(loc).ExportToWkb()))
                fg.write('\n')

            if loc in tdat:
                for element in tdat[loc]:
                    #write out id, foreign keys (time then geometry) and the variable value
                    fd.write(','.join([repr(element['id']),repr(ii+1),repr(jj+1),repr(element['properties'][var])]))
                    #write out level if appropriate
                    if 'level' in element['properties']:
                        fd.write(','+repr(element['properties']['level']))
                    fd.write('\n')

    ft.close()
    fg.close()
    fd.close()
    
    return(paths)
