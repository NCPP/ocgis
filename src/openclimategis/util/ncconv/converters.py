import geojson
from util.helpers import get_temp_path
from util.toshp import OpenClimateShp


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

def as_tabular(elements,var,wkt=False,wkb=False,path = None):
    '''writes output in a tabular, CSV format geometry output is optional'''
    import osgeo.ogr as ogr

    if path is None:
        path = get_temp_path(suffix='.txt')

    #define spatial references for the projection
    sr = ogr.osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    sr2 = ogr.osr.SpatialReference()
    sr2.ImportFromEPSG(3005) #Albers Equal Area is used to ensure legitimate area values

    with open(path,'w') as f:
        
        #prepare column header
        header = ['id','timestamp',var]
        if 'level' in elements[0]['properties'].keys():
                header += ['level']
        header += ['area']
        if wkb:
            header += ['wkb']

        if wkt:
            header += ['wkt']
        
                
        f.write(','.join(header))
        f.write('\n')
        
        for ii,element in enumerate(elements):

            #convert area from degrees to m^2
            geo = ogr.CreateGeometryFromWkb(element['geometry'].wkb)
            geo.AssignSpatialReference(sr)
            geo.TransformTo(sr2)
            area = geo.GetArea()

            #write id, timestamp, variable
            f.write(','.join([repr(ii+1),element['properties']['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),repr(element['properties'][var])]))

            #write level if the dataset has levels
            if 'level' in element['properties'].keys():
                f.write(','+repr(element['properties']['level']))

            #write the area
            f.write(','+repr(area))

            #write wkb if requested
            if wkb:
                f.write(','+repr(element['geometry'].wkb))

            #write wkt if requested
            if wkt:
                f.write(','+repr(element['geometry'].wkt))

            f.write('\n')
        f.close()

    return path

def as_keyTabular(elements,var,wkt=False,wkb=False,path = None):
    '''writes output as tabular csv files, but uses foreign keys
on time and geometry to reduce file size'''
    import osgeo.ogr as ogr

    if path is None:
        path = get_temp_path(suffix='')

    if len(path)>4 and path[-4] == '.':
        path = path[:-4]

    patht = path+"_time.txt"
    pathg = path+"_geometry.txt"
    pathd = path+"_data.txt"

    #define spatial references for the projection
    sr = ogr.osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    sr2 = ogr.osr.SpatialReference()
    sr2.ImportFromEPSG(3005)
    data = {}

    #sort the data into dictionaries so common times and geometries can be identified
    for ii,element in enumerate(elements):

        #record new element ids (otherwise threads will produce copies of ids)
        element['id']=ii

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
    
    fgheader = ['gid','area']
    
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