import csv
from ocgis.conv.base import OcgConverter
from csv import excel
from ocgis.util.shp_cabinet import ShpCabinet
import os
from ocgis import env, constants


class OcgDialect(excel):
    lineterminator = '\n'


class CsvConverter(OcgConverter):
    _ext = 'csv'
    
    def _write_(self):
        build = True
        with open(self.path,'w') as f:
            writer = csv.writer(f,dialect=OcgDialect)
            for coll in self:
                if build:
                    headers = coll.get_headers(upper=True)
                    writer.writerow(headers)
                    build = False
                for geom,row in coll.get_iter():
                    writer.writerow(row)


class CsvPlusConverter(CsvConverter):
    _add_ugeom = True
    
    def _write_(self):
        gid_file = []
        build = True
        is_aggregated = self.ops.aggregate
        with open(self.path,'w') as f:
            writer = csv.writer(f,dialect=OcgDialect)
            for coll in self:
                if build:
                    headers = coll.get_headers(upper=True)
                    if env.WRITE_TO_REFERENCE_PROJECTION:
                        projection = constants.reference_projection
                    else:
                        projection = coll._archetype.spatial.projection
                    ugid_idx = headers.index('UGID')
                    gid_idx = headers.index('GID')
                    did_idx = headers.index('DID')
                    writer.writerow(headers)
                    build = False
                for geom,row in coll.get_iter():
                    if not is_aggregated:
                        gid_file.append({'geom':geom,'did':row[did_idx],
                                         'ugid':row[ugid_idx],'gid':row[gid_idx]})
                    writer.writerow(row)
        
        if is_aggregated is True:
            if env.VERBOSE:
                print('creating a UGID-GID shapefile is not necessary for aggregated data. use UGID shapefile.')
        else:
            sc = ShpCabinet()
            shp_dir = os.path.join(self.outdir,'shp')
            try:
                os.mkdir(shp_dir)
            ## catch if the directory exists
            except OSError:
                if os.path.exists(shp_dir):
                    pass
                else:
                    raise
            shp_path = os.path.join(shp_dir,self.prefix+'_gid.shp')
            sc.write(gid_file,shp_path,sr=projection.sr)
