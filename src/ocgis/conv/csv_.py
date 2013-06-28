import csv
from ocgis.conv.base import OcgConverter
from csv import excel
from ocgis.util.shp_cabinet import ShpCabinet
import os
from ocgis import env, constants
from collections import OrderedDict
import logging
from ocgis.util.logging_ocgis import ocgis_lh


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
        gid_file = OrderedDict()
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
#                    ugid_idx = headers.index('UGID')
#                    gid_idx = headers.index('GID')
#                    did_idx = headers.index('DID')
                    writer.writerow(headers)
                    build = False
                for geom,row,geom_ids in coll.get_iter(with_geometry_ids=True):
                    if not is_aggregated:
                        ugid = geom_ids['ugid']
                        did = geom_ids['did']
                        gid = geom_ids['gid']
                        if ugid not in gid_file:
                            gid_file[ugid] = OrderedDict()
                        if did not in gid_file[ugid]:
                            gid_file[ugid][did] = OrderedDict()
                        gid_file[ugid][did][gid] = geom
#                        gid_file.append({'geom':geom,'did':row[did_idx],
#                                         'ugid':row[ugid_idx],'gid':row[gid_idx]})
                    writer.writerow(row)
        
        if is_aggregated is True:
            ocgis_lh('creating a UGID-GID shapefile is not necessary for aggregated data. use UGID shapefile.',
                     'conv',
                     logging.WARN)
        else:
            ocgis_lh('writing UGID-GID shapefile','conv',logging.DEBUG)
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
            
            def iter_gid_file():
                for ugid,did_gid in gid_file.iteritems():
                    for did,gid_geom in did_gid.iteritems():
                        for gid,geom in gid_geom.iteritems():
                            yield({'geom':geom,'DID':did,
                                   'UGID':ugid,'GID':gid})
            
            sc.write(iter_gid_file(),shp_path,sr=projection.sr)
