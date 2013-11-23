import csv
from ocgis.conv.base import OcgConverter
from csv import excel
import os
from collections import OrderedDict
import logging
from ocgis.util.logging_ocgis import ocgis_lh
import fiona
from shapely.geometry.geo import mapping


class OcgDialect(excel):
    lineterminator = '\n'


class CsvConverter(OcgConverter):
    _ext = 'csv'
                    
    def _build_(self,coll):
        headers = [h.upper() for h in coll.headers]
        f = open(self.path,'w')
        writer = csv.DictWriter(f,headers,dialect=OcgDialect)
        writer.writeheader()
        ret = {'file_object':f,'csv_writer':writer}
        return(ret)
        
    def _write_coll_(self,f,coll):
        writer = f['csv_writer']
        
        for geom,row in coll.get_iter_dict(use_upper_keys=True):
            writer.writerow(row)

    def _finalize_(self,f):
        for fobj in f.itervalues():
            try:
                fobj.close()
            except:
                pass

class CsvPlusConverter(CsvConverter):
    _add_ugeom = True

    def _build_(self,coll):
        ret = CsvConverter._build_(self,coll)
        
        self._ugid_gid_store = {}
        
        if not self.ops.aggregate:
            fiona_path = os.path.join(self._get_or_create_shp_folder_(),self.prefix+'_gid.shp')
            archetype_field = coll._archetype_field
            fiona_crs = archetype_field.spatial.crs.value
            fiona_schema = {'geometry':archetype_field.spatial.abstraction_geometry._geom_type,
                            'properties':OrderedDict([['DID','int'],['UGID','int'],['GID','int']])}
            fiona_object = fiona.open(fiona_path,'w',driver='ESRI Shapefile',crs=fiona_crs,schema=fiona_schema)
        else:
            ocgis_lh('creating a UGID-GID shapefile is not necessary for aggregated data. use UGID shapefile.',
                     'conv.csv+',
                     logging.WARN)
            fiona_object = None
        
        ret.update({'fiona_object':fiona_object})
        
        return(ret)
    
    def _write_coll_(self,f,coll):
        writer = f['csv_writer']
        file_fiona = f['fiona_object']
        rstore = self._ugid_gid_store
        is_aggregated = self.ops.aggregate
        
        for geom,row in coll.get_iter_dict(use_upper_keys=True):
            writer.writerow(row)
            if not is_aggregated:
                did,gid,ugid = row['DID'],row['GID'],row['UGID']
                try:
                    if gid in rstore[did][ugid]:
                        continue
                    else:
                        raise(KeyError)
                except KeyError:
                    if did not in rstore:
                        rstore[did] = {}
                    if ugid not in rstore[did]:
                        rstore[did][ugid] = []
                    if gid not in rstore[did][ugid]:
                        rstore[did][ugid].append(gid)
                    
                    ## for multivariate calculation outputs the dataset identifier
                    ## is None.
                    try:
                        converted_did = int(did)
                    except TypeError:
                        converted_did = None
                    feature = {'properties':{'GID':int(gid),'UGID':int(ugid),'DID':converted_did},
                               'geometry':mapping(geom)}
                    file_fiona.write(feature)
