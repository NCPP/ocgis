#from util.ncconv.experimental.ocg_converter.ocg_converter import OcgConverter
import geojson
from util.ncconv.experimental.ocg_converter.subocg_converter import SubOcgConverter


class GeojsonConverter(SubOcgConverter):
    
    def _convert_(self):
#        if self.use_geom:
#            raise(NotImplementedError)
#            headers = ['GID','WKT']
#        else:
##            if self.use_stat:
##                adds = ['WKT']
##            else:
##                adds = ['WKT','TIME']
#            headers = self.get_headers()
        features = [attrs for attrs in self.get_iter(wkt=True)]
        for feat in features:
            feat['geometry'] = feat.pop('WKT')
            if 'TIME' in feat:
                feat['TIME'] = str(feat['TIME'])
        fc = geojson.FeatureCollection(features)
        return(geojson.dumps(fc))