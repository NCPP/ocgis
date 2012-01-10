from util.ncconv.experimental.ocg_converter.ocg_converter import OcgConverter
import geojson


class GeojsonConverter(OcgConverter):
    
    def _convert_(self):
        if self.use_geom:
            headers = ['GID','WKT']
        else:
            if self.use_stat:
                adds = ['WKT']
            else:
                adds = ['WKT','TIME']
            headers = self.get_headers(self.value_table,adds=adds)
        features = [attrs for attrs in self.get_iter(self.value_table,headers)]
        for feat in features:
            feat['geometry'] = feat.pop('WKT')
            if 'TIME' in feat:
                feat['TIME'] = str(feat['TIME'])
        fc = geojson.FeatureCollection(features)
        return(geojson.dumps(fc))