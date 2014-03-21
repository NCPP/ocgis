import abc
from ocgis.util.shp_cabinet import ShpCabinetIterator
from exc import *


class AbstractLabelMaker(object):
    '''
    :param key: The shapefile's key or name.
    :type key: str
    :param filter_geometries: Sequence of geometries to use as a not filter.
    :type filter_geometries: sequence
    :param make_labels: If True, added geometry and subcategory labels. 
    :type make_labels: bool
    '''
    __metaclass__ = abc.ABCMeta
    _malformed_map = {}
    
    def __init__(self,key,filter_geometries=None,make_labels=True):
        self.key = key
        self.filter_geometries = filter_geometries
        self.make_labels = make_labels
        
    def __iter__(self):
        sci = ShpCabinetIterator(self.key)
        for row in sci:
            
            ## if there is overlap with a filter geometry, the geometry should
            ## not be included
            if self.filter_geometries is not None:
                from ocgis.util.shp_scanner.shp_scanner import get_does_intersect
                if not get_does_intersect(self.filter_geometries,row['geom']):
                    continue
                
            row['envelope'] = row['geom'].envelope.wkt
            
            if self.make_labels:
                row['geometry_label'] = self.get_geometry_label(row['properties'])
                try:
                    row['subcategory_label'] = self.get_subcategory_label(row['properties'])
                except NoSubcategoryError:
                    pass
            yield(row)
    
    def get_geometry_label(self,properties):
        ret = self._get_geometry_label_(properties)
        ret = self.format_label(ret)
        return(ret)
        
    def get_subcategory_label(self,properties):
        ret = self._get_subcategory_label_(properties)
        ret = self.format_label(ret)
        return(ret)
    
    def format_label(self,label):
        ret = label
        if label is not None:
            if ',' in label:
                try:
                    ret = self._malformed_map[label]
                except KeyError:
                    raise(MalformedLabel(label,self.key,reason='comma'))
        return(ret)
            
    @abc.abstractmethod
    def _get_geometry_label_(self,properties): str
        
    def _get_subcategory_label_(self,properties):
        raise(NoSubcategoryError)


class Huc8Boundaries(AbstractLabelMaker):
    
    def __init__(self,*args,**kwds):
        AbstractLabelMaker.__init__(self,*args,**kwds)
        
        ## collect state boundary geometries
        self.states = list(StateBoundaries('state_boundaries',
                                           filter_geometries=kwds.get('filter_geometries'),
                                           make_labels=False))
        
    def __iter__(self):
        sci = ShpCabinetIterator(self.key)
        for row in sci:
            row['envelope'] = row['geom'].envelope.wkt
            for state,label in self.get_geometry_label(row['geom'],row['properties']):
                yld = row.copy()
                yld['geometry_label'] = label
                self.format_label(yld['geometry_label'])
                try:
                    yld['subcategory_label'] = self.get_subcategory_label(state,row['properties'])
                    self.format_label(yld['subcategory_label'])
                except NoSubcategoryError:
                    pass
                yield(yld)
                
    def format_label(self,label):
        ret = label
        if label is not None:
            if ',' in label:
                ret = label.replace(', ',' - ')
                print(' changed label: "{0}" to "{1}"'.format(label,ret))
        return(ret)
        
    def get_geometry_label(self,geom,properties): 
        states = [s for s in self.states if s['geom'].intersects(geom)]
        for state in states:
            yield(state,properties['Name'])
            
    def get_subcategory_label(self,state,properties):
        return(state['properties']['STATE_NAME'])
        
    def _get_geometry_label_(self):
        raise(NotImplementedError)
        

class StateBoundaries(AbstractLabelMaker):
    
    def _get_geometry_label_(self,properties):
        ret = '{0} ({1})'.format(properties['STATE_NAME'],properties['STATE_ABBR'])
        return(ret)


class UsCounties(AbstractLabelMaker):
    
    def _get_geometry_label_(self,properties):
        return(properties['COUNTYNAME'])
    
    def _get_subcategory_label_(self,properties):
        ret = '{0} Counties'.format(properties['STATE'])
        return(ret)
    
    
class UsLevel3Ecoregions(AbstractLabelMaker):
    
    def _get_geometry_label_(self,properties):
        return(properties['US_L3NAME'])
    
    def _get_subcategory_label_(self,properties):
        ret = properties['NA_L1NAME'].title()
        return(ret)


class WorldCountries(AbstractLabelMaker):
    _malformed_map = {'Bahamas, The':'The Bahamas',
                      'Gambia, The':'The Gambia',
                      "Korea, Peoples Republic of":'Peoples Republic of Korea',
                      'Korea, Republic of':'Republic of Korea',
                      "Tanzania, United Republic of":"United Republic of Tanzania"}
    
    def _get_geometry_label_(self,properties):
        return(properties['NAME'])
    
    def _get_subcategory_label_(self,properties):
        ret = properties['REGION']
        if ret == 'NorthAfrica':
            ret = 'North Africa'
        if ret == 'Sub Saharan Africa':
            ret = 'Sub-Saharan Africa'
        return(ret)
        

class ClimateDivisions(AbstractLabelMaker):
    _malformed_map = {"Powder, Little Missouri, Tongu":"Powder - Little Missouri - Tongu"}
    
    def _get_geometry_label_(self,properties):
        ret = properties['NAME']
        return(ret.title())
    
    def _get_subcategory_label_(self,properties):
        ret = properties['STATE']
        return(ret)
