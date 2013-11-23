from ocgis.util.helpers import iter_array


class MeltedIterator(object):
    
    def __init__(self,add_bounds=True):
        self.add_bounds = add_bounds
        
    def iter_spatial_dimension(self,dim):
        geoms = dim.vector.geom
        name_id = dim._name_id
        uid = dim.vector.uid
        
        ret = {}
        for (ii,jj),geom in iter_array(geoms,return_value=True):
            ret[name_id] = uid[ii,jj]
            yield(((ii,jj),geom,ret))
    
    def iter_vector_dimension(self,dim):
        value = dim.value
        uid = dim.uid
        bounds = dim.bounds
        has_bounds = False if bounds is None else True
        if not self.add_bounds and has_bounds:
            has_bounds = False
        name_id = dim._name_id
        name_value = dim._name_long
        name_left_bound = 'bnd_left_'+name_value
        name_right_bound = 'bnd_right_'+name_value
        
        ret = {}
        for idx in range(value.shape[0]):
            ret[name_value] = value[idx]
            ret[name_id] = uid[idx]
            if has_bounds:
                ret[name_left_bound] = bounds[idx,0]
                ret[name_right_bound] = bounds[idx,1]
            yield(idx,ret)
