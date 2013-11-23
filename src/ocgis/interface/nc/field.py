from ocgis.interface.base.field import Field
import numpy as np
from copy import deepcopy
from ocgis.util.logging_ocgis import ocgis_lh


class NcField(Field):
    
    def _get_value_from_source_(self,data,variable_name):
        ## collect the dimension slices
        axis_slc = {}
        axis_slc['T'] = self.temporal._src_idx
        try:
            axis_slc['Y'] = self.spatial.grid.row._src_idx
            axis_slc['X'] = self.spatial.grid.col._src_idx
        ## if grid and row are not present on the GridDimesion object. the source
        ## indices are attached to the grid object itself.
        except AttributeError:
            axis_slc['Y'] = self.spatial.grid._row_src_idx
            axis_slc['X'] = self.spatial.grid._col_src_idx
        if self.realization is not None:
            axis_slc['R'] = self.realization._src_idx
        if self.level is not None:
            axis_slc['Z'] = self.level._src_idx
        ## check for singletons in the indices and convert those from NumPy arrays.
        ## an index error is raised otherwise.
        axis_slc = {k:v if len(v) > 1 else slice(v[0],v[0]+1) for k,v in axis_slc.iteritems()}
        
        dim_map = data._source_metadata['dim_map']
        slc = [None for v in dim_map.values() if v is not None]
        axes = deepcopy(slc)
        for k,v in dim_map.iteritems():
            if v is not None:
                slc[v['pos']] = axis_slc[k]
                axes[v['pos']] = k
        
        ## ensure axes ordering is as expected
        possible = [['T','Y','X'],['T','Z','Y','X'],['R','T','Y','X'],['R','T','Z','Y','X']]
        check = [axes == poss for poss in possible]
        assert(any(check))
        
        ds = data._open_()
        try:
            try:
                raw = ds.variables[variable_name].__getitem__(slc)
            except IndexError:
                ocgis_lh(logger='nc.field',exc=IndexError('variable: {0}'.format(variable_name)))
            if not isinstance(raw,np.ma.MaskedArray):
                raw = np.ma.array(raw,mask=False)
            ## reshape the data adding singleton axes where necessary
            new_shape = []
            for axis in self._axes:
                if axis in axes:
                    try:    
                        to_append = raw.shape[axes.index(axis)]
                    except IndexError as e:
                        ## it may be a singleton index request
                        if len(slc[axes.index(axis)]) == 1:
                            to_append = 1
                        else:
                            ocgis_lh(logger='nc.field',exc=e)
                else:
                    to_append = 1
                new_shape.append(to_append)
            raw = raw.reshape(new_shape)
            
            return(raw)
#            ## apply any spatial mask if the geometries have been loaded
#            if self.spatial._geom is not None:
#                self._set_new_value_mask_(self,self.spatial.get_mask())
        finally:
            ds.close()
