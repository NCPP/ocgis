from ocgis.interface.projection import WGS84


name_bounds = ['bounds','bnds','bound','bnd']
name_bounds.extend(['d_'+b for b in name_bounds])
ocgis_bounds = 'bounds'

fill_value = 1e20

raw_headers = ['did','vid','ugid','tid','lid','gid','variable','alias','time','level','value']
calc_headers = ['did','vid','cid','ugid','tgid','lid','gid','variable','alias','calc_name','year','month','day','hour','minute','level','value']
multi_headers = ['ugid','tid','lid','gid','calc_name','time','level','value']

reference_projection = WGS84()