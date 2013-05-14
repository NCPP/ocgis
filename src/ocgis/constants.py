from ocgis.interface.projection import WGS84


name_bounds = ['bounds','bnds','bound','bnd']
name_bounds.extend(['d_'+b for b in name_bounds])
ocgis_bounds = 'bounds'

fill_value = 1e20

raw_headers = ['vid','ugid','tid','lid','gid','var_name','time','level','value']
calc_headers = ['vid','cid','ugid','tgid','lid','gid','var_name','calc_name','year','month','day','hour','minute','level','value']
multi_headers = ['ugid','tid','lid','gid','calc_name','time','level','value']

reference_projection = WGS84()