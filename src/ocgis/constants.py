#from ocgis.interface.projection import WGS84


name_bounds = ['bounds','bnds','bound','bnd']
name_bounds.extend(['d_'+b for b in name_bounds])
ocgis_bounds = 'bounds'

fill_value = 1e20

raw_headers = ['did','vid','ugid','tid','lid','gid','variable','alias','time','level','value']
calc_headers = ['did','vid','cid','ugid','tgid','lid','gid','variable','alias','calc_name','year','month','day','level','value']
multi_headers = ['ugid','tid','lid','gid','calc_name','time','level','value']

#reference_projection = WGS84()

#test_data_download_url_prefix = 'https://dl.dropboxusercontent.com/u/867854/test_data_download/'
test_data_download_url_prefix = 'http://www.earthsystemmodeling.org/download/data/ocgis/nc/'

## the day value to use for month centroid
calc_month_centroid = 16
calc_year_centroid_month = 7
calc_year_centroid_day = 1
