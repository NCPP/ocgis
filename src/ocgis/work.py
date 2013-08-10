import ocgis
import netCDF4 as nc
from ocgis.util.large_array import compute
from ocgis.api.request import RequestDatasetCollection, RequestDataset


def compute_sfwe_maurer():
    def maurer_pr():
        ret = {'uri':'Maurer02new_OBS_pr_daily.1971-2000.nc','variable':'pr'}
        return(ret)
        
    def maurer_tas():
        ret = {'uri':'Maurer02new_OBS_tas_daily.1971-2000.nc','variable':'tas'}
        return(ret)
    
    
    ocgis.env.DIR_DATA = '/usr/local/climate_data/'
    ocgis.env.DIR_OUTPUT = '/home/local/WX/ben.koziol/climate_data/QED-2013/sfwe/maurer02v2'
    ocgis.env.OVERWRITE = True
    
    calc = [{'func':'sfwe','name':'sfwe','kwds':{'tas':'tas','pr':'pr'}}]
    time_range = None
    rds = []
    for var in [maurer_pr(),maurer_tas()]:
        var.update({'time_range':time_range})
        rds.append(var)
    rdc = RequestDatasetCollection(rds)
    sfwe = compute(rdc,calc,['month','year'],175,verbose=True,prefix='sfwe')
    print(sfwe)
    #sfwe = '/home/local/WX/ben.koziol/climate_data/QED-2013/sfwe/maurer02v2/sfwe/sfwe.nc'
    
    calc = [{'func':'sum','name':'p'}]
    time_range = None
    rds = [maurer_pr()]
    rdc = RequestDatasetCollection(rds)
    pr = compute(rdc,calc,['month','year'],175,verbose=True,prefix='pr')
    print(pr)
    #pr = '/home/local/WX/ben.koziol/climate_data/QED-2013/sfwe/maurer02v2/pr/pr.nc'
    
    calc = [{'func':'ratio_sfwe_p','name':'sfwe_p','kwds':{'sfwe':'sfwe','p':'p'}}]
    rds = [RequestDataset(sfwe,'sfwe'),RequestDataset(pr,'p')]
    rdc = RequestDatasetCollection(rds)
    sfwe_p = compute(rdc,calc,None,175,verbose=True,prefix='sfwe_p')
    print(sfwe_p)
    
def compute_sfwe_other():
    ocgis.env.DIR_DATA = '/data/ben.koziol/sfwe/data'
    ocgis.env.DIR_OUTPUT = '/home/local/WX/ben.koziol'
    ocgis.env.OVERWRITE = False
    
    tas = [['bcca_gfdl_cm2_1.gregorian.20c3m.run1.tas.1971-2000.nc','tas',None,'bcca_gfdl'],
     ['bcca_cccma_cgcm3_1.gregorian.20c3m.run1.tas.1971-2000.nc','tas',None,'bcca_cccma_cgcm3'],
     ['arrm_cgcm3_t63.20c3m.tas.NAm.1971-2000.nc','tas','365_day','arrm_cgcm3'],
     ['arrm_gfdl_2.1.20c3m.tas.NAm.1971-2000.nc','tas','365_day','arrm_gfdl']]
    
    pr = [['bcca_gfdl_cm2_1.gregorian.20c3m.run1.pr.1971-2000.nc','pr',None,'bcca_gfdl'],
     ['bcca_cccma_cgcm3_1.gregorian.20c3m.run1.pr.1971-2000.nc','pr',None,'bcca_cccma_cgcm3'],
     ['arrm_cgcm3_t63.20c3m.pr.NAm.1971-2000.nc','pr','365_day','arrm_cgcm3'],
     ['arrm_gfdl_2.1.20c3m.pr.NAm.1971-2000.nc','pr','365_day','arrm_gfdl']]
    
    for t,p in zip(tas,pr):
        
        tas_rd = ocgis.RequestDataset(t[0],t[1],t_calendar=t[2])
        pr_rd = ocgis.RequestDataset(p[0],p[1],t_calendar=p[2])
        
        calc = [{'func':'sfwe','name':'sfwe','kwds':{'tas':'tas','pr':'pr'}}]
        rdc = RequestDatasetCollection([tas_rd,pr_rd])
        sfwe = compute(rdc,calc,['month','year'],175,verbose=True,prefix='sfwe_'+t[3])
        print(sfwe)
        #sfwe = '/home/local/WX/ben.koziol/climate_data/QED-2013/sfwe/maurer02v2/sfwe/sfwe.nc'
        
        calc = [{'func':'sum','name':'p'}]
        rds = [pr_rd]
        rdc = RequestDatasetCollection(rds)
        pr = compute(rdc,calc,['month','year'],175,verbose=True,prefix='pr_'+t[3])
        print(pr)
        #pr = '/home/local/WX/ben.koziol/climate_data/QED-2013/sfwe/maurer02v2/pr/pr.nc'
        
        calc = [{'func':'ratio_sfwe_p','name':'sfwe_p','kwds':{'sfwe':'sfwe','p':'p'}}]
        rds = [RequestDataset(sfwe,'sfwe',t_calendar=t[2]),RequestDataset(pr,'p',t_calendar=t[2])]
        rdc = RequestDatasetCollection(rds)
        sfwe_p = compute(rdc,calc,None,175,verbose=True,prefix='sfwe_p_'+t[3])
        print(sfwe_p)


if __name__ == '__main__':
#    compute_sfwe_maurer()
    compute_sfwe_other()