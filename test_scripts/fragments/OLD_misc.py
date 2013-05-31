from collections import OrderedDict
from itertools import product
import numpy as np
import datetime
from shapely.geometry.polygon import Polygon


def pause_test(f):
    print('{0} test paused..'.format(f.__name__))
    
def make_random_geom(n):
    geom = []
    for ii in range(0,n):
        geom.append([ii,'...'])
    return(geom)
    
def gen_descriptor_classes(niter=1,opts=None):
        '''generates an exhaustive set of Descriptor Class forms'''
        
        if opts is None:
            ## this is some metadata options. pulled out for easier editing.
            meta = [{'uri':('/home/local/WX/ben.koziol/Dropbox/nesii/project/'
                            'ocg/nc/test.nc'),
                     'spatial_row_bounds':'bounds_latitude',
                     'spatial_col_bounds':'bounds_longitude',
                     'calendar':'proleptic_gregorian',
                     'time_units':'days since 2000-01-01 00:00:00',
                     'time_name':'time',
                     'level_name':'level',
                     'variable':'foo'}]
            
            ## set of options for each entry in the Descriptor Class
            opts = OrderedDict({
                    'meta':meta,
    #                'mode':['ocg','gdp'],
                    'mode':['ocg'],
                    'time_range':[None,
                                  [datetime.datetime(2000,1,1),
                                   datetime.datetime(2001,12,31)]
                                  ],
                    'level_range':[[1],[1,3],[1],[2],[1,3],None],
                    'geom':[[{'id':1,
                              'geom':Polygon(((-103.053,39.967),(-100.068,39.984),
                                              (-99.945 ,38.026),(-103.009,37.912)))
                              }],
                             None],
                    'output_format':['shp','csv'],
                    'output_grouping':[None],
                    'spatial_operation':['intersects','clip',],
                    'aggregate':[True,False],
                    'calc_raw':[False,True],
                    'calc_grouping':[['month'],['month','year'],['year']],
                    'calc':[None,
                            [{'func':'median','name':'my_median'},
                             {'func':'mean','name':'the_mean'}]],
    #                'abstraction':['vector','grid']
                    'abstraction':'vector'
                                })
        ## loop for the combinations
        for idx in gen_select_idx(opts,niter=niter):
            topts = OrderedDict()
            select = 0
            for key,value in opts.iteritems():
                topts.update({key:value[idx[select]]})
                select += 1
            yield(topts)
                
def gen_select_idx(opts,niter=1):
    '''generate a list of attribute selection indices for a Descriptor
    class'''
    ## generate list of iterables
    iters = [range(len(ii)) for ii in opts.itervalues()]
    ii = 0
    for x in product(*iters):
        try:
            yield(x)
        finally:
            ii += 1
            if not ii < niter:
                break
        
def gen_example_data(niter=1):
    '''generate example data dictionaries originating from spatial operations
    niter=None :: number of iterations. if None, assume infinite.
    '''
    
#    gid:[],
#        tid:[],
#        lid:[],
#        geom:[],
#        timevec:[],
#        levelvec:[],
#        value:[],
#        weights:[], (not normalized)
#        calc:{}

    if niter == None:
        niter = float('inf')
    ii = 0
    while ii < niter:
        data = dict(
         gid = np.array([10,20,30,40,50]),
         tid = np.array([1,2,3,4]),
         lid = np.array([1]),
         geom = np.array([str(ii)+'...' for ii in range(5)]),
         timevec = np.array([datetime.date(2000,1,15),
                             datetime.date(2000,2,15),
                             datetime.date(2001,1,15),
                             datetime.date(2001,2,15)]),
         levelvec = np.array([11]),
         value = np.array([100,200,300,400,500,
                           110,220,330,440,550,
                           111,222,333,444,555,
                           1,2,3,4,5],dtype=float).reshape(4,1,5),
#         weights = np.array([1,1,1,1,1],dtype=float),
         raw_value = make_raw_value(5,4,1,[5,10]),
         attr = {}
                    )
        ii += 1
        yield(data)
        
def make_example_coll():
    for data in gen_example_data():
        return(data)
        
def make_raw_value(n,tdim,ldim,grng,vrng=[0,100],arng=[100,100]):
    
    def rand(shape,rng):
        ary = np.random.random_integers(rng[0],rng[1],size=shape).astype(float)
        return(ary)
    
    raw_value = []
    for ii in range(n):
        val = rand((tdim,ldim,np.random.random_integers(grng[0],grng[1],size=1)),
                   vrng)
        area = rand(val.shape,arng)
        weight = area/area.max()
        raw_value.append({'value':val,'weight':weight,'area':area})
    return(raw_value)
    