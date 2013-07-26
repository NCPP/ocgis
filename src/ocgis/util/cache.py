import os
from pickle import dump, load
from ocgis import constants
from ConfigParser import SafeConfigParser, NoOptionError, NoSectionError
import time
from decimal import Decimal, InvalidOperation
from ocgis.util.logging_ocgis import ocgis_lh
import logging
import numpy as np
from tempfile import mkstemp
from ocgis import env
from ocgis.exc import DataNotCached


class CacheCabinet(object):
    
    def __init__(self,path,limit=500):
        self.path = path
        self.limit = limit
        self._cfg_path = os.path.join(path,constants.cache_cfg_name)
        self._config = SafeConfigParser()
        
        ## attempt to make the directory if it does not exists
        if not os.path.exists(self.path):
            os.makedirs(path)
        ## look for cfg file
        if not os.path.exists(self._cfg_path):
            with open(self._cfg_path,'w') as f:
                self._config.add_section('metadata')
                self._config.set('metadata','created',str(self._quantize_(time.time())))
                self._config.write(f)
        self._config.read(self._cfg_path)
        
    def __iter__(self):
        for section in self._config.sections():
            if section != 'metadata':
                obj_path = self._config.get(section,'path')
                obj_mtime = self._config.getfloat(section,'mtime')
                obj_size = os.path.getsize(obj_path)*9.53674e-7
                obj_accessed = self._config.getfloat(section,'accessed')
                yield(section,obj_path,obj_mtime,obj_size,obj_accessed)
                
    @property
    def array(self):
        dtype = [('key',object),('path',object),('mtime',float),('size',float),('accessed',float)]
        ret = np.empty(0,dtype=dtype)
        for row in self:
            to_append = np.array(row,dtype=dtype)
            ret = np.append(ret,to_append)
        return(ret)
        
    @property
    def size_megabytes(self):
        return(self.array['size'].sum())
    
    def keys(self):
        ret = self._config.sections()
        ret.remove('metadata')
        return(ret)
    
    def add(self,*args,**kwargs):
        ## if we are over the limit, delete oldest files
        size_mb = self.size_megabytes
        if size_mb > self.limit:
            arr = np.sort(self.array,order='accessed')
            ## how much needs to be removed
            mb_to_remove = size_mb - (self.limit - np.mean(arr['size']))
            ## get file modification times and sizes
            amount_queued = 0.0
            idx_to_remove = []
            idx_ctr = 0
            while amount_queued < mb_to_remove:
                amount_queued += arr[idx_ctr]['size']
                idx_to_remove.append(idx_ctr)
                idx_ctr += 1
            for remove_idx in idx_to_remove:
                self.remove(arr[remove_idx]['key'])
        self._add_object_(*args,**kwargs)
        
    def remove(self,key):
        obj_path = self._config.get(key,'path')
        os.remove(obj_path)
        self._config.remove_section(key)
        self._write_config_()
    
    def _add_object_(self,key,obj,mtime=None):
        if self._config.has_section(key):
            ## if modification time is None, we have to overwrite
            if mtime is not None:
                mtime = self._quantize_(mtime)
                ## get the modification time from the target
                cfg_mtime = self._config.get(key,'mtime')
                try:
                    cached_mtime = self._quantize_(cfg_mtime)
                except InvalidOperation:
                    if cfg_mtime == 'None':
                        cached_mtime = None
                    else:
                        raise
                if cached_mtime is None:
                    obj_path = self._config.get(key,'path')
                    co = CachedObject(obj_path)
                    co.write(obj)
                    self._config.set(key,'mtime',str(mtime))
                else:
                    if mtime == cached_mtime:
                        ocgis_lh(msg='cached object with key={0} and mtime={1} already in cache'.format(key,mtime),
                                 logger='cache',level=logging.INFO)
                    else:
                        obj_path = self._config.get(key,'path')
                        co = CachedObject(obj_path)
                        co.write(obj)
                        self._config.set(key,'mtime',str(mtime))
            else:
                obj_path = self._config.get(key,'path')
                co = CachedObject(obj_path)
                co.write(obj)
        else:
            self._config.add_section(key)
            if mtime is None:
                write_mtime = str(self._quantize_(time.time()))
            else:
                write_mtime = str(self._quantize_(mtime))
            self._config.set(key,'mtime',write_mtime)
            obj_path = self.get_path(key)
            self._config.set(key,'path',obj_path)
            co = CachedObject(obj_path)
            co.write(obj)
        ## always write out the configuration file
        self._write_config_(key)
        
    def get(self,key):
        path = self.get_path(key)
        co = CachedObject(path)
        ret = co.get()
        self._write_config_(key)
        return(ret)
        
    def get_path(self,key):
        try:
            path = self._config.get(key,'path')
        except NoOptionError:
            path = mkstemp(dir=self.path,suffix='.pkl')[1]
            self._config.set(key,'path',path)
            self._write_config_()
        return(path)
    
    def _quantize_(self,flt):
        ret = Decimal(flt).quantize(Decimal('0.0000'))
        return(ret)
    
    def _write_config_(self,key=None):
        if key is not None:
            self._config.set(key,'accessed',str(self._quantize_(time.time())))
        with open(self._cfg_path,'w') as f:
            self._config.write(f)


class CachedObject(object):
    
    def __init__(self,path):
        self.path = path
    
    def get(self):
        with open(self.path,'r') as f:
            ret = load(f)
        return(ret)
    
    def write(self,obj):
        with open(self.path,'wb') as f:
            dump(obj,f)


def get_cache_state():
        ret = False
        if env.USE_CACHING:
            if env.ops is not None:
                if not env.ops.snippet:
                    ret = True
                if env.ops.calc is not None:
                    ret = True
            else:
                ret = True
        return(ret)
    
def get_cached_temporal(request_dataset,mtime=None):
    key,mtime = get_key_mtime(request_dataset,mtime=mtime)
    try:
        cc = CacheCabinet(env.DIR_CACHE)
        ret = cc.get(key)
    except NoSectionError:
        raise(DataNotCached)
    return(ret)
    
def get_key_mtime(request_dataset,mtime=None):
    try:
        ret = '_'.join(request_dataset._uri)
    except AttributeError:
        ## it is likely a string key passed for temporal group caching
        if isinstance(request_dataset,basestring):
            ret = request_dataset
            assert(mtime is not None)
            mtime = mtime
        else:
            raise
    try:
        if mtime is None:
            mtime = max(map(os.path.getmtime,request_dataset._uri))
    except OSError:
        ## data may be remote
        mtime = None
    if env.ops is not None:
        if env.ops.calc is not None and env.ops.calc_grouping is not None:
            ret += '_ocgis_grouping_' + '_'.join(env.ops.calc_grouping)
    return(ret,mtime)

def add_to_cache(obj,request_dataset,mtime=None):
    key,mtime = get_key_mtime(request_dataset,mtime=mtime)
    cc = CacheCabinet(env.DIR_CACHE)
    cc.add(key,obj,mtime)
