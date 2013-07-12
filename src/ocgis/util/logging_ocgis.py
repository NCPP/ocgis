import logging
import os

## try to turn off fiona logging
try:
    import fiona
    fiona_logger = logging.getLogger('Fiona')
    fiona_logger.setLevel(logging.ERROR)
except ImportError:
    pass

class OcgisLogging(object):
    
    def __init__(self):
        self.level = None
        self.null = True ## pass through if not configured
        self.parent = None
        self.duplicates = set()
        
    def __call__(self,msg=None,logger=None,level=logging.INFO,alias=None,ugid=None,exc=None,
                 check_duplicate=False):
        if self.null:
            if exc is None:
                pass
            else:
                raise(exc)
        else:
            if check_duplicate:
                if msg in self.duplicates:
                    return()
                else:
                    self.duplicates.add(msg)
            dest_level = level or self.level
            ## get the logger by string name
            if isinstance(logger,basestring):
                dest_logger = self.get_logger(logger)
            else:
                dest_logger = logger or self.parent
            if alias is not None:
                msg = self.get_formatted_msg(msg,alias,ugid=ugid)
            if exc is None:
                dest_logger.log(dest_level,msg)
            else:
                dest_logger.exception(msg)
                raise(exc)
    
    def configure(self,to_file=None,to_stream=False,level=logging.INFO):
        ## no need to configure loggers
        if to_file is None and not to_stream:
            self.null = True
        else:
            self.level = level
            self.null = False
            ## add the filehandler if request
            if to_file is None:
                filename = os.devnull
            else:
                filename = to_file
            ## create the root logger
            self.loggers = {}
            self.parent = logging.getLogger('ocgis')
            self.parent.parent = None
            self.parent.setLevel(level)
            self.parent.handlers = []
            ## add the file handler
            fh = logging.FileHandler(filename,'w')
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(fmt='%(name)12s: %(levelname)s: %(asctime)s: %(message)s',
                                              datefmt='%Y-%m-%d %H:%M'))
            self.parent.addHandler(fh)
            ## add the stream handler if requested
            if to_stream:
                console = logging.StreamHandler()
                console.setLevel(level)
                console.setFormatter(logging.Formatter('%(name)12s: %(levelname)s: %(message)s'))
                self.parent.addHandler(console)
    
    def get_formatted_msg(self,msg,alias,ugid=None):
        if ugid is None:
            ret = 'alias={0}: {1}'.format(alias,msg)
        else:
            ret = 'alias={0}, ugid={1}: {2}'.format(alias,ugid,msg)
        return(ret)
    
    def get_logger(self,name):
        if self.null:
            ret = None
        else:
            ret = logging.getLogger('ocgis').getChild(name)
        return(ret)
    
    def shutdown(self):
        self.null = True
        try:
            logging.shutdown()
        except:
            pass
    
#    def _reset_handlers_(self):
#        logging.getLogger('ocgis').handlers = []


ocgis_lh = OcgisLogging()
