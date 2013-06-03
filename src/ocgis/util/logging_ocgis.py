import logging


class OcgisLogging(object):
    
    def __init__(self):
        self.to_file = None
        self.filename = None
        self.to_stream = None
        self.level = None
        self.null = True ## pass through if not configured
        self.root_logger = None
        
        self._reset_handlers_()
        
    def __call__(self,msg=None,logger=None,level=None,alias=None,ugid=None,exc=None):
        if self.null:
            if exc is None:
                pass
            else:
                raise(exc)
        else:
            dest_level = level or self.level
            dest_logger = logger or self.root_logger
            if alias is not None:
                msg = self.get_formatted_msg(msg,alias,ugid=ugid)
            if exc is None:
                dest_logger.log(dest_level,msg)
            else:
                dest_logger.exception(msg)
                raise(exc)
            
    def __del__(self):
        try:
            logging.shutdown()
        except:
            pass
    
    def configure(self,to_file=True,filename=None,to_stream=True,
                  level=logging.INFO):
        self.level = level
        self.to_file = to_file
        self.filename = filename
        self.to_stream = to_stream
        self._reset_handlers_()
        
        ## no need to configure loggers
        if not to_file and not to_stream:
            self.null = True
        else:
            if to_file and filename is None:
                raise(ValueError('a filename is required when writing to file'))
            self.null = False
            ## have the logging capture any emitted warning messages
            logging.captureWarnings(True)
            ## add the filehandler if request
            if to_file:
                fmt = '%(name)s: %(levelname)s: %(message)s'
                fileh = logging.FileHandler(filename,mode='w')
                fileh.setFormatter(logging.Formatter(fmt))
                fileh.setLevel(level)
                self.root_logger.addHandler(fileh)
            ## add the stream handler if requested
            if to_stream:
                console = logging.StreamHandler()
                console.setLevel(level)
                console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
                self.root_logger.addHandler(console)
    
    def get_formatted_msg(self,msg,alias,ugid=None):
        if ugid is None:
            ret = 'alias={0} : {1}'.format(alias,msg)
        else:
            ret = 'alias={0}, ugid={1} : {2}'.format(alias,ugid,msg)
        return(ret)
    
    def get_logger(self,name):
        return(logging.getLogger(name))        
    
    def _reset_handlers_(self):
        root_logger = logging.getLogger()
        root_logger.handlers = []
        self.root_logger = root_logger
        
ocgis_lh = OcgisLogging()
