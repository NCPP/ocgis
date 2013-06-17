import logging
import os


class OcgisLogging(object):
    
    def __init__(self):
        self.to_file = None
        self.to_stream = None
        self.level = None
        self.null = True ## pass through if not configured
        
        self._reset_handlers_()
        
    def __call__(self,msg=None,logger=None,level=None,alias=None,ugid=None,exc=None):
        if self.null:
            if exc is None:
                pass
            else:
                raise(exc)
        else:
            dest_level = level or self.level
            dest_logger = logger or logging.getLogger()
            if alias is not None:
                msg = self.get_formatted_msg(msg,alias,ugid=ugid)
            if exc is None:
                dest_logger.log(dest_level,msg)
            else:
                dest_logger.exception(msg)
                raise(exc)
    
    def configure(self,to_file=None,to_stream=False,level=logging.INFO):
        self._reset_handlers_()
        ## no need to configure loggers
        if to_file is None and not to_stream:
            self.null = True
        else:
            self.level = level
            self.to_file = to_file
            self.to_stream = to_stream
            self.null = False
            ## add the filehandler if request
            if to_file is None:
                filename = os.devnull
            else:
                filename = to_file
            logging.basicConfig(level=level,
                    format='%(name)s: %(levelname)s: %(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M',
                    filename=filename,
                    filemode='w')
            ## have the logging capture any emitted warning messages
            if to_file is not None:
                logging.captureWarnings(True)
            ## add the stream handler if requested
            if to_stream:
                console = logging.StreamHandler()
                console.setLevel(level)
                console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
                logging.getLogger().addHandler(console)
    
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
            ret = logging.getLogger(name)
        return(ret)
    
    def shutdown(self):
        self.null = True
        logging.captureWarnings(False)
        try:
            logging.shutdown()
        except:
            pass
    
    def _reset_handlers_(self):
        logging.captureWarnings(False)
        logging.getLogger().handlers = []
        
ocgis_lh = OcgisLogging()
