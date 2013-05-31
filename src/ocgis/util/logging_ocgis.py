from ocgis import constants, env
import logging


def configure_logging(add_filehandler=True,filename=None):
    ## tell other modules to load loggers
    env._use_logging = True
    ## reset the handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []
    ## if file logging is enable or verbose is active, push warnings to the
    ## logger
    if env.VERBOSE or env.ENABLE_FILE_LOGGING:
        logging.captureWarnings(True)
    else:
        logging.captureWarnings(False)
    ## add a file handler if requested
    if add_filehandler:
        fmt = '%(name)s: %(levelname)s: %(message)s'
        fileh = logging.FileHandler(filename,mode='w')
        fileh.setFormatter(logging.Formatter(fmt))
        fileh.setLevel(constants.logging_level)
        root_logger.addHandler(fileh)
    if env.VERBOSE:
        console = logging.StreamHandler()
        console.setLevel(constants.logging_level)
        console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        root_logger.addHandler(console)
            
def get_formatted_msg(msg,alias,ugid=None):
    if ugid is None:
        ret = 'alias={0} : {1}'.format(alias,msg)
    else:
        ret = 'alias={0}, ugid={1} : {2}'.format(alias,ugid,msg)
    return(ret)

def ocgis_lh(msg,logger,level=logging.INFO,alias=None,exc=None,ugid=None):
    if exc is not None:
        msg = exc.message
    if alias is not None:
        msg = get_formatted_msg(msg,alias,ugid=ugid)
    if exc is None:
        if logger is not None and (env.VERBOSE or env.ENABLE_FILE_LOGGING):
            logger.log(level,msg)
    else:
        if logger is not None and (env.VERBOSE or env.ENABLE_FILE_LOGGING):
            logger.exception(msg)
        raise(exc)
