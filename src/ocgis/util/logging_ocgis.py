from ocgis import constants, env
import logging


def configure_logging(add_filehandler=True,filename=None):
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
        logging.getLogger().addHandler(fileh)
    if env.VERBOSE:
        root_logger = logging.getLogger()
        ## multiple stream handlers are possible with this init setup,
        ## do not add more than one...
        if not any([isinstance(hdlr,logging.StreamHandler) for hdlr in root_logger.handlers]):
            console = logging.StreamHandler()
            console.setLevel(constants.logging_level)
            console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            logging.getLogger().addHandler(console)
            
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
    if exc is None and (env.VERBOSE or env.ENABLE_FILE_LOGGING):
        logger.log(level,msg)
    else:
        if env.VERBOSE or env.ENABLE_FILE_LOGGING:
            logger.exception(msg)
        raise(exc)