# Django settings for openclimategis project.

# Python's ConfigParser is used to access a configuration file in the /etc
# directory, so that sensitive information is not stored under version control.
# References:
# http://code.djangoproject.com/wiki/SplitSettings#ini-stylefilefordeployment
# http://docs.python.org/library/configparser.html
import os

from ConfigParser import RawConfigParser
config = RawConfigParser()
config.read('/etc/openclimategis/settings.ini')
logfile = config.get('logging', 'LOG_FILENAME')

## settings for parallel execution. set to greater than one for in-parallel
## execution.
MAXPROCESSES = 8
MAXPROCESSES_PER_POLY = 2

DEBUG = config.getboolean('debug','DEBUG')
TEMPLATE_DEBUG = config.getboolean('debug','TEMPLATE_DEBUG')

ADMINS = (
    tuple(config.items('error mail'))
)

MANAGERS = tuple(config.items('404 mail'))

DATABASES = {
    'default': {
        'ENGINE':   'django.contrib.gis.db.backends.postgis', 
        'NAME':     config.get('database', 'DATABASE_NAME'),
        'USER':     config.get('database', 'DATABASE_USER'),
        'PASSWORD': config.get('database', 'DATABASE_PASSWORD'),
        'HOST':     config.get('database', 'DATABASE_HOST'),
        'PORT':     config.get('database', 'DATABASE_PORT'),
    }
}
POSTGIS_VERSION = (1, 5, 2)
POSTGIS_TEMPLATE = 'postgis-1.5.2-template'

# Use the GeoDjango Test Suite Runner to test the geospatial dependencies
TEST_RUNNER = 'django.test.simple.DjangoTestSuiteRunner'
#TEST_RUNNER = 'django.contrib.gis.tests.GeoDjangoTestSuiteRunner'

#GEOS_LIBRARY_PATH = '/home/bob/local/lib/libgeos_c.so'
#GDAL_LIBRARY_PATH = '/home/sue/local/lib/libgdal.so'

# Local time zone for this installation. Choices can be found here:
# http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
# although not all choices may be available on all operating systems.
# On Unix systems, a value of None will cause Django to use the same
# timezone as the operating system.
# If running in a Windows environment this must be set to the same as your
# system time zone.
TIME_ZONE = 'UTC'

# Language code for this installation. All choices can be found here:
# http://www.i18nguy.com/unicode/language-identifiers.html
LANGUAGE_CODE = 'en-us'

SITE_ID = 1

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = True

# If you set this to False, Django will not format dates, numbers and
# calendars according to the current locale
USE_L10N = True

# Absolute filesystem path to the directory that will hold user-uploaded files.
# Example: "/home/media/media.lawrence.com/"
MEDIA_ROOT = config.get('media','MEDIA_ROOT')

# location where APACHE serves non project specific static file
APACHE_STATIC_ROOT = config.get('media','APACHE_ROOT')

# URL that handles the media served from MEDIA_ROOT. Make sure to use a
# trailing slash if there is a path component (optional in other cases).
# Examples: "http://media.lawrence.com", "http://example.com/media/"
MEDIA_URL = ''

# URL prefix for admin media -- CSS, JavaScript and images. Make sure to use a
# trailing slash.
# Examples: "http://foo.com/media/", "/media/".
ADMIN_MEDIA_PREFIX = '/admin_media/'

# Make this unique, and don't share it with anybody.
SECRET_KEY = config.get('secrets','SECRET_KEY')

# List of callables that know how to import templates from various sources.
TEMPLATE_LOADERS = (
    'django.template.loaders.filesystem.Loader',
    'django.template.loaders.app_directories.Loader',
#     'django.template.loaders.eggs.Loader',
)

MIDDLEWARE_CLASSES = (
    'django.middleware.common.CommonMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
)

ROOT_URLCONF = 'openclimategis.urls'

TEMPLATE_DIRS = (
    # Put strings here, like "/home/html/django_templates" or "C:/www/django/templates".
    # Always use forward slashes, even on Windows.
    # Don't forget to use absolute paths, not relative paths.
)

INSTALLED_APPS = (
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.sites',
    'django.contrib.messages',
    'django.contrib.gis',
    'django.contrib.admin',
    'django.contrib.admindocs',
    #'django_extensions', # enables extra django admin commands 
    'climatedata',
    'api',
)

# configure logging
PROJECT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(PROJECT_DIR)
LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'file':{
            'level':'DEBUG',
            'class':'logging.FileHandler',
            #'filename':os.path.join(PARENT_DIR, 'django.log'),
            'filename':config.get('logging', 'LOG_FILENAME'),
            'formatter': 'default',
        },
        'console':{
            'level':'DEBUG',
            'class':'logging.StreamHandler',
            'formatter': 'default',
        },
    },
    'loggers': {
        '': { # default logger
            'handlers':['file',],
            'propagate': True,
            'level':'DEBUG',
        },
        'django.request': {
            'handlers':['console',],
            'propagate': False,
            'level': 'DEBUG',
        },
    }
}
