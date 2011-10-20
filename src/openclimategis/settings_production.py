import os

# production-specific Django settings for the OpenClimateGIS project
from settings import *

DEBUG = True  # set to false to load large shapefiles
TEMPLATE_DEBUG = DEBUG

# This allows us to construct the needed absolute paths dynamically,
# e.g., for the GIS_DATA_DIR, MEDIA_ROOT, and TEMPLATE_DIRS settings.
DJANGO_APP_DIR = os.path.dirname(__file__)

DATABASES = {
    'default': {
        'ENGINE':   'django.contrib.gis.db.backends.postgis', 
        'NAME':     config.get('database-production', 'DATABASE_NAME'),
        'USER':     config.get('database-production', 'DATABASE_USER'),
        'PASSWORD': config.get('database-production', 'DATABASE_PASSWORD'),
        'HOST':     config.get('database-production', 'DATABASE_HOST'),
        'PORT':     config.get('database-production', 'DATABASE_PORT'),
    }
}
POSTGIS_VERSION = (1, 5, 2)
POSTGIS_TEMPLATE = 'postgis-1.5.2-template'

# Absolute path to the directory that holds media.
# Example: "/home/media/media.lawrence.com/"
MEDIA_ROOT = os.path.join(DJANGO_APP_DIR,'media/')

# URL that handles the media served from MEDIA_ROOT. Make sure to use a
# trailing slash if there is a path component (optional in other cases).
# Examples: "http://media.lawrence.com", "http://example.com/media/"
MEDIA_URL = '/media/'

# SERVE STATIC FILES from the path to media files
# 'site_media' to STATIC_DOC_ROOT
STATIC_DOC_ROOT = 'media'

# URL prefix for admin media -- CSS, JavaScript and images. Make sure to use a
# trailing slash.
# Examples: "http://foo.com/media/", "/media/".
ADMIN_MEDIA_PREFIX = '/admin_media/'