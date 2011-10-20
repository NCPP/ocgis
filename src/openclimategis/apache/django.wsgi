# WSGI configuration for running Django projects
# see: http://docs.djangoproject.com/en/dev/howto/deployment/modwsgi/

import os
import sys

## add the path of the Django version
#sys.path.append('/usr/local/django/1.3')

os.environ['DJANGO_SETTINGS_MODULE'] = 'openclimategis.settings_production'

# append the django project root directory to the python path
path = '/home/ubuntu/.virtualenvs/openclimategis/src/openclimategis/src/openclimategis'
if path not in sys.path:
    sys.path.append(path)

import django.core.handlers.wsgi
application = django.core.handlers.wsgi.WSGIHandler()
