# WSGI configuration for running Django projects
# see: http://docs.djangoproject.com/en/dev/howto/deployment/modwsgi/

import os
import sys

os.environ['DJANGO_SETTINGS_MODULE'] = 'openclimategis.settings_production'

paths = [
    '/home/ubuntu/.virtualenvs/openclimategis/lib/python2.6/site-packages',
    '/home/ubuntu/.virtualenvs/openclimategis/src/openclimategis/src',
    '/home/ubuntu/.virtualenvs/openclimategis/src/openclimategis/src/openclimategis',
]
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

import django.core.handlers.wsgi
application = django.core.handlers.wsgi.WSGIHandler()
