# WSGI configuration for running Django projects
# see: http://docs.djangoproject.com/en/dev/howto/deployment/modwsgi/

import os
import sys

os.environ['DJANGO_SETTINGS_MODULE'] = 'openclimategis.settings'

paths = [
    # production (AWS) paths
    '/home/ubuntu/.virtualenvs/openclimategis/lib/python2.6/site-packages',
    '/home/ubuntu/.virtualenvs/openclimategis/src/openclimategis/src',
    '/home/ubuntu/.virtualenvs/openclimategis/src/openclimategis/src/openclimategis',
    '/home/ubuntu/.virtualenvs/openclimategis/src/piston',
    ## Tyler's development paths
    #'/home/terickson/.virtualenvs/openclimategis/lib/python2.6/site-packages',
    ##'/home/terickson/.virtualenvs/openclimategis/src/openclimategis/src',
    #'/home/terickson/Dropbox/project/openclimategis/git/OpenClimateGIS/src',
    ##'/home/terickson/.virtualenvs/openclimategis/src/openclimategis/src/openclimategis',
    #'/home/terickson/Dropbox/project/openclimategis/git/OpenClimateGIS/src/openclimategis',
    #'/home/terickson/.virtualenvs/openclimategis/src/piston',
]
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

import django.core.handlers.wsgi
application = django.core.handlers.wsgi.WSGIHandler()

print >> sys.stderr, '############ DEBUG from django.wsgi config ##########'
print >> sys.stderr, '# sys.path ='
print >> sys.stderr, sys.path
