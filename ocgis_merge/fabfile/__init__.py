from fabric.api import task
from fabric.api import env

import aws
import tasks_system as system
import database
import apache2
import virtualenv
import django_tasks

# Fabric environment parameters
#env.user = 'ubuntu'
#env.hosts = [
#    'localhost',
#    '{user}@{host}'.format(user='ubuntu',host=aws_elastic_ip), # AWS Instance
#]
env.key_filename = '/home/terickson/.ssh/aws_openclimategis/ec2-keypair.pem'

def get_settings_value(filename, section, key):
    '''Retrieve a settings value from a settings INI file'''
    from ConfigParser import RawConfigParser
    config = RawConfigParser()
    config.read(filename)
    
    return config.get(section, key)


@task
def deploy(settings_file):
    '''Deploy OpenClimateGIS System

    Note that if deploying to AWS you need to:
    1. create the AWS instance 
       fab aws.create_aws_instance
    2. update the system
       fab -H ubuntu@PUBLICDNS system.update_system())
    3. restart the AWS image system
    '''
    
    system.install_system_dependencies()
    
    database.install_postgresql()
    database.install_psycopg2()
    
    database.create_postgresql_user(
        username=get_settings_value(settings_file, 'database', 'DATABASE_USER'),
        password=get_settings_value(settings_file, 'database', 'DATABASE_PASSWORD'),
    )
    database.install_postgis()
    database.create_postgis_template_db()
    database.create_openclimategis_db(
        databasename=get_settings_value(settings_file, 'database', 'DATABASE_NAME'),
        owner=get_settings_value(settings_file, 'database', 'DATABASE_USER'),
    )
    
    django.install_openclimategis_django()
    django.copy_django_settings_config(settings_file)
    django.syncdb()
    django.register_archive_usgs_cida_maurer()
    
    apache2.install()
    apache2.config_openclimategis()
