from fabric.api import task
from time import sleep
from boto.ec2.connection import EC2Connection

# Amazon Web Services parameters
aws_elastic_ip = '107.22.251.99'
USER = 'ubuntu'

@task
def create_aws_instance():
    '''Initialize an AWS instance'''
    
    print('Creating an AWS instance...')
    conn = EC2Connection()
    # start an instance of Ubuntu 10.04
    ami_ubuntu10_04 = conn.get_all_images(image_ids=['ami-3202f25b'])
    reservation = ami_ubuntu10_04[0].run( \
        key_name='ec2-keypair', \
        security_groups=['OCG_group'], \
        instance_type='m1.large', \
    )
    instance = reservation.instances[0]
    sleep(1)
    while instance.state!=u'running':
        print("Instance state = {0}".format(instance.state))
        instance.update()
        sleep(5)
    print("Instance state = {0}".format(instance.state))
    sleep(5)
    
    # add a tag to name the instance
    instance.add_tag('Name','OpenClimateGIS')
    
    print("DNS = {0}".format(instance.dns_name))
    print('ID  = {0}'.format(instance.id))
    
    return instance.id

def get_instance(instance_id):
    '''Retrieves an AWS instance'''
    conn = EC2Connection()
    reservation = conn.get_all_instances(instance_ids=[instance_id])[0]
    return reservation.instances[0]

@task
def reboot_instance(instance_id):
    '''Reboot an AWS instance'''
    instance = get_instance(instance_id)
    instance.reboot()

@task
def associate_address(id, ip):
    '''Associate an IP address with an AWS instance'''
    conn = EC2Connection()
    instance = get_instance(id)
    if conn.associate_address(instance.id, ip):
        print('Success. Instance is now available at:{0}'.format(ip))
    else:
        print('Failed. Unable to associate instance with the IP.')