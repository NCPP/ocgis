==============
OpenClimateGIS
==============

OpenClimateGIS is a Python-based web server that distributes climate model data
in geospatial (vector) formats.

------------
Dependencies
------------

* PostgreSQL_ - a client-server relational database
* PostGIS_ - adds geospatial functionality to PostgreSQL databases
* psycopg2_ - a Python interface for working with PostgreSQL databases
* numpy_ - a multi-dimensionsal data library for Python
* netcdf4-python_ - a Python interface for working with netCDF4 files
* pyKML_ - a library for manipulating KML documents

.. _PostgreSQL: http://www.postgresql.org/
.. _PostGIS: http://postgis.refractions.net/
.. _psycopg2: http://initd.org/psycopg/
.. _numpy: http://numpy.scipy.org/
.. _netcdf4-python: http://code.google.com/p/netcdf4-python/
.. _pyKML: http://pypi.python.org/pypi/pykml/

------------
Installation
------------

The following instructions for installing OpenClimateGIS on Ubuntu 10.04 
running on Amazon's `Elastic Compute Cloud (EC2)`_, a component of 
`Amazon Web Services (AWS)`_.

.. _Elastic Compute Cloud (EC2): http://aws.amazon.com/ec2/
.. _Amazon Web Services (AWS): http://aws.amazon.com/

~~~~~~~~~~~~~~~~~~~~~~~~
Creating an AWS Instance
~~~~~~~~~~~~~~~~~~~~~~~~

Although it is not required, installing OpenClimateGIS on an AWS has the 
benefits having an isolate instance with specific library versions and the
ability to easily scale to multiple servers in the future.

An EC2 instance can be created from within Python, using boto_, a Python 
interface to Amazon Web Services.  The following is an example script that
creates an EC2 instance and returns the Public DNS.
Note that this assumes that you have set the AWS_ACCESS_KEY_ID and 
AWS_SECRET_ACCESS_KEY environment variables as described in the boto 
documentation.::

    from time import sleep
    from boto.ec2.connection import EC2Connection
    conn = EC2Connection()

    # start an instance of Ubuntu 10.04
    ami_ubuntu10_04 = conn.get_all_images(image_ids=['ami-3202f25b'])
    reservation = ami_ubuntu10_04[0].run( \
        key_name='ec2-keypair', \
        security_groups=['OCG_group'], \
        instance_type='m1.large', \
    )
    instance = reservation.instances[0]

    while instance.state!=u'running':
        print("Instance state = {0}".format(instance.state))
        instance.update()
        sleep(10)

    print "Instance state = {0}".format(instance.state)

    instance.add_tag('Name','OpenClimateGIS')

    print "DNS="
    print instance.dns_name

Once you configured the Security Group for the instance to allow access on 
port 22 and created an public/private key pair (See: 'AWS Security Credentials`_)
you can connect to the instance using ssh::

    ssh -i ~/.ssh/aws_openclimategis/ec2-keypair.pem ubuntu@DNSNAME

.. _boto: http://code.google.com/p/boto/
.. _AWS Security Credentials: https://aws-portal.amazon.com/gp/aws/developer/account/index.html?action=access-key

~~~~~~~~~~~~~~~~~~~~~~~~~
Installing OpenClimateGIS
~~~~~~~~~~~~~~~~~~~~~~~~~

The dependencies for OpenClimateGIS are installed via a series of bash scripts.
The main install script (INSTALL.sh) calls numerous other scripts found in the
install_scripts directory.  To download (clone) the respository, including the
install script::

    # install version control tools
    sudo apt-get install git-core
    
    # clone the OpenClimateGIS repository
    git clone http://github.com/tylere/OpenClimateGIS.git
    
    # run the script to install the dependencies
    cd OpenClimateGIS
    chmod u+x INSTALL.sh
    ./INSTALL.sh

------------
Source Code
------------

The source code for OpenClimateGIS is available at::

    https://github.com/tylere/OpenClimateGIS

