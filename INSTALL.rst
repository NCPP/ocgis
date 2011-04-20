============================
OpenClimateGIS INSTALL Steps
============================

1. Install the non-Python geospatial dependencies::

    source install_geospatial_dependencies.sh

2. Install the database dependencies::

    source install_database_dependencies.sh

3. Create a Python Virtual Environment

    VIRTUALENVNAME=openclimategis-test
    mkvirtualenv --no-site-packages $VIRTUALENVNAME

4. Create symbolic links in the Python Virtual Environment

    source install_create_virtualenv_symbolic_links.sh

5. Install the Python dependencies into a virtual environment::

    pip -E $VIRTUALENVNAME install git+http://github.com/tylere/OpenClimateGIS

6. Create a database user for the Django project

    DBOWNER=openclimategis_user
    sudo su -c "createuser --no-superuser --no-createrole --createdb --pwprompt $DBOWNER" - postgres

7. Create a database for the Django project::

    DBNAME=openclimategis_sql
    sudo su -c "createdb $DBNAME -T $POSTGIS_TEMPLATE" - postgres
    sudo -u postgres psql -d postgres -c "ALTER DATABASE $DBNAME OWNER TO $DBOWNER;"

8. Create the settings.ini configuration file and set passwords, etc.

    mkdir /etc/openclimategis
    sudo cp src/openclimategis/settings.ini.TEMPLATE /etc/openclimategis/settings.ini
    sudo nano /etc/openclimategis/settings.ini

9. Test the Django project::

    ./manage.py test
