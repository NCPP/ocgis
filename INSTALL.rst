============================
OpenClimateGIS INSTALL Steps
============================

1. Install the non-Python dependencies::

    ./install_dependencies.sh

2. Create a Python Virtual Environment

    mkvirtualenv --no-site-packages openclimategis

3. Install the Python dependencies into a virtual environment::

    pip -E openclimategis install -r http://  /openclimategis/requirements4pip.txt (TODO: update URL once posted online)

4. Create a database user for the Django project

    DBOWNER=openclimategis_user
    sudo su -c "createuser --no-superuser --no-createrole --createdb --pwprompt $DBOWNER" - postgres

5. Create a database for the Django project::

    DBNAME=openclimategis_sql
    sudo su -c "createdb $DBNAME -T $POSTGIS_TEMPLATE" - postgres
    sudo -u postgres psql -d postgres -c "ALTER DATABASE $DBNAME OWNER TO $DBOWNER;"

6. Create the settings.ini configuration file and set passwords, etc.

    mkdir /etc/openclimategis
    sudo cp src/openclimategis/settings.ini.TEMPLATE /etc/openclimategis/settings.ini
    sudo nano /etc/openclimategis/settings.ini

7. Test the Django project::

    ./manage.py test
