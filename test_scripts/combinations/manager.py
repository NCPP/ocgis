import db
import os
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import MetaData
from sqlalchemy.orm.session import sessionmaker
import subprocess


def make_load_test_data():
    ## the output database
    db_path = '/home/local/WX/ben.koziol/links/git/test_scripts/combinations/data.sqlite'
    ## where to write the subset data
    subset_path = '/home/local/WX/ben.koziol/climate_data/subset'
    ## remove all data in the subset folder
    subprocess.call(['rm',subset_path+'/*'])
    ## prefix to apply to subset data
    subset_prefix = 'subset'
    ## try to remove the original database
    try:
        os.remove(db_path)
    except OSError:
        if not os.path.exists(db_path):
            pass
        else:
            raise
    ## four slashes for absolute paths - three for relative
    connstr = 'sqlite:///{0}'.format(db_path)
    ## create the database tables
    engine = create_engine(connstr)
    db.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
#    metadata = MetaData(bind=engine)
    data_dir = '/usr/local/climate_data'
    for (dirpath,dirnames,filenames) in os.walk(data_dir,followlinks=True):
        if 'hostetler' in dirpath: continue
        for filename in filenames:
            try:
                print(dirpath,filename)
                data = db.OriginalTestData(dirpath,filename)
#                subset_data = db.SubsetTestData(data,subset_path,subset_prefix)
            except RuntimeError:
                if not filename.endswith('nc'):
                    continue
                else:
                    raise
            session.add(data)
#            session.add(subset_data)
            session.commit()
    session.close()
#            import ipdb;ipdb.set_trace()

def run_combinations():
    nretries = 100
    curr_retry = 0
    combo = 92159
    combo_log = '../test_scripts/combinations/combination.log'
    
    while curr_retry <= nretries:
        try:
            run_subprocess(run_subprocess(combo))
        except subprocess.CalledProcessError:
            curr_retry += 1
            with open(combo_log,'r') as f:
                lines = f.readlines()
                combo = lines[-1].strip()
    if curr_retry >= nretries:
        raise(RuntimeError('maximum number of retries reached'))
            
def run_subprocess(combo):
    subprocess.check_call(['python','../test_scripts/combinations/run.py','-c',str(combo)])

if __name__ == '__main__':
#    make_load_test_data()
    run_combinations()