import os

import sys
from pstats import Stats

profile_dat = os.path.expanduser('/home/benkoziol/.PyCharm50/system/snapshots/ocgis9.pstat')


def profile_target():
    argv = [sys.argv[0], ]


# cProfile.run('profile_target()', filename=os.path.expanduser(profile_dat))

stats = Stats(profile_dat)
stats.strip_dirs()
stats.sort_stats('time', 'name')
stats.print_stats(0.01)
# stats.print_callers(0.01)
