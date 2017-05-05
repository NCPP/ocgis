import time


class ProcessManager(object):
    """
    Manage multiprocess execution with adaptive checking (if requested).
    
    procs -- Process sequence
    maxprocs -- maximum number of allowable processes
    polling -- the following options are available:
        'adaptive' :: poll the running processes adaptively
              None :: un-delayed while loop checking alive processes
             float :: manually set delay in seconds
    conn -- Pipe object. sends signals:
        1 = get
        2 = complete
    """

    _max_poll = 1
    _poll_interval = 0.001
    _curr_poll = 0.0
    _ii = None

    def __init__(self, procs, maxprocs=1, polling='adaptive'):
        self.procs = procs
        self.maxprocs = maxprocs
        self.polling = polling

    def run(self):
        if len(self.procs) < self.maxprocs:
            rng = len(self.procs)
        else:
            rng = self.maxprocs
        for ii in range(0, rng):
            self.procs[ii].start()
        self._ii = ii + 1
        while True:
            poll = self.poll()
            if poll is False:
                try:
                    self.start()
                except IndexError:
                    self.join()
                    break
                    #        codes = [bool(p.exitcode) for p in self.procs]
                    #        if any(codes):
                    #            raise(RuntimeError('{0} processes had a non-zero exit status.'.format(sum(codes))))

    def alive(self):
        count = sum([p.is_alive() for p in self.procs])
        if count < self.maxprocs:
            ret = False
        else:
            ret = True
        return (ret)

    def start(self):
        self.procs[self._ii].start()
        self._ii += 1

    def join(self):
        for proc in self.procs:
            proc.join()

    def poll(self):
        if self.polling == 'adaptive':
            self.adaptive()
        elif self.polling is None:
            pass
        else:
            time.sleep(self.polling)
        ret = self.alive()
        if ret is False:
            self._curr_poll = 0.0
        return (ret)

    def adaptive(self):
        time.sleep(self._curr_poll)
        self._curr_poll += self._poll_interval
        if self._curr_poll > self._max_poll:
            self._curr_poll = self._max_poll


            # if __name__ == '__main__':
            #    import multiprocessing as mp
            #
            #    def ft(dur):
            #        time.sleep(dur)
            #        print('done!!')
            #
            #    dur = 5
            #    nprocs = 4
            #    maxprocs = 4
            #
            #    procs = [mp.Process(target=ft,args=[dur]) for ii in range(0,nprocs)]
            #    pmanager = ProcessManager(procs,maxprocs)
            #    pmanager.run()
