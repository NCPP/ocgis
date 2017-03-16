from ocgis.base import AbstractOcgisObject
from ocgis.vm.mpi import get_standard_comm_state


class OcgVM(AbstractOcgisObject):
    """

    :param ompi:
    :type ompi: :class:`ocgis.new_interface.mpi.OcgMpi`
    """

    def __init__(self, ompi=None):
        self._is_finalized = False
        self._live_ranks = None

        self.initialize(ompi=ompi)

    def __del__(self):
        self.finalize()

    @property
    def live_ranks(self):
        assert not self._is_finalized
        return self._live_ranks

    @property
    def root(self):
        return self.live_ranks[0]

    def initialize(self, ompi=None):
        if ompi is None:
            _, _, size = get_standard_comm_state()
            live_ranks = list(range(size))
        else:
            assert ompi.has_updated_dimensions
            live_ranks = ompi.get_empty_ranks(inverse=True)
        self.set_live_ranks(live_ranks)

    def finalize(self):
        self._is_finalized = True
        self._live_ranks = None

    def set_live_ranks(self, ranks):
        self._live_ranks = ranks


ovm = OcgVM()
