from ocgis.contrib.library_icclim import AbstractIcclimFunction


class MockAbstractIcclimFunction(AbstractIcclimFunction):
    key = 'icclim_fillme'

    def __init__(self, field, tgd):
        self.field = field
        self.tgd = tgd
