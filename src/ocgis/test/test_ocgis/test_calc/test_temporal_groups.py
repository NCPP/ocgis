from ocgis.calc.temporal_groups import SeasonalTemporalGroup, AbstractTemporalGroup
from ocgis.test.base import TestBase
from ocgis.test.base import attr


class TestSeasonalTemporalGroup(TestBase):
    @attr('data')
    def test_init(self):
        actual = [[12, 1, 2], [3, 4, 5], 'unique']
        st = SeasonalTemporalGroup(actual)
        self.assertIsInstances(st, [list, AbstractTemporalGroup])
        self.assertEqual(st, actual)

    def test_icclim_mode(self):
        actual = [[12, 1, 2], [3, 4, 5], 'unique']
        st = SeasonalTemporalGroup(actual)
        self.assertEqual(st.icclim_mode, 'DJF-MAM (unique)')

        actual = [[12, 1, 2], [3, 4, 5]]
        st = SeasonalTemporalGroup(actual)
        self.assertEqual(st.icclim_mode, 'DJF-MAM')
