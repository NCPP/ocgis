from ocgis import env
from ocgis.calc import base
from ocgis.util.helpers import iter_array
from ocgis.util.units import get_are_units_equal_by_string_or_cfunits
import numpy as np
import datetime as dt

class FreezeThaw(base.AbstractUnivariateSetFunction, base.AbstractParameterizedFunction):
    key = 'freezethaw'
    description = "Number of freeze-thaw events, where freezing and thawing occurs once a threshold of degree days below or above 0C is reached. A complete cycle (freeze-thaw-freeze) will return a value of 2. "
    long_name = "Number of freeze-thaw events"
    standard_name = "freeze-thaw"
    required_units = ['K', 'C']

    parms_definition = {'threshold': float}

    def calculate(self, values, threshold=15):
        """
        Return the number of freeze-thaw transitions. A value of 2 corresponds
        to a complete cycle (frozen-thawed-frozen).

        :param threshold: The number of degree-days above or below the freezing point after which the ground is considered frozen or thawed.
        """

        assert (len(values.shape) == 3)

        # Check temporal resolution
        t = self.field['time'].get_value()[:2]
        step = t[1] - t[0]
        if type(step) == dt.timedelta:
            assert step == dt.timedelta(days=1)
        else:
            if self.field['time'].units.startswith('days'):
                assert step == 1
            else:
                raise NotImplementedError



        # Unit conversion
        units = self.field.data_variables[0].units
        if get_are_units_equal_by_string_or_cfunits(units, 'C',
                                            try_cfunits=env.USE_CFUNITS):
            tas = values
        elif get_are_units_equal_by_string_or_cfunits(units, 'K',
                                                      try_cfunits=env.USE_CFUNITS):
            tas = values - 273.15

        out = np.apply_along_axis(freezethaw1d, 0, tas, threshold=threshold)
        return np.ma.masked_invalid(out)


def freezethaw1d(x, threshold):
    """
    Return the number of freeze-thaw transitions.

    Parameters
    ----------
    x : ndarray
      The daily temperature series (C).
    threshold : float
      The threshold in degree-days above or below the freezing point at
      which we consider the soil thawed or frozen.

    Returns
    -------
    out : int
      The number of times the medium thawed or froze, so that a complete
      freeze-thaw cycle corresponds to a count of 2.
    """

    # Masked values are just compressed.
    if isinstance(x, np.ma.MaskedArray):
        if x.mask.all():
            return np.nan
        else:
            x = x.compressed()

    # This avoids issues when the threshold is reached right at the first value.
    x = np.concatenate([[0,], x])

    # Compute the cumulative degree days relative to the freezing point.
    cx = np.cumsum(x)

    # Find the places where the temperature crosses the freezing point (FP).
    over = x >= 0
    cross = np.concatenate([[0,], np.nonzero(np.diff(over) != 0)[0]])

    cycles = [0,]
    for ci in cross:

        # Skip FP crossing if it occurs before the threshold is reached.
        if ci < np.abs(cycles[-1]):
            continue

        # Otherwise reset the cumulative sum starting from the crossing.
        d = cx[ci:] - cx[ci]

        # Find the first place where t is exceeded (from above or below)
        w = np.nonzero(np.abs(d) >= threshold)[0]
        if len(w) > 0:
            w = w[0]  # <-- This is where it occurs.
            s = np.sign(d[w])  # <-- This indicates if its thawing or freezing.

            # Test for the alternance of freeze and thaw events.
            # Store only an event if its different from the last.
            if s != np.sign(cycles[-1]):
                cycles.append(s * (w + ci))

    # Return the number of transitions from frozen to thawed or vice-versa
    return float(len(cycles) - 2) # There are two "artificial" transitions
