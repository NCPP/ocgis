from ocgis.calc import base
from ocgis.util.helpers import iter_array
import numpy as np


class FreezeThawCycles(base.AbstractUnivariateSetFunction, base.AbstractParameterizedFunction):
    key = 'freezethawcycles'
    description = "Number of freeze thaw cycles, where freezing and thawing occurs once a threshold of degree days below or above 0C is reached."
    long_name = "Number of freeze thaw cycles"
    standard_name = "Number of freeze thaw cycles"
    required_units = ['kelvin', 'K']

    parms_definition = {'threshold': float}

    def calculate(self, values, threshold=15):

        # TODO: Check temporal resolution

        assert (len(values.shape) == 3)

        # storage array for counts
        shp_out = values.shape[-2:]
        out = np.zeros(shp_out, dtype=int).flatten()

        for ii, (rowidx, colidx) in enumerate(iter_array(values[0, :, :], use_mask=True)):
            x = values[:, rowidx, colidx].reshape(-1)
            out[ii] = freezethaw1d(x, threshold)

        out.resize(shp_out)

        # update the output mask. this only applies to geometries so pick the
        # first masked time field
        out = np.ma.array(out, mask=values.mask[0, :, :])
        return out


def freezethaw1d(tas, threshold):
    """
    Return the number of freeze-thaw cycles.

    Parameters
    ----------
    tas : array
      The daily temperature series (Kelvins).
    threshold : float
      The threshold in degree-days above or below the freezing point at
      which we consider the soil thawed or frozen.

    Returns
    -------
    out : int
      The number of times the medium thawed or froze, so that a complete
      freeze-thaw cycle corresponds to a count of 2.
    """

    # Ignore masked values.
    if hasattr(tas, 'mask'):
        if tas.mask.all():
            return np.nan
        else:
            tas = tas.compressed()

    # Convert the units to C
    x = tas - 273.15

    # Compute the cumulative degree days relative to the freezing point.
    cx = np.cumsum(x)

    # Find the places where the temperature crosses the freezing point (FP).
    over = x >= 0
    cross = [0, ] + np.nonzero(np.diff(over) != 0)[0].tolist()

    cycles = [0, ]
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

    # Remove the artificial cycle starting at 0.
    cycles.pop(0)

    # Return the number of transitions from frozen to thawed or vice-versa
    return len(cycles)-1
