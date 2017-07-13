import logging

import numpy as np

from ocgis.base import get_variable_names, orphaned
from ocgis.calc.base import AbstractMultivariateFunction
from ocgis.calc.eval_function import EvalFunction, MultivariateEvalFunction
from ocgis.util.logging_ocgis import ocgis_lh


class CalculationEngine(object):
    """
    Manages calculation execution.
    
    :type grouping: list of temporal groupings (e.g. ['month','year'])
    :type funcs: :class:`list` of `function dictionaries`
    :param bool calc_sample_size: If ``True``, calculation sample sizes for the calculations.
    :param progress:  A progress object to update.
    :type progress: :class:`~ocgis.util.logging_ocgis.ProgressOcgOperations`
    """

    def __init__(self, grouping, funcs, calc_sample_size=False, spatial_aggregation=False, progress=None):
        self.grouping = grouping
        self.funcs = funcs
        self.calc_sample_size = calc_sample_size
        self.spatial_aggregation = spatial_aggregation

        self._tgds = {}
        self._progress = progress

    @property
    def has_multivariate_functions(self):
        multivariate_classes = [AbstractMultivariateFunction, MultivariateEvalFunction]
        return any([self._check_calculation_members_(self.funcs, k) for k in multivariate_classes])

    @staticmethod
    def _check_calculation_members_(funcs, klass):
        """
        Return True if a subclass of type `klass` is contained in the calculation
        list.

        :param funcs: Sequence of calculation dictionaries.
        :param klass: `ocgis.calc.base.OcgFunction`
        """
        check = [issubclass(f['ref'], klass) for f in funcs]
        ret = True if any(check) else False
        return ret

    def execute(self, coll, file_only=False, tgds=None):
        """
        :param :class:~`ocgis.SpatialCollection` coll:
        :param bool file_only:
        :param dict tgds: {'field_alias': :class:`ocgis.interface.base.dimension.temporal.TemporalGroupDimension`,...}
        """
        from ocgis import VariableCollection

        # Select which dictionary will hold the temporal group dimensions.
        if tgds is None:
            tgds_to_use = self._tgds
            tgds_overloaded = False
        else:
            tgds_to_use = tgds
            tgds_overloaded = True

        # Group the variables. If grouping is None, calculations are performed on each element.
        if self.grouping is not None:
            ocgis_lh('Setting temporal groups: {0}'.format(self.grouping), 'calc.engine')
            for field in coll.iter_fields():
                if tgds_overloaded:
                    assert field.name in tgds_to_use
                else:
                    if field.name not in tgds_to_use:
                        tgds_to_use[field.name] = field.time.get_grouping(self.grouping)

        # Iterate over functions.
        for ugid, container in list(coll.children.items()):
            for field_name, field in list(container.children.items()):
                new_temporal = tgds_to_use.get(field_name)
                if new_temporal is not None:
                    new_temporal = new_temporal.copy()
                # If the engine has a grouping, ensure it is equivalent to the new temporal dimension.
                if self.grouping is not None:
                    try:
                        compare = set(new_temporal.grouping) == set(self.grouping)
                    # Types may be unhashable, compare directly.
                    except TypeError:
                        compare = new_temporal.grouping == self.grouping
                    if not compare:
                        msg = 'Engine temporal grouping and field temporal grouping are not equivalent. Perhaps ' \
                              'optimizations are incorrect?'
                        ocgis_lh(logger='calc.engine', exc=ValueError(msg))

                out_vc = VariableCollection()

                for f in self.funcs:
                    try:
                        ocgis_lh('Calculating: {0}'.format(f['func']), logger='calc.engine')
                        # Initialize the function.
                        function = f['ref'](alias=f['name'], dtype=None, field=field, file_only=file_only, vc=out_vc,
                                            parms=f['kwds'], tgd=new_temporal, calc_sample_size=self.calc_sample_size,
                                            meta_attrs=f.get('meta_attrs'),
                                            spatial_aggregation=self.spatial_aggregation)
                        # Allow a calculation to create a temporal aggregation after initialization.
                        if new_temporal is None and function.tgd is not None:
                            new_temporal = function.tgd.extract()
                    except KeyError:
                        # Likely an eval function which does not have the name key.
                        function = EvalFunction(field=field, file_only=file_only, vc=out_vc,
                                                expr=self.funcs[0]['func'], meta_attrs=self.funcs[0].get('meta_attrs'))

                    ocgis_lh('calculation initialized', logger='calc.engine', level=logging.DEBUG)

                    # Return the variable collection from the calculations.
                    out_vc = function.execute()

                    for dv in out_vc.values():
                        # Any outgoing variables from a calculation must have an associated data type.
                        try:
                            assert dv.dtype is not None
                        except AssertionError:
                            assert isinstance(dv.dtype, np.dtype)
                        # If this is a file only operation, there should be no computed values.
                        if file_only:
                            assert dv._value is None

                    ocgis_lh('calculation finished', logger='calc.engine', level=logging.DEBUG)

                    # Try to mark progress. Okay if it is not there.
                    try:
                        self._progress.mark()
                    except AttributeError:
                        pass

                out_field = function.field.copy()
                function_tag = function.tag

                # Format the returned field. Doing things like removing original data variables and modifying the
                # time dimension if necessary. Field functions handle all field modifications on their own, so bypass
                # in that case.
                if new_temporal is not None:
                    new_temporal = new_temporal.extract()
                format_return_field(function_tag, out_field, new_temporal=new_temporal)

                # Add the calculation variables.
                for variable in list(out_vc.values()):
                    with orphaned(variable):
                        out_field.add_variable(variable)
                # Tag the calculation data as data variables.
                out_field.append_to_tags(function_tag, list(out_vc.keys()))

                coll.children[ugid].children[field_name] = out_field
        return coll


def format_return_field(function_tag, out_field, new_temporal=None):
    # Remove the variables used by the calculation.
    try:
        to_remove = get_variable_names(out_field.get_by_tag(function_tag))
    except KeyError:
        # Let this fail quietly as the tag may not exist on incoming fields.
        pass
    else:
        for tr in to_remove:
            out_field.remove_variable(tr)

    # Remove the original time variable and replace with the new one if there is a new time dimension. New
    # time dimensions may not be present for calculations that do not compute one.
    if new_temporal is not None:
        out_field.remove_variable(out_field.time)
        out_field.set_time(new_temporal, force=True)
