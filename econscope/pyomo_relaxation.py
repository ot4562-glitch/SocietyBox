"""Explicit original MILP versus original Pyomo continuous relaxation comparison."""
from fractions import Fraction
import math

from . import highs_native, pyomo_model


def validate(spec):
    pyomo_model.relaxed_spec(spec)
    return spec


def check_transformation(original, relaxed):
    expected = pyomo_model.relaxed_spec(original)
    pyomo_model.check_representation(expected, relaxed)


def compare_results(spec, integer, relaxed):
    """Independent exact models establish each status and the bound direction."""
    validate(spec)
    continuous = pyomo_model.relaxed_spec(spec)
    first = highs_native.compare(spec, integer)
    second = highs_native.compare(continuous, relaxed)
    i, r = integer['status'], relaxed['status']
    result = {'status': 'BOTH_INFEASIBLE', 'integer_objective': integer['objective'],
              'relaxed_objective': relaxed['objective'], 'lower_bound': None, 'upper_bound': None,
              'gap_decimal': None, 'gap_exact': None, 'sense': spec['sense'],
              'integer_check': first, 'relaxation_check': second,
              'interpretation': 'Declared optimization bound; different feasible sets, not equal-model parity or empirical effect.'}
    if i == 'OPTIMAL' and r != 'OPTIMAL': raise ValueError('RELAXATION_FEASIBILITY_CONTRADICTION')
    if i != 'OPTIMAL' and r == 'OPTIMAL':
        result['status'] = 'INTEGER_INFEASIBLE_RELAXATION_FEASIBLE'
        result['upper_bound' if spec['sense'] == 'maximize' else 'lower_bound'] = relaxed['objective']
    elif i == r == 'OPTIMAL':
        exact_i = Fraction(first['oracle']['objective_exact'])
        exact_r = Fraction(second['oracle']['objective_exact'])
        exact_gap = exact_r-exact_i if spec['sense'] == 'maximize' else exact_i-exact_r
        gap = relaxed['objective']-integer['objective'] if spec['sense'] == 'maximize' else integer['objective']-relaxed['objective']
        if exact_gap < 0 or not math.isclose(gap, float(exact_gap), abs_tol=highs_native.ABS_TOL, rel_tol=highs_native.REL_TOL):
            raise ValueError('RELAXATION_BOUND_OR_GAP_MISMATCH')
        result.update(status='OPTIMAL_BOUND_COMPARISON', gap_decimal=gap, gap_exact=str(exact_gap),
                      lower_bound=integer['objective'] if spec['sense'] == 'maximize' else relaxed['objective'],
                      upper_bound=relaxed['objective'] if spec['sense'] == 'maximize' else integer['objective'])
    return result


def run(spec, sources, runtime):
    validate(spec)
    # One pinned original Pyomo worker builds the integer model, records its
    # representation, clones it and relaxes only the clone. Do not repeat the
    # archive audit/extraction/worker startup for the unchanged original.
    relaxed_model = pyomo_model.canonicalize(spec, sources, transformation='RELAX_INTEGER_VARS')
    original_canonical = relaxed_model['transformation']['original_before']
    pyomo_model.check_representation(spec, original_canonical)
    check_transformation(spec, relaxed_model['canonical'])
    original_model = {key: value for key, value in relaxed_model.items() if key != 'transformation'}
    original_model.update(canonical=original_canonical,
                          native_program=pyomo_model._native_encoding(spec, original_canonical),
                          derivation='PRESERVED_ORIGINAL_FROM_VERIFIED_CLONE_TRANSFORM')
    integer = highs_native.run(original_model['native_program'], runtime)
    continuous = highs_native.run(relaxed_model['native_program'], runtime)
    comparison = compare_results(spec, integer['actual'], continuous['actual'])
    return {'engine': 'PYOMO_INTEGER_RELAXATION_HIGHS', 'mode': 'NO_LLM',
            'native_execution': 'EXECUTED', 'native_parity': 'EACH_MODEL_WITHIN_DECLARED_ORACLE_TOLERANCE',
            'comparison': comparison, 'models': {'integer': original_model, 'relaxation': relaxed_model},
            'runs': {'integer': integer, 'relaxation': continuous}, 'os_isolation': 'NOT_ATTESTED',
            'assumptions': ['Original integer model is preserved; only a cloned model is relaxed.',
                'Both variants retain the same objective, bounds and constraint coefficients.',
                'Relaxed fractional decisions are not feasible integer recommendations.',
                'No causal identification, empirical policy effect or full upstream API coverage.']}
