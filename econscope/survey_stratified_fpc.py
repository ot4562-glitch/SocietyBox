"""First-stage finite-population correction for a declared PSU SRSWOR design.

Uses the exact, uncorrected PSU Taylor components from survey_stratified_mean
as intermediate arithmetic, then applies (N_h - m_h)/N_h per stratum. This is
conditional on equal-probability SRS without replacement of PSUs, complete
observation within selected PSUs, and correctly specified survey weights.
Neither the PSU population frame nor these assumptions are authenticated.
For PPS, second-stage subsampling, uncertain frame size or lonely PSU strata,
this model is unsupported: do not silently apply this correction.
"""
from fractions import Fraction as F
import hashlib
import json

from .execution_control import checkpoint
from .survey_stratified_mean import SCHEMA as WR_SCHEMA, stratified_psu_mean

SCHEMA = 'societybox-survey-psu-srswor-mean-v1'
DESIGN = 'EQUAL_PROBABILITY_PSU_SRSWOR_CENSUS_WITHIN_SELECTED_PSU_DECLARED'
_MAX_POPULATION_PSUS_PER_STRATUM = 1000000


def _fail(code):
    raise ValueError('SURVEY_FPC_' + code)


def stratified_psu_fpc_mean(spec):
    """Calculate FPC-corrected stratified first-stage Taylor mean variance.

    Input retains the same declared rows, units and rights as the original
    survey_stratified_mean contract, but requires a *different* sampling
    declaration plus a stratum-to-population-PSU-count mapping. The unadjusted
    WR arithmetic is an internal intermediate, NOT the declared study design.
    """
    required = {'schema','data_kind','source_declared_sha256','rights',
                'outcome_unit','psu_sampling','rows','population_psus_by_stratum'}
    if type(spec) is not dict or set(spec) != required or spec['schema'] != SCHEMA:
        _fail('SCHEMA_INVALID')
    if type(spec['psu_sampling']) is not str or spec['psu_sampling'] != DESIGN:
        _fail('SRSWOR_CENSUS_DESIGN_DECLARATION_REQUIRED')
    counts = spec['population_psus_by_stratum']
    if type(counts) is not dict or not 1 <= len(counts) <= 64:
        _fail('STRATUM_POPULATION_COUNTS_REQUIRED')
    # We require the original validated inputs, including the *full sampled
    # PSU frame* when calculating a domain mean. No private base API changes.
    original = {key:value for key,value in spec.items()
                if key != 'population_psus_by_stratum'}
    original['schema'] = WR_SCHEMA
    original['psu_sampling'] = 'WITH_REPLACEMENT_FIRST_STAGE_APPROXIMATION_DECLARED'
    approximation = stratified_psu_mean(original)
    strata = approximation['stratum_variance_components']
    names = {entry['stratum'] for entry in strata}
    if set(counts) != names:
        _fail('POPULATION_STRATA_NOT_EXACTLY_MATCHED')

    variance = F(0)
    corrections = []
    for entry in strata:
        checkpoint()
        stratum = entry['stratum']
        sampled = entry['psu_count']
        population = counts[stratum]
        if type(population) is not int or not 2 <= population <= _MAX_POPULATION_PSUS_PER_STRATUM:
            _fail('POPULATION_PSU_COUNT_INVALID')
        if population < sampled:
            _fail('POPULATION_PSU_COUNT_SMALLER_THAN_SAMPLED')
        fpc = F(population - sampled, population)
        uncorrected = F(entry['variance_component_exact'])
        corrected = fpc * uncorrected
        variance += corrected
        corrections.append(dict(stratum=stratum, sampled_psus=sampled,
                                population_psus=population,
                                first_stage_sampling_fraction_exact=str(F(sampled, population)),
                                fpc_multiplier_exact=str(fpc),
                                uncorrected_variance_component_exact=str(uncorrected),
                                corrected_variance_component_exact=str(corrected)))
    uncorrected_total = F(approximation['variance_exact'])
    if (variance < 0 or variance > uncorrected_total or
            variance != sum((F(v['corrected_variance_component_exact'])
                             for v in corrections), F(0))):
        _fail('CORRECTION_IDENTITY_FAILED')
    full_digest = hashlib.sha256(json.dumps(spec, sort_keys=True,
                                            separators=(',', ':'), ensure_ascii=False,
                                            allow_nan=False).encode('utf-8')).hexdigest()
    return dict(schema='societybox-survey-psu-srswor-mean-result-v1',
                status='DECLARED_SRSWOR_DESIGN_ARITHMETIC_ASSUMPTIONS_UNVERIFIED',
                estimator='FIRST_STAGE_STRATIFIED_PSU_SRSWOR_FPC_TAYLOR_RATIO_MEAN',
                input_sha256=full_digest,
                uncorrected_input_sha256=approximation['input_sha256'],
                source_declared_sha256=spec['source_declared_sha256'],
                source_authentication='NOT_ESTABLISHED',
                design_authentication='NOT_ESTABLISHED',
                inference_validity='NOT_ESTABLISHED',
                confidence_interval='NOT_CALCULATED',
                original_upstream_parity='NOT_RUN',
                second_stage='DECLARED_PSU_CENSUS_NOT_AUTHENTICATED',
                psu_selection='EQUAL_PROBABILITY_FIRST_STAGE_SRSWOR_DECLARED',
                supported_scope='NO_PPS_NO_SECOND_STAGE_SAMPLING_NO_LONELY_PSUS',
                finite_population_correction='APPLIED_BY_DECLARED_STRATUM_PSU_FRAME',
                weight_semantics=approximation['weight_semantics'],
                data_kind=spec['data_kind'], outcome_unit=spec['outcome_unit'],
                observations=approximation['observations'],
                domain_observations=approximation['domain_observations'],
                total_psus=approximation['total_psus'],
                strata_count=approximation['strata_count'],
                degrees_of_freedom=approximation['degrees_of_freedom'],
                domain_population_weight_exact=approximation['domain_population_weight_exact'],
                weighted_mean_exact=approximation['weighted_mean_exact'],
                uncorrected_variance_exact=approximation['variance_exact'],
                fpc_variance_exact=str(variance), stratum_corrections=corrections)
