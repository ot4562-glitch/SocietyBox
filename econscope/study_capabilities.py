"""Read-only feature negotiation for the current offline study implementation.

This describes implemented tools, not study readiness, an institution's policy,
OS network enforcement, or permission to execute or export anything.
"""
from .workspace import ScopeError
from .feature_catalog import feature_catalog


def _mechanism_capabilities():
    """Discovery only; these Python/CLI tools are not registered LLM tools."""
    return {
        "schema": "societybox-mechanism-capabilities-v1",
        "assurance": "SYNTHETIC_LEDGER_CONTRACT_ONLY",
        "data_kind": "SYNTHETIC", "read_only": True,
        "available": ["synthetic_ledger_review", "synthetic_ledger_as_of_snapshot",
                      "mechanism_study_review"],
        "study_review_scope": "PROPOSAL_CONTRACT_ONLY",
        "blocked": {
            "mechanism_estimation": "NOT_IMPLEMENTED",
            "mechanism_forecast": "NOT_IMPLEMENTED",
            "mechanism_identification": "NOT_IMPLEMENTED",
            "mechanism_causal_claim": "NOT_IMPLEMENTED",
            "historical_as_of": "HISTORICAL_KNOWLEDGE_AND_ACCESS_NOT_ESTABLISHED",
            "verified_researcher_information_set": "RESEARCHER_ACCESS_NOT_ESTABLISHED",
        },
        "interfaces": {
            "python": ["econscope.mechanism_vintage.review_ledger",
                       "econscope.mechanism_vintage.as_of_snapshot",
                       "econscope.mechanism_study.review_mechanism_study"],
            "cli": ["ledger-review", "ledger-snapshot", "mechanism-study-review"],
            "llm_tool_registration": "NOT_IMPLEMENTED",
        },
        "vintage_targets": ["AS_OF"],
        "claim_level": "SIMULATED_RETROSPECTIVE_ANALYSIS",
        "source_authentication": "NOT_ESTABLISHED",
        "researcher_information_set": "NOT_ESTABLISHED",
        "limitations": [
            "Discovery does not review inputs, execute tools, grant approval or register LLM tools.",
            "Only synthetic ledgers are supported; every call enforces its own cutoff, proof and rights checks.",
            "Mechanism study review checks inert synthetic proposal contracts only; READY grants no execution or authorization.",
            "Caller-supplied proof and rights declarations do not authenticate sources or institutional permission.",
            "Transform declarations do not prove actual preprocessing isolation or historical model/knowledge availability.",
            "AS_OF selects a synthetic vintage; it does not establish HISTORICAL_AS_OF or researcher access.",
            "No estimator, forecast, causal identification or empirical forecast skill is implemented by these tools.",
        ],
    }


def _ols_capabilities():
    """One real read-only estimator, not an all-purpose inverse mechanism engine.

    Its source digest/rights are declared rather than authenticated. This
    discovery contract does not authorize an LLM tool or a native adapter.
    """
    return {
        "schema": "societybox-ols-capabilities-v1",
        "available": ["bounded_ols_inverse_coefficients"],
        "data_kinds": ["SYNTHETIC", "DECLARED_PUBLIC_OBSERVATIONS"],
        "scope": "ONE_REGRESSOR_INTERCEPT_ONLY_EXACT_RATIONAL_FORWARD_REPLAY",
        "interfaces": {
            "python": ["econscope.mechanism_regression.reconstruct_ols"],
            "cli": ["ols-reconstruct"],
            "llm_tool_registration": "NOT_IMPLEMENTED",
        },
        "estimator": "READ_ONLY_LOCAL_OLS",
        "variance": "EXPLICIT_CLASSICAL_OR_HC0_HC1_HC2_HC3_ASSUMPTIONS_UNVERIFIED",
        "variance_estimators": ["CLASSICAL_HOMOSKEDASTIC", "HC0", "HC1", "HC2", "HC3"],
        "sample_selection": "EXPLICIT_OBSERVATION_IDS_OR_ALL_ROWS_NO_AUTOMATIC_DROPPING",
        "result_exports": ["EXACT_COVARIANCE", "RESIDUAL_CSV_IN_JSON_RECEIPT", "FIT_TRACE"],
        "back_test": "NOT_RUN",
        "source_authentication": "NOT_ESTABLISHED",
        "original_paper_spec_recovery": "NOT_ESTABLISHED",
        "causal_identification": "NOT_ESTABLISHED",
        "native_upstream_parity": "NOT_RUN",
        "historical_as_of": "NOT_ESTABLISHED",
        "limitations": [
            "Requires explicit researcher-declared observations, variable units, model and variance estimator.",
            "Does not select the regression specification, omitted variables, publication sample or exclusions.",
            "A verified numeric fit does not establish causal explanation, real-time forecasts or a paper reproduction.",
        ],
    }


def execution_capabilities():
    """Fresh actual execution context; no requested features or readiness claim."""
    contract = negotiate_capabilities()
    return {key: contract[key] for key in (
        "schema", "scope", "provider", "mode", "connected_to_llm",
        "network_policy", "network_enforcement", "data_classification",
        "external_transfer", "engine_capabilities", "automatic_model_selection")}


def _iv_capabilities():
    return {
        "available": ["exact_linear_iv"],
        "python": ["econscope.instrumental_variables.calculate",
                   "econscope.instrumental_variables.anderson_rubin",
                   "econscope.instrumental_variables.invert_anderson_rubin",
                   "econscope.instrumental_variables.anderson_rubin_confidence_set",
                   "econscope.instrumental_variables.anderson_rubin_robust",
                   "econscope.instrumental_variables.linear_gmm",
                   "econscope.instrumental_variables.linear_wald",
                   "econscope.instrumental_variables_likelihood.liml",
                   "econscope.iterative_gmm.iterated_gmm", "econscope.iterative_gmm.cue_objective", "econscope.iterative_gmm.cue_gmm",
                   "econscope.instrumental_variables.compare_highs"],
        "implementation_status": "INDEPENDENT_IMPLEMENTED",
        "methods": ["2SLS", "CLASSICAL_ANDERSON_RUBIN_F", "CLASSICAL_AR_DECLARED_CUTOFF_INVERSION",
                    "CLASSICAL_AR_NOMINAL_CONFIDENCE_SET", "ROBUST_REDUCED_FORM_AR_WALD",
                    "LINEAR_GMM_ONE_STEP", "IID_ROBUST_LINEAR_GMM_TWO_STEP", "ONE_WAY_CLUSTER_LINEAR_GMM_TWO_STEP",
                    "HAC_BARTLETT_2SLS", "HAC_BARTLETT_LINEAR_GMM_TWO_STEP", "HAC_BARTLETT_AR_WALD",
                    "LINEAR_RESTRICTION_WALD", "LIML_CERTIFIED_KAPPA_INTERVAL", "FULLER_MODIFIED_LIML",
                    "ITERATED_LINEAR_GMM", "CUE_EXACT_OBJECTIVE_GRADIENT_HESSIAN", "CONTINUOUSLY_UPDATED_LINEAR_GMM"],
        "iterated_gmm_scope": "BINARY64_ITERATES_EXACT_FIXED_POINT_PREDICTOR_RESIDUAL_NOT_PARAMETER_ERROR_CERTIFICATE",
        "cue_scope": "UNCENTERED_IID_CLUSTER_BARTLETT_EXACT_DERIVATIVES_LOCAL_POSITIVE_CURVATURE_STATIONARITY",
        "cue_global_optimality": "NOT_ESTABLISHED",
        "iterative_gmm_limits": {"max_iterations": 200, "max_backtracks": 40, "classical_weights": "UNSUPPORTED"},
        "liml_scope": "BOUNDED_EXACT_GENERALIZED_EIGEN_BRACKET_CERTIFIED_COEFFICIENT_AND_COVARIANCE_ERROR_NOT_WEAK_IV_ROBUST",
        "liml_limits": {"reduced_form_covariance": "POSITIVE_DEFINITE_REQUIRED", "maximum_bisections": 160,
                        "fuller": "NONNEGATIVE_ALPHA_OVER_N_MINUS_TOTAL_INSTRUMENTS",
                        "failure": "NO_RESULT_IF_REQUESTED_ACCURACY_NOT_CERTIFIED"},
        "linear_wald_scope": "EXACT_STATISTIC_ASYMPTOTIC_CHI_SQUARED_NOT_WEAK_IV_ROBUST",
        "covariance": ["CLASSICAL", "HC0", "HC1", "CLUSTER_CR0", "CLUSTER_CR1", "HAC_BARTLETT"],
        "hac_contract": {
            "bandwidth": "EXPLICIT_INTEGER_0_TO_N_MINUS_1",
            "time_index": "EXPLICIT_CONTIGUOUS_INCREASING_INTEGER_PERIODS_IN_ROW_ORDER",
            "scores": "UNCENTERED_STRUCTURAL_RESIDUALS_NO_DF_SCALING_NO_PREWHITENING",
            "kernel": "1_MINUS_LAG_OVER_BANDWIDTH_PLUS_1",
            "scientific_validity": "WEAK_DEPENDENCE_AND_BANDWIDTH_ASYMPTOTICS_NOT_ESTABLISHED",
        },
        "scope": "BOUNDED_EXACT_RATIONAL_EXPLICIT_UNWEIGHTED_DESIGN",
        "ar_scope": "EXACT_JOINT_NULL_CLASSICAL_F_STATISTIC_AND_BINARY64_REFERENCE_P_VALUE",
        "ar_inversion_scope": "SCALAR_ENDOGENOUS_EXACT_QUADRATIC_ACCEPTANCE_AT_DECLARED_CUTOFF_NO_NOMINAL_COVERAGE_CLAIM",
        "ar_nominal_confidence_scope": "F_QUANTILE_CALIBRATED_SCALAR_SET_COVERAGE_CONDITIONAL_ON_REFERENCE_ASSUMPTIONS",
        "robust_ar_scope": "HC0_HC1_ONE_WAY_CR0_CR1_BARTLETT_HAC_WALD_ASYMPTOTIC_CHI_SQUARED_NOT_FINITE_SAMPLE_F",
        "reference_distributions": {
            "python": ["econscope.inference_distributions.f_survival",
                       "econscope.inference_distributions.log_f_survival",
                       "econscope.inference_distributions.chi_square_survival",
                       "econscope.inference_distributions.log_chi_square_survival",
                       "econscope.inference_distributions.f_critical_value"],
            "implementation": "INDEPENDENT_BETA_GAMMA_BINARY64_INTEGER_DF",
            "native_parity": "NOT_RUN", "study_design_validity": "NOT_ESTABLISHED",
        },
        "identification": "NOT_ESTABLISHED",
        "native_upstream_parity": "NOT_RUN",
        "upstream_status": "DEPENDENCY_BLOCKED",
        "upstream_reason": "PINNED_LINEARMODELS_RUNTIME_AND_EXECUTION_ADAPTER_NOT_CONFIGURED",
        "upstream_reference_commit": "181ec8d598cd2f59812f63cc103428687b7f4437",
        "optional_native_comparison": {
            "source": "HiGHS", "version": "1.15.1",
            "commit": "73cac48c5340d775a477087198611862559be250",
            "status": "EXPLICIT_MATCHED_RUNTIME_REQUIRED",
            "scope": "2SLS_COEFFICIENTS_AND_PROJECTED_QP_OBJECTIVE_ONLY",
            "native_iv_covariance": "NOT_RUN",
        },
        "chat_ui": "NOT_CONNECTED", "llm_tool_registration": "NOT_IMPLEMENTED",
        "engine_study_bundle": "NOT_CONNECTED",
        "unsupported": ["WEIGHTED_IV", "HAC_AUTOMATIC_BANDWIDTH", "HAC_PREWHITENING",
                        "WINDMEIJER_CORRECTION",
                        "ROBUST_AR_SET_INVERSION", "SMALL_CLUSTER_AR_CALIBRATION",
                        "CLR", "WEAK_IV_STRENGTH_CERTIFICATION"],
    }


def _dynamic_panel_capabilities():
    return {
        "available": ["exact_difference_gmm", "exact_system_gmm"],
        "python": ["econscope.dynamic_panel.validate", "econscope.dynamic_panel.calculate",
                   "econscope.dynamic_panel.compare_highs", "econscope.system_gmm.validate", "econscope.system_gmm.calculate",
                   "econscope.system_gmm.test_level_moments", "econscope.panel_inference.arellano_bond", "econscope.panel_inference.windmeijer"],
        "implementation_status": "INDEPENDENT_IMPLEMENTED",
        "methods": ["ARELLANO_BOND_STYLE_DIFFERENCE_GMM_ONE_STEP", "ENTITY_CLUSTER_TWO_STEP_GMM",
                    "DIFFERENCE_AND_LEVEL_SYSTEM_GMM", "COMMON_WEIGHT_NESTED_LEVEL_MOMENT_C_TEST",
                    "ARELLANO_BOND_ROBUST_PLUGIN_M_TEST", "WINDMEIJER_TWO_STEP_PANEL_GMM_COVARIANCE"],
        "ar_scope": "DIFFERENCE_RESIDUALS_FINAL_NUMERATOR_EXPLICIT_FIRST_OR_FINAL_COVARIANCE_BASIS_UNCORRECTED_GMM_PLUGIN",
        "ar_limits": {"maximum_orders": 16, "maximum_period_lag": 1000000},
        "windmeijer_scope": "TWO_STEP_FIRST_STEP_MOMENT_PLUGIN_EXACT_CORRECTION_NOT_COVERAGE_CERTIFICATION",
        "system_initial_weight": "JOINT_INNOVATION_AND_DECLARED_ENTITY_VARIANCE_INCLUDING_DIFFERENCE_LEVEL_COVARIANCE",
        "system_limits": {"input_rows": 512, "stacked_rows": 512, "parameters": 8, "instruments": 12},
        "level_moments": "EXPLICIT_LAGS_AND_DIFFERENCE_ENTITY_ORTHOGONALITY_DECLARED_NOT_VERIFIED",
        "initial_weight": "INVERSE_SUM_ZI_TRANSPOSE_DI_DI_TRANSPOSE_ZI",
        "instrument_styles": ["COLLAPSED", "TIME_SPECIFIC"],
        "regressor_timing": ["STRICTLY_EXOGENOUS", "PREDETERMINED", "ENDOGENOUS"],
        "covariance": ["CLUSTER_CR0", "CLUSTER_CR1"],
        "identification": "NOT_ESTABLISHED", "source_authentication": "NOT_ESTABLISHED",
        "system_gmm": "INDEPENDENT_IMPLEMENTED_PYTHON_API", "windmeijer_correction": "INDEPENDENT_IMPLEMENTED_PYTHON_API",
        "arellano_bond_m_test": "INDEPENDENT_ROBUST_PLUGIN_PYTHON_API",
        "native_upstream_parity": "NOT_RUN", "upstream_status": "DEPENDENCY_BLOCKED",
        "upstream_reason": "ORIGINAL_DYNAMIC_PANEL_RUNTIME_NOT_CONFIGURED",
        "optional_native_comparison": "EXPLICIT_PINNED_HIGHS_FIXED_WEIGHT_COEFFICIENTS_AND_OBJECTIVE_ONLY",
        "chat_ui": "NOT_CONNECTED", "engine_study_bundle": "NOT_CONNECTED",
        "llm_tool_registration": "NOT_IMPLEMENTED",
    }


def _random_effects_capabilities():
    return {
        "available": ["exact_random_effects_gls", "exact_correlated_random_effects"],
        "python": ["econscope.random_effects.validate", "econscope.random_effects.calculate",
                   "econscope.correlated_random_effects.validate", "econscope.correlated_random_effects.calculate"],
        "implementation_status": "INDEPENDENT_IMPLEMENTED",
        "variance_components": ["DECLARED", "BETWEEN_TRACE_MOMENTS"],
        "covariance": ["MODEL_BASED", "ENTITY_CR0", "ENTITY_CR1"],
        "scope": "STATIC_UNBALANCED_RANDOM_INTERCEPT_GLS_TIME_INVARIANT_REGRESSORS_EXPLICIT_BETWEEN_TRACE_MOMENT_FGLS",
        "correlated_effects": "OBSERVED_SAMPLE_MUNDLAK_MEAN_SPAN_EXACT_WITHIN_SLOPE_EQUIVALENCE",
        "mundlak_test": "IDENTIFIABLE_ADDITIONAL_MEAN_DIRECTIONS_MODEL_OR_ENTITY_CR0_CR1_WALD_ASYMPTOTIC_CHI_SQUARED",
        "mundlak_limits": "AUGMENTED_PARAMETERS_LE_8_SINGULAR_COVARIANCE_OR_NO_ADDITIONAL_MEANS_BLOCK_PROBABILITY",
        "limits": {"rows": 512, "entities": 128, "parameters": 8, "exact_output_bits": 12000},
        "scientific_validity": "STRICT_EXOGENEITY_VARIANCE_MODEL_AND_CLUSTER_INDEPENDENCE_NOT_ESTABLISHED",
        "original_default_convention_parity": "NOT_RUN", "upstream_status": "DEPENDENCY_BLOCKED",
        "chat_ui": "NOT_CONNECTED", "engine_study_bundle": "NOT_CONNECTED",
    }


def _var_capabilities():
    return {
        "available": ["exact_var_dynamics"],
        "python": ["econscope.vector_autoregression.validate", "econscope.vector_autoregression.calculate",
                   "econscope.vector_autoregression.forecast", "econscope.vector_autoregression.impulse_responses",
                   "econscope.vector_autoregression.variance_decomposition", "econscope.vector_autoregression.historical_decomposition",
                   "econscope.vector_autoregression.stationary_moments"],
        "implementation_status": "INDEPENDENT_IMPLEMENTED",
        "scope": "DECLARED_CUTOFF_EXACT_VAR_REFERENCE_CORE",
        "stability": "EXACT_COMPANION_DISCRETE_LYAPUNOV_POSITIVE_DEFINITE_WITNESS",
        "covariance_divisors": ["RESIDUAL_DF", "ML"],
        "recursive_identification": "EXPLICIT_ORDERING_REQUIRED_NOT_CAUSAL_CERTIFICATION",
        "historical_decomposition": "EX_POST_FULL_ESTIMATION_SAMPLE_EXACT_INNOVATION_ACCOUNTING",
        "parameter_uncertainty": "NOT_INCLUDED",
        "per_call_limits": {"variables": 3, "lags": 3, "companion_dimension": 6, "observations": 128, "horizon": 64},
        "native_upstream_parity": "NOT_RUN", "upstream_status": "DEPENDENCY_BLOCKED",
        "upstream_reason": "PINNED_STATSMODELS_NUMPY_SCIPY_RUNTIME_NOT_CONFIGURED",
        "upstream_reference_commit": "aa61cc148b820898247052008652b4dc4bda665e",
        "chat_ui": "NOT_CONNECTED", "engine_study_bundle": "NOT_CONNECTED", "llm_tool_registration": "NOT_IMPLEMENTED",
        "unsupported": ["AUTOMATIC_LAG_SELECTION", "VARX", "VECM", "BVAR", "GENERAL_SVAR", "IRF_CONFIDENCE_BANDS"],
    }


def _glm_capabilities():
    return {
        "available": ["binary_logit_poisson"],
        "python": ["econscope.generalized_linear.validate", "econscope.generalized_linear.fit", "econscope.generalized_linear.predict",
                   "econscope.generalized_linear.discrete_changes"],
        "families": ["BINOMIAL_LOGIT", "POISSON_COUNT", "POISSON_PSEUDO"],
        "implementation_status": "INDEPENDENT_IMPLEMENTED",
        "scope": "BERNOULLI_OR_POISSON_CANONICAL_LINK_WITH_DECLARED_OFFSETS_UNWEIGHTED_OBSERVATIONS",
        "optimizer": "COLUMN_EQUILIBRATED_PIVOTED_HOUSEHOLDER_QR_FISHER_SCORING_WITH_LINE_SEARCH",
        "covariance": ["MODEL_BASED", "HC0", "HC1", "CLUSTER_CR0", "CLUSTER_CR1"],
        "separation": "EXACT_VERIFIED_CANDIDATE_DIRECTIONS_NOT_COMPLETE_LP_DETECTOR",
        "prediction_intervals": "CONDITIONAL_MEAN_CONFIDENCE_ONLY_NOT_FUTURE_RESPONSE_INTERVALS",
        "inference_resolution": "CANCELLED_SCORE_OR_UNREPRESENTABLE_INTERVALS_BLOCKED_NOT_INTERVAL_CERTIFIED",
        "marginal_effects": "CONTINUOUS_PARTIAL_DERIVATIVES_HOLDING_OFFSETS_FIXED_NOT_CAUSAL",
        "marginal_effect_inference": "ANALYTIC_JACOBIAN_DELTA_NORMAL_REFERENCE_FIXED_EMPIRICAL_DESIGN_AND_OFFSETS",
        "discrete_changes": "AVERAGE_BINARY_ZERO_TO_ONE_FIXED_OTHER_COLUMNS_AND_OFFSETS_DELTA_NORMAL_REFERENCE",
        "upstream_marginal_effect_offsets": "GET_MARGEFF_OFFSET_NOT_IMPLEMENTED_IN_PINNED_SOURCE",
        "native_upstream_parity": "NOT_RUN", "upstream_status": "DEPENDENCY_BLOCKED",
        "upstream_reason": "PINNED_STATSMODELS_NUMPY_SCIPY_RUNTIME_NOT_CONFIGURED",
        "upstream_reference_commit": "aa61cc148b820898247052008652b4dc4bda665e",
        "chat_ui": "NOT_CONNECTED", "engine_study_bundle": "NOT_CONNECTED", "llm_tool_registration": "NOT_IMPLEMENTED",
        "unsupported": ["FREQUENCY_OR_SURVEY_WEIGHTS", "AGGREGATED_BINOMIAL_TRIALS", "PROBIT", "NEGATIVE_BINOMIAL",
                        "MULTINOMIAL_OR_ORDERED_LOGIT", "RANDOM_EFFECTS", "COMPLETE_SEPARATION_LP_SOLVER",
                        "UNCONDITIONAL_MARGINAL_EFFECT_VARIANCE", "NONBINARY_DISCRETE_CHANGES", "AUTOMATIC_INTERACTION_RECONSTRUCTION",
                        "SECOND_ORDER_DELTA_METHOD"],
    }


def _quantile_capabilities():
    return {
        "available": ["quantile_regression"],
        "python": ["econscope.quantile_regression.validate", "econscope.quantile_regression.calculate", "econscope.quantile_regression.compare_highs"],
        "implementation_status": "INDEPENDENT_IMPLEMENTED",
        "scope": "POSITIVE_WEIGHTED_LINEAR_CHECK_LOSS_AT_ONE_DECLARED_INTERIOR_QUANTILE",
        "independent_solver": "EXACT_FULL_RANK_CONTACT_ENUMERATION",
        "native_solver": "PINNED_HIGHS_DUAL_BOX_LP_WITH_EXACT_PRIMAL_DUAL_KKT_CERTIFICATE",
        "native_execution": "PER_RUN_RECEIPT_ONLY_NOT_PROBED",
        "upstream_reference_commit": "73cac48c5340d775a477087198611862559be250",
        "per_call_limits": {"observations": 256, "coefficients": 8, "oracle_contact_sets": 20000, "oracle_work_units": 5000000, "rational_bits": 8192},
        "coefficient_uniqueness": "NOT_ASSESSED_MULTIPLE_OPTIMA_POSSIBLE",
        "unsupported": ["COVARIANCE_OR_BOOTSTRAP_INFERENCE", "INSTRUMENTAL_QUANTILE_REGRESSION", "CENSORED_QUANTILE_REGRESSION", "SURVEY_DESIGN_INFERENCE", "GENERAL_LINEAR_CONSTRAINTS", "NONCROSSING_JOINT_QUANTILES"],
        "chat_ui": "NOT_CONNECTED", "engine_study_bundle": "NOT_CONNECTED", "llm_tool_registration": "NOT_IMPLEMENTED",
    }


def _state_space_capabilities():
    return {
        "available": ["linear_gaussian_state_space", "state_space_likelihood_information", "state_space_wald_test", "state_space_constrained_em"],
        "python": ["econscope.state_space.validate", "econscope.state_space.filter", "econscope.state_space.smooth", "econscope.state_space.forecast", "econscope.state_space.em_step", "econscope.state_space.fit_em", "econscope.state_space_inference.likelihood_information", "econscope.state_space_inference.wald_test"],
        "implementation_status": "INDEPENDENT_IMPLEMENTED",
        "scope": "KNOWN_PARAMETER_TIME_VARYING_LINEAR_GAUSSIAN_WITH_PARTIAL_MISSING_OBSERVATIONS",
        "filter": "EXACT_RATIONAL_JOSEPH_UPDATE_SEMIDEFINITE_LDL_CONDITIONING",
        "smoother": "GENERALIZED_RTS_WITH_EXACT_LAG_ONE_CONDITIONAL_COVARIANCES",
        "parameter_estimation": "ONE_EXACT_EM_STEP_TIME_INVARIANT_PARTIAL_MISSING_FIXED_INITIAL_PD_Q_R",
        "em_missing_data": "EXACT_CONDITIONAL_EMISSION_FIRST_SECOND_STATE_CROSS_MOMENTS_WITH_CORRELATED_R",
        "em_missingness_assumption": "IGNORABLE_OBSERVATION_PATTERN_NOT_SELECTION_MODEL",
        "em_blocks": ["STATE_DYNAMICS", "OBSERVATION_DYNAMICS"],
        "em_fixed_parameter_groups": ["transition", "state_intercept", "process_covariance", "measurement", "observation_intercept", "measurement_covariance"],
        "em_fixed_parameter_semantics": "WHOLE_MATRIX_VECTOR_OR_COVARIANCE_GROUPS_CONDITIONAL_M_STEP_NO_ROUNDING_OF_FIXED_VALUES",
        "em_fixed_entry_fields": ["transition", "state_intercept", "measurement", "observation_intercept"],
        "em_fixed_entry_semantics": "DECLARED_COEFFICIENT_ENTRIES_PRECISION_WEIGHTED_ECM_AT_OLD_NOISE_THEN_OPTIONAL_COVARIANCE_UPDATE_FIXED_VALUES_RESTORED_AFTER_QUANTIZATION_AND_MIXING",
        "em_fixed_entry_limits": {"fixed_entries": 70, "coupled_free_coefficients_per_block": 42},
        "parameter_information": "EXACT_SCORE_OBSERVED_HESSIAN_LOCAL_RANK_AND_CONDITIONAL_CURVATURE_SE_REFERENCE_AT_SMALL_SCORE_PD_POINT",
        "wald_test": "EXACT_LINEAR_RESTRICTION_STATISTIC_CONDITIONAL_ASYMPTOTIC_CHI_SQUARE_AT_SMALL_SCORE_PD_POINT",
        "wald_null_check": "DECLARED_INTERIOR_NULL_WITNESS_FULL_RANK_REALIZED_PREDICTABLE_INFORMATION_NOT_REGULARITY_PROOF",
        "information_coordinates": "SHARED_TRAINING_ENTRIES_ALL_OTHER_ENTRIES_FIXED_SYMMETRIC_COVARIANCE_COORDINATES_INTERIOR",
        "information_limits": {"parameter_directions": 8, "work_units": 5000000, "rational_bits": 8192},
        "information_native_adapter": "NOT_IMPLEMENTED",
        "em_convergence": "DECLARED_CHANGE_CRITERIA_NOT_GLOBAL_OR_EXACT_MLE_CERTIFICATION",
        "em_optimizer": "DECLARED_COEFFICIENT_LDL_QUANTIZATION_DUAL_Q_LIKELIHOOD_ASCENT_INPUT_BOUND_FORWARD_REUSE",
        "em_upstream_status": "DEPENDENCY_BLOCKED",
        "em_upstream_reference": "DYNAMAX_ffc626bf89bf07c7ae7064922f6a373d013c4b17",
        "singular_innovation": "EXACT_SUPPORT_CHECK_NO_FULL_DIMENSIONAL_DENSITY",
        "forecast": "DECLARED_ORIGIN_AND_FUTURE_DYNAMICS_NO_FUTURE_OBSERVATION_ASSIMILATION",
        "per_call_limits": {"states": 6, "observables": 4, "steps": 128, "forecast_horizon": 64, "rational_bits": 8192},
        "native_upstream_parity": "NOT_RUN", "upstream_status": "DEPENDENCY_BLOCKED",
        "upstream_reference_commit": "aa61cc148b820898247052008652b4dc4bda665e",
        "upstream_reason": "PINNED_STATSMODELS_NUMPY_SCIPY_RUNTIME_NOT_CONFIGURED",
        "chat_ui": "NOT_CONNECTED", "engine_study_bundle": "NOT_CONNECTED", "llm_tool_registration": "NOT_IMPLEMENTED",
        "unsupported": ["GLOBAL_MLE_CERTIFICATION", "AUTOMATIC_FIT_UNCERTAINTY", "STATE_SPACE_LR_TESTS", "EM_ELEMENTWISE_COVARIANCE_MASKS", "EM_GENERAL_LINEAR_PARAMETER_RESTRICTIONS", "STATIONARITY_CONSTRAINED_MLE", "EM_WITH_TIME_VARYING_PARAMETERS", "NONIGNORABLE_MISSINGNESS", "EM_INITIAL_PRIOR_ESTIMATION", "DIFFUSE_INITIALIZATION", "CROSS_TIME_OR_STATE_MEASUREMENT_NOISE_CORRELATION",
                        "NONLINEAR_OR_NON_GAUSSIAN_FILTERS", "JOINT_MULTI_HORIZON_OBSERVATION_COVARIANCE", "STATE_SIMULATION"],
    }


def negotiate_capabilities(required=()):
    """Report actual no-LLM capabilities; never choose a model or probe engines."""
    if not isinstance(required, (list, tuple)) or any(
            not isinstance(item, str) for item in required):
        raise ScopeError("STUDY_CAPABILITY_REQUEST_INVALID")
    available = [
        "declarative_model_review", "bounded_scalar_execution",
        "local_receipt_reexecution", "paired_comparison",
        "rights_gated_export", "svg_charts", "html_reports",
        "ascii_vector_pdf", "exact_trajectory_csv",
    ]
    blocked = {
        "ai_conversation": "LLM_NOT_CONNECTED",
        "ai_model_generation": "LLM_NOT_CONNECTED",
        "ai_literature_interpretation": "LLM_NOT_CONNECTED",
        "native_dynare": "ADAPTER_NOT_IMPLEMENTED",
        "native_mesa": "ADAPTER_NOT_IMPLEMENTED",
        "native_quantecon": "ADAPTER_NOT_IMPLEMENTED",
        "native_hark": "ADAPTER_NOT_IMPLEMENTED",
        "untrusted_code_execution": "OS_ISOLATION_NOT_ESTABLISHED",
    }
    # Keep the independent ledger contract outside scalar execution receipts.
    # Agent context carries this full negotiation result, without tool grants.
    mechanism = _mechanism_capabilities()
    ols = _ols_capabilities()
    iv = _iv_capabilities()
    dynamic_panel = _dynamic_panel_capabilities()
    random_effects = _random_effects_capabilities()
    var = _var_capabilities()
    glm = _glm_capabilities()
    state_space = _state_space_capabilities()
    quantile = _quantile_capabilities()
    known = set(available) | set(blocked) | set(mechanism["available"]) | set(mechanism["blocked"]) | set(ols["available"]) | set(iv["available"]) | set(dynamic_panel["available"]) | set(random_effects["available"]) | set(var["available"]) | set(glm["available"]) | set(state_space["available"]) | set(quantile["available"])
    if set(required) - known:
        raise ScopeError("STUDY_CAPABILITY_UNKNOWN")
    all_blocked = {**blocked, **mechanism["blocked"]}
    missing = {name: all_blocked[name] for name in sorted(set(required)) if name in all_blocked}
    return {
        "schema": "societybox-study-capabilities-v1",
        "scope": "IMPLEMENTED_TOOLS_NOT_STUDY_READINESS_OR_AUTHORIZATION",
        "provider": "NONE", "mode": "NO_LLM", "connected_to_llm": False,
        "network_policy": "DENY", "network_enforcement": "NOT_ESTABLISHED",
        "data_classification": "UNSPECIFIED", "external_transfer": "DISABLED",
        "status": "BLOCKED" if missing else "SUPPORTED",
        "required": sorted(set(required)), "missing": missing,
        "mechanism_capabilities": mechanism,
        "ols_capabilities": ols,
        "iv_capabilities": iv,
        "dynamic_panel_capabilities": dynamic_panel,
        "random_effects_capabilities": random_effects,
        "var_capabilities": var,
        "glm_capabilities": glm,
        "state_space_capabilities": state_space,
        "quantile_capabilities": quantile,
        "exact_study_capabilities": {
            "families": ["input_output", "survey_replicate", "vintage_comparison"],
            "workflow": "ENGINE_STUDY_REVIEW_ACKNOWLEDGE_RUN_EXPORT_RECOMPUTE",
            "engine": "SOCIETY_BOX_INDEPENDENT_EXACT", "native_execution": "NOT_RUN",
            "numerical_verification": "CRAMER_AND_EXACT_IDENTITIES_OR_REPLICATE_ALGEBRA_OR_REVISION_SELECTION_DECIMAL_DIFFERENCES",
            "source_authentication": "NOT_ESTABLISHED", "survey_design": "NOT_ESTABLISHED",
            "causal_identification": "NOT_ESTABLISHED", "llm_tool_registration": "NOT_IMPLEMENTED",
        },
        "inverse_screen_capabilities": {
            "family": "dynamic_inverse", "feature_id": "lagged-hypothesis-comparison",
            "python": "econscope.inverse_screen.screen_inverse_hypotheses",
            "cli": "python -m societybox inverse-screen --spec LOCAL_JSON",
            "execution": "OFFLINE_NO_LLM", "model_selection": "TRAINING_ONLY",
            "candidate_budget": "UP_TO_24_SINGLE_LAG_HYPOTHESES_PLUS_ONE_JOINT",
            "holdout": "RETROSPECTIVE_DECLARED_RELEASES_ONLY",
            "residual_proxies": "TRAINING_ONLY_SAME_DAY_EX_POST_NOT_FORECAST_FEATURES",
            "automatic_causal_identification": "NOT_ESTABLISHED",
            "external_source_authentication": "NOT_ESTABLISHED",
            "research_evidence": "INLINE_CONTENT_BOUND_EVIDENCE_UNKNOWNS_OBLIGATIONS_COVERAGE",
            "local_reexecution": "--verify-result RAW_RESULT_JSON; NOT_INDEPENDENT_REPLICATION",
            "researcher_report": "--format html --locale en|ko|zh-Hans|ja; RECOMPUTED_OFFLINE",
            "native_study_workflow": "engine_study dynamic_inverse: REVIEW_ACKNOWLEDGE_RUN_EXPORT_RECOMPUTE",
            "native_selected_regression": "PINNED_HIGHS_QP_WITH_EXACT_OLS_AND_CRAMER_CHECKS; REQUIRES_RUNTIME_AND_COEFFICIENT_BOUNDS",
            "native_scope": "SELECTED_FULL_RANK_OLS_ONLY; NOT_ALL_CANDIDATES_OR_ALL_PUBLIC_APIS",
            "llm_tool_registration": "NOT_IMPLEMENTED",
        },
        "coefficient_identification_capabilities": {
            "feature_id": "coefficient-equivalence-ranges", "family": "coefficient_identification",
            "python": "econscope.inverse_identification", "workflow": "EXISTING_ENGINE_STUDY_REVIEW_ACK_RUN_EXPORT_VERIFY",
            "execution": "PINNED_HIGHS_LP_EXTREMA_AND_EXACT_RANK_NULLSPACE",
            "equivalence": "FINITE_TRAINING_DESIGN_PREDICTIONS_WITH_DECLARED_REFERENCE_AND_BOUNDS",
            "confidence_interval": "NOT_A_STATISTICAL_CONFIDENCE_REGION",
            "causal_identification": "NOT_ESTABLISHED", "probe_coverage": "PARTIAL_ENDPOINT_WITNESSES_ONLY",
            "holdout_targets_used": False, "llm_tool_registration": "NOT_IMPLEMENTED",
            "independent_installed_replication": "NOT_RUN",
        },
        "weighted_regression_capabilities": {
            "feature_id": "weighted-regression-covariance", "family": "regression_sensitivity",
            "python": "econscope.regression_sensitivity", "workflow": "EXISTING_REVIEW_ACK_RUN_EXPORT_RECOMPUTE",
            "estimator": "DECLARED_MULTIVARIATE_ANALYTIC_WEIGHT_WLS_EXACT_AND_PINNED_HIGHS_QP",
            "covariance": ["CLASSICAL_WLS", "HC0", "HC1", "HC2", "HC3", "HAC_BARTLETT", "CLUSTER_CR0", "CLUSTER_CR1"],
            "oracle": "INDEPENDENT_CRAMER_ADJUGATE_AND_PAIRWISE_INFLUENCE_KERNELS",
            "weight_meaning": "POSITIVE_ANALYTIC_NOT_SURVEY_OR_FREQUENCY",
            "inferential_validity": "NOT_ESTABLISHED", "post_selection_validity": "NOT_ESTABLISHED",
            "causal_identification": "NOT_ESTABLISHED", "historical_as_of": "NOT_ESTABLISHED",
            "statsmodels_original_execution": "DEPENDENCY_BLOCKED_NUMPY_SCIPY_NOT_INSTALLED",
            "llm_tool_registration": "NOT_IMPLEMENTED", "independent_installed_replication": "NOT_RUN",
        },
        "fixed_effects_capabilities": {
            "family":"fixed_effects_regression", "feature_id":"weighted-fixed-effects",
            "execution":"INDEPENDENT_EXACT_WEIGHTED_PROJECTION_AND_PINNED_HIGHS_SLOPES",
            "verification":"INDEPENDENT_FULL_DUMMY_RANK_COEFFICIENTS_LEVERAGE_COVARIANCE",
            "degrees_of_freedom":"N_MINUS_EXACT_RANK_OF_FIXED_EFFECTS_AND_PREDICTORS",
            "cluster_correction":"TOTAL_MODEL_RANK_NO_NESTED_EFFECT_ADJUSTMENT_OR_PACKAGE_PARITY",
            "original_statistics_packages":"BLOCKED_NO_APPROVED_PINNED_STATISTICS_RUNTIME",
            "large_data_parity":"NOT_ESTABLISHED_DENSE_REFERENCE_OPERATION_BUDGETS",
            "inferential_validity":"NOT_ESTABLISHED", "causal_identification":"NOT_ESTABLISHED",
        },
        "originwise_forecast_capabilities": {
            "feature_id": "originwise-forecast-evaluation", "family": "forecast_validation",
            "python": "econscope.forecast_validation", "workflow": "EXISTING_REVIEW_ACK_RUN_EXPORT_RECOMPUTE",
            "estimation": "ORIGINWISE_COMMON_AVAILABLE_SAMPLE_EXACT_OLS_AND_CRAMER_WITH_PINNED_HIGHS_QP",
            "baseline": "LAST_AVAILABLE_TARGET_AT_ORIGIN", "scoring": "PAIRED_BY_HORIZON_AT_DECLARED_EVALUATION_CUTOFF",
            "future_regressors": "BLOCKED_UNLESS_ALREADY_OBSERVED_AND_AVAILABLE_AT_ORIGIN",
            "historical_as_of": "NOT_ESTABLISHED_DECLARED_SNAPSHOT_OR_REVISION_HISTORY", "probabilistic_calibration": "NOT_RUN_POINT_FORECASTS_ONLY",
            "revision_history": "OPTIONAL_DATE_ONLY_CALENDAR_WITH_ORIGIN_SPECIFIC_RELEASE_OR_LOCAL_ACQUISITION_SNAPSHOTS",
            "evaluation_vintage": "SEPARATE_EXPLICIT_CUTOFF_NEVER_USED_FOR_SELECTION_OR_REFIT",
            "preprocessing": "OPTIONAL_EXACT_PREDICTOR_WINSORIZATION_FIT_ON_ORIGIN_COMMON_TRAINING_ROWS_ONLY",
            "preprocessing_target_and_baseline": "UNTRANSFORMED; NO_ROBUSTNESS_OR_CAUSAL_GUARANTEE",
            "external_forecast_validity": "NOT_ESTABLISHED", "independent_installed_replication": "NOT_RUN",
            "llm_tool_registration": "NOT_IMPLEMENTED",
        },
        "network_study_capabilities": {
            "scope": "SEPARATE_ENGINE_STUDY_REVIEW_APPROVAL_RUN_VERIFY",
            "python": "econscope.engine_study",
            "cli": "python -m econscope.engine_study",
            "family": "supply_network", "upstream": "PINNED_NETWORKX",
            "methods": ["directed_minimum_delivery_delay", "declared_node_closure_comparison",
                        "directed_supply_capacity_maximum_flow", "directed_minimum_cost_flow"],
            "additional_family": "supply_capacity",
            "additional_families": ["supply_capacity", "supply_cost"],
            "minimum_cost_apis": ["networkx.network_simplex", "networkx.cost_of_flow_ON_OPTIMAL_ONLY"],
            "minimum_cost_oracle": "EXACT_DEMAND_CUTS_AND_RESIDUAL_BELLMAN_FORD_PRIMAL_DUAL_CERTIFICATE",
            "prerequisites": ["PINNED_LOCAL_ARCHIVE_AND_MANIFEST", "SOURCE_AND_RIGHTS_DECLARATION",
                              "EXPLICIT_REVIEW_DIGEST_ACKNOWLEDGMENT"],
            "units": "POSITIVE_INTEGER_MODEL_TICKS; CLOSURE_COMPARISON_USES_DAYS",
            "native_execution": "PER_RUN_RECEIPT_ONLY_NOT_PROBED",
            "oracle": "INDEPENDENT_BELLMAN_FORD; CAPACITY_EXHAUSTIVE_CUT_AND_CONSERVATION",
            "llm_tool_registration": "NOT_IMPLEMENTED",
            "causal_identification": "NOT_ESTABLISHED", "os_isolation": "NOT_ATTESTED",
        },
        "staffing_study_capabilities": {
            "scope": "SEPARATE_ENGINE_STUDY_REVIEW_APPROVAL_RUN_VERIFY",
            "python": "econscope.engine_study", "cli": "python -m econscope.engine_study",
            "family": "service_staffing", "upstream": "PINNED_SIMPY",
            "methods": ["finite_fifo_parallel_staffing_cost_comparison"],
            "prerequisites": ["PINNED_LOCAL_ARCHIVE_AND_MANIFEST", "SOURCE_AND_RIGHTS_DECLARATION",
                              "EXPLICIT_REVIEW_DIGEST_ACKNOWLEDGMENT"],
            "native_execution": "PER_RUN_RECEIPT_ONLY_NOT_PROBED",
            "oracle": "INDEPENDENT_FIFO_INTERVAL_RECURRENCE",
            "units": "DECLARED_COMMON_COST_UNIT_FOR_ENTIRE_WORKLOAD",
            "llm_tool_registration": "NOT_IMPLEMENTED", "os_isolation": "NOT_ATTESTED",
            "causal_identification": "NOT_ESTABLISHED",
        },
        "mathematical_program_capabilities": {
            "scope": "SEPARATE_ENGINE_STUDY_REVIEW_APPROVAL_RUN_VERIFY",
            "python": "econscope.engine_study", "cli": "python -m econscope.engine_study",
            "family": "mathematical_program", "upstream": "PINNED_LOCAL_HIGHS_C_API",
            "backends": ["HiGHS", "Pyomo+HiGHS"],
            "pyomo_prerequisite": "PINNED_SOURCE_ARCHIVE_AND_MANIFEST_FOR_ACTUAL_CORE_REPN_CONSTRUCTION",
            "kinds": ["LP", "MILP", "QP"],
            "constraints": "FINITE_SIGNED_BOUNDS_TWO_SIDED_ROWS; STRICT_CURVATURE_QP",
            "oracle": "EXACT_VERTEX_INTEGER_ENUMERATION_OR_ACTIVE_SET_KKT",
            "limits": "1_TO_4_VARIABLES_0_TO_8_ROWS_20000_ORACLE_SYSTEMS",
            "prerequisites": ["REVIEWED_HASH_PINNED_WINDOWS_RUNTIME_AND_DLL_CLOSURE",
                              "SOURCE_AND_RIGHTS_DECLARATION", "EXPLICIT_REVIEW_DIGEST_ACKNOWLEDGMENT"],
            "native_execution": "PER_RUN_RECEIPT_ONLY_NOT_PROBED",
            "numerical_comparison": "ABS_1E-7_REL_1E-8_PLUS_PRIMAL_FEASIBILITY",
            "llm_tool_registration": "NOT_IMPLEMENTED", "os_isolation": "NOT_ATTESTED",
            "clean_installed_replication": "NOT_ESTABLISHED",
        },
        "optimization_capabilities": {
            "scope": "STANDALONE_EXPLICIT_FALLBACK_ONLY_NOT_STUDYSPEC_EXECUTION",
            "python": "econscope.optimization_policy.optional_optimization",
            "cli": "python -B -m econscope.optimization_policy --allow-fallback",
            "method": "EXACT_PRIMAL_DUAL_LINEAR_PROGRAM_MAX",
            "limits": "1_TO_4_NONNEGATIVE_VARIABLES_1_TO_8_LE_CONSTRAINTS",
            "native_execution": "NOT_RUN", "native_parity": "NOT_RUN",
            "causal_effect": "UNKNOWN", "real_policy_counterfactual": "UNKNOWN",
            "llm_tool_registration": "NOT_IMPLEMENTED",
        },
        "engine_capabilities": {
            "engine": "BOUNDED_DECLARATIVE_SCALAR_INTERPRETER",
            "available": available, "blocked": blocked,
            "native_upstream_parity": "NOT_RUN",
        },
        "automatic_model_selection": False,
        "product_feature_catalog": feature_catalog("ko"),
        "legacy_keyword_planner": "HEURISTIC_FIXTURE_NOT_AI_OR_STUDY_CAPABILITY",
        "limitations": [
            "Available tools still require valid researcher inputs and their own approval and data-rights checks.",
            "Separate economic-study adapters require their own pinned runtime, review and rights; installed metadata grants no execution authority.",
            "No LLM provider is connected, including local, organization and remote providers.",
            "DENY describes this interface's no-transfer policy, not measured host network isolation or institutional authorization.",
            "ASCII PDF and scalar model support do not imply multilingual reporting or general-purpose solver support.",
        ],
    }
