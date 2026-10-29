"""Origin-wise declared-information forecasting with exact/native OLS refits.

Declared single-snapshot or revision-history simulation, not authenticated history.
Future targets do not enter selection. Unsupported future regressors remain
blocked; unobserved/unreleased target values remain unscored.
"""
from copy import deepcopy
from datetime import timedelta
from fractions import Fraction as F
import math
from pathlib import Path
import time

from . import highs_native,inverse_native,mechanism_dynamic
from .inverse_screen import _validate_request,_single_candidates
from .inverse_numeric_verifier import _determinant
from .workspace import canonical,digest


def _source_receipt():
    receipt=inverse_native._source_receipt()
    path=Path(__file__)
    if path.is_symlink() or not path.is_file(): raise ValueError('FORECAST_CODE_UNAVAILABLE')
    receipt[path.name]=digest(path.read_bytes())
    for name in ('forecast_vintages.py','research_vintages.py','mechanism_vintage.py','forecast_preprocessing.py'):
        path=Path(__file__).with_name(name)
        if path.is_symlink() or not path.is_file(): raise ValueError('FORECAST_CODE_UNAVAILABLE')
        receipt[name]=digest(path.read_bytes())
    return receipt


def _validated(parameters):
    fields={'screen','coefficient_bounds','origins','horizons','evaluation_as_of_day'}
    if (type(parameters) is not dict or not fields<=set(parameters)
            or not set(parameters)<=fields|{'revision_history','preprocessing'}):
        raise ValueError('FORECAST_SCHEMA_INVALID')
    if 'preprocessing' in parameters:
        from .forecast_preprocessing import validate as validate_preprocessing
        validate_preprocessing(parameters['preprocessing'])
    screen=parameters['screen']
    if 'revision_history' in parameters:
        from .forecast_vintages import validation_screen
        screen=validation_screen(parameters)
    data,lags=_validate_request(screen)
    singles=_single_candidates(data,lags,parameters['screen']['search']['include_target_history'])
    anchor=dict(candidate_id='window_anchor',features=[dict(series=singles[0]['features'][0]['series'],lag=max(lags))])
    rows,_,start=mechanism_dynamic._validate_structure(dict(data,candidates=singles+[anchor]),candidate_limit=25,
        input_size_cap=mechanism_dynamic.MAX_BYTES+8192,minimum_train=0,minimum_holdout=0)
    origins=parameters['origins']; horizons=parameters['horizons']
    if type(origins) is not list or not 1<=len(origins)<=16 or any(type(day) is not str for day in origins):
        raise ValueError('FORECAST_ORIGINS_INVALID')
    if sorted(set(origins))!=origins or origins[0]!=data['training_end_day']:
        raise ValueError('FORECAST_ORIGINS_INVALID')
    by_day={row[0].isoformat():i for i,row in enumerate(rows)}
    if any(day not in by_day for day in origins): raise ValueError('FORECAST_ORIGIN_NOT_OBSERVED')
    if (type(horizons) is not list or not 1<=len(horizons)<=8
            or any(type(h) is not int or not 1<=h<=3650 for h in horizons) or sorted(set(horizons))!=horizons):
        raise ValueError('FORECAST_HORIZONS_INVALID')
    if len(origins)*len(horizons)>32: raise ValueError('FORECAST_GRID_BUDGET')
    try:
        rows[by_day[origins[-1]]][0]+timedelta(days=max(horizons))
    except OverflowError:
        raise ValueError('FORECAST_DATE_RANGE_INVALID') from None
    evaluation=mechanism_dynamic._day(parameters['evaluation_as_of_day'])
    if evaluation<rows[by_day[origins[-1]]][0]: raise ValueError('FORECAST_EVALUATION_CUTOFF_INVALID')
    bounds=parameters['coefficient_bounds']; required={'intercept'}|{inverse_native._label(c['features'][0]) for c in singles}
    if type(bounds) is not dict or set(bounds)!=required: raise ValueError('FORECAST_COEFFICIENT_BOUNDS_REQUIRED')
    for pair in bounds.values():
        if type(pair) is not list or len(pair)!=2 or not highs_native._num(pair[0])<highs_native._num(pair[1]):
            raise ValueError('FORECAST_COEFFICIENT_BOUNDS_INVALID')
    return rows,start,singles,by_day,evaluation


def validate(parameters):
    """No estimation, forecast, native worker or approval during review."""
    _validated(parameters)
    return parameters


def _checked_fit(x,y):
    fit=mechanism_dynamic._fit_design_exact(x,y)
    width=len(x[0]); gram=[[sum(row[i]*row[j] for row in x) for j in range(width)] for i in range(width)]
    rhs=[sum(row[j]*value for row,value in zip(x,y)) for j in range(width)]
    det=_determinant(gram)
    if not det:
        if fit is not None: raise ValueError('FORECAST_CRAMER_RANK_MISMATCH')
        return None
    beta=[_determinant([[rhs[r] if c==j else gram[r][c] for c in range(width)] for r in range(width)])/det for j in range(width)]
    residuals=[value-sum(v*b for v,b in zip(row,beta)) for row,value in zip(x,y)]
    sse=sum(e*e for e in residuals); score=sse/(len(y)-width)
    if fit is None or fit!=(beta,residuals,sse,score): raise ValueError('FORECAST_CRAMER_NUMERIC_MISMATCH')
    return fit


def _cell(parameters,rows,start,singles,origin,horizon,evaluation,runtime,evaluation_rows=None):
    day=rows[origin][0]; target=parameters['screen']['study']['target']
    forecast_day=day+timedelta(days=horizon); target_index=origin+horizon
    available_target=[i for i in range(origin+1) if rows[i][2][target]<=day]
    baseline_index=available_target[-1] if available_target else None
    cell=dict(origin_day=day.isoformat(),horizon=horizon,forecast_day=forecast_day.isoformat(),status='BLOCKED',blocker=None,
        selected_candidate_id=None,selected_features=[],coefficients_exact=None,prediction_exact=None,native_prediction_decimal=None,
        native=None,native_verification='NOT_RUN',candidate_coverage=[],candidates=[],training_observations=0,
        excluded_training_rows=[],baseline_source_day=None if baseline_index is None else rows[baseline_index][0].isoformat(),
        baseline_prediction_exact=None if baseline_index is None else str(rows[baseline_index][1][target]),
        observed_target_exact=None,squared_error_exact=None,baseline_squared_error_exact=None,
        score_status='NOT_SCORED_FORECAST_BLOCKED',future_target_used_for_selection=False)
    eligible=[]
    for candidate in singles:
        feature=candidate['features'][0]; index=target_index-feature['lag']; reason=None
        if index>origin: reason='FUTURE_PREDICTOR_PATH_NOT_PROVIDED'
        elif index<0 or rows[index][2][feature['series']]>day: reason='PREDICTOR_UNAVAILABLE_AT_ORIGIN'
        cell['candidate_coverage'].append(dict(candidate_id=candidate['candidate_id'],features=candidate['features'],
            state='ELIGIBLE' if reason is None else 'BLOCKED',reason=reason))
        if reason is None: eligible.append(candidate)
    known=[dict(day=row[0].isoformat(),series=name,value_exact=str(row[1][name]),available_on=row[2][name].isoformat())
           for row in rows[:origin+1] for name in sorted(row[1]) if row[2][name]<=day]
    cell['information_set_sha256']=digest(canonical(dict(origin_day=day.isoformat(),horizon=horizon,
        known_observations=known,eligible_candidates=eligible,common_max_lag=start)))
    if 'preprocessing' in parameters:
        cell['information_set_sha256']=digest(canonical(dict(
            raw_information_set_sha256=cell['information_set_sha256'],preprocessing_plan=parameters['preprocessing'])))
    if not eligible:
        cell['blocker']='NO_ORIGIN_AVAILABLE_CANDIDATE'; return cell
    train=[]
    for index in range(start,origin+1):
        reasons=[]
        if rows[index][2][target]>day: reasons.append('TRAINING_TARGET_UNAVAILABLE_AT_ORIGIN')
        if any(rows[index-c['features'][0]['lag']][2][c['features'][0]['series']]>day for c in eligible):
            reasons.append('TRAINING_PREDICTOR_UNAVAILABLE_AT_ORIGIN')
        if reasons: cell['excluded_training_rows'].append(dict(day=rows[index][0].isoformat(),reasons=reasons))
        else: train.append(index)
    cell['training_observations']=len(train)
    cell['training_days']=[rows[i][0].isoformat() for i in train]
    if len(train)<12:
        cell['blocker']='INSUFFICIENT_COMMON_AS_OF_TRAINING_ROWS'; return cell
    y=[rows[i][1][target] for i in train]; fits={}
    learned=None
    if 'preprocessing' in parameters:
        from .forecast_preprocessing import fit_predictors,transform
        features={inverse_native._label(f):f for candidate in eligible for f in candidate['features']}
        columns={label:[rows[i-f['lag']][1][f['series']] for i in train] for label,f in features.items()}
        learned=fit_predictors(columns,cell['training_days'],parameters['preprocessing'])
        cell['preprocessing']=learned
    def predictor(value,feature):
        return value if learned is None else transform(value,learned['features'][inverse_native._label(feature)])
    def fit(candidate):
        x=[[F(1)]+[predictor(rows[i-f['lag']][1][f['series']],f) for f in candidate['features']] for i in train]
        computed=_checked_fit(x,y)
        rec=dict(candidate_id=candidate['candidate_id'],features=candidate['features'],status='RANK_DEFICIENT')
        if computed is not None:
            beta,residuals,sse,score=computed
            rec.update(status='FIT',training_sse_exact=str(sse),training_score_exact=str(score),
                       residual_degrees_of_freedom=len(train)-len(beta))
            fits[candidate['candidate_id']]=(candidate,x,beta,residuals,sse,score)
        cell['candidates'].append(rec)
    for candidate in eligible: fit(candidate)
    ranking=sorted((entry[5],len(entry[0]['features']),name) for name,entry in fits.items())
    if len(ranking)>=2:
        joint=dict(candidate_id='joint_top_two',features=fits[ranking[0][2]][0]['features']+fits[ranking[1][2]][0]['features'])
        fit(joint)
        cell['candidate_coverage'].append(dict(joint,state='ELIGIBLE',reason=None))
    ranking=sorted((entry[5],len(entry[0]['features']),name) for name,entry in fits.items())
    if not ranking:
        cell.update(status='INDETERMINATE',blocker='NO_FULL_RANK_ORIGIN_MODEL'); return cell
    winner=ranking[0][2]; candidate,x,beta,residuals,sse,score=fits[winner]
    cell.update(selected_candidate_id=winner,selected_features=candidate['features'],coefficients_exact=list(map(str,beta)),
                training_sse_exact=str(sse),training_score_exact=str(score),arithmetic_oracle='EXACT_CRAMER_ALL_FITTED_CANDIDATES')
    raw_values=[rows[target_index-f['lag']][1][f['series']] for f in candidate['features']]
    values=[F(1)]+[predictor(value,feature) for value,feature in zip(raw_values,candidate['features'])]
    prediction=sum(v*b for v,b in zip(values,beta))
    labels=['intercept']+[inverse_native._label(f) for f in candidate['features']]
    lower=[highs_native._num(parameters['coefficient_bounds'][name][0]) for name in labels]
    upper=[highs_native._num(parameters['coefficient_bounds'][name][1]) for name in labels]
    try:
        program=inverse_native._make_ols_program(x,y,beta,lower,upper)
        # Positive scaling preserves the minimizer and coefficient bounds.
        # Choose a bounded, validated scale before the sole native invocation;
        # do not turn an otherwise valid problem into a numeric-limit failure.
        objective_scale=F(1)
        if learned is not None:
            for scale in (1000,100,10,1):
                candidate_program=dict(program,
                    objective=[str(F(value)*scale) for value in program['objective']],
                    offset=str(F(program['offset'])*scale),
                    hessian=[[str(F(value)*scale) for value in row] for row in program['hessian']])
                try: highs_native.validate(candidate_program)
                except ValueError as error:
                    if str(error) not in ('HIGHS_NUMBER_LIMIT','HIGHS_NUMBER_FORMAT_INVALID'): raise
                    continue
                program=candidate_program;objective_scale=F(scale);break
            else: raise ValueError('HIGHS_NUMBER_LIMIT')
            cell['native_objective_scale']=str(objective_scale)
            cell['native_objective_scale_policy']='LARGEST_VALID_OF_1000_100_10_1_BEFORE_ONE_NATIVE_EXECUTION'
        cell['native']=highs_native.run(program,runtime)
        proxy=dict(selected_coefficients_exact=cell['coefficients_exact'],selected_candidate_id=winner,candidates=cell['candidates'])
        comparison_native=cell['native']
        if objective_scale!=1:
            # The raw native receipt remains unchanged. Only this comparison
            # view converts the scaled solver objective back to target-unit MSE.
            comparison_native=dict(cell['native'],actual=dict(cell['native']['actual'],
                objective=cell['native']['actual']['objective']/float(objective_scale)))
        cell['native_comparison']=inverse_native._compare(proxy,comparison_native,x,y)
        if learned is not None:
            cell['native_comparison'].update(native_objective_scale=str(objective_scale),
                raw_native_objective=cell['native']['actual']['objective'])
        native_prediction=sum(float(v)*b for v,b in zip(values,cell['native']['actual']['solution']))
        error=abs(native_prediction-float(prediction))
        if not math.isfinite(native_prediction) or error>1e-6+1e-7*abs(float(prediction)):
            raise ValueError('FORECAST_NATIVE_PREDICTION_PARITY_FAILED')
    except ValueError as error:
        if str(error) not in {'INVERSE_NATIVE_BOUNDS_EXCLUDE_OLS','HIGHS_NUMBER_LIMIT',
                              'INVERSE_NATIVE_REGRESSION_PARITY_FAILED','FORECAST_NATIVE_PREDICTION_PARITY_FAILED'}: raise
        cell.update(blocker=str(error),native_verification='FAILED' if cell['native'] is not None else 'NOT_RUN')
        return cell
    cell.update(status='FORECAST_VERIFIED',native_verification='PASS',prediction_exact=str(prediction),
        native_prediction_decimal=str(native_prediction),native_prediction_absolute_error=error,
        score_status='TARGET_NOT_AVAILABLE_AT_EVALUATION_CUTOFF')
    cell['forecast_inputs']=[dict(series=f['series'],lag=f['lag'],observation_day=rows[target_index-f['lag']][0].isoformat(),
        available_on=rows[target_index-f['lag']][2][f['series']].isoformat(),value_exact=str(v))
        for f,v in zip(candidate['features'],values[1:])]
    if learned is not None:
        for row,feature,raw in zip(cell['forecast_inputs'],candidate['features'],raw_values):
            row.update(raw_value_exact=str(raw),preprocessing_parameter_id=inverse_native._label(feature),
                       preprocessing_action='CLIPPED' if row['value_exact']!=str(raw) else 'UNCHANGED')
    scoring=rows if evaluation_rows is None else evaluation_rows
    if (target_index<len(scoring) and scoring[target_index][1][target] is not None
            and scoring[target_index][2][target]<=evaluation):
        observed=scoring[target_index][1][target]
        cell.update(observed_target_exact=str(observed),squared_error_exact=str((observed-prediction)**2),
                    score_status='SCORED_AT_DECLARED_EVALUATION_CUTOFF')
        if baseline_index is not None:
            cell['baseline_squared_error_exact']=str((observed-rows[baseline_index][1][target])**2)
    return cell


def _summaries(cells,horizons):
    result=[]
    for horizon in horizons:
        selected=[cell for cell in cells if cell['horizon']==horizon]
        paired=[cell for cell in selected if cell['status']=='FORECAST_VERIFIED' and cell['squared_error_exact'] is not None
                and cell['baseline_squared_error_exact'] is not None]
        model=sum(F(c['squared_error_exact']) for c in paired)/len(paired) if paired else None
        baseline=sum(F(c['baseline_squared_error_exact']) for c in paired)/len(paired) if paired else None
        result.append(dict(horizon=horizon,attempted_cells=len(selected),verified_cells=sum(c['status']=='FORECAST_VERIFIED' for c in selected),
            scored_pairs=len(paired),model_mse_exact=None if model is None else str(model),
            baseline_mse_exact=None if baseline is None else str(baseline),
            paired_loss_difference_exact=None if model is None else str(model-baseline),
            uncertainty='NOT_CALCULATED_POINT_FORECASTS_AND_SELECTED_ORIGINS_ONLY'))
    return result


def _evidence(parameters,result,code):
    from .research_evidence import EvidenceLedger,UnknownRegistry,make_evidence,make_obligation,evaluate_coverage
    subject=result['input_sha256']; scope=dict(input_sha256=subject,origins=parameters['origins'],horizons=parameters['horizons'],
        evaluation_as_of_day=parameters['evaluation_as_of_day'],time_policy=result['time_policy'])
    producer=dict(tool_id='forecast.originwise_validation',version='1',code_sha256=digest(canonical(code)))
    declaration=make_evidence(subject_sha256=subject,producer=producer,operation='DECLARED_FORECAST_PLAN',authority='DECLARED_ONLY',
        epistemic_state='OBSERVED',proof_kind='INPUT_DECLARATION',scope=scope,
        coverage=dict(status='PARTIAL',unknown_dimensions=['source_vintage_authentication']),
        result=dict(source_declared_sha256=parameters['screen']['study']['source_declared_sha256'],rights=parameters['screen']['rights']),
        parent_evidence_ids=[],limitations=['Declared snapshot/revision inputs; no authenticated first-release/access history'])
    calculation=make_evidence(subject_sha256=subject,producer=producer,operation='ORIGINWISE_EXACT_REFIT_AND_NATIVE_FORECAST_CHECK',
        authority='REVIEWED_LOCAL_NUMERIC_EXECUTION',epistemic_state='DERIVED',proof_kind='ORIGINWISE_EXECUTION',scope=scope,
        coverage=dict(status='PARTIAL',unknown_dimensions=['blocked_cells','historical_as_of','external_forecast_validity','probabilistic_calibration']),
        result=result,parent_evidence_ids=[declaration['evidence_id']],limitations=['Origin filtering checks declarations, not release authenticity',
        'Selection uses training only; specification-search/multiple-comparison uncertainty is uncorrected'])
    ledger=EvidenceLedger([declaration,calculation]); unknowns=UnknownRegistry(ledger)
    unknown=unknowns.register('Does this origin-wise result establish externally valid calibrated forecast skill?',scope,
        'INDEPENDENT_FORECAST_VALIDATION','FORECAST_VALIDATION')
    unknowns.revise(unknown['unknown_id'],unknown['revision_digest'],status='OPEN',supporting_evidence_ids=[calculation['evidence_id']],
        recommended_probes=['Authenticate real vintage/access records; prespecify independent origin/horizon holdouts and calibration tests'])
    obligations=[]
    for claim,outcome,proof,blocks in (
        ('originwise_plan_execution','PASS','ORIGINWISE_EXECUTION',[]),
        ('all_declared_forecasts_verified','PASS' if result['verified_cells']==len(result['cells']) else 'UNKNOWN',
         'DECLARED_GRID_COVERAGE',[] if result['verified_cells']==len(result['cells']) else ['Some requested origin/horizon cells remain blocked or indeterminate']),
        ('historical_as_of','UNKNOWN','AUTHENTICATED_INFORMATION_SET',['Snapshot/revision release and access declarations are unauthenticated']),
        ('external_forecast_validity','UNKNOWN','FORECAST_VALIDATION',[]),
        ('probabilistic_calibration','UNKNOWN','PROBABILISTIC_CALIBRATION',['Only point forecasts are implemented in this workflow']),
        ('causal_identification','UNKNOWN','CAUSAL_DESIGN',['No independent causal identification strategy'])):
        local=claim in ('originwise_plan_execution','all_declared_forecasts_verified')
        authority=('REVIEWED_LOCAL_NUMERIC_EXECUTION' if local else 'AUTHENTICATED_SOURCE_REVIEW' if claim=='historical_as_of'
                   else 'EXTERNAL_IDENTIFICATION' if claim=='causal_identification' else 'INDEPENDENT_FORECAST_VALIDATION')
        ob=make_obligation(claim_id=claim,subject_sha256=subject,scope=scope,
            required_authority=authority,
            required_proof_kind=proof,verifier_id=producer['tool_id'],dependencies=[],
            unknown_ids=[unknown['unknown_id']] if claim=='external_forecast_validity' else [],blocking_reasons=blocks,
            next_probe='Inspect origin-specific availability, blocked candidates, scoring cutoff and independent vintage/forecast evidence')
        obligations.append(ob)
        if local:
            ledger.merge([make_evidence(subject_sha256=subject,producer=producer,operation='VERIFY_'+claim.upper(),
                authority='REVIEWED_LOCAL_NUMERIC_EXECUTION',epistemic_state='DERIVED',proof_kind=proof,scope=scope,
                coverage=dict(status='COMPLETE',unknown_dimensions=[]),
                result=dict(obligation_id=ob['obligation_id'],revision=1,previous_verification_id=None,outcome=outcome),
                parent_evidence_ids=[calculation['evidence_id']],limitations=['Coverage scoped to the declared plan; blockers remain binding'])])
    return dict(evidence=ledger.records(),unknowns=unknowns.records(),obligations=obligations,
        unknown_histories={u['unknown_id']:unknowns.history(u['unknown_id']) for u in unknowns.records()},
        coverage=evaluate_coverage(obligations,ledger,unknowns))


def run(parameters,runtime):
    from .execution_control import checkpoint
    original=parameters; parameters=deepcopy(parameters); rows,start,singles,by_day,evaluation=_validated(parameters)
    code=_source_receipt(); began=time.monotonic(); cells=[]
    vintage='revision_history' in parameters
    if vintage:
        from . import forecast_vintages as vintages
        groups=vintages.prepare(parameters)
        scoring,scoring_sha,score_selection=vintages.select(parameters,groups,
            parameters['revision_history']['evaluation_as_of'],rows,evaluation)
    for origin in parameters['origins']:
        origin_rows=rows
        if vintage:
            origin_rows,origin_sha,origin_selection=vintages.select(parameters,groups,
                parameters['revision_history']['origin_as_of'][origin],rows,mechanism_dynamic._day(origin))
        for horizon in parameters['horizons']:
            checkpoint()
            if time.monotonic()-began>180: raise ValueError('FORECAST_RUN_TIME_BUDGET')
            cell=_cell(parameters,origin_rows,start,singles,by_day[origin],horizon,evaluation,runtime,
                       evaluation_rows=scoring if vintage else None)
            if vintage:
                cell.update(origin_as_of=parameters['revision_history']['origin_as_of'][origin],
                    evaluation_as_of=parameters['revision_history']['evaluation_as_of'],
                    origin_snapshot_sha256=origin_sha,evaluation_snapshot_sha256=scoring_sha,
                    information_set_sha256=digest(canonical(dict(
                        daily_information_sha256=cell['information_set_sha256'],origin_snapshot_sha256=origin_sha))),
                    availability_semantics='DATE_MARKERS_MEAN_ELIGIBLE_AT_EXACT_ORIGIN_SEE_REVISION_TIMESTAMPS',
                    scored_target_vintage=None)
                target=parameters['screen']['study']['target']
                if cell['observed_target_exact'] is not None:
                    cell['scored_target_vintage']=deepcopy(score_selection.get((cell['forecast_day'],target)))
                if cell['baseline_source_day'] is not None:
                    cell['baseline_vintage']=deepcopy(origin_selection.get((cell['baseline_source_day'],target)))
                for value in cell.get('forecast_inputs',[]):
                    value['selected_vintage']=deepcopy(origin_selection.get((value['observation_day'],value['series'])))
            cells.append(cell)
    if time.monotonic()-began>180: raise ValueError('FORECAST_RUN_TIME_BUDGET')
    if code!=_source_receipt(): raise ValueError('FORECAST_CODE_CHANGED_DURING_RUN')
    try: unchanged=canonical(original)==canonical(parameters)
    except (ValueError,TypeError,RecursionError,UnicodeError): unchanged=False
    if not unchanged: raise ValueError('FORECAST_INPUT_CHANGED_DURING_RUN')
    native=sum(cell['native'] is not None for cell in cells); verified=sum(cell['status']=='FORECAST_VERIFIED' for cell in cells)
    result=dict(engine='ORIGINWISE_EXACT_CRAMER_AND_PINNED_HIGHS_FORECAST',mode='NO_LLM',
        status='READY' if verified==len(cells) else 'PARTIAL',native_execution='EXECUTED' if native else 'NOT_RUN',
        native_parity='WITHIN_DECLARED_TOLERANCE_FOR_VERIFIED_CELLS_ONLY' if verified else 'NOT_VERIFIED' if native else 'NOT_RUN',
        native_runs=native,verified_cells=verified,input_sha256=digest(canonical(parameters)),code_files=code,
        time_policy='DECLARED_SINGLE_SNAPSHOT_ORIGIN_INFORMATION_FILTER_NOT_AUTHENTICATED_HISTORICAL_AS_OF',
        selection='ORIGINWISE_TRAIN_ONLY_SSE_PER_DF_COMMON_AVAILABLE_SAMPLE',baseline='LAST_AVAILABLE_TARGET_AT_EACH_ORIGIN',
        cells=cells,horizon_summaries=_summaries(cells,parameters['horizons']),historical_as_of='NOT_ESTABLISHED',
        external_forecast_validity='NOT_ESTABLISHED',probabilistic_calibration='NOT_RUN_POINT_FORECASTS_ONLY',
        source_authentication='NOT_ESTABLISHED',scientific_acceptance='NOT_ESTABLISHED',causal_identification='NOT_ESTABLISHED',
        independent_installed_replication='NOT_RUN',os_isolation='NOT_ATTESTED')
    if vintage:
        result['time_policy']='DECLARED_ORIGIN_SPECIFIC_REVISION_SNAPSHOTS_SEPARATE_EVALUATION_VINTAGE_NOT_AUTHENTICATED_HISTORY'
        result['revision_history']=vintages.binding_receipt(parameters)
    result.update(_evidence(parameters,result,code))
    return result


def export_artifacts(parameters,result,chart):
    import csv
    import io
    def text(value): return "'"+value if value.lstrip()[:1] in ('=','+','-','@') else value
    def blob(header,rows):
        stream=io.StringIO(newline=''); writer=csv.writer(stream); writer.writerow(header); writer.writerows(rows)
        return stream.getvalue().encode('utf-8')
    fields=['origin_day','horizon','forecast_day','status','blocker','training_observations','selected_candidate_id',
            'prediction_exact','native_prediction_decimal','baseline_source_day','baseline_prediction_exact','observed_target_exact',
            'squared_error_exact','baseline_squared_error_exact','score_status','information_set_sha256','future_target_used_for_selection']
    if 'revision_history' in parameters:
        fields+=['origin_as_of','evaluation_as_of','origin_snapshot_sha256','evaluation_snapshot_sha256']
    forecasts=blob(fields,[[cell.get(key) for key in fields] for cell in result['cells']])
    candidates=blob(['origin_day','horizon','candidate_id','features','state','reason','fit_status','training_score_exact'],
        [[cell['origin_day'],cell['horizon'],row['candidate_id'],
          ';'.join(inverse_native._label(f) for f in row['features']),row['state'],row['reason'],
          next((fit['status'] for fit in cell['candidates'] if fit['candidate_id']==row['candidate_id']),''),
          next((fit.get('training_score_exact','') for fit in cell['candidates'] if fit['candidate_id']==row['candidate_id']),'')]
         for cell in result['cells'] for row in cell['candidate_coverage']])
    keys=list(result['horizon_summaries'][0])
    summary=blob(keys,[[row[key] for key in keys] for row in result['horizon_summaries']])
    values=blob(['variable','value_exact','unit','source_id'],
        [[label,value,text(chart['unit']),text(chart['source_id'])] for label,value in zip(chart['labels'],chart['values_exact'])])
    evidence=canonical({key:result[key] for key in ('evidence','unknowns','unknown_histories','obligations','coverage')})
    artifacts={'forecasts.csv':forecasts,'forecast-candidates.csv':candidates,'forecast-summary.csv':summary,
               'forecast-evidence.json':evidence,'values.csv':values}
    if 'preprocessing' in parameters:
        artifacts['forecast-preprocessing.csv']=blob(
            ['origin_day','horizon','predictor','lower_exact','upper_exact','training_observations',
             'clipped_training_values','fit_scope','training_days_sha256','raw_training_predictors_sha256'],
            [[cell['origin_day'],cell['horizon'],text(label),limits['lower_exact'],limits['upper_exact'],
              cell['preprocessing']['training_observations'],limits['clipped_training_values'],cell['preprocessing']['fit_scope'],
              cell['preprocessing']['training_days_sha256'],cell['preprocessing']['raw_training_predictors_sha256']]
             for cell in result['cells'] if 'preprocessing' in cell for label,limits in cell['preprocessing']['features'].items()])
    return artifacts


_REVIEW={
    'en':('Origin-wise forecasts','Horizon-specific paired scores','Origin information and blockers','Evidence obligations',
          'Every origin selects and refits on its available training information. Unknown future predictors remain blocked; unavailable targets remain unscored. This single-snapshot simulation is not authenticated historical as-of, calibrated forecast skill or causal identification.'),
    'ko':('원점별 예측','예측기간별 짝지은 평가','원점 정보와 차단 사유','증거·검증 의무',
          '각 원점에서 이용 가능한 학습 정보로 후보를 선택하고 다시 추정합니다. 알려지지 않은 미래 예측변수는 차단하고 이용 불가능한 실제값은 채점하지 않습니다. 단일 snapshot 모의평가는 공식 과거 빈티지·보정된 예측력·인과 식별의 증거가 아닙니다.'),
    'zh-Hans':('逐预测原点','分预测期的配对评价','原点信息与阻碍','证据与验证义务',
               '每个原点仅使用当时可用的训练信息选择并重估。未知未来预测变量保持阻碍，不可用实际值不计分。单一快照模拟不证明真实历史信息集、校准预测能力或因果识别。'),
    'ja':('起点ごとの予測','予測期間別の対応評価','起点の情報と阻害理由','証拠・検証義務',
          '各起点で利用可能な訓練情報だけで候補を選び再推定します。未知の将来説明変数は阻害として残し、利用できない実績値は採点しません。単一スナップショットの模擬評価は真正な過去情報・較正された予測力・因果識別の証明ではありません。'),
}

_TABLE_TEXT={
    'en':('Horizon (days)','Paired origins','Verified cells','Model MSE','Baseline MSE','Model minus baseline','Exact value',
          'Origin','Forecast day','Status','Prediction','Baseline','Scored target','Unscored','Origin details'),
    'ko':('예측기간(일)','짝지은 원점 수','검증된 칸','모형 MSE','기준모형 MSE','모형 − 기준모형','정확값',
          '예측 원점','예측 대상일','상태','예측값','기준모형','채점된 실제값','미채점','원점별 세부 근거'),
    'zh-Hans':('预测期（天）','配对原点数','已验证单元','模型MSE','基准MSE','模型减基准','精确值',
               '预测原点','预测日期','状态','预测值','基准','已计分实际值','未计分','原点详细证据'),
    'ja':('予測期間（日）','対応する起点数','検証済みセル','モデルMSE','基準MSE','モデル − 基準','正確値',
          '予測起点','予測対象日','状態','予測値','基準モデル','採点済み実績値','未採点','起点ごとの詳細証拠'),
}


def render_review_section(parameters,result,locale):
    import json
    from html import escape
    if locale not in _REVIEW: raise ValueError('FORECAST_LOCALE_INVALID')
    title,summary,cells,evidence,limits=_REVIEW[locale]
    if 'revision_history' in parameters:
        limits={
            'en':'Each origin selects and refits from its own declared revision snapshot. The scoring target is selected at a separate evaluation cutoff. Missing or unresolved cells are not filled with future values. Release/access authenticity and external forecast validity remain NOT_ESTABLISHED.',
            'ko':'각 원점의 선언된 개정 스냅샷으로 후보를 선택하고 재추정합니다. 채점할 실제값은 별도 평가 시각에서 선택합니다. 누락·시점 불명확 값을 미래 값으로 채우지 않습니다. 발표·접근 진위와 외부 예측 타당성은 NOT_ESTABLISHED입니다.',
            'zh-Hans':'每个原点从各自声明的修订快照中选择模型并重新估计。评分目标值在独立评价时点选择。缺失或时点不明的值不以未来值填充。发布与访问真实性及外部预测有效性仍为 NOT_ESTABLISHED。',
            'ja':'各起点の申告改定スナップショットから候補を選び再推定します。採点対象の実績値は別の評価時点で選択します。欠測や時点不明の値を将来値で補完しません。公表・アクセスの真正性と外部予測妥当性は NOT_ESTABLISHED です。',
        }[locale]
    def block(value): return '<pre>'+escape(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2))+'</pre>'
    headings=_TABLE_TEXT[locale]
    def numeric(value):
        if value is None: return '<span>'+escape(headings[13])+'</span>'
        approximate=format(float(F(value)),'.8g')
        return ('<span>≈ '+escape(approximate)+'</span><details><summary>'+escape(headings[6])+'</summary><code>'
                +escape(str(value))+'</code></details>')
    table=['<table class="forecast-summary"><thead><tr>']
    table.extend('<th scope="col">'+escape(label)+'</th>' for label in headings[:6])
    table.append('</tr></thead><tbody>')
    for row in result['horizon_summaries']:
        table.append('<tr><th scope="row">'+str(row['horizon'])+'</th><td>'+str(row['scored_pairs'])+'</td><td>'
            +str(row['verified_cells'])+'/'+str(row['attempted_cells'])+'</td>')
        table.extend('<td>'+numeric(row[key])+'</td>' for key in ('model_mse_exact','baseline_mse_exact','paired_loss_difference_exact'))
        table.append('</tr>')
    table.append('</tbody></table><table class="forecast-cells"><thead><tr>')
    table.extend('<th scope="col">'+escape(headings[index])+'</th>' for index in (7,0,8,9,10,11,12))
    table.append('</tr></thead><tbody>')
    for cell in result['cells']:
        table.append('<tr><th scope="row">'+escape(cell['origin_day'])+'</th><td>'+str(cell['horizon'])+'</td><td>'
            +escape(cell['forecast_day'])+'</td><td><code>'+escape(cell['status'])+'</code>'
            +('<p>'+escape(cell['blocker'])+'</p>' if cell['blocker'] else '')+'</td>')
        table.extend('<td>'+numeric(cell[key])+'</td>' for key in ('prediction_exact','baseline_prediction_exact','observed_target_exact'))
        table.append('</tr>')
    table.append('</tbody></table>')
    compact=[{key:value for key,value in cell.items() if key!='native'} for cell in result['cells']]
    data=parameters['screen']['study']; unit=data['series_units'][data['target']]+' squared'
    preprocessing=''
    if 'preprocessing' in parameters:
        from .forecast_preprocessing import render_review_section as render_preprocessing
        preprocessing=render_preprocessing(result,locale)
    return ('<section class="originwise-forecast-review"><h2>'+escape(title)+'</h2><p>'+escape(limits)+'</p>'
        +preprocessing+'<h3>'+escape(summary)+'</h3><p>MSE: '+escape(unit)+'</p>'+''.join(table)
        +'<details><summary>'+escape(headings[14])+'</summary><h3>'+escape(cells)+'</h3>'+block(compact)+'</details>'
        +'<h3>'+escape(evidence)+'</h3>'+block(result['coverage'])+block(result['unknowns'])+'</section>')
