"""Declared lagged WLS and exact covariance sensitivity; assumptions stay open.

Positive analytic weights, not frequency/survey weights. Original HiGHS solves
the declared weighted-MSE QP; covariance is independently authored arithmetic,
not execution of statsmodels or proof of inferential validity.
"""
from copy import deepcopy
import csv
from fractions import Fraction as F
from html import escape
import io
import math
from pathlib import Path

from . import highs_native
from .execution_control import checkpoint
from .inverse_native import _make_ols_program, COEFFICIENT_ABS_TOL, COEFFICIENT_REL_TOL, MSE_ABS_TOL, MSE_REL_TOL
from .mechanism_dynamic import _solve, _dot
from .research_evidence import EvidenceLedger, UnknownRegistry, make_evidence, make_obligation, evaluate_coverage
from .workspace import canonical, digest

METHODS=('CLASSICAL_WLS','HC0','HC1','HC2','HC3','HAC_BARTLETT','CLUSTER_CR0','CLUSTER_CR1')


def _source_receipt():
    names=('regression_sensitivity.py','regression_covariance_verifier.py','mechanism_dynamic.py',
           'inverse_numeric_verifier.py','inverse_native.py','highs_native.py','highs_native_worker.py',
           'research_evidence.py','workspace.py','feature_catalog.py','strict_json.py','execution_control.py')
    out={}
    for name in names:
        path=Path(__file__).with_name(name)
        if path.is_symlink() or not path.is_file(): raise ValueError('REGRESSION_CODE_UNAVAILABLE')
        out[name]=digest(path.read_bytes())
    return out


def _validated(parameters):
    from .mechanism_dynamic import _validate
    if type(parameters) is not dict or set(parameters)!={'data','rights','features','weights','coefficient_bounds','covariance'}:
        raise ValueError('REGRESSION_PARAMETERS_INVALID')
    if parameters['rights']!={'local_processing':'PERMITTED','report_export':'PERMITTED'}:
        raise ValueError('REGRESSION_RIGHTS_REQUIRED')
    data=parameters['data']
    if type(data) is not dict or 'candidates' in data: raise ValueError('REGRESSION_DATA_INVALID')
    rows,end,start=_validate(dict(data,candidates=[dict(candidate_id='declared_wls',features=parameters['features'])]))
    width=1+len(parameters['features'])
    if width>4: raise ValueError('REGRESSION_NATIVE_DIMENSION_BLOCKED')
    days=[row[0].isoformat() for row in rows[start:end+1]]
    weights=parameters['weights']
    if type(weights) is not dict or set(weights)!=set(days): raise ValueError('REGRESSION_TRAINING_WEIGHTS_REQUIRED')
    weights=[highs_native._num(weights[day]) for day in days]
    if any(w<=0 for w in weights): raise ValueError('REGRESSION_POSITIVE_ANALYTIC_WEIGHTS_REQUIRED')
    bounds=parameters['coefficient_bounds']
    if type(bounds) is not list or len(bounds)!=width or any(type(pair) is not list or len(pair)!=2 for pair in bounds):
        raise ValueError('REGRESSION_BOUNDS_REQUIRED')
    low=[highs_native._num(pair[0]) for pair in bounds]; high=[highs_native._num(pair[1]) for pair in bounds]
    if any(lo>=hi for lo,hi in zip(low,high)): raise ValueError('REGRESSION_BOUNDS_INVALID')
    cov=parameters['covariance']
    if type(cov) is not dict or set(cov)!={'estimators','bandwidth','clusters'}:
        raise ValueError('REGRESSION_COVARIANCE_REQUIRED')
    methods=cov['estimators']
    if (type(methods) is not list or not methods or len(methods)>len(METHODS)
            or any(type(m) is not str or m not in METHODS for m in methods) or len(set(methods))!=len(methods)):
        raise ValueError('REGRESSION_COVARIANCE_INVALID')
    bandwidth=cov['bandwidth']; clusters=cov['clusters']
    if 'HAC_BARTLETT' in methods:
        if type(bandwidth) is not int or not 0<=bandwidth<len(days): raise ValueError('REGRESSION_BANDWIDTH_INVALID')
    elif bandwidth is not None: raise ValueError('REGRESSION_UNUSED_BANDWIDTH')
    if any(m.startswith('CLUSTER_') for m in methods):
        if (type(clusters) is not dict or set(clusters)!=set(days) or
                any(type(c) is not str or not 1<=len(c)<=64 or not c.isascii() or
                    any(not(ch.isalnum() or ch in '_-') for ch in c) for c in clusters.values())):
            raise ValueError('REGRESSION_CLUSTER_LABELS_REQUIRED')
    elif clusters is not None: raise ValueError('REGRESSION_UNUSED_CLUSTERS')
    if len(canonical(parameters))>65536: raise ValueError('REGRESSION_INPUT_SIZE')
    return rows,end,start,days,weights,low,high


def validate(parameters):
    """Review only, no fit, covariance calculation or native invocation."""
    _validated(parameters)
    return parameters


def _multiply(a,b):
    return [[sum((x*y for x,y in zip(row,col)),F(0)) for col in zip(*b)] for row in a]


def _bartlett_meat(scores, bandwidth):
    """Exact Bartlett score meat via zero-padded moving windows.

    Each score pair at distance h occurs in bandwidth+1-h windows. Thus
    sum(window_total * window_total')/(bandwidth+1) equals the usual lag
    kernel with weight 1-h/(bandwidth+1), including both boundary tails.
    Trusted callers validate dimensions and 0 <= bandwidth < n. Cost is
    O((n+bandwidth)*p**2), rather than O(n*bandwidth*p**2); no n-by-n matrix.
    """
    n, p = len(scores), len(scores[0])
    total = [F(0)]*p
    meat = [[F(0)]*p for _ in range(p)]
    for end in range(n+bandwidth):
        checkpoint()
        if end < n:
            for j in range(p): total[j] += scores[end][j]
        if end > bandwidth:
            for j in range(p): total[j] -= scores[end-bandwidth-1][j]
        for i in range(p):
            for j in range(i, p): meat[i][j] += total[i]*total[j]
    return [[meat[min(i,j)][max(i,j)]/(bandwidth+1) for j in range(p)] for i in range(p)]


def _bartlett_apply(values,bandwidth):
    """Apply the same Bartlett kernel to a vector in O(n+bandwidth) space/time.

    This is the moving-window operator followed by its adjoint, with both
    zero-padded tails, divided by bandwidth+1. Trusted callers validate lags.
    """
    n=len(values); windows=[]; total=F(0)
    for end in range(n+bandwidth):
        checkpoint()
        if end<n: total+=values[end]
        if end>bandwidth: total-=values[end-bandwidth-1]
        windows.append(total)
    total=sum(windows[:bandwidth+1],F(0)); result=[total/(bandwidth+1)]
    for i in range(1,n):
        total+=windows[i+bandwidth]-windows[i-1]
        result.append(total/(bandwidth+1))
    return result


def _covariances(x,residuals,weights,bread,days,spec,*,degrees_of_freedom=None,leverage_override=None):
    n,p=len(x),len(x[0]); scores=[[w*e*v for v in row] for row,e,w in zip(x,residuals,weights)]
    df=n-p if degrees_of_freedom is None else degrees_of_freedom
    if type(df) is not int or not 0<df<=n: raise ValueError('REGRESSION_COVARIANCE_DF_INVALID')
    needs_leverage=any(method in ('HC2','HC3') for method in spec['estimators'])
    leverage=(leverage_override if leverage_override is not None else
              [w*_dot(row,[_dot(b,row) for b in bread]) for row,w in zip(x,weights)]
              if needs_leverage else None)
    if leverage is not None and (len(leverage)!=n or any(not 0<=h<=1 for h in leverage)):
        raise ValueError('REGRESSION_COVARIANCE_LEVERAGE_INVALID')
    results={}; clusters=spec['clusters']; groups=len(set(clusters.values())) if clusters else 0
    for method in spec['estimators']:
        reason=('LEVERAGE_ONE_CORRECTION_UNDEFINED' if method in ('HC2','HC3') and any(h==1 for h in leverage)
                else 'AT_LEAST_TWO_CLUSTERS_REQUIRED' if method.startswith('CLUSTER_') and groups<2 else None)
        record=dict(status='BLOCKED' if reason else 'COMPUTED_ASSUMPTIONS_UNVERIFIED',reason=reason,matrix_exact=None)
        if reason: results[method]=record; continue
        if method=='CLASSICAL_WLS':
            scale=sum(w*e*e for w,e in zip(weights,residuals))/df
            matrix=[[scale*v for v in row] for row in bread]
        else:
            meat=[[F(0)]*p for _ in range(p)]
            def add(a,b,scale):
                for i in range(p):
                    for j in range(p): meat[i][j]+=scale*a[i]*b[j]
            if method.startswith('CLUSTER_'):
                for group in sorted(set(clusters.values())):
                    score=[sum((s[j] for day,s in zip(days,scores) if clusters[day]==group),F(0)) for j in range(p)]
                    add(score,score,F(1))
            elif method=='HAC_BARTLETT':
                meat=_bartlett_meat(scores,spec['bandwidth'])
            else:
                for t,score in enumerate(scores):
                    correction=(1-leverage[t])**({'HC2':1,'HC3':2}[method]) if method in ('HC2','HC3') else F(1)
                    add(score,score,1/correction)
            factor=F(n,df) if method=='HC1' else F(groups,groups-1)*F(n-1,df) if method=='CLUSTER_CR1' else F(1)
            matrix=[[factor*v for v in row] for row in _multiply(_multiply(bread,meat),bread)]
        if any(matrix[i][i]<0 or matrix[i][j]!=matrix[j][i] for i in range(p) for j in range(p)):
            raise ValueError('REGRESSION_COVARIANCE_ARITHMETIC_FAILED')
        record['matrix_exact']=[list(map(str,row)) for row in matrix]; results[method]=record
    return results


def _native_wls_program(x, y, beta, low, high, weights):
    """Scale only the objective to improve small-curvature QP native accuracy.

    The scale is the count of training rows, not a function of the fitted beta
    or holdout results. A common positive factor preserves the minimizer and
    all original coefficient bounds; HiGHS independently checks the scaled
    model via its exact KKT oracle. Never weaken the coefficient tolerance.
    """
    ordinary = _make_ols_program(x, y, beta, low, high, weights=weights)
    factor = F(len(x))
    proposed = dict(ordinary,
        objective=[str(F(value)*factor) for value in ordinary['objective']],
        hessian=[[str(F(value)*factor) for value in row] for row in ordinary['hessian']],
        offset=str(F(ordinary['offset'])*factor))
    try:
        highs_native.validate(proposed)
    except ValueError as error:
        # Do not sacrifice a numerically valid original QP because its
        # optional conditioning scale exceeds the native numeric contract.
        if str(error) not in ('HIGHS_NUMBER_LIMIT', 'HIGHS_NUMBER_FORMAT_INVALID'):
            raise
        return ordinary, F(1)
    return proposed, factor


def _compare(beta,x,y,weights,native,loss,objective_scale):
    actual=native['actual']; returned=actual['solution']
    if (actual['status']!='OPTIMAL' or type(returned) is not list or len(returned)!=len(beta)
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in returned)):
        raise ValueError('REGRESSION_NATIVE_PARITY_FAILED')
    errors=[abs(float(b)-v) for b,v in zip(beta,returned)]
    direct=sum(w*(target-_dot(row,list(map(lambda v:F(str(v)),returned))))**2 for row,target,w in zip(x,y,weights))/sum(weights)
    objective=actual['objective']
    # Native receipts retain the actual scaled objective. Compare its value
    # only after converting back to the declared weighted-MSE units.
    normalized_objective=(objective/float(objective_scale) if type(objective) in (int,float)
                          and math.isfinite(objective) else None)
    failed=(any(err>COEFFICIENT_ABS_TOL+COEFFICIENT_REL_TOL*abs(float(b)) for b,err in zip(beta,errors))
            or normalized_objective is None
            or abs(float(direct-loss))>MSE_ABS_TOL+MSE_REL_TOL*abs(float(loss))
            or abs(normalized_objective-float(loss))>MSE_ABS_TOL+MSE_REL_TOL*abs(float(loss)))
    return dict(status='FAILED' if failed else 'WITHIN_DECLARED_WLS_TOLERANCE',
        reason='REGRESSION_NATIVE_PARITY_FAILED' if failed else None,maximum_coefficient_error=max(errors),
        original_data_weighted_mse_error=abs(float(direct-loss)),
        native_objective_error=(None if normalized_objective is None else abs(normalized_objective-float(loss))),
        native_objective_scale=str(objective_scale), native_objective_in_weighted_mse_units=normalized_objective,
        coefficient_abs_tolerance=COEFFICIENT_ABS_TOL,coefficient_rel_tolerance=COEFFICIENT_REL_TOL,
        mse_abs_tolerance=MSE_ABS_TOL,mse_rel_tolerance=MSE_REL_TOL)


def _evidence(parameters,result,code):
    subject=result['input_sha256']; scope=dict(training=result['training'],parameter_order=result['parameter_order'],
        weights=parameters['weights'],covariance=parameters['covariance'],model='DECLARED_LAGGED_ANALYTIC_WEIGHT_WLS')
    producer=dict(tool_id='regression.sensitivity',version='1',code_sha256=digest(canonical(code)))
    declared=make_evidence(subject_sha256=subject,producer=producer,operation='DECLARED_WEIGHTED_REGRESSION',
        authority='DECLARED_ONLY',epistemic_state='OBSERVED',proof_kind='INPUT_DECLARATION',scope=scope,
        coverage=dict(status='PARTIAL',unknown_dimensions=['source_authentication','weight_meaning','model_selection_history']),
        result=dict(source_declared_sha256=parameters['data']['source_declared_sha256'],rights=parameters['rights']),
        parent_evidence_ids=[],limitations=['Analytic weights are not survey, frequency or authenticated inverse-variance weights'])
    computed=make_evidence(subject_sha256=subject,producer=producer,operation='WLS_AND_COVARIANCE_ARITHMETIC',
        authority='LOCAL_EXACT_ARITHMETIC',epistemic_state='DERIVED',proof_kind='WLS_COVARIANCE_RESULT',scope=scope,
        coverage=dict(status='PARTIAL',unknown_dimensions=['inferential_validity','causal_identification','external_replication']),
        result=result,parent_evidence_ids=[declared['evidence_id']],limitations=['Exact arithmetic does not verify sampling or exogeneity assumptions'])
    ledger=EvidenceLedger([declared,computed]); unknowns=UnknownRegistry(ledger); obligations=[]
    def obligation(claim,authority,proof,passed,reason=None):
        ob=make_obligation(claim_id=claim,subject_sha256=subject,scope=scope,required_authority=authority,
            required_proof_kind=proof,verifier_id=producer['tool_id'],dependencies=[],unknown_ids=[],
            blocking_reasons=[reason] if reason else [],next_probe='Inspect row weights, covariance assumptions, source authentication and independent design/replication evidence')
        obligations.append(ob)
        if passed is not None:
            checked=make_evidence(subject_sha256=subject,producer=producer,operation='VERIFY_'+claim.upper(),
                authority=authority,epistemic_state='DERIVED',proof_kind=proof,scope=scope,
                coverage=dict(status='COMPLETE',unknown_dimensions=[]),
                result=dict(obligation_id=ob['obligation_id'],revision=1,previous_verification_id=None,outcome='PASS' if passed else 'FAIL'),
                parent_evidence_ids=[computed['evidence_id']],limitations=['Only the explicitly declared arithmetic or native claim is verified'])
            ledger.merge([checked])
    fitted=result['coefficients_exact'] is not None
    obligation('declared_design_fit_or_rank_arithmetic','LOCAL_EXACT_ARITHMETIC','WLS_COVARIANCE_CHECK',True)
    obligation('original_native_weighted_fit','PINNED_UPSTREAM_NUMERIC_EXECUTION','WLS_NATIVE_CHECK',
               result['native_parity']!='FAILED' if fitted else None,
               None if fitted else 'RANK_DEFICIENT_NO_NATIVE_FIT')
    for method,rec in result['covariances'].items():
        obligation('covariance_'+method.lower(),'LOCAL_EXACT_ARITHMETIC','COVARIANCE_'+method,True if rec['matrix_exact'] is not None else None,rec['reason'])
    for claim,question in [('inferential_validity','Are weights, exogeneity, dependence, sampling and cluster asymptotics justified?'),
                           ('post_selection_validity','Was this model specified independently of examined outcomes?'),
                           ('causal_identification','What independent causal design excludes competing mechanisms?'),
                           ('historical_as_of','Are source-specific vintages and researcher access authenticated?'),
                           ('independent_replication','Can an independent installed environment reproduce and challenge this study?')]:
        unk=unknowns.register(question,scope,'EXTERNAL_IDENTIFICATION','SCIENTIFIC_ASSUMPTION_CHECK')
        unknowns.revise(unk['unknown_id'],unk['revision_digest'],status='OPEN',supporting_evidence_ids=[computed['evidence_id']],
            recommended_probes=['Acquire independent design, source and replication evidence; do not infer validity from fit'])
        obligations.append(make_obligation(claim_id=claim,subject_sha256=subject,scope=scope,
            required_authority='EXTERNAL_IDENTIFICATION',required_proof_kind='SCIENTIFIC_ASSUMPTION_CHECK',
            verifier_id=producer['tool_id'],dependencies=[],unknown_ids=[unk['unknown_id']],
            blocking_reasons=['NO_INDEPENDENT_ASSUMPTION_OR_REPLICATION_EVIDENCE'],next_probe=question))
    return dict(evidence=ledger.records(),unknowns=unknowns.records(),obligations=obligations,
                coverage=evaluate_coverage(obligations,ledger,unknowns))


def run(parameters,runtime):
    original=parameters; parameters=deepcopy(parameters)
    rows,end,start,days,weights,low,high=_validated(parameters); code=_source_receipt()
    features=parameters['features']; p=1+len(features)
    x=[[F(1)]+[rows[t-f['lag']][1][f['series']] for f in features] for t in range(start,end+1)]
    y=[rows[t][1][parameters['data']['target']] for t in range(start,end+1)]
    gram=[[sum(w*row[i]*row[j] for row,w in zip(x,weights)) for j in range(p)] for i in range(p)]
    rhs=[sum(w*row[j]*v for row,v,w in zip(x,y,weights)) for j in range(p)]
    beta=_solve(gram,rhs)
    result=dict(engine='EXACT_WLS_COVARIANCE_WITH_PINNED_HIGHS',mode='NO_LLM',input_sha256=digest(canonical(parameters)),code_files=code,
        parameter_order=['intercept']+[f['series']+'[t-'+str(f['lag'])+']' for f in features],
        parameter_units=[parameters['data']['series_units'][parameters['data']['target']]]+[
            '('+parameters['data']['series_units'][parameters['data']['target']]+')/('+parameters['data']['series_units'][f['series']]+')' for f in features],
        training=dict(start_day=days[0],end_day=days[-1],count=len(days),days=days),
        weight_semantics='POSITIVE_ANALYTIC_NOT_FREQUENCY_OR_SURVEY_WEIGHTS',selection='RESEARCHER_DECLARED_MODEL_NO_AUTOMATIC_SELECTION',
        coefficients_exact=None,residuals_exact=None,weighted_mse_exact=None,covariances={},native=None,derived_program=None,comparison=None,
        native_objective_scale=None, native_objective_semantics='SCALED_WEIGHTED_MEAN_SQUARED_RESIDUAL',
        native_execution='NOT_RUN',native_parity='NOT_RUN',status='INDETERMINATE',blocker='RANK_DEFICIENT' if beta is None else None,
        p_values=None,confidence_intervals=None,inferential_validity='NOT_ESTABLISHED',causal_identification='NOT_ESTABLISHED',
        historical_as_of='NOT_ESTABLISHED',scientific_acceptance='NOT_ESTABLISHED',independent_installed_replication='NOT_RUN',os_isolation='NOT_ATTESTED',
        covariance_contract=dict(classical='CONDITIONAL_VARIANCE_PROPORTIONAL_TO_INVERSE_ANALYTIC_WEIGHT',
            hc='INDEPENDENT_OBSERVATIONS_EXOGENEITY_AND_LEVERAGE_REGULARITY_REQUIRED',
            hac='EQUALLY_SPACED_SINGLE_SERIES_BARTLETT_NO_FINITE_SAMPLE_CORRECTION_WEAK_DEPENDENCE_REQUIRED',
            cluster='ONE_WAY_DECLARED_GROUPS_INDEPENDENT_ACROSS_GROUPS_MANY_CLUSTERS_REQUIRED',
            cr1_factor='G/(G-1)*(n-1)/(n-p)',hc1_factor='n/(n-p)',
            selection='NO_POST_SELECTION_OR_MULTIPLE_TESTING_CORRECTION',assumptions='DECLARED_NOT_VERIFIED'))
    if beta is None:
        result['covariances']={method:dict(status='BLOCKED',reason='RANK_DEFICIENT',matrix_exact=None)
                               for method in parameters['covariance']['estimators']}
    if beta is not None:
        residuals=[v-_dot(row,beta) for row,v in zip(x,y)]
        if any(sum(w*row[j]*e for w,row,e in zip(weights,x,residuals)) for j in range(p)):
            raise ValueError('REGRESSION_WEIGHTED_NORMAL_EQUATIONS_FAILED')
        bread_columns=[_solve(gram,[F(int(i==j)) for i in range(p)]) for j in range(p)]
        bread=[list(row) for row in zip(*bread_columns)]
        covariances=_covariances(x,residuals,weights,bread,days,parameters['covariance'])
        loss=sum(w*e*e for w,e in zip(weights,residuals))/sum(weights)
        result.update(coefficients_exact=list(map(str,beta)),residuals_exact=list(map(str,residuals)),weighted_mse_exact=str(loss),covariances=covariances)
    from .regression_covariance_verifier import verify_computed
    result['oracle']=verify_computed(parameters,result)
    if beta is not None:
        program,objective_scale=_native_wls_program(x,y,beta,low,high,weights)
        native=highs_native.run(program,runtime)
        try: comparison=_compare(beta,x,y,weights,native,loss,objective_scale)
        except ValueError as exc:
            if str(exc)!='REGRESSION_NATIVE_PARITY_FAILED': raise
            comparison=dict(status='FAILED',reason=str(exc),coefficient_abs_tolerance=COEFFICIENT_ABS_TOL,
                coefficient_rel_tolerance=COEFFICIENT_REL_TOL,mse_abs_tolerance=MSE_ABS_TOL,mse_rel_tolerance=MSE_REL_TOL)
        failed=comparison['status']=='FAILED'
        result.update(native=native,derived_program=program,comparison=comparison,
            native_objective_scale=str(objective_scale), native_execution='EXECUTED',
            native_parity='FAILED' if failed else 'WITHIN_DECLARED_WLS_TOLERANCE',
            blocker='REGRESSION_NATIVE_PARITY_FAILED' if failed else None,
            status='BLOCKED' if failed else 'READY' if all(c['matrix_exact'] is not None for c in covariances.values()) else 'PARTIAL')
    if code!=_source_receipt(): raise ValueError('REGRESSION_CODE_CHANGED_DURING_RUN')
    try: unchanged=canonical(original)==canonical(parameters)
    except (TypeError,ValueError,RecursionError,UnicodeError): unchanged=False
    if not unchanged: raise ValueError('REGRESSION_INPUT_CHANGED_DURING_RUN')
    result.update(_evidence(parameters,result,code))
    return result


def export_artifacts(parameters,result,chart):
    def blob(header,rows):
        out=io.StringIO(newline=''); writer=csv.writer(out); writer.writerow(header); writer.writerows(rows)
        return out.getvalue().encode('utf-8')
    labels=result['parameter_order']; units=result['parameter_units']; beta=result['coefficients_exact'] or []
    def text(value): return "'"+value if value.lstrip()[:1] in ('=','+','-','@') else value
    actual=result['native']['actual']['solution'] if result['native'] else None
    # An indeterminate native run may have no solution. Preserve every exact
    # coefficient and the BLOCKED receipt instead of dropping CSV rows or
    # fabricating a native estimate.
    if type(actual) is not list: actual=[]
    covariance=[[method,rec['status'],labels[i],labels[j],value,text('('+units[i]+')*('+units[j]+')'),rec['reason'] or '']
        for method,rec in result['covariances'].items() for i,row in enumerate(rec['matrix_exact'] or []) for j,value in enumerate(row)]
    covariance += [[method,rec['status'],'','','','',rec['reason']] for method,rec in result['covariances'].items() if rec['matrix_exact'] is None]
    days=result['training']['days']; clusters=parameters['covariance']['clusters'] or {}
    return {'coefficients.csv':blob(['parameter','value_exact','native_decimal','unit','causal_identification'],
                [[name,b,actual[i] if i<len(actual) else '',text(unit),'NOT_ESTABLISHED']
                 for i,(name,b,unit) in enumerate(zip(labels,beta,units))]),
        'covariances.csv':blob(['estimator','status','row_parameter','column_parameter','covariance_exact','unit','blocker'],covariance),
        'regression-residuals.csv':blob(['target_day','analytic_weight','cluster','residual_exact'],
            [[day,parameters['weights'][day],text(clusters.get(day,'')),e] for day,e in zip(days,result['residuals_exact'] or [])]),
        'regression-evidence.json':canonical({key:result[key] for key in ('evidence','unknowns','obligations','coverage','oracle')})}


def render_review_section(parameters,result,locale):
    titles={'en':('Weighted regression and covariance sensitivity','Coefficient','Covariance diagonal','Assumptions and causal identification remain unverified; no p-values or confidence coverage are established.'),
        'ko':('가중회귀·공분산 민감도','회귀계수','공분산 대각 성분','통계 가정과 인과 식별은 미검증입니다. p값이나 신뢰구간의 포함률을 확립하지 않습니다.'),
        'zh-Hans':('加权回归与协方差敏感性','回归系数','协方差对角线','统计假设和因果识别尚未验证；未确立p值或置信覆盖率。'),
        'ja':('加重回帰・共分散の感度','回帰係数','共分散の対角成分','統計的仮定と因果識別は未検証です。p値や信頼区間の被覆率は確立していません。')}
    title,coefficient,covariance,warning=titles[locale]
    def number(value):
        approx=format(float(F(value)),'.8g')
        return '≈ '+escape(approx)+'<details><summary>exact</summary>'+escape(value)+'</details>'
    content='<section class="regression-sensitivity-review"><h3>'+title+'</h3><p>'+warning+'</p><p>'+escape(result['status'])+' / '+escape(result['native_parity'])+'</p>'
    if result['blocker']: content+='<p>'+escape(result['blocker'])+'</p>'
    content+='<table class="regression-coefficients"><caption>'+coefficient+'</caption><tbody>'
    for name,value,unit in zip(result['parameter_order'],result['coefficients_exact'] or [],result['parameter_units']):
        content+='<tr><th>'+escape(name)+'</th><td>'+number(value)+'</td><td>'+escape(unit)+'</td></tr>'
    content+='</tbody></table><table class="regression-covariances"><caption>'+covariance+'</caption><tbody>'
    for method,rec in result['covariances'].items():
        values='; '.join(escape(name)+': '+number(rec['matrix_exact'][i][i])+' '+escape('('+result['parameter_units'][i]+')^2') for i,name in enumerate(result['parameter_order'])) if rec['matrix_exact'] else escape(rec['reason'])
        content+='<tr><th>'+escape(method)+'</th><td>'+escape(rec['status'])+'</td><td>'+values+'</td></tr>'
    return content+'</tbody></table><p>'+escape(result['weight_semantics'])+'</p><p>inferential_validity / post_selection_validity / causal_identification / historical_as_of / independent_replication: NOT_ESTABLISHED</p></section>'
