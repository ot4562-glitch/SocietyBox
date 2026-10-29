"""Weighted linear quantile regression: exact contact oracle and pinned HiGHS.

Minimize sum w*rho_tau(y-Xb). The native path solves the bounded dual LP,
max y'z subject to X'z=0 and w*(tau-1)<=z<=w*tau. An independently reconstructed
rational primal/dual witness must satisfy exact feasibility, subgradient signs
and zero gap. Numeric proximity alone never certifies the regression solution.
"""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import json
import math
import re
import time

from . import highs_native as native
from . import instrumental_variables as iv
from .inverse_identification import _row_basis,_source_receipt as _rank_receipt
from .optimization_policy import _solve_square
from .execution_control import checkpoint
from .research_evidence import EvidenceLedger,UnknownRegistry,make_evidence,make_obligation,evaluate_coverage
from .workspace import canonical,digest


def _fail(reason):raise ValueError('QUANTILE_'+reason)


def _bounded(values):
    for value in values:
        if type(value) is list:_bounded(value)
        elif max(value.numerator.bit_length(),value.denominator.bit_length())>8192:_fail('EXACT_ARITHMETIC_RESOURCE_LIMIT')
    return values


def _snapshot(q):
    try:raw=canonical(q)
    except (ValueError,TypeError,OverflowError,RecursionError):_fail('JSON_INPUT_REQUIRED')
    if len(raw)>65536:_fail('INPUT_SIZE_LIMIT')
    return json.loads(raw)


def _validated(q):
    if type(q) is not dict or set(q)!={'schema','quantile','model','observations','units','source_declared_sha256','rights'} or q['schema']!='societybox-quantile-regression-v1':_fail('SCHEMA_INVALID')
    if q['rights']!=dict(local_processing='PERMITTED',report_export='PERMITTED'):_fail('RIGHTS_REQUIRED')
    if type(q['source_declared_sha256']) is not str or not re.fullmatch('[a-f0-9]{64}',q['source_declared_sha256']):_fail('SOURCE_REQUIRED')
    model=q['model']
    if type(model) is not dict or set(model)!={'outcome','regressors','intercept'}:_fail('MODEL_INVALID')
    names=model['regressors'];outcome=model['outcome']
    if (type(names) is not list or type(model['intercept']) is not bool or not 1<=len(names)+int(model['intercept'])<=8
        or not iv._text(outcome) or any(not iv._text(v) for v in names) or len(set(names+[outcome]))!=len(names)+1
        or '(Intercept)' in names):_fail('MODEL_NAMES_INVALID')
    if type(q['units']) is not dict or set(q['units'])!=set(names+[outcome]) or any(not iv._text(v) for v in q['units'].values()):_fail('UNITS_REQUIRED')
    tau=native._num(q['quantile'])
    if not 0<tau<1:_fail('QUANTILE_INTERIOR_REQUIRED')
    rows=q['observations'];p=len(names)+int(model['intercept'])
    if type(rows) is not list or not p<=len(rows)<=256:_fail('OBSERVATION_DIMENSIONS_INVALID')
    x=[];y=[];weights=[];ids=set()
    for row in rows:
        if type(row) is not dict or set(row)!={'observation_id','response','regressors','weight'}:_fail('OBSERVATION_INVALID')
        if not iv._text(row['observation_id']) or row['observation_id'] in ids:_fail('OBSERVATION_ID_INVALID')
        ids.add(row['observation_id'])
        if type(row['regressors']) is not dict or set(row['regressors'])!=set(names):_fail('REGRESSORS_REQUIRED')
        x.append([F(1)]*int(model['intercept'])+[native._num(row['regressors'][v]) for v in names])
        y.append(native._num(row['response']));weights.append(native._num(row['weight']))
        if weights[-1]<=0:_fail('POSITIVE_WEIGHTS_REQUIRED')
    basis,_,null=_row_basis(x);_bounded([row for _,row in basis]+null)
    if len(basis)!=p:_fail('DESIGN_RANK_DEFICIENT')
    return x,y,weights,tau,['(Intercept)']*int(model['intercept'])+names


def validate(q):_validated(_snapshot(q));return q


def _loss(x,y,w,tau,beta):
    residual=_bounded([v-iv._dot(row,beta) for row,v in zip(x,y)])
    terms=_bounded([weight*(tau*r if r>=0 else (tau-1)*r) for weight,r in zip(w,residual)])
    value=sum(terms,F(0));_bounded([value]);return value,residual


def _dual_model(q):
    x,y,w,tau,_=_validated(q);n=len(y);p=len(x[0])
    return dict(kind='LP',sense='maximize',objective=y,constraints=[list(col) for col in zip(*x)],
        lower=[v*(tau-1) for v in w],upper=[v*tau for v in w],row_lower=[F(0)]*p,row_upper=[F(0)]*p,
        integrality=[0]*n,hessian=[],offset=F(0))


def _certificate(x,y,w,tau,beta,z):
    if len(beta)!=len(x[0]) or len(z)!=len(y):_fail('CERTIFICATE_DIMENSIONS_INVALID')
    value,residual=_loss(x,y,w,tau,beta)
    if any(not weight*(tau-1)<=v<=weight*tau for weight,v in zip(w,z)):_fail('CERTIFICATE_DUAL_BOUNDS')
    stationarity=_bounded([iv._dot(col,z) for col in zip(*x)])
    if any(stationarity):_fail('CERTIFICATE_DUAL_STATIONARITY')
    if any((r>0 and v!=weight*tau) or (r<0 and v!=weight*(tau-1)) for r,v,weight in zip(residual,z,w)):_fail('CERTIFICATE_SUBGRADIENT')
    dual=iv._dot(y,z);gap=value-dual;_bounded([dual,gap])
    if gap!=0:_fail('CERTIFICATE_NONZERO_GAP')
    return dict(verified_exact=True,primal_objective_exact=str(value),dual_objective_exact=str(dual),gap_exact='0',
        stationarity_exact=list(map(str,stationarity)),dual_weights_exact=list(map(str,z)),
        bounds_and_residual_subgradient='VERIFIED_EXACT',scope='GLOBAL_OPTIMALITY_OF_DECLARED_FINITE_CHECK_LOSS_PROBLEM_ONLY')


def _linear_correction(a,b):
    """Rectangular exact Gauss-Jordan; unpivoted correction coordinates stay zero."""
    n=len(a[0]);aug=[row+[value] for row,value in zip(a,b)];r=0;pivots=[]
    for j in range(n):
        pivot=next((i for i in range(r,len(aug)) if aug[i][j]),None)
        if pivot is None:continue
        aug[r],aug[pivot]=aug[pivot],aug[r];scale=aug[r][j];aug[r]=[v/scale for v in aug[r]]
        for i in range(len(aug)):
            if i!=r:
                scale=aug[i][j];aug[i]=[v-scale*u for v,u in zip(aug[i],aug[r])]
        _bounded(aug);pivots.append(j);r+=1
        if r==len(aug):break
    if any(not any(row[:n]) and row[n] for row in aug):_fail('CERTIFICATE_RECONSTRUCTION_INCONSISTENT')
    result=[F(0)]*n
    for i,j in enumerate(pivots):result[j]=aug[i][-1]
    return result


def _close(actual,expected):
    try:value=float(expected)
    except OverflowError:_fail('NATIVE_EXACT_WITNESS_OUTSIDE_BINARY64_RANGE')
    if not math.isfinite(value):_fail('NATIVE_EXACT_WITNESS_OUTSIDE_BINARY64_RANGE')
    return abs(actual-value)<=native.ABS_TOL+native.REL_TOL*max(abs(actual),abs(value))


def _recover(x,y,w,tau,actual):
    n,p=len(y),len(x[0]);keys={'status','solution','objective','row_values','row_duals','column_duals'}
    if type(actual) is not dict or set(actual)!=keys or actual['status']!='OPTIMAL':_fail('NATIVE_RESULT_SCHEMA_INVALID')
    for key,size in (('solution',n),('column_duals',n),('row_values',p),('row_duals',p)):
        if type(actual[key]) is not list or len(actual[key])!=size:_fail('NATIVE_RESULT_DIMENSIONS_INVALID')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in actual[key]):_fail('NATIVE_RESULT_NONFINITE')
    if type(actual['objective']) not in (int,float) or not math.isfinite(actual['objective']):_fail('NATIVE_RESULT_NONFINITE')
    approx=actual['row_duals'];contacts=[]
    for i,(row,value) in enumerate(zip(x,y)):
        error=float(value)-math.fsum(float(v)*b for v,b in zip(row,approx))
        scale=abs(float(value))+math.fsum(abs(float(v)*b) for v,b in zip(row,approx))
        if abs(error)<=native.ABS_TOL+native.REL_TOL*scale:contacts.append(i)
    if not contacts:_fail('CERTIFICATE_PRIMAL_CONTACTS_NOT_RECOVERED')
    basis,indices,_=_row_basis([x[i] for i in contacts]);_bounded([row for _,row in basis])
    if len(basis)!=p:_fail('CERTIFICATE_PRIMAL_CONTACTS_NOT_RECOVERED')
    chosen=[contacts[j] for j in indices];beta=_solve_square([x[i] for i in chosen],[y[i] for i in chosen]);_bounded(beta)
    _,residual=_loss(x,y,w,tau,beta);z=[];zero=[]
    for i,(r,weight,v) in enumerate(zip(residual,w,actual['solution'])):
        lo,hi=weight*(tau-1),weight*tau
        if r>0:z.append(hi)
        elif r<0:z.append(lo)
        else:
            zero.append(i)
            near=lo if abs(v-float(lo))<=abs(v-float(hi)) else hi
            z.append(near if _close(v,near) else F(str(v)).limit_denominator(10**9))
    # Correct only zero-residual subgradients, preferring coordinates with slack.
    zero.sort(key=lambda i:min(float(z[i]-w[i]*(tau-1)),float(w[i]*tau-z[i])),reverse=True)
    rhs=_bounded([-iv._dot(col,z) for col in zip(*x)])
    if any(rhs):
        if not zero:_fail('CERTIFICATE_DUAL_CONTACTS_NOT_RECOVERED')
        correction=_linear_correction([[x[i][j] for i in zero] for j in range(p)],rhs)
        for i,delta in zip(zero,correction):z[i]+=delta
    _bounded(z);proof=_certificate(x,y,w,tau,beta,z)
    comparisons=[(actual['objective'],F(proof['primal_objective_exact']))]
    comparisons+=list(zip(actual['row_duals'],beta))+list(zip(actual['solution'],z))+list(zip(actual['column_duals'],residual))
    comparisons+=list(zip(actual['row_values'],[F(0)]*p))
    if any(not _close(a,b) for a,b in comparisons):_fail('NATIVE_EXACT_WITNESS_ALIGNMENT_FAILED')
    return beta,proof


def _code_receipt():
    result=iv._code_receipt();result.update(_rank_receipt());result['quantile_regression.py']=digest(Path(__file__).read_bytes());return result


def _assurance(q,result,code):
    current=_code_receipt()
    if current!=code:_fail('CODE_CHANGED_DURING_CALCULATION')
    subject=digest(canonical(q));scope=dict(model=q['model'],quantile=q['quantile'],units=q['units'],weight_scope='POSITIVE_OBJECTIVE_WEIGHTS_NOT_SURVEY_DESIGN')
    evidence=make_evidence(subject_sha256=subject,producer=dict(tool_id='econ.quantile_regression',version='1',code_sha256=digest(canonical(code))),
        operation=result['method'],authority='INDEPENDENT_EXACT_FINITE_CHECK_LOSS_OPTIMALITY',epistemic_state='DERIVED',
        proof_kind='EXHAUSTIVE_CONTACT_OR_PRIMAL_DUAL_CERTIFICATE',scope=scope,result=result,parent_evidence_ids=[],
        coverage=dict(status='PARTIAL',unknown_dimensions=['conditional_quantile_model','weights','source_vintage','causal_identification','independent_replication']),
        limitations=['Exact finite-objective optimality does not prove model specification, causal identification or estimator uncertainty.',
                    'Multiple optima are possible; no covariance, bootstrap, survey design or instrument correction is implemented.'])
    ledger=EvidenceLedger([evidence]);registry=UnknownRegistry(ledger);unknowns=[];obligations=[]
    for claim,question in (
        ('conditional_quantile_model','Is the linear conditional quantile specification appropriate, with justified sampling and dependence assumptions?'),
        ('weights','What sampling or objective interpretation justifies these positive weights? Survey-design inference is not implemented.'),
        ('source_vintage','Are values, units, source rights and historical availability verified?'),
        ('causal_identification','What independent research design supports causal interpretation rather than conditional association?'),
        ('independent_replication','Have external replication and substantive diagnostics validated the data and model?')):
        unknown=registry.register(question,scope,'EXTERNAL_SCIENTIFIC_VALIDATION','QUANTILE_DESIGN_CHECK');unknowns.append(unknown)
        obligations.append(make_obligation(claim_id=claim,subject_sha256=subject,scope=scope,required_authority='EXTERNAL_SCIENTIFIC_VALIDATION',
            required_proof_kind='QUANTILE_DESIGN_CHECK',verifier_id='econ.quantile_regression',dependencies=[],unknown_ids=[unknown['unknown_id']],blocking_reasons=[],next_probe=question))
    return dict(input_sha256=subject,code_sha256=code,evidence=[evidence],unknowns=unknowns,obligations=obligations,coverage=evaluate_coverage(obligations,ledger,registry))


def _result(q,x,y,w,tau,order,beta,method):
    value,residual=_loss(x,y,w,tau,beta)
    return dict(schema='societybox-quantile-result-v1',method=method,status='CALCULATED',implementation_status='INDEPENDENT_IMPLEMENTED',
        parameter_order=order,coefficients_exact=list(map(str,beta)),check_loss_exact=str(value),quantile_exact=str(tau),
        residuals_exact=list(map(str,residual)),observation_ids=[row['observation_id'] for row in q['observations']],
        observations=len(y),design_rank=len(order),coefficient_uniqueness='NOT_ASSESSED_MULTIPLE_OPTIMA_POSSIBLE',
        moment_arithmetic='BOUNDED_EXACT_RATIONAL',weights_scope='POSITIVE_CHECK_LOSS_OBJECTIVE_WEIGHTS_NOT_SURVEY_DESIGN',
        covariance='NOT_IMPLEMENTED',causal_identification='NOT_ESTABLISHED',ui_status='NOT_CONNECTED',native_execution='NOT_RUN',native_parity='NOT_RUN')


def calculate(q):
    """Independent exact finite contact enumeration for small reference problems."""
    q=_snapshot(q);code=_code_receipt();x,y,w,tau,order=_validated(q);n,p=len(y),len(x[0]);work=math.comb(n,p)
    if work>20000 or work*n*p>5000000:_fail('ORACLE_WORK_LIMIT')
    best=None;beta=None;started=time.monotonic();examined=0
    for indices in combinations(range(n),p):
        checkpoint();examined+=1
        if examined%64==0 and time.monotonic()-started>15:_fail('ORACLE_TIME_LIMIT')
        candidate=_solve_square([x[i] for i in indices],[y[i] for i in indices])
        if candidate is None:continue
        _bounded(candidate);value,_=_loss(x,y,w,tau,candidate)
        if best is None or (value,candidate)<(best,beta):best,beta=value,candidate
    if beta is None:_fail('INTERNAL_NO_FULL_RANK_CONTACT_SET')
    result=_result(q,x,y,w,tau,order,beta,'EXACT_EXHAUSTIVE_FULL_RANK_CONTACT_ENUMERATION')
    result['oracle_contact_sets_examined']=examined
    result.update(_assurance(q,result,code));return result


def compare_highs(q,runtime):
    """Pinned original dual LP and independent exact primal/dual optimality proof.

    Native row duals supply a candidate coefficient point; arbitrary numerical
    solutions, unverified provenance or unrecoverable rational witnesses fail.
    This path does not enumerate contact combinations as the sample grows.
    """
    q=_snapshot(q);code=_code_receipt();x,y,w,tau,order=_validated(q)
    if len(native._bytes(q))>65536:_fail('NATIVE_INPUT_SIZE_LIMIT')
    raw,worker=native._checked_worker(q,runtime,'--quantile-dual')
    beta,proof=_recover(x,y,w,tau,raw['actual'])
    result=_result(q,x,y,w,tau,order,beta,'HIGHS_BOUNDED_DUAL_LP_WITH_EXACT_PRIMAL_DUAL_CERTIFICATE')
    result.update(native_execution='EXECUTED',native_parity='NUMERICAL_PARITY_VERIFIED',upstream_status='ORIGINAL_API_EXECUTED',
        upstream=raw['upstream'],actual=raw['actual'],certificate=proof,primal_dual_gap_exact=proof['gap_exact'],
        worker_sha256=digest(worker.read_bytes()),native_alignment_tolerance=dict(absolute=native.ABS_TOL,relative=native.REL_TOL),
        native_parity_scope='EXACT_FINITE_OBJECTIVE_KKT_WITNESS_WITH_BINARY64_NATIVE_ALIGNMENT_NOT_UNIQUE_COEFFICIENT_OR_VENDOR_INFERENCE_PARITY',
        os_isolation='NOT_ATTESTED')
    result.update(_assurance(q,result,code));return result
