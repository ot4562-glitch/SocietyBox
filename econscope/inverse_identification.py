"""Finite training-prediction equivalence sets, not statistical confidence sets.

Original pinned HiGHS LPs compute bounded coefficient projections. Exact row
reduction verifies all original training rows, rank and nullspace. Reference
coefficients and bounds are declarations; no estimator or causal model is inferred.
"""
import csv
from copy import deepcopy
from fractions import Fraction as F
from html import escape
import io
import math
from pathlib import Path

from . import highs_native
from .feature_catalog import describe_feature
from .research_evidence import EvidenceLedger, UnknownRegistry, make_evidence, make_obligation, evaluate_coverage
from .workspace import canonical, digest


def _source_receipt():
    names=('inverse_identification.py','mechanism_dynamic.py','highs_native.py','highs_native_worker.py',
           'research_evidence.py','workspace.py','feature_catalog.py','strict_json.py')
    result={}
    for name in names:
        path=Path(__file__).with_name(name)
        if path.is_symlink() or not path.is_file(): raise ValueError('IDENTIFICATION_CODE_UNAVAILABLE')
        result[name]=digest(path.read_bytes())
    return result


def _label(feature): return feature['series']+'[t-'+str(feature['lag'])+']'
def _dot(a,b): return sum((x*y for x,y in zip(a,b)),F(0))


def _validated(parameters):
    from .mechanism_dynamic import _validate
    if type(parameters) is not dict or set(parameters)!={'data','rights','features','reference_coefficients','coefficient_bounds'}:
        raise ValueError('IDENTIFICATION_SCHEMA_INVALID')
    if parameters['rights']!={'local_processing':'PERMITTED','report_export':'PERMITTED'}:
        raise ValueError('IDENTIFICATION_RIGHTS_REQUIRED')
    data=parameters['data']
    if type(data) is not dict or 'candidates' in data: raise ValueError('IDENTIFICATION_DATA_INVALID')
    query=dict(data,candidates=[dict(candidate_id='declared_reference',features=parameters['features'])])
    rows,end,start=_validate(query)
    width=1+len(parameters['features'])
    if width>4: raise ValueError('IDENTIFICATION_NATIVE_DIMENSION_BLOCKED')
    beta=parameters['reference_coefficients']; bounds=parameters['coefficient_bounds']
    if (type(beta) is not list or len(beta)!=width or type(bounds) is not list or len(bounds)!=width
            or any(type(pair) is not list or len(pair)!=2 for pair in bounds)):
        raise ValueError('IDENTIFICATION_COEFFICIENTS_OR_BOUNDS_INVALID')
    beta=list(map(highs_native._num,beta))
    low=[highs_native._num(pair[0]) for pair in bounds]; high=[highs_native._num(pair[1]) for pair in bounds]
    if any(not lo<=b<=hi for lo,b,hi in zip(low,beta,high)):
        raise ValueError('IDENTIFICATION_REFERENCE_OUTSIDE_BOUNDS')
    return rows,end,start,beta,low,high


def validate(parameters):
    """Review-only data, rights, release and numeric validation; no fitting/LP."""
    _validated(parameters)
    return parameters


def _row_basis(design):
    """Exact reduced echelon rows, preserving indices of independent originals."""
    basis=[]; indices=[]
    for index,original in enumerate(design):
        row=list(original)
        for pivot,base in basis:
            scale=row[pivot]; row=[a-scale*b for a,b in zip(row,base)]
        pivot=next((j for j,value in enumerate(row) if value),None)
        if pivot is None: continue
        scale=row[pivot]; row=[value/scale for value in row]
        basis=[(col,[a-base[pivot]*b for a,b in zip(base,row)]) for col,base in basis]
        basis.append((pivot,row)); basis.sort(key=lambda entry:entry[0]); indices.append(index)
    for original in design:
        row=list(original)
        for pivot,base in basis:
            scale=row[pivot]; row=[a-scale*b for a,b in zip(row,base)]
        if any(row): raise ValueError('IDENTIFICATION_ROW_COMPRESSION_FAILED')
    pivots={col for col,_ in basis}; nullspace=[]
    for free in range(len(design[0])):
        if free not in pivots:
            vector=[F(0)]*len(design[0]); vector[free]=F(1)
            for pivot,row in basis: vector[pivot]=-row[free]
            if any(_dot(row,vector) for row in design): raise ValueError('IDENTIFICATION_NULLSPACE_FAILED')
            nullspace.append(vector)
    return basis,indices,nullspace


def _verify_endpoint(native,design,reference,low,high,index):
    expected=native['comparison']['oracle']
    if expected['status']!='OPTIMAL' or native['actual']['status']!='OPTIMAL':
        raise ValueError('IDENTIFICATION_ENDPOINT_NOT_OPTIMAL')
    exact=list(map(F,expected['solution_exact']))
    if (len(exact)!=len(reference) or any(not lo<=x<=hi for lo,x,hi in zip(low,exact,high))
            or F(expected['objective_exact'])!=exact[index]
            or any(_dot(row,exact)!=_dot(row,reference) for row in design)):
        raise ValueError('IDENTIFICATION_EXACT_WITNESS_INVALID')
    returned=native['actual']['solution']
    if (type(returned) is not list or len(returned)!=len(reference)
            or any(type(value) not in (int,float) or not math.isfinite(value) for value in returned)):
        raise ValueError('IDENTIFICATION_FULL_DESIGN_NATIVE_PARITY_FAILED')
    floating=list(map(lambda value:F(str(value)),returned))
    objective=native['actual']['objective']
    if (type(objective) not in (int,float) or not math.isfinite(objective)
            or not math.isclose(returned[index],float(exact[index]),abs_tol=highs_native.ABS_TOL,rel_tol=highs_native.REL_TOL)
            or not math.isclose(objective,returned[index],abs_tol=highs_native.ABS_TOL,rel_tol=highs_native.REL_TOL)):
        raise ValueError('IDENTIFICATION_ENDPOINT_OBJECTIVE_MISMATCH')
    for row in design:
        target=_dot(row,reference)
        error=abs(float(_dot(row,floating)-target))
        if error>highs_native.ABS_TOL+highs_native.REL_TOL*abs(float(target)):
            raise ValueError('IDENTIFICATION_FULL_DESIGN_NATIVE_PARITY_FAILED')
    return exact


def _probes(rows,end,features,witnesses):
    probes=[]
    for t in range(end+1,len(rows)):
        row=[F(1)]+[rows[t-f['lag']][1][f['series']] for f in features]
        projections=sorted((_dot(row,list(map(F,w['coefficients_exact']))),w['witness_id']) for w in witnesses)
        if projections and projections[-1][0]>projections[0][0]:
            probes.append(dict(origin_day=rows[t-1][0].isoformat(),target_day=rows[t][0].isoformat(),
                lower_prediction_exact=str(projections[0][0]),upper_prediction_exact=str(projections[-1][0]),
                prediction_gap_exact=str(projections[-1][0]-projections[0][0]),
                lower_witness_id=projections[0][1],upper_witness_id=projections[-1][1],
                interpretation='RETROSPECTIVE_SEPARATING_PREDICTOR_CONFIGURATION_NOT_A_CAUSAL_OR_FORECAST_CLAIM'))
    return probes


def _evidence(parameters,computed,code):
    subject=computed['input_sha256']
    scope=dict(input_sha256=subject,training=computed['training'],parameter_order=computed['parameter_order'],
               reference_coefficients=computed['reference_coefficients_exact'],
               bounds=parameters['coefficient_bounds'],equivalence_scope=computed['equivalence_scope'])
    producer=dict(tool_id='identification.coefficient_ranges',version='1',code_sha256=digest(canonical(code)))
    declared=make_evidence(subject_sha256=subject,producer=producer,operation='DECLARED_REFERENCE_AND_DESIGN',
        authority='DECLARED_ONLY',epistemic_state='OBSERVED',proof_kind='INPUT_DECLARATION',scope=scope,
        coverage=dict(status='PARTIAL',unknown_dimensions=['source_authentication','release_authentication','rights_authentication']),
        result=dict(source_declared_sha256=parameters['data']['source_declared_sha256'],rights=parameters['rights']),
        parent_evidence_ids=[],limitations=['The reference vector need not be an estimated or best-fitting model'])
    calculation=make_evidence(subject_sha256=subject,producer=producer,operation='EXACT_RANK_AND_ORIGINAL_LP_EXTREMA',
        authority='PINNED_UPSTREAM_NUMERIC_EXECUTION',epistemic_state='DERIVED',proof_kind='FINITE_COEFFICIENT_RANGES',scope=scope,
        coverage=dict(status='PARTIAL',unknown_dimensions=['structural_causal_identification','external_replication','unchecked_competitor_probes']),
        result=computed,parent_evidence_ids=[declared['evidence_id']],
        limitations=['Finite training-prediction equivalence is not equality of probability laws or a confidence region',
                     'Native endpoints and exact witnesses cover coefficient projections, not every mechanism or probe'])
    ledger=EvidenceLedger([declared,calculation]); unknowns=UnknownRegistry(ledger)
    unk=unknowns.register('Is the structural or causal interpretation independently identified?',scope,
                          'EXTERNAL_IDENTIFICATION','CAUSAL_DESIGN')
    unknowns.revise(unk['unknown_id'],unk['revision_digest'],status='OPEN',
        supporting_evidence_ids=[calculation['evidence_id']],
        recommended_probes=['Authenticate independent measurements and the identification strategy; challenge alternative mechanisms',
                            'Observe predictor configurations that separate the retained exact endpoint witnesses'])
    obligations=[]
    def check(claim,outcome,proof,dependencies=(),blocked=False):
        ob=make_obligation(claim_id=claim,subject_sha256=subject,scope=scope,
            required_authority='EXTERNAL_IDENTIFICATION' if blocked else 'PINNED_UPSTREAM_NUMERIC_EXECUTION',
            required_proof_kind=proof,verifier_id=producer['tool_id'],dependencies=list(dependencies),
            unknown_ids=[unk['unknown_id']] if blocked else [],
            blocking_reasons=['No independent structural/causal design evidence'] if blocked else [],
            next_probe='Inspect exact witnesses, bounds, separating observations and remaining independent identification evidence')
        obligations.append(ob)
        if not blocked:
            ledger.merge([make_evidence(subject_sha256=subject,producer=producer,operation='VERIFY_'+claim.upper(),
                authority='PINNED_UPSTREAM_NUMERIC_EXECUTION',epistemic_state='DERIVED',proof_kind=proof,scope=scope,
                coverage=dict(status='COMPLETE',unknown_dimensions=[]),
                result=dict(obligation_id=ob['obligation_id'],revision=1,previous_verification_id=None,outcome=outcome),
                parent_evidence_ids=[calculation['evidence_id']],limitations=['Claim is scoped to the declared finite design and bounds'])])
        return ob
    numeric=check('coefficient_range_computation','PASS','FINITE_COEFFICIENT_RANGES')
    check('data_only_parameter_identification','PASS' if computed['nullity']==0 else 'FAIL','FINITE_DESIGN_RANK',[numeric['obligation_id']])
    check('coefficient_uniqueness_under_declared_bounds','PASS' if computed['bounded_unique'] else 'FAIL',
          'BOUNDED_COEFFICIENT_UNIQUENESS',[numeric['obligation_id']])
    check('structural_causal_identification','UNKNOWN','CAUSAL_DESIGN',blocked=True)
    return dict(evidence=ledger.records(),unknowns=unknowns.records(),
                unknown_histories={u['unknown_id']:unknowns.history(u['unknown_id']) for u in unknowns.records()},
                obligations=obligations,coverage=evaluate_coverage(obligations,ledger,unknowns))


def run(parameters,runtime):
    original_parameters=parameters
    parameters=deepcopy(parameters)
    rows,end,start,reference,low,high=_validated(parameters)
    code=_source_receipt(); features=parameters['features']; width=len(reference)
    design=[[F(1)]+[rows[t-f['lag']][1][f['series']] for f in features] for t in range(start,end+1)]
    basis,indices,nullspace=_row_basis(design)
    reduced=[design[i] for i in indices]
    targets=[_dot(row,reference) for row in reduced]
    template=dict(kind='LP',sense='minimize',objective=[0]*width,
        constraints=[list(map(str,row)) for row in reduced],row_lower=list(map(str,targets)),row_upper=list(map(str,targets)),
        lower=list(map(str,low)),upper=list(map(str,high)),integrality=[0]*width,hessian=[],offset=0)
    ranges=[]; native_runs=[]; witnesses={}; labels=['intercept']+[_label(f) for f in features]
    for index,label in enumerate(labels):
        endpoints=[]
        for sense in ('minimize','maximize'):
            program=dict(template,sense=sense,objective=[int(j==index) for j in range(width)])
            native=highs_native.run(program,runtime)
            witness=_verify_endpoint(native,design,reference,low,high,index)
            key='wit_'+digest(canonical(list(map(str,witness))))
            witnesses[key]=dict(witness_id=key,coefficients_exact=list(map(str,witness)))
            endpoints.append((witness[index],native['actual']['solution'][index],key))
            native_runs.append(native)
        if endpoints[0][0]>endpoints[1][0]: raise ValueError('IDENTIFICATION_RANGE_ORDER_FAILED')
        ranges.append(dict(parameter=label,lower_exact=str(endpoints[0][0]),upper_exact=str(endpoints[1][0]),
            lower_native_decimal=str(endpoints[0][1]),upper_native_decimal=str(endpoints[1][1]),
            lower_witness_id=endpoints[0][2],upper_witness_id=endpoints[1][2]))
    witness_list=[witnesses[key] for key in sorted(witnesses)]
    probes=_probes(rows,end,features,witness_list)
    unique=all(F(row['lower_exact'])==F(row['upper_exact']) for row in ranges)
    result=dict(engine='EXACT_DESIGN_EQUIVALENCE_WITH_PINNED_HIGHS_LP',mode='NO_LLM',
        native_execution='EXECUTED',native_parity='EXACT_VERTEX_RANGES_AND_FULL_DESIGN_NATIVE_FEASIBILITY',
        product_feature=describe_feature('coefficient_identification'),input_sha256=digest(canonical(parameters)),
        design_rank=len(basis),parameter_count=width,nullity=width-len(basis),
        nullspace_exact=[list(map(str,vector)) for vector in nullspace],parameter_order=labels,
        training=dict(start_day=rows[start][0].isoformat(),end_day=rows[end][0].isoformat(),observations=len(design)),
        independent_training_row_indices=[start+i for i in indices],
        reference_coefficients_exact=list(map(str,reference)),reference_status='RESEARCHER_DECLARED_NOT_ESTIMATED',
        equivalence_scope='FINITE_TRAINING_DESIGN_PREDICTIONS_NOT_PROBABILITY_LAW_OR_CAUSAL_EQUIVALENCE',
        bounded_unique=unique,conclusion='UNIQUE_WITHIN_DECLARED_BOUNDS_NOT_CAUSAL_IDENTIFICATION' if unique else 'COEFFICIENTS_UNDERDETERMINED',
        uniqueness_basis='FULL_COLUMN_RANK_FINITE_DESIGN' if not nullspace else 'DECLARED_BOUNDS_WITH_RANK_DEFICIENT_DATA' if unique else 'MULTIPLE_FEASIBLE_COEFFICIENT_VECTORS',
        ranges=ranges,witnesses=witness_list,native_runs=native_runs,compressed_program_template=template,
        separating_observations=probes,probe_coverage='PARTIAL_CHECKED_ENDPOINT_WITNESSES_ONLY',
        probe_conclusion='SEPARATING_CONFIGURATIONS_OBSERVED_RETROSPECTIVELY' if probes else 'NO_SEPARATION_IN_CHECKED_WITNESSES_NOT_PROOF_OF_ABSENCE',
        holdout_targets_used=False,causal_identification='NOT_ESTABLISHED',historical_as_of='NOT_ESTABLISHED',
        source_authentication='NOT_ESTABLISHED',independent_installed_replication='NOT_RUN',
        scientific_acceptance='NOT_ESTABLISHED',os_isolation='NOT_ATTESTED',code_files=code)
    if code!=_source_receipt(): raise ValueError('IDENTIFICATION_CODE_CHANGED_DURING_RUN')
    try:
        unchanged=canonical(original_parameters)==canonical(parameters)
    except (ValueError,TypeError,RecursionError,UnicodeError):
        unchanged=False
    if not unchanged: raise ValueError('IDENTIFICATION_INPUT_CHANGED_DURING_RUN')
    result.update(_evidence(parameters,result,code))
    return result


def export_artifacts(parameters,result,chart):
    """Pure exports of the freshly computed artifact snapshot."""
    def csv_bytes(header,rows):
        stream=io.StringIO(newline=''); writer=csv.writer(stream)
        writer.writerow(header); writer.writerows(rows)
        return stream.getvalue().encode('utf-8')
    def text(value): return "'"+value if value.lstrip()[:1] in ('=','+','-','@') else value
    data=parameters['data']; target_unit=data['series_units'][data['target']]
    units=[target_unit]+[target_unit+'/'+data['series_units'][f['series']] for f in parameters['features']]
    ranges=csv_bytes(['parameter','lower_exact','upper_exact','lower_native_decimal','upper_native_decimal','unit','scope'],
        [[row[key] for key in ('parameter','lower_exact','upper_exact','lower_native_decimal','upper_native_decimal')]
         +[text(unit),'FINITE_PREDICTION_EQUIVALENCE_NOT_CONFIDENCE_INTERVAL'] for row,unit in zip(result['ranges'],units)])
    witnesses=csv_bytes(['witness_id','parameter','coefficient_exact','unit'],
        [[row['witness_id'],label,value,text(unit)] for row in result['witnesses']
         for label,value,unit in zip(result['parameter_order'],row['coefficients_exact'],units)])
    fields=['origin_day','target_day','lower_prediction_exact','upper_prediction_exact','prediction_gap_exact',
            'lower_witness_id','upper_witness_id','interpretation']
    probes=csv_bytes(fields,[[row[key] for key in fields] for row in result['separating_observations']])
    values=csv_bytes(['variable','value_exact','unit','source_id'],
        [[label,value,chart['unit'],text(chart['source_id'])] for label,value in zip(chart['labels'],chart['values_exact'])])
    return {'ranges.csv':ranges,'witnesses.csv':witnesses,'separating-observations.csv':probes,
            'identification.json':canonical(result),'values.csv':values}


_REVIEW_TEXT={
    'en':('Prediction equivalence','Coefficient ranges','Separating observations','Evidence, unknowns and verification',
          'These ranges preserve a declared reference prediction on this finite training design and within declared bounds. They are not confidence intervals or causal identification. Additional observations are retrospective examples among checked witnesses; an empty list is not proof of absence.'),
    'ko':('예측 동치','계수 범위','후보를 구별하는 관측','증거·미지수·검증 의무',
          '이 범위는 유한 학습 표본과 선언한 제약 안에서 기준 계수의 예측을 보존합니다. 신뢰구간이나 인과 식별이 아닙니다. 추가 관측은 검토한 증인 후보들의 사후 구별 사례이며, 빈 목록은 차이가 없다는 증명이 아닙니다.'),
    'zh-Hans':('预测等价','系数范围','区分候选的观测','证据、未解问题与验证义务',
               '这些范围仅在有限训练设计和声明约束内保持参考预测。它们不是置信区间或因果识别。额外观测是已检查候选的事后区分案例，空列表不证明不存在差异。'),
    'ja':('予測同値','係数範囲','候補を区別する観測','証拠・未解決事項・検証義務',
          'この範囲は有限の訓練設計と申告された制約の中で参照予測を保ちます。信頼区間や因果識別ではありません。追加観測は確認した候補の事後的な区別例であり、空の一覧は差がないことの証明ではありません。'),
}


def render_review_section(parameters,result,locale):
    """Escaped projection only; caller must verify its existing engine bundle."""
    import json
    if locale not in _REVIEW_TEXT: raise ValueError('IDENTIFICATION_LOCALE_INVALID')
    title,ranges_title,probes_title,evidence_title,limits=_REVIEW_TEXT[locale]
    def block(value): return '<pre>'+escape(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2))+'</pre>'
    fields=('conclusion','uniqueness_basis','design_rank','parameter_count','nullity','reference_status',
            'equivalence_scope','probe_coverage','probe_conclusion','holdout_targets_used','causal_identification')
    return ('<section class="coefficient-identification-review"><h2>'+escape(title)+'</h2><p>'+escape(limits)+'</p>'
        +block({key:result[key] for key in fields})+'<h3>'+escape(ranges_title)+'</h3>'+block(result['ranges'])
        +'<h3>'+escape(probes_title)+'</h3>'+block(result['separating_observations'])
        +'<h3>'+escape(evidence_title)+'</h3>'+block(result['coverage'])+block(result['unknowns'])+'</section>')
