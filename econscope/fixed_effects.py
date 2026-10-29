"""Exact weighted multiway projection with original HiGHS slope estimation.

This is an independently authored dense reference workflow, not execution or
large-data parity of PyFixest/linearmodels/Stata. Finite operation budgets keep
the independent full-dummy oracle bounded. Analytic weights are not survey or
frequency weights; computed covariance does not authenticate its assumptions.
"""
from copy import deepcopy
from fractions import Fraction as F
from hashlib import sha256
from html import escape
from pathlib import Path
import re

from . import highs_native,regression_sensitivity as wls
from .mechanism_dynamic import _solve,_dot
from .execution_control import checkpoint
from .workspace import canonical,digest

METHODS=tuple(method for method in wls.METHODS if method!='HAC_BARTLETT')
_NAME=re.compile(r'[A-Za-z][A-Za-z0-9_]{0,31}\Z',re.ASCII)


def _text(value,cap=128):
    if type(value) is not str or not value.strip() or len(value)>cap or any(ord(c)<32 for c in value):
        raise ValueError('FIXED_EFFECTS_TEXT_INVALID')


def _validated(q):
    fields={'schema','data_kind','source_declared_sha256','rights','target_unit','predictor_units',
            'fixed_effects','rows','coefficient_bounds','covariance'}
    if (type(q) is not dict or set(q)!=fields or q['schema']!='societybox-fixed-effects-regression-v1'
            or q['data_kind'] not in ('SYNTHETIC','RESEARCHER_SUPPLIED')):
        raise ValueError('FIXED_EFFECTS_SCHEMA_INVALID')
    if q['rights']!={'local_processing':'PERMITTED','report_export':'PERMITTED'}:
        raise ValueError('FIXED_EFFECTS_RIGHTS_REQUIRED')
    if type(q['source_declared_sha256']) is not str or re.fullmatch('[a-f0-9]{64}',q['source_declared_sha256']) is None:
        raise ValueError('FIXED_EFFECTS_SOURCE_DIGEST_REQUIRED')
    _text(q['target_unit']); units=q['predictor_units']; dimensions=q['fixed_effects']; rows=q['rows']
    if (type(units) is not dict or not 1<=len(units)<=4
            or any(type(name) is not str or _NAME.fullmatch(name) is None for name in units)):
        raise ValueError('FIXED_EFFECTS_PREDICTOR_DECLARATION_INVALID')
    for unit in units.values(): _text(unit)
    if (type(dimensions) is not list or not 1<=len(dimensions)<=8
            or any(type(name) is not str or _NAME.fullmatch(name) is None for name in dimensions)
            or len(set(dimensions))!=len(dimensions)):
        raise ValueError('FIXED_EFFECTS_DIMENSIONS_INVALID')
    if type(rows) is not list or not 4<=len(rows)<=64:
        raise ValueError('FIXED_EFFECTS_DENSE_REFERENCE_ROW_BUDGET')
    labels=sorted(units); y=[]; x=[]; weights=[]; ids=[]; levels={dim:set() for dim in dimensions}
    def number(value):
        if type(value) is not str: raise ValueError('FIXED_EFFECTS_EXACT_NUMERIC_STRING_REQUIRED')
        return highs_native._num(value)
    for row in rows:
        if type(row) is not dict or set(row)!={'id','y','x','analytic_weight','fixed_effects'}:
            raise ValueError('FIXED_EFFECTS_ROW_INVALID')
        _text(row['id']);ids.append(row['id'])
        if (type(row['x']) is not dict or set(row['x'])!=set(labels)
                or type(row['fixed_effects']) is not dict or set(row['fixed_effects'])!=set(dimensions)):
            raise ValueError('FIXED_EFFECTS_ROW_FIELDS_INVALID')
        y.append(number(row['y']));x.append([number(row['x'][name]) for name in labels]);weights.append(number(row['analytic_weight']))
        if weights[-1]<=0: raise ValueError('FIXED_EFFECTS_POSITIVE_ANALYTIC_WEIGHT_REQUIRED')
        for dim in dimensions:
            _text(row['fixed_effects'][dim]);levels[dim].add(row['fixed_effects'][dim])
    if len(set(ids))!=len(ids): raise ValueError('FIXED_EFFECTS_ROW_ID_CONFLICT')
    if sum(map(len,levels.values()))>24: raise ValueError('FIXED_EFFECTS_DENSE_DUMMY_ORACLE_BUDGET')
    bounds=q['coefficient_bounds']
    if type(bounds) is not dict or set(bounds)!=set(labels): raise ValueError('FIXED_EFFECTS_BOUNDS_REQUIRED')
    low=[];high=[]
    for name in labels:
        pair=bounds[name]
        if type(pair) is not list or len(pair)!=2: raise ValueError('FIXED_EFFECTS_BOUNDS_INVALID')
        low.append(number(pair[0]));high.append(number(pair[1]))
        if low[-1]>=high[-1]: raise ValueError('FIXED_EFFECTS_BOUNDS_INVALID')
    cov=q['covariance']
    if type(cov) is not dict or set(cov)!={'estimators','clusters'}: raise ValueError('FIXED_EFFECTS_COVARIANCE_INVALID')
    methods=cov['estimators'];groups=cov['clusters']
    if (type(methods) is not list or not methods or any(type(m) is not str or m not in METHODS for m in methods)
            or len(set(methods))!=len(methods)):
        raise ValueError('FIXED_EFFECTS_COVARIANCE_METHOD_INVALID')
    if any(method.startswith('CLUSTER_') for method in methods):
        if type(groups) is not dict or set(groups)!=set(ids): raise ValueError('FIXED_EFFECTS_CLUSTER_LABELS_REQUIRED')
        for value in groups.values(): _text(value)
    elif groups is not None: raise ValueError('FIXED_EFFECTS_UNUSED_CLUSTER_LABELS')
    if len(canonical(q))>60000: raise ValueError('FIXED_EFFECTS_INPUT_BUDGET')
    return labels,ids,y,x,weights,levels,low,high


def validate(q):
    """Review declarations only, never project, estimate or invoke native code."""
    _validated(q)
    return q


def _orthogonal(columns,weights):
    basis=[];norms=[]
    for column in columns:
        checkpoint();v=list(column)
        for b,norm in zip(basis,norms):
            coefficient=sum(w*a*c for w,a,c in zip(weights,v,b))/norm
            v=[a-coefficient*c for a,c in zip(v,b)]
        norm=sum(w*a*a for w,a in zip(weights,v))
        if norm: basis.append(v);norms.append(norm)
    return basis,norms


def _absorb(q,x,y,weights,levels):
    columns=[[F(row['fixed_effects'][dim]==level) for row in q['rows']]
             for dim in q['fixed_effects'] for level in sorted(levels[dim])]
    basis,norms=_orthogonal(columns,weights)
    def residual(column):
        v=list(column)
        for b,norm in zip(basis,norms):
            coefficient=sum(w*a*c for w,a,c in zip(weights,v,b))/norm
            v=[a-coefficient*c for a,c in zip(v,b)]
        return v
    projected=[residual(column) for column in zip(*x)]
    within_x=list(map(list,zip(*projected)));within_y=residual(y)
    rank_x=len(_orthogonal(projected,weights)[0])
    h_fe=[w*sum(b[i]*b[i]/norm for b,norm in zip(basis,norms)) for i,w in enumerate(weights)]
    return within_x,within_y,len(basis),rank_x,h_fe


def _code():
    receipt=wls._source_receipt()
    for name in ('fixed_effects.py','fixed_effects_verifier.py','execution_control.py'):
        receipt[name]=sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    return receipt


def run(q,runtime):
    original=q;q=deepcopy(q);labels,ids,y,x,w,levels,low,high=_validated(q);code=_code()
    z,v,rank_fe,rank_x,h_fe=_absorb(q,x,y,w,levels);p=len(labels);n=len(y);rank=rank_fe+rank_x;df=n-rank
    gram=[[sum(weight*row[i]*row[j] for row,weight in zip(z,w)) for j in range(p)] for i in range(p)]
    rhs=[sum(weight*row[j]*value for row,value,weight in zip(z,v,w)) for j in range(p)]
    beta=_solve(gram,rhs)
    result=dict(schema='societybox-fixed-effects-result-v1',engine='EXACT_WEIGHTED_FWL_WITH_PINNED_HIGHS_SLOPES',mode='NO_LLM',
        input_sha256=digest(canonical(q)),code_files=code,parameter_order=labels,
        parameter_units=['('+q['target_unit']+')/('+q['predictor_units'][name]+')' for name in labels],
        training=dict(count=n,days=ids),fixed_effects_rank=rank_fe,identified_slope_rank=rank_x,total_model_rank=rank,
        residual_degrees_of_freedom=df,fixed_effect_level_counts={dim:len(values) for dim,values in levels.items()},
        weight_semantics='POSITIVE_ANALYTIC_NOT_SURVEY_OR_FREQUENCY_WEIGHTS',
        fixed_effect_normalization='COMBINED_ROW_EFFECT_ONLY_NO_SEPARATE_INTERCEPT_OR_GAUGE_CLAIM',
        slope_problem='UNCONSTRAINED_FWL_OLS_NATIVE_BOUNDS_MUST_CONTAIN_ITS_SOLUTION',
        covariance_df_policy='EXACT_TOTAL_MODEL_RANK_NO_NESTED_CLUSTER_ADJUSTMENT_OR_PACKAGE_PARITY',
        original_statistics_package='NOT_EXECUTED_NO_APPROVED_PINNED_STATISTICS_RUNTIME',
        coefficients_exact=None,residuals_exact=None,absorbed_effect_exact=None,weighted_mse_exact=None,
        full_model_leverage_exact=None,covariances={},native=None,derived_program=None,comparison=None,
        native_execution='NOT_RUN',native_parity='NOT_RUN',status='INDETERMINATE',blocker='REGRESSORS_ABSORBED_OR_COLLINEAR' if beta is None else None,
        inferential_validity='NOT_ESTABLISHED',causal_identification='NOT_ESTABLISHED',source_authentication='NOT_ESTABLISHED',
        historical_as_of='NOT_ESTABLISHED',scientific_acceptance='NOT_ESTABLISHED',os_isolation='NOT_ATTESTED',
        independent_installed_replication='NOT_RUN',p_values=None,confidence_intervals=None)
    if beta is None:
        result['covariances']={m:dict(status='BLOCKED',reason='REGRESSORS_ABSORBED_OR_COLLINEAR',matrix_exact=None) for m in q['covariance']['estimators']}
    else:
        residual=[value-_dot(row,beta) for row,value in zip(z,v)]
        columns=[_solve(gram,[F(i==j) for i in range(p)]) for j in range(p)];bread=list(map(list,zip(*columns)))
        hat=[h+weight*_dot(row,[_dot(b,row) for b in bread]) for h,weight,row in zip(h_fe,w,z)]
        if sum(hat)!=rank or any(not 0<=h<=1 for h in hat): raise ValueError('FIXED_EFFECTS_LEVERAGE_IDENTITY_FAILED')
        covariance=(wls._covariances(z,residual,w,bread,ids,dict(q['covariance'],bandwidth=None),
                                    degrees_of_freedom=df,leverage_override=hat) if df>0 else
                    {m:dict(status='BLOCKED',reason='NO_RESIDUAL_DEGREES_OF_FREEDOM',matrix_exact=None) for m in q['covariance']['estimators']})
        loss=sum(weight*e*e for weight,e in zip(w,residual))/sum(w)
        result.update(coefficients_exact=list(map(str,beta)),residuals_exact=list(map(str,residual)),
            absorbed_effect_exact=[str(value-_dot(row,beta)-e) for value,row,e in zip(y,x,residual)],
            weighted_mse_exact=str(loss),full_model_leverage_exact=list(map(str,hat)),covariances=covariance)
        try:
            program,scale=wls._native_wls_program(z,v,beta,low,high,w)
            native=highs_native.run(program,runtime);comparison=wls._compare(beta,z,v,w,native,loss,scale)
            failed=comparison['status']=='FAILED'
            result.update(derived_program=program,native=native,comparison=comparison,native_objective_scale=str(scale),
                native_execution='EXECUTED',native_parity='FAILED' if failed else 'WITHIN_DECLARED_WLS_TOLERANCE',
                status='BLOCKED' if failed else 'READY' if all(rec['matrix_exact'] is not None for rec in covariance.values()) else 'PARTIAL',
                blocker='FIXED_EFFECTS_NATIVE_PARITY_FAILED' if failed else None)
        except ValueError as error:
            if str(error) not in ('INVERSE_NATIVE_BOUNDS_EXCLUDE_OLS','HIGHS_NUMBER_LIMIT','HIGHS_NUMBER_FORMAT_INVALID'):raise
            result.update(status='BLOCKED',blocker=str(error))
    from .fixed_effects_verifier import verify_computed
    result['oracle']=verify_computed(q,result)
    result.update(_evidence(q,result))
    if code!=_code() or canonical(original)!=canonical(q): raise ValueError('FIXED_EFFECTS_INPUT_OR_CODE_CHANGED')
    return result


def _evidence(q,result):
    from .research_evidence import make_evidence,EvidenceLedger,UnknownRegistry,make_obligation,evaluate_coverage
    subject=result['input_sha256'];scope=dict(input_sha256=subject,model='DECLARED_WEIGHTED_FIXED_EFFECTS')
    record=make_evidence(subject_sha256=subject,producer=dict(tool_id='societybox.fixed_effects',version='1',code_sha256=digest(canonical(result['code_files']))),
        operation='WEIGHTED_ABSORPTION_AND_EXPLICIT_DUMMY_PARITY',authority='LOCAL_EXACT_ARITHMETIC',epistemic_state='DERIVED',
        proof_kind='EXPLICIT_DUMMY_REGRESSION_AND_COVARIANCE',scope=scope,
        coverage=dict(status='PARTIAL',unknown_dimensions=['source_authentication','weight_meaning','model_and_covariance_assumptions','causal_identification','large_data_and_original_package_parity']),
        result=dict(coefficients_exact=result['coefficients_exact'],fixed_effects_rank=result['fixed_effects_rank'],
                    total_model_rank=result['total_model_rank'],oracle=result['oracle'],native_parity=result['native_parity']),
        parent_evidence_ids=[],limitations=['Dense independently authored reference, not original statistics package execution',
            'Declared analytic weights and fixed effects do not establish exogeneity or cluster independence',
            'No p-values, confidence coverage, nested-cluster DF adjustment or causal effect certification'])
    ledger=EvidenceLedger([record]);unknowns=UnknownRegistry(ledger);obligations=[]
    for dimension in record['coverage']['unknown_dimensions']:
        sub=dict(scope,dimension=dimension);u=unknowns.register('Establish '+dimension,sub,'INDEPENDENT_EXTERNAL_REVIEW','SOURCE_AND_METHOD_VALIDATION')
        obligations.append(make_obligation(claim_id=dimension,subject_sha256=subject,scope=sub,required_authority='INDEPENDENT_EXTERNAL_REVIEW',
            required_proof_kind='SOURCE_AND_METHOD_VALIDATION',verifier_id='external.'+dimension,dependencies=[],unknown_ids=[u['unknown_id']],
            blocking_reasons=[],next_probe='Obtain independent source, statistical assumption and original package evidence'))
    return dict(evidence=ledger.records(),unknowns=unknowns.records(),obligations=obligations,coverage=evaluate_coverage(obligations,ledger,unknowns))


def export_artifacts(q,result,chart):
    import csv
    import io
    def text(value):return "'"+value if value.lstrip()[:1] in ('=','+','-','@') else value
    def blob(header,rows):
        stream=io.StringIO(newline='');writer=csv.writer(stream);writer.writerow(header);writer.writerows(rows)
        return stream.getvalue().encode('utf-8')
    beta=result['coefficients_exact'] or [];actual=(result['native'] or {}).get('actual',{}).get('solution',[])
    names=result['parameter_order'];units=result['parameter_units']
    covariance=[[method,record['status'],names[i],names[j],value,text('('+units[i]+')*('+units[j]+')'),record['reason'] or '']
        for method,record in result['covariances'].items() for i,row in enumerate(record['matrix_exact'] or []) for j,value in enumerate(row)]
    covariance.extend([method,record['status'],'','','','',record['reason']] for method,record in result['covariances'].items() if record['matrix_exact'] is None)
    residual=result['residuals_exact'] or [];effect=result['absorbed_effect_exact'] or [];hat=result['full_model_leverage_exact'] or []
    return {'coefficients.csv':blob(['parameter','value_exact','native_decimal','unit'],
                [[name,b,actual[i] if i<len(actual) else '',text(unit)] for i,(name,b,unit) in enumerate(zip(names,beta,units))]),
        'covariances.csv':blob(['estimator','status','row_parameter','column_parameter','covariance_exact','unit','blocker'],covariance),
        'fixed-effects-residuals.csv':blob(['row_id','analytic_weight','fixed_effects','absorbed_row_effect_exact','residual_exact','full_model_leverage_exact'],
            [[text(row['id']),row['analytic_weight'],canonical(row['fixed_effects']).decode(),effect[i] if i<len(effect) else '',
              residual[i] if i<len(residual) else '',hat[i] if i<len(hat) else ''] for i,row in enumerate(q['rows'])]),
        'fixed-effects-evidence.json':canonical({key:result[key] for key in ('evidence','unknowns','obligations','coverage','oracle')})}


def render_review_section(q,result,locale):
    words={
        'en':('Weighted fixed-effects projection','Observations','Fixed-effects rank','Total model rank','Residual degrees of freedom',
              'Independent dense projection and explicit-dummy check. Only the slope problem executes in original HiGHS. No original statistical package or large-data parity is established. Cluster correction uses total model rank, with no nested-effect adjustment.'),
        'ko':('가중 고정효과 투영','관측 수','고정효과 행렬의 계수(rank)','전체 모형 행렬의 계수(rank)','잔차 자유도',
              '독립 구현한 조밀 투영과 명시적 더미 회귀를 대조합니다. 원본 HiGHS는 설명변수 계수 문제만 풉니다. 원본 통계 패키지 실행·대규모 자료 동등성은 확립되지 않았습니다. 군집 보정은 전체 모형 계수를 쓰며 중첩 고정효과를 별도 조정하지 않습니다.'),
        'zh-Hans':('加权固定效应投影','观测数','固定效应矩阵秩','完整模型矩阵秩','残差自由度',
              '独立实现的稠密投影与显式虚拟变量回归相核对。原始HiGHS仅求解斜率问题；未确立原始统计软件执行或大数据等价性。聚类修正使用完整模型秩，不进行嵌套效应调整。'),
        'ja':('加重固定効果の射影','観測数','固定効果行列の階数','全モデル行列の階数','残差自由度',
              '独立実装の密行列射影と明示的ダミー回帰を照合します。原版HiGHSは傾きの問題のみを解きます。元の統計パッケージの実行や大規模データでの同等性は未確立です。クラスタ補正は全モデルの階数を用い、入れ子効果の調整は行いません。'),
    }[locale]
    values=(result['training']['count'],result['fixed_effects_rank'],result['total_model_rank'],result['residual_degrees_of_freedom'])
    rank=('<section class="fixed-effects-rank"><h3>'+escape(words[0])+'</h3><p>'+escape(words[5])+'</p><table><tbody>'
          +''.join('<tr><th scope="row">'+escape(label)+'</th><td>'+str(value)+'</td></tr>' for label,value in zip(words[1:5],values))+'</tbody></table></section>')
    return rank+wls.render_review_section(q,result,locale)


def example_parameters():
    """Owned balanced toy panel, never observed or causal research evidence."""
    signs=[1,-1,-1,1];rows=[]
    for i in range(4):
        for j in range(4):
            rows.append(dict(id='r'+str(i)+str(j),y=str(2*i*j+3*i-j+signs[i]*signs[j]),
                x={'x':str(i*j)},analytic_weight='1',fixed_effects={'entity':'e'+str(i),'period':'t'+str(j)}))
    return dict(schema='societybox-fixed-effects-regression-v1',data_kind='SYNTHETIC',source_declared_sha256='a'*64,
        rights=dict(local_processing='PERMITTED',report_export='PERMITTED'),target_unit='currency',predictor_units={'x':'index'},
        fixed_effects=['entity','period'],rows=rows,coefficient_bounds={'x':['-100','100']},
        covariance=dict(estimators=list(METHODS),clusters={row['id']:row['fixed_effects']['entity'] for row in rows}))
