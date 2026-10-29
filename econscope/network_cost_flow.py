"""Declared one-commodity minimum-cost flow with exact independent certificates.

Pure validation/certification functions also load in the fixed original-source
worker. Product evidence imports occur only in the parent-side run function.
"""
from copy import deepcopy
import hashlib
import re
from pathlib import Path


def validate(parameters):
    if type(parameters) is not dict or set(parameters)!={'nodes','links','demands','flow_unit','cost_unit'}:
        raise ValueError('COST_FLOW_SCHEMA_INVALID')
    nodes=parameters['nodes']; links=parameters['links']; demands=parameters['demands']
    if (type(nodes) is not list or not 1<=len(nodes)<=12
            or any(type(n) is not str or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,31}',n) for n in nodes)
            or len(set(nodes))!=len(nodes) or type(links) is not list or len(links)>64
            or type(demands) is not dict or set(demands)!=set(nodes)
            or any(type(v) is not int or abs(v)>1000000 for v in demands.values())):
        raise ValueError('COST_FLOW_NODES_OR_DEMANDS_INVALID')
    for field in ('flow_unit','cost_unit'):
        value=parameters[field]
        if type(value) is not str or not value.strip() or len(value)>64 or any(ord(c)<32 for c in value):
            raise ValueError('COST_FLOW_UNITS_INVALID')
    seen=set()
    for edge in links:
        if (type(edge) is not list or len(edge)!=4 or any(type(n) is not str or n not in nodes for n in edge[:2])
                or type(edge[2]) is not int or not 0<=edge[2]<=1000000
                or type(edge[3]) is not int or abs(edge[3])>1000000):
            raise ValueError('COST_FLOW_LINK_INVALID')
        key=tuple(edge[:2])
        if key in seen: raise ValueError('COST_FLOW_DUPLICATE_LINK')
        seen.add(key)
    return parameters


def feasibility_oracle(parameters):
    """Exact cut characterization for finite balanced integer transshipment."""
    validate(parameters)
    total=sum(parameters['demands'].values())
    if total:
        return dict(feasibility='INFEASIBLE',method='GLOBAL_DEMAND_IMBALANCE',total_demand=total,cuts_checked=0)
    nodes=parameters['nodes']; checked=0
    for mask in range(1<<len(nodes)):
        selected={n for i,n in enumerate(nodes) if mask&(1<<i)}
        demand=sum(parameters['demands'][n] for n in selected)
        incoming=sum(cap for a,b,cap,_ in parameters['links'] if a not in selected and b in selected)
        checked+=1
        if demand>incoming:
            return dict(feasibility='INFEASIBLE',method='EXHAUSTIVE_NET_DEMAND_CUTS',cuts_checked=checked,
                        violated_subset=sorted(selected),net_demand=demand,incoming_capacity=incoming)
    return dict(feasibility='FEASIBLE',method='EXHAUSTIVE_NET_DEMAND_CUTS',cuts_checked=checked)


def check_result(parameters,actual):
    """Verify cost optimality without network simplex or upstream dual values."""
    validate(parameters)
    if type(actual) is not dict or set(actual)!={'status','cost','recomputed_cost','edge_flows'}:
        raise ValueError('COST_FLOW_RESULT_INVALID')
    feasible=feasibility_oracle(parameters)
    if feasible['feasibility']=='INFEASIBLE':
        if actual['status']!='INFEASIBLE' or any(actual[k] is not None for k in ('cost','recomputed_cost','edge_flows')):
            raise ValueError('COST_FLOW_INFEASIBILITY_MISMATCH')
        return dict(feasible,method='EXACT_INFEASIBILITY_CUT_CERTIFICATE',feasibility_method=feasible['method'])
    if actual['status']!='OPTIMAL' or type(actual['edge_flows']) is not list or len(actual['edge_flows'])!=len(parameters['links']):
        raise ValueError('COST_FLOW_FEASIBILITY_MISMATCH')
    balance=dict.fromkeys(parameters['nodes'],0); primal=0; residual=[]
    for declared,row in zip(parameters['links'],actual['edge_flows']):
        if (type(row) is not list or len(row)!=3 or row[:2]!=declared[:2]
                or type(row[2]) is not int or not 0<=row[2]<=declared[2]):
            raise ValueError('COST_FLOW_EDGE_FLOW_INVALID')
        a,b,cap,cost=declared; flow=row[2]
        balance[a]-=flow; balance[b]+=flow; primal+=flow*cost
        if flow<cap: residual.append((a,b,cost))
        if flow>0: residual.append((b,a,-cost))
    if balance!=parameters['demands']: raise ValueError('COST_FLOW_CONSERVATION_MISMATCH')
    if any(type(actual[k]) is not int or actual[k]!=primal for k in ('cost','recomputed_cost')):
        raise ValueError('COST_FLOW_COST_MISMATCH')
    potential=dict.fromkeys(parameters['nodes'],0)
    for _ in parameters['nodes']:
        changed=False
        for a,b,cost in residual:
            if potential[b]>potential[a]+cost:
                potential[b]=potential[a]+cost; changed=True
        if not changed: break
    else:
        raise ValueError('COST_FLOW_NEGATIVE_RESIDUAL_CYCLE')
    reduced=[cost+potential[a]-potential[b] for a,b,cost in residual]
    dual=sum(potential[n]*parameters['demands'][n] for n in parameters['nodes'])
    dual+=sum(cap*min(0,cost+potential[a]-potential[b]) for a,b,cap,cost in parameters['links'])
    if any(value<0 for value in reduced) or dual!=primal:
        raise ValueError('COST_FLOW_DUAL_CERTIFICATE_MISMATCH')
    return dict(method='EXACT_RESIDUAL_BELLMAN_FORD_PRIMAL_DUAL',feasibility='FEASIBLE',
        feasibility_certificate=feasible,primal_cost=primal,dual_lower_bound=dual,potentials=potential,
        residual_edges_checked=len(residual),minimum_residual_reduced_cost=min(reduced,default=0),
        nonunique_solution_policy='COMPARE_PRIMAL_FEASIBILITY_AND_OPTIMAL_COST_NOT_FLOW_VECTOR_EQUALITY')


def _source_receipt():
    names=('network_cost_flow.py','network_paths.py','network_paths_worker.py','research_evidence.py','workspace.py','feature_catalog.py')
    result={}
    for name in names:
        path=Path(__file__).with_name(name)
        if path.is_symlink() or not path.is_file(): raise ValueError('COST_FLOW_CODE_UNAVAILABLE')
        result[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _evidence(parameters,result,code):
    from .workspace import canonical,digest
    from .research_evidence import EvidenceLedger,UnknownRegistry,make_evidence,make_obligation,evaluate_coverage
    subject=digest(canonical(parameters))
    scope=dict(input_sha256=subject,flow_unit=parameters['flow_unit'],cost_unit=parameters['cost_unit'],
               semantics='DECLARED_SINGLE_COMMODITY_STATIC_MINIMUM_COST_FLOW_NOT_EQUILIBRIUM_OR_CAUSAL_EFFECT')
    producer=dict(tool_id='supply.minimum_cost_flow',version='1',code_sha256=digest(canonical(code)))
    declared=make_evidence(subject_sha256=subject,producer=producer,operation='DECLARED_NETWORK_INPUT',
        authority='DECLARED_ONLY',epistemic_state='OBSERVED',proof_kind='INPUT_DECLARATION',scope=scope,
        coverage=dict(status='PARTIAL',unknown_dimensions=['source_and_rights_authentication']),
        result=parameters,parent_evidence_ids=[],limitations=['Observed local declaration, not authenticated economic facts or rights'])
    record=make_evidence(subject_sha256=subject,producer=producer,operation='ORIGINAL_MIN_COST_FLOW_AND_EXACT_CERTIFICATE',
        authority='PINNED_UPSTREAM_NUMERIC_EXECUTION',epistemic_state='DERIVED',proof_kind='MIN_COST_FLOW_CERTIFICATE',scope=scope,
        coverage=dict(status='PARTIAL',unknown_dimensions=['source_and_rights_authentication','causal_effect','external_replication']),
        result=result,parent_evidence_ids=[declared['evidence_id']],limitations=['Source, demand, capacities, costs and units are researcher declarations',
           'Static finite flow/circulation optimization is not observed welfare, market equilibrium or a forecast'])
    ledger=EvidenceLedger([declared,record]); unknowns=UnknownRegistry(ledger)
    unk=unknowns.register('Do these declarations identify an empirical causal or welfare effect?',scope,'EXTERNAL_IDENTIFICATION','CAUSAL_DESIGN')
    unknowns.revise(unk['unknown_id'],unk['revision_digest'],status='OPEN',supporting_evidence_ids=[record['evidence_id']],
        recommended_probes=['Authenticate capacity, demand and cost measurements and challenge alternative mechanisms'])
    obligations=[]
    for claim,outcome,proof,blocked in (
            ('minimum_cost_flow_verification','PASS','MIN_COST_FLOW_CERTIFICATE',False),
            ('declared_demands_feasible','PASS' if result['actual']['status']=='OPTIMAL' else 'FAIL','DEMAND_FEASIBILITY',False),
            ('empirical_causal_effect','UNKNOWN','CAUSAL_DESIGN',True)):
        ob=make_obligation(claim_id=claim,subject_sha256=subject,scope=scope,
            required_authority='EXTERNAL_IDENTIFICATION' if blocked else 'PINNED_UPSTREAM_NUMERIC_EXECUTION',
            required_proof_kind=proof,verifier_id=producer['tool_id'],dependencies=[],
            unknown_ids=[unk['unknown_id']] if blocked else [],
            blocking_reasons=['No authenticated empirical identification design'] if blocked else [],
            next_probe='Inspect primal/dual or infeasible-cut evidence; authenticate data and independent replication')
        obligations.append(ob)
        if not blocked:
            ledger.merge([make_evidence(subject_sha256=subject,producer=producer,operation='VERIFY_'+claim.upper(),
                authority='PINNED_UPSTREAM_NUMERIC_EXECUTION',epistemic_state='DERIVED',proof_kind=proof,scope=scope,
                coverage=dict(status='COMPLETE',unknown_dimensions=[]),
                result=dict(obligation_id=ob['obligation_id'],revision=1,previous_verification_id=None,outcome=outcome),
                parent_evidence_ids=[record['evidence_id']],limitations=['Declared model and exact certificate only'])])
    return dict(evidence=ledger.records(),unknowns=unknowns.records(),
        unknown_histories={u['unknown_id']:unknowns.history(u['unknown_id']) for u in unknowns.records()},
        obligations=obligations,coverage=evaluate_coverage(obligations,ledger,unknowns))


def run(parameters,sources):
    from .network_paths import run as run_pinned
    from .workspace import canonical
    original=parameters; parameters=deepcopy(parameters); validate(parameters); code=_source_receipt()
    result=run_pinned(parameters,sources,calculation='cost_flow')
    from .feature_catalog import describe_feature
    result['product_feature']=describe_feature('supply_cost')
    if code!=_source_receipt(): raise ValueError('COST_FLOW_CODE_CHANGED_DURING_RUN')
    try: unchanged=canonical(original)==canonical(parameters)
    except (ValueError,TypeError,RecursionError,UnicodeError): unchanged=False
    if not unchanged: raise ValueError('COST_FLOW_INPUT_CHANGED_DURING_RUN')
    result['code_files']=code
    result.update(_evidence(parameters,result,code))
    return result


def export_artifacts(parameters,result,chart):
    import csv
    import io
    import json
    from .workspace import canonical
    def text(value): return "'"+value if value.lstrip()[:1] in ('=','+','-','@') else value
    def blob(header,rows):
        stream=io.StringIO(newline=''); writer=csv.writer(stream); writer.writerow(header); writer.writerows(rows)
        return stream.getvalue().encode('utf-8')
    feasible=result['actual']['status']=='OPTIMAL'
    flow_values=[row[2] for row in result['actual']['edge_flows']] if feasible else ['']*len(parameters['links'])
    balance=dict.fromkeys(parameters['nodes'],0)
    if feasible:
        for (a,b,_,_),flow in zip(parameters['links'],flow_values): balance[a]-=flow; balance[b]+=flow
    flows=blob(['supplier','customer','capacity','unit_cost','flow','cost_contribution','flow_unit','cost_unit','status'],
        [[a,b,cap,cost,flow,cost*flow if feasible else '',text(parameters['flow_unit']),text(parameters['cost_unit']),result['actual']['status']]
         for (a,b,cap,cost),flow in zip(parameters['links'],flow_values)])
    nodes=blob(['node','declared_net_demand','actual_net_inflow','flow_unit','status'],
        [[n,parameters['demands'][n],balance[n] if feasible else '',text(parameters['flow_unit']),result['actual']['status']] for n in parameters['nodes']])
    certificate=blob(['quantity','value'],[[key,json.dumps(value,sort_keys=True,ensure_ascii=False)] for key,value in sorted(result['oracle'].items())])
    values=blob(['variable','value_exact','unit','source_id'],
        [[label,value,text(chart['unit']),text(chart['source_id'])] for label,value in zip(chart['labels'],chart['values_exact'])])
    return {'flows.csv':flows,'node-balances.csv':nodes,'certificate.csv':certificate,
            'evidence.json':canonical({key:result[key] for key in ('evidence','unknowns','unknown_histories','obligations','coverage')}),
            'values.csv':values}


_REVIEW={
    'en':('Minimum-cost allocation','Exact feasibility and optimality certificate','Remaining evidence obligations',
          'Positive net demand receives flow and negative demand supplies it. Costs and capacities are declarations. Bounded negative-cost circulation is allowed; computed cost is not observed profit, welfare, causal policy impact or a forecast.'),
    'ko':('최소비용 배분','정확한 실행 가능성·최적성 인증서','남은 증거·검증 의무',
          '양의 순수요는 수령, 음의 순수요는 공급입니다. 비용과 용량은 선언값입니다. 용량이 제한된 음의 비용 순환을 허용하며, 계산된 비용은 실제 이익·후생·인과적 정책효과·예측을 뜻하지 않습니다.'),
    'zh-Hans':('最低成本分配','精确可行性与最优性证书','剩余证据与验证义务',
               '正净需求表示接收，负净需求表示供应。成本和容量为声明值。允许有限容量的负成本循环；计算成本不代表实际利润、福利、因果政策效果或预测。'),
    'ja':('最小費用配分','正確な実行可能性・最適性証明','残る証拠・検証義務',
          '正の純需要は受領、負の純需要は供給です。費用と容量は申告値です。容量が有限の負費用循環を許容しますが、計算費用は実際の利益・厚生・因果的な政策効果・予測を意味しません。'),
}


def render_review_section(parameters,result,locale):
    import json
    from html import escape
    if locale not in _REVIEW: raise ValueError('COST_FLOW_LOCALE_INVALID')
    title,proof,remaining,limits=_REVIEW[locale]
    def block(value): return '<pre>'+escape(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2))+'</pre>'
    return ('<section class="minimum-cost-flow-review"><h2>'+escape(title)+'</h2><p>'+escape(limits)+'</p>'
            +block(result['actual'])+'<h3>'+escape(proof)+'</h3>'+block(result['oracle'])
            +'<h3>'+escape(remaining)+'</h3>'+block(result['coverage'])+block(result['unknowns'])+'</section>')
