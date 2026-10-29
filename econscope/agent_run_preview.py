"""Selected local receipt inspection; recomputation is explicit and data-only."""
from copy import deepcopy
from html import escape
import json

from .study_claims import build_claim_graph
from .study_runs import verify
from .workspace import ScopeError, SHA, canonical, digest, get_root, load_json


RUN_TEXT = {
    'en': ('Selected run: computational verification',
           'Separate local inspection; run data are not sent to the host. Verification recomputes the supported deterministic model from stored data. No arbitrary code execution or new execution approval is granted. Sources, rights and causal validity are not independently verified.',
           'Verification', 'Source and assumption declarations', 'Model and effective parameters',
           'Computational lineage', 'Computed results'),
    'zh-Hans': ('所选运行：计算验证',
                '独立的本地检查；运行数据不会发送到主机。验证使用存储数据重新计算受支持的确定性模型。不执行任意代码，也不授予新的执行批准。来源、权利和因果有效性未经独立验证。',
                '验证', '来源与假设声明', '模型与有效参数', '计算溯源', '计算结果'),
    'ko': ('선택한 실행: 계산 검증',
           '별도의 로컬 검사이며 실행 데이터를 호스트에 보내지 않습니다. 검증은 저장된 데이터로 지원되는 결정론적 모델을 재계산합니다. 임의 코드 실행이나 새 실행 승인을 부여하지 않습니다. 출처, 권리 및 인과적 타당성은 독립적으로 검증되지 않았습니다.',
           '검증', '출처 및 가정 선언', '모델 및 유효 파라미터', '계산 계보', '계산 결과'),
    'ja': ('選択した実行：計算の検証',
           '独立したローカル検査であり、実行データをホストに送信しません。検証は保存データから対応する決定論的モデルを再計算します。任意コードの実行や新たな実行承認は付与しません。出典、権利、因果的妥当性は独立に検証されていません。',
           '検証', '出典と仮定の宣言', 'モデルと有効パラメータ', '計算の系譜', '計算結果'),
}


TABLE_TEXT = {
    'en': ('Exact model-state trajectory', 'Recorded conditional model events',
           'Model step', 'Declared time unit', 'State variable', 'Declared unit',
           'Exact value', 'ClaimGraph node', 'Event ID', 'Researcher label',
           'Declared detector', 'Trigger state, units and cell nodes',
           'No recorded model events.',
           'Model steps are not historical calendar dates. Events are observations inside the declared model only.'),
    'zh-Hans': ('精确模型状态轨迹', '已记录的条件模型事件',
                '模型步', '声明的时间单位', '状态变量', '声明的单位',
                '精确值', 'ClaimGraph 节点', '事件 ID', '研究者标签',
                '声明的检测条件', '触发状态、单位与单元节点',
                '没有已记录的模型事件。',
                '模型步不代表历史日历日期。事件仅为所声明模型内部的观测。'),
    'ko': ('정확한 모델 상태 궤적', '기록된 조건부 모델 사건',
           '모델 단계', '선언된 시간 단위', '상태 변수', '선언된 단위',
           '정확한 값', 'ClaimGraph 노드', '사건 ID', '연구자 레이블',
           '선언된 탐지 조건', '발생 시 상태, 단위 및 셀 노드',
           '기록된 모델 사건이 없습니다.',
           '모델 단계는 역사적 달력 날짜가 아닙니다. 사건은 선언된 모델 내부의 관측에 한정됩니다.'),
    'ja': ('正確なモデル状態の軌跡', '記録された条件付きモデル事象',
           'モデルステップ', '宣言された時間単位', '状態変数', '宣言された単位',
           '正確な値', 'ClaimGraph ノード', '事象 ID', '研究者のラベル',
           '宣言された検出条件', '検出時の状態、単位とセルノード',
           '記録されたモデル事象はありません。',
           'モデルステップは歴史上の暦日を表しません。事象は宣言されたモデル内の観測に限られます。'),
}


PROVENANCE_TEXT = {
    'en': ('Run ID', 'Claim graph SHA-256'),
    'zh-Hans': ('运行 ID', '声明图 SHA-256'),
    'ko': ('실행 ID', '주장 그래프 SHA-256'),
    'ja': ('実行 ID', '主張グラフ SHA-256'),
}


NUMERICAL_TEXT = {
    'en': ('Numerical assurance',
           'Values reproduce the recorded Decimal calculation. Flags cover the whole calculation, including data initialization and event detection; they do not locate rounding or measure its error. No observed flags does not establish accuracy. Convergence, stability and error bounds remain unsupported.',
           'Decimal precision (significant digits)', 'Rounding policy',
           'Observed arithmetic flags', 'Numerical accuracy'),
    'zh-Hans': ('数值保障',
                '数值复现已记录的 Decimal 计算。标志涵盖整个计算，包括数据初始化和事件检测；它们不能定位舍入或衡量误差。未观察到标志不代表精度已获证实。收敛性、稳定性和误差界限仍不受支持。',
                '十进制精度（有效数字）', '舍入策略', '已观察到的算术标志', '数值精度'),
    'ko': ('수치 보증',
           '값은 기록된 Decimal 계산을 재현합니다. 플래그는 데이터 초기화와 사건 탐지를 포함한 전체 계산을 대상으로 하며 반올림 위치나 오차 크기를 나타내지 않습니다. 관측된 플래그가 없어도 정확성이 입증되지는 않습니다. 수렴성, 안정성 및 오차 한계는 지원되지 않습니다.',
           '십진수 정밀도(유효 자릿수)', '반올림 정책', '관측된 산술 플래그', '수치 정확성'),
    'ja': ('数値保証',
           '値は記録された Decimal 計算を再現します。フラグはデータ初期化と事象検出を含む計算全体を対象とし、丸めの位置や誤差の大きさを示しません。観測されたフラグがなくても精度は立証されません。収束性、安定性、誤差限界は未対応です。',
           '十進精度（有効桁数）', '丸め方針', '観測された算術フラグ', '数値精度'),
}


def _numerical_summary(record, locale, run_id):
    """Display the verified snapshot's policy and observations without promotion."""
    labels = NUMERICAL_TEXT[locale]
    policy = record['runtime']['numerical_assurance']['policy']
    diagnostics = record['results']['numerical_diagnostics']
    fields = (
        (PROVENANCE_TEXT[locale][0], run_id),
        (labels[2], str(policy['precision'])),
        (labels[3], policy['rounding']),
        (labels[4], ', '.join(diagnostics['raised_flags']) or 'NONE_OBSERVED'),
        (labels[5], diagnostics['numerical_accuracy']),
    )
    return ('<section class="numerical-assurance"><h4>' + escape(labels[0])
            + '</h4><p>' + escape(labels[1]) + '</p><dl>'
            + ''.join('<dt>' + escape(label) + '</dt><dd><code>' + escape(value)
                      + '</code></dd>' for label, value in fields)
            + '</dl></section>')


def _result_tables(record, locale, run_id, graph_sha256):
    """Project exact receipt strings after verification; no data reads or calculations."""
    labels = TABLE_TEXT[locale]
    model, results = record['spec']['model'], record['results']

    def table(caption, headings, rows):
        run_label, graph_label = PROVENANCE_TEXT[locale]
        return ('<table><caption>' + escape(caption)
                + '<br>' + escape(run_label) + ': <code>' + escape(run_id) + '</code>'
                + '<br>' + escape(graph_label) + ': <code>' + escape(graph_sha256) + '</code>'
                + '</caption><thead><tr>'
                + ''.join('<th scope="col">' + escape(h) + '</th>' for h in headings)
                + '</tr></thead><tbody>'
                + ''.join('<tr><th scope="row">' + escape(str(row[0])) + '</th>'
                          + ''.join('<td>' + escape(str(value)) + '</td>'
                                    for value in row[1:]) + '</tr>' for row in rows)
                + '</tbody></table>')

    trajectory = (
        (row['step'], model['time_unit'], name, model['states'][name]['unit'],
         value, f"cell/{row['step']}/{name}")
        for row in results['trajectory'] for name, value in sorted(row['state'].items()))
    events = (
        (event['step'], model['time_unit'], event['event_id'], event['label'],
         event['detector'],
         '; '.join(f"{name} = {value} [{model['states'][name]['unit']}] (cell/{event['step']}/{name})"
                   for name, value in sorted(event['trigger_state'].items())), f'event/{index}')
        for index, event in enumerate(results['events']))
    return ('<p>' + escape(labels[13]) + '</p>'
            + table(labels[0], labels[2:8], trajectory)
            + table(labels[1], (labels[2], labels[3], *labels[8:12], labels[7]), events)
            + ('<p>' + escape(labels[12]) + '</p>' if not results['events'] else ''))


def selected_run_section(workspace, run_id, spec, overrides, locale):
    """Fail closed before rendering results; use one hash-bound receipt snapshot."""
    if not isinstance(run_id, str) or not SHA.fullmatch(run_id):
        raise ScopeError('STUDY_RUN_ID_INVALID')
    spec, overrides = deepcopy(spec), deepcopy(overrides)
    path, _, _ = get_root(workspace)
    file = path / 'runs' / (run_id + '.json')
    record = load_json(file)
    if digest(canonical(record)) != run_id:
        raise ScopeError('STUDY_RUN_RECEIPT_TAMPERED')
    if not isinstance(record, dict) or canonical(record.get('spec')) != canonical(spec):
        raise ScopeError('STUDY_PREVIEW_RUN_STUDY_MISMATCH')
    if overrides is not None and canonical(record.get('overrides')) != canonical(overrides):
        raise ScopeError('STUDY_PREVIEW_RUN_OVERRIDES_MISMATCH')
    verification = verify(workspace, run_id)
    if canonical(load_json(file)) != canonical(record):
        raise ScopeError('STUDY_PREVIEW_RUN_CHANGED_DURING_INSPECTION')
    graph = build_claim_graph(record, run_id)
    graph_sha256 = digest(canonical(graph))
    labels = RUN_TEXT[locale]
    cards = [
        {**verification, 'authorization': 'NOT_GRANTED',
         'verification_execution': 'SUPPORTED_DETERMINISTIC_MODEL_RECOMPUTED',
         'evidence_verification': 'NOT_INDEPENDENTLY_VERIFIED'},
        record['spec']['evidence'],
        {'definition': record['spec']['model'], 'overrides': record['overrides'],
         'effective_parameters': record['results']['parameters'],
         'limitations': record['spec']['limitations']},
        graph,
        record['results'],
    ]
    parts = ['<section><h2>' + escape(labels[0]) + '</h2><p>' + escape(labels[1]) + '</p>']
    for index, (heading, card) in enumerate(zip(labels[2:], cards)):
        parts.extend(['<section><h3>' + escape(heading) + '</h3>',
                      (_numerical_summary(record, locale, run_id)
                       + _result_tables(record, locale, run_id, graph_sha256))
                      if index == len(cards) - 1 else '',
                      '<pre><code>',
                      escape(json.dumps(card, ensure_ascii=False, allow_nan=False, indent=2)),
                      '</code></pre></section>'])
    return ''.join(parts) + '</section>'
