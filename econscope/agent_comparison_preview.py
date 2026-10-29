"""Read-only portable comparison inspection from one verified byte snapshot."""
import base64
from html import escape
import json

from .agent_run_preview import RUN_TEXT
from .study_charts import _trajectory_geometry, _trajectory_svg
from .study_runs import _project_claim_graph, _verified_comparison_snapshot
from .study_recompute import _inspect_receipt
from .study_csv import _comparison_trajectory_bytes
from .study_pdf import _pdf_with_record
from .study_html import _document_with_record
from .workspace import ScopeError, canonical, digest


COMPARISON_TEXT = {
    'en': ('Portable baseline / alternative comparison',
           'Separate local inspection; comparison data are not sent to the host. Only artifact consistency is checked; no engine execution or new approval occurs. Original receipt authenticity, source authenticity, legal rights and causal validity are not established. Consistently rewritten artifacts cannot be independently detected. Deltas are exact alternative minus baseline within the declared model; model steps are not historical dates.',
           'Exact conditional model contrasts'),
    'zh-Hans': ('便携基线 / 替代方案比较',
                '独立的本地检查；比较数据不会发送到主机。仅检查文件一致性；不运行引擎，也不授予新的批准。原始运行凭据真实性、来源真实性、法律权利和因果有效性尚未确立。无法独立发现被一致改写的文件。差值为声明模型内精确的替代值减基线值；模型步不是历史日期。',
                '精确的条件模型对比'),
    'ko': ('이동식 기준 / 대안 비교',
           '별도의 로컬 검사이며 비교 데이터를 호스트에 보내지 않습니다. 산출물 일관성만 검사하며 엔진 실행이나 새 승인은 없습니다. 원본 실행 기록과 출처의 진위, 법적 권리 및 인과적 타당성은 확립되지 않았습니다. 일관되게 변조된 산출물은 독립적으로 탐지할 수 없습니다. 차이는 선언된 모델 안에서 대안에서 기준을 뺀 정확한 값이며 모델 단계는 역사적 날짜가 아닙니다.',
           '정확한 조건부 모델 비교'),
    'ja': ('ポータブルな基準 / 代替比較',
           '独立したローカル検査であり、比較データをホストに送信しません。成果物の整合性のみを確認し、エンジン実行や新たな承認は行いません。元の実行記録と出典の真正性、法的権利、因果的妥当性は確立されていません。一貫して改変された成果物は独立に検出できません。差は宣言されたモデル内の正確な代替値から基準値を引いた値であり、モデルステップは歴史上の日付ではありません。',
           '正確な条件付きモデル比較'),
}


RECEIPT_TEXT = {
    'en': ('Separate local recomputation record',
           'Unsigned recorded outcome; anyone can rewrite and rehash this record. Binding checks do not execute the engine or recheck source bytes. Independent replication and authenticated identity are not established.'),
    'zh-Hans': ('独立的本地重计算记录',
                '记录的结果未经签名；任何人都可以改写记录并重新计算哈希。绑定检查不运行引擎，也不重新检查源文件字节。独立复现和身份认证尚未确立。'),
    'ko': ('별도의 로컬 재계산 기록',
           '서명되지 않은 기록 결과이며 누구나 기록을 수정하고 해시를 다시 계산할 수 있습니다. 바인딩 검사는 엔진을 실행하거나 원본 바이트를 다시 검사하지 않습니다. 독립 재현과 인증된 신원은 확립되지 않았습니다.'),
    'ja': ('別個のローカル再計算記録',
           '記録された結果は未署名であり、誰でも記録を書き換えてハッシュを再計算できます。結合の確認はエンジンを実行せず、元データのバイトも再検査しません。独立した再現と認証された身元は確立されていません。'),
}


TRAJECTORY_TEXT = {
    'en': ('Recorded conditional model trajectories',
           'Straight segments join recorded model cells only; they are not continuous-time estimates. Coordinates are rounded to 0.001 canvas units, not numerical error bounds. Exact recorded decimal values, cell nodes, run IDs and graph hashes are embedded in each figure’s metadata.',
           'Conditional model trajectory: ', 'Unit: ', 'Model time: ',
           'Baseline: solid gray. Alternative: dashed blue.'),
    'zh-Hans': ('已记录的条件模型轨迹',
                '直线仅连接已记录的模型单元，并非连续时间估计。坐标舍入至 0.001 画布单位，不代表数值误差界限。每个图的元数据包含精确记录的十进制值、单元节点、运行 ID 和图哈希。',
                '条件模型轨迹：', '单位：', '模型时间：',
                '基线：灰色实线。替代方案：蓝色虚线。'),
    'ko': ('기록된 조건부 모델 궤적',
           '직선은 기록된 모델 셀만 연결하며 연속 시간 추정이 아닙니다. 좌표는 캔버스 단위 0.001로 반올림되며 수치 오차 한계가 아닙니다. 각 그림의 메타데이터에 기록된 정확한 십진수 값, 셀 노드, 실행 ID 및 그래프 해시가 포함됩니다.',
           '조건부 모델 궤적: ', '단위: ', '모델 시간: ',
           '기준: 회색 실선. 대안: 파란색 점선.'),
    'ja': ('記録された条件付きモデル軌跡',
           '直線は記録されたモデルセルのみを結び、連続時間の推定ではありません。座標はキャンバス単位の 0.001 に丸められ、数値誤差の限界を表しません。各図のメタデータには記録された正確な十進値、セルノード、実行 ID、グラフハッシュが含まれます。',
           '条件付きモデル軌跡：', '単位：', 'モデル時間：',
           '基準：灰色の実線。代替：青色の破線。'),
}


CSV_TEXT = {
    'en': ('Download exact trajectory CSV', 'Local recorded cells and provenance only. Keep this download private. Numerical accuracy: NOT_ESTABLISHED. Provider transfer: NOT_GRANTED.'),
    'zh-Hans': ('下载精确轨迹 CSV', '仅包含本地记录的单元与来源。请保密保存此下载。数值准确性：NOT_ESTABLISHED。提供方传输：NOT_GRANTED。'),
    'ko': ('정확한 궤적 CSV 다운로드', '로컬에 기록된 셀과 출처만 포함합니다. 다운로드를 비공개로 보관하세요. 수치 정확성: NOT_ESTABLISHED. 제공자 전송: NOT_GRANTED.'),
    'ja': ('正確な軌跡 CSV をダウンロード', 'ローカルに記録されたセルと来歴のみを含みます。ダウンロードは非公開で保管してください。数値精度：NOT_ESTABLISHED。プロバイダー転送：NOT_GRANTED。'),
}


TABLE_TEXT = {
    'en': ('Exact recorded cells: ', 'Model step', 'Baseline', 'Alternative',
           'Cell node', 'Run ID', 'Claim graph SHA-256'),
    'zh-Hans': ('精确记录的单元：', '模型步', '基线', '替代方案',
                '单元节点', '运行 ID', '声明图 SHA-256'),
    'ko': ('기록된 정확한 셀: ', '모델 단계', '기준', '대안',
           '셀 노드', '실행 ID', '주장 그래프 SHA-256'),
    'ja': ('記録された正確なセル：', 'モデルステップ', '基準', '代替',
           'セルノード', '実行 ID', '主張グラフ SHA-256'),
}


def _trajectory_table(geometry, locale):
    """Accessible exact cells and visible provenance from the figure's snapshot."""
    title, step, baseline, alternative, node, run_id, graph = TABLE_TEXT[locale]
    unit, time = TRAJECTORY_TEXT[locale][3:5]
    parts = ['<table><caption>', escape(title + geometry['variable']),
             '<br>', escape(unit + geometry['unit']), '; ',
             escape(time + geometry['time_unit'])]
    for role, label in (('baseline', baseline), ('alternative', alternative)):
        origin = geometry['runs'][role]
        parts.extend(['<br>', escape(label + ' — ' + run_id + ': '),
                      '<code>', escape(origin['run_id']), '</code>; ',
                      escape(graph + ': '), '<code>',
                      escape(origin['claim_graph_sha256']), '</code>'])
    parts.append('</caption><thead><tr>')
    for label in (step, baseline, baseline + ' — ' + node,
                  alternative, alternative + ' — ' + node):
        parts.extend(['<th scope="col">', escape(label), '</th>'])
    parts.append('</tr></thead><tbody>')
    for left, right in zip(geometry['series']['baseline'], geometry['series']['alternative']):
        parts.extend(['<tr><th scope="row">', escape(str(left['step'])), '</th>'])
        for cell in (left, right):
            parts.extend(['<td>', escape(cell['value']), '</td><td><code>',
                          escape(cell['graph_node']), '</code></td>'])
        parts.append('</tr>')
    return ''.join(parts) + '</tbody></table>'


def _trajectory_figures(comparison, records, locale):
    """Render the verified in-memory cells using the portable report geometry."""
    heading, caption, *labels = TRAJECTORY_TEXT[locale]
    parts = ['<section><h3>' + escape(heading) + '</h3>']
    for index, variable in enumerate(sorted(comparison['lineage']['final_cells'])):
        geometry = _trajectory_geometry(comparison, records, variable)
        parts.extend(['<figure>', _trajectory_svg(
            geometry, f'conversation-comparison-trajectory-{index}', labels=labels),
            '<figcaption>' + escape(variable) + ': ' + escape(caption) + '</figcaption></figure>',
            _trajectory_table(geometry, locale)])
    return ''.join(parts) + '</section>'


HTML_TEXT = {
    'en': ('Download evidence-first HTML report', 'Private local download. Report language: English. Provider transfer: NOT_GRANTED.'),
    'ko': ('근거 우선 HTML 보고서 다운로드', '비공개 로컬 다운로드. 보고서 언어: 영어. 제공자 전송: NOT_GRANTED.'),
    'zh-Hans': ('下载证据优先 HTML 报告', '私有本地下载。报告语言：英语。提供商传输：NOT_GRANTED。'),
    'ja': ('根拠を先に示す HTML レポートをダウンロード', '非公開のローカルダウンロード。レポートの言語：英語。プロバイダー転送：NOT_GRANTED。'),
}


PDF_TEXT = {
    'en': ('Download evidence-first PDF report', 'Private local download. English ASCII PDF only; multilingual fonts: UNSUPPORTED. Rendered-page QA: NOT_RUN. Provider transfer: NOT_GRANTED.'),
    'ko': ('근거 우선 PDF 보고서 다운로드', '비공개 로컬 다운로드. 영어 ASCII PDF만 지원; 다국어 글꼴: UNSUPPORTED. 렌더링 검토: NOT_RUN. 제공자 전송: NOT_GRANTED.'),
    'zh-Hans': ('下载证据优先 PDF 报告', '私密本地下载。仅支持英语 ASCII PDF；多语言字体：UNSUPPORTED。页面渲染检查：NOT_RUN。提供商传输：NOT_GRANTED。'),
    'ja': ('根拠優先 PDF レポートをダウンロード', '非公開のローカルダウンロード。英語 ASCII PDF のみ。多言語フォント：UNSUPPORTED。描画確認：NOT_RUN。プロバイダー転送：NOT_GRANTED。'),
}


def comparison_section(folder, spec, locale, recomputation_receipt=None, *, export_csv=False, export_html=False, export_pdf=False):
    """Public boundary: always verify files and receipt before rendering."""
    snapshot = _verified_comparison_snapshot(folder)
    selected = snapshot[2]['baseline']['spec']
    if spec is not None and canonical(spec) != canonical(selected):
        raise ScopeError('STUDY_PREVIEW_COMPARISON_STUDY_MISMATCH')
    checked_record = (_inspect_receipt(snapshot, recomputation_receipt)
                      if recomputation_receipt is not None else None)
    return _comparison_section_from_verified(
        snapshot, spec, locale, local_record=checked_record, export_csv=export_csv,
        export_html=export_html, export_pdf=export_pdf)


def _comparison_section_from_verified(snapshot, spec, locale, *, local_record=None,
                                      export_csv=False, export_html=False, export_pdf=False):
    """Internal renderer for a previously verified snapshot and inspected receipt."""

    assurance, comparison, records = snapshot
    selected = records['baseline']['spec']
    if spec is not None and canonical(spec) != canonical(selected):
        raise ScopeError('STUDY_PREVIEW_COMPARISON_STUDY_MISMATCH')
    # Read-only rendering from the trusted caller-provided snapshot.
    pdf_download = ''
    if export_pdf:
        blob, details = _pdf_with_record(snapshot, local_record)
        label, disclosure = PDF_TEXT[locale]
        pdf_download = ('<section class="local-comparison-pdf"><p>' + escape(disclosure)
                        + '</p><p>PDF SHA-256: <code>' + details['sha256']
                        + '</code></p><a download="societybox-comparison-'
                        + assurance['comparison_id'] + '.pdf" href="data:application/pdf;base64,'
                        + base64.b64encode(blob).decode('ascii') + '">' + escape(label)
                        + '</a></section>')
    html_download = ''
    if export_html:
        blob, _ = _document_with_record(snapshot, local_record)
        label, disclosure = HTML_TEXT[locale]
        html_download = ('<section class="local-comparison-html"><p>' + escape(disclosure)
                         + '</p><p>HTML SHA-256: <code>' + digest(blob)
                         + '</code></p><a download="societybox-comparison-'
                         + assurance['comparison_id'] + '.html" href="data:text/html;base64,'
                         + base64.b64encode(blob).decode('ascii') + '">' + escape(label)
                         + '</a></section>')
    csv_download = ''
    if export_csv:
        blob, rows = _comparison_trajectory_bytes(*snapshot)
        label, disclosure = CSV_TEXT[locale]
        csv_download = ('<section class="local-comparison-csv"><p>' + escape(disclosure)
                        + '</p><p>CSV SHA-256: <code>' + digest(blob)
                        + '</code>; rows: ' + str(rows) + '</p><a download="societybox-comparison-'
                        + assurance['comparison_id'] + '.csv" href="data:text/csv;base64,'
                        + base64.b64encode(blob).decode('ascii') + '">' + escape(label)
                        + '</a></section>')
    # Keep computed deltas after the evidence and model conditions.
    assurance = {key: value for key, value in assurance.items() if key != 'final_deltas'}
    assurance.update(authorization='NOT_GRANTED', host_transfer='NOT_RUN')
    conditions = {
        'title': selected['title'], 'question': selected['question'],
        'definition': selected['model'], 'limitations': selected['limitations'],
        'export_rights': selected['export_rights'],
        'runs': {role: {'run_id': record['original_run_id'],
                        'overrides': record['overrides'],
                        'effective_parameters': record['results']['parameters']}
                 for role, record in records.items()},
    }
    graphs = {role: _project_claim_graph(record, record['original_run_id'])
              for role, record in records.items()}
    labels, common = COMPARISON_TEXT[locale], RUN_TEXT[locale]
    cards = ((common[2], assurance), (common[3], selected['evidence']),
             (common[4], conditions), (labels[2], comparison), (common[5], graphs))
    parts = ['<section><h2>' + escape(labels[0]) + '</h2><p>' + escape(labels[1]) + '</p>']
    for heading, card in cards:
        parts.extend(['<section><h3>' + escape(heading) + '</h3><pre><code>',
                      escape(json.dumps(card, ensure_ascii=False, allow_nan=False, indent=2)),
                      '</code></pre></section>'])
    parts.append(_trajectory_figures(comparison, records, locale))
    if local_record is not None:
        heading, limits = RECEIPT_TEXT[locale]
        parts.extend(['<section><h3>' + escape(heading) + '</h3><p>' + escape(limits)
                      + '</p><pre><code>',
                      escape(json.dumps(local_record, ensure_ascii=False, allow_nan=False, indent=2)),
                      '</code></pre></section>'])
    return ''.join(parts) + csv_download + html_download + pdf_download + '</section>'
