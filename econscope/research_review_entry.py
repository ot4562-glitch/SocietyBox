"""Self-contained offline selection of precomputed, read-only review snapshots.

Only the authored selection program runs in the browser. Research/model content
is escaped by the existing Python review renderer and is never executable code.
No server, provider, upload, approval or filesystem access is exposed in the browser.
Explicit run snapshots are verified during generation by deterministic recomputation.
"""
import base64
import hashlib
from html import escape
from pathlib import Path

from .agent_preview import _study_section
from .feature_catalog import describe_feature
from .agent_preview_labels import PREVIEW_TEXT
from .agent_host import prepare_host_request, parse_host_request
from .agent_turn import receive_host_turn
from .agent_review_html import render_host_turn_section
from .agent_model_preview import selected_model_section, compared_model_section
from .agent_run_preview import selected_run_section
from .agent_comparison_preview import comparison_section
from . import agent_comparison_preview as _comparison_preview
from .workspace import ScopeError, SHA, canonical, load_json


MAX_CHOICES = 4
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024


def render_research_chat(locale='ko'):
    """Live local entry over the same study reports, without owner HTML assets."""
    from .research_chat_ui import render
    return render(locale)


def research_workflow_view_model(spec, receipt, locale="en"):
    """Read-only conversation/evidence layout contract; never grants execution.

    Designed for the existing user-supplied chat-first interface; this function
    only produces labels and already computed, receipt-bound facts. It does not
    make an AI response or choose the next researcher's actions.
    """
    labels = {
        "en": ("Research question", "Evidence and authorization",
               "Verification obligations", "Conditional finding", "Next investigation",
               "No live LLM. A calculation is not a causal finding.",
               "No verification obligations were recorded in this receipt.",
               "Confirm source rights, vintage, assumptions and external validity."),
        "ko": ("연구 질문", "증거와 실행 승인", "검증 의무", "조건부 계산 결과", "다음 조사",
               "실제 LLM 추론 없음. 계산값은 인과적 결론이 아닙니다.",
               "이 영수증에는 검증 의무가 기록되지 않았습니다.",
               "자료 권리·공표 시점·가정·외부 타당성을 확인하세요."),
        "zh-Hans": ("研究问题", "证据与授权", "验证义务", "条件计算结果", "下一步调查",
                    "未执行实时语言模型推理。计算不等于因果结论。",
                    "本记录未提供验证义务。",
                    "核对来源权利、发布时点、假设和外部有效性。"),
        "ja": ("研究上の問い", "証拠と承認", "検証義務", "条件付き計算結果", "次の調査",
               "ライブLLM推論なし。計算結果は因果的結論ではありません。",
               "この記録に検証義務はありません。",
               "出典の権利、公開時点、仮定、外部妥当性を確認してください。"),
    }
    if locale not in labels:
        raise ValueError("ENGINE_STUDY_LOCALE_INVALID")
    result = receipt.get("result", {})
    result = result if type(result) is dict else {}
    coverage = result.get("coverage", result.get("native_coverage"))
    rows = coverage.get("obligations", ()) if type(coverage) is dict else ()
    rows = rows if type(rows) is list else ()
    # Do not display untrusted, unbounded receipt material in the summary.
    pending = [{"claim": row.get("claim_id", "UNSPECIFIED"),
                "state": row.get("state", "UNKNOWN"),
                "next_probe": row.get("next_probe", "")}
               for row in rows if type(row) is dict and row.get("state") != "PASS"]
    execution = receipt.get("native_execution", "NOT_RUN")
    parity = receipt.get("native_parity", "NOT_RUN")
    status = result.get("status", "NOT_REPORTED")
    return {"schema": "societybox-workflow-readonly-preview-v1",
            "labels": labels[locale], "question": spec.get("question", ""),
            "feature": describe_feature(spec["family"], locale)["name"],
            "source_id": spec.get("source", {}).get("id", "UNDECLARED"),
            "rights": spec.get("rights", {}),
            "authorization": receipt.get("review", {}).get("approval", "NOT_GRANTED"),
            "source_clearance": receipt.get("review", {}).get(
                "source_clearance", "OPERATOR_DECLARATION_NOT_VERIFIED"),
            "native_execution": execution, "native_parity": parity,
            "result_status": status, "coverage_status": (
                coverage.get("status", "NOT_RECORDED") if type(coverage) is dict else "NOT_RECORDED"),
            "pending_obligations": pending[:5],
            "pending_count": len(pending), "obligations_recorded": len(rows),
            "epistemic_state": "CAUSAL_AND_EXTERNAL_VALIDITY_NOT_ESTABLISHED",
            "llm": "NO_LLM", "controls": "READ_ONLY_NO_EXECUTION_NO_TRANSFER"}


def render_research_workflow_preview(spec, receipt, locale="en"):
    """Compact HTML summary at the front of existing static engine reports.

    Full evidence, numeric tables and technical receipts remain unchanged and
    follow this first-glance area. This is not a new standalone mockup/UI app.
    """
    view = research_workflow_view_model(spec, receipt, locale)
    labels = view["labels"]
    def safe(value):
        return escape(str(value), quote=True)
    pending = view["pending_obligations"]
    if pending:
        obligations = "".join(
            '<li><strong>' + safe(row["claim"]) + '</strong>: ' +
            safe(row["state"]) +
            (' — ' + safe(row["next_probe"]) if row["next_probe"] else '') +
            '</li>' for row in pending)
    else:
        obligations = '<li>' + safe(labels[6] if not view["obligations_recorded"]
                                    else view["coverage_status"]) + '</li>'
    return ('<section class="research-workflow-summary" aria-label="' + safe(labels[0]) + '">'
            '<div class="research-workflow-primary">'
            '<article class="research-question"><h3>' + safe(labels[0]) + '</h3><p>' +
            safe(view["question"]) + '</p></article>'
            '<article class="research-evidence-summary"><h3>' + safe(labels[1]) + '</h3><p>' +
            safe(view["source_id"]) + ' · ' + safe(view["source_clearance"]) +
            ' · ' + safe(view["authorization"]) + ' · local_processing: ' +
            safe(view["rights"].get("local_processing", "NOT_ESTABLISHED")) +
            ' · report_export: ' + safe(view["rights"].get("report_export", "NOT_ESTABLISHED")) +
            '</p></article>'
            '<article class="research-conditional-result"><h3>' + safe(labels[3]) +
            '</h3><p>' + safe(view["feature"]) + ' · ' + safe(view["result_status"]) +
            ' · ' + safe(view["native_parity"]) + ' · ' +
            safe(view["native_execution"]) + '</p><p>' + safe(labels[5]) +
            '</p></article></div>'
            '<aside class="research-workflow-assurance"><h3>' + safe(labels[2]) +
            '</h3><p>' + safe(view["coverage_status"]) + ' · ' +
            safe(view["pending_count"]) + '</p><ol>' + obligations + '</ol>'
            '<h3>' + safe(labels[4]) + '</h3><p>' + safe(labels[7]) +
            '</p><p>' + safe(view["epistemic_state"]) + '</p></aside></section>')


def render_engine_study_review(spec, receipt, locale='en', *, as_section=False):
    """Static, escaped report for the explicit offline economic study workflow.

    Reuses the research review page styling. This report contains private input
    data and grants no authority to execute or contact a model provider.
    """
    import json
    from .engine_study import chart_svg
    labels = {
        'en': ('Economic study review', 'Source declaration', 'Model and data',
               'Conditional result', 'Private local report. Operator declarations are unverified.'),
        'zh-Hans': ('经济研究审查', '来源声明', '模型与数据', '条件计算结果',
                    '本地私有报告。操作员声明未经核实。'),
        'ko': ('경제 연구 검토', '자료 출처 선언', '모형과 자료', '조건부 계산 결과',
               '비공개 로컬 보고서입니다. 운영자 선언은 검증되지 않았습니다.'),
        'ja': ('経済研究レビュー', '出典の申告', 'モデルとデータ', '条件付き計算結果',
               '非公開のローカルレポート。操作者の申告は未検証です。'),
    }
    if locale not in labels:
        raise ValueError('ENGINE_STUDY_LOCALE_INVALID')
    title, source, model, result, privacy = labels[locale]
    feature = describe_feature(spec['family'], locale)
    feature_heading = '<h2>'+escape(feature['name'])+'</h2>'
    feature_id = '<p>Society Box feature: '+escape(feature['feature_id'])+'</p>'
    def block(value):
        return '<pre>'+escape(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))+'</pre>'
    status = ('<p>NO_LLM · LLM_REASONING: NOT_PERFORMED · '
              'native_execution: '+escape(receipt['native_execution'])+' · native_parity: '
              +escape(receipt['native_parity'])+' · source_clearance: '
              'OPERATOR_DECLARATION_NOT_VERIFIED</p>')
    content = ('' if spec['family'] in ('forecast_validation','regression_sensitivity','fixed_effects_regression','input_output','survey_replicate','vintage_comparison') else feature_heading+feature_id+'<h3>'+source+'</h3>'
               +block({'source': spec['source'], 'rights': spec['rights'],
                       'study_sha256': receipt['review']['binding']['study_sha256']})
               +'<h3>'+model+'</h3>'+block({'question': spec['question'], 'request': spec['request']})
               +'<h3>'+result+'</h3><p>'+escape(receipt['interpretation'])+'</p>'
               +block(receipt)+chart_svg(receipt['chart']))
    if spec['family'] == 'dynamic_inverse':
        from .inverse_investigation import render_inverse_screen
        content += render_inverse_screen(spec['request']['parameters']['screen'], receipt['result']['screen'],
                                         locale, as_section=True)
    if spec['family'] == 'coefficient_identification':
        from .inverse_identification import render_review_section
        content += render_review_section(spec['request']['parameters'], receipt['result'], locale)
    if spec['family'] == 'supply_cost':
        from .network_cost_flow import render_review_section
        content += render_review_section(spec['request']['parameters'], receipt['result'], locale)
    if spec['family'] in ('forecast_validation','regression_sensitivity','fixed_effects_regression','vintage_comparison'):
        if spec['family'] == 'forecast_validation':
            from .forecast_validation import render_review_section
        elif spec['family'] == 'regression_sensitivity':
            from .regression_sensitivity import render_review_section
        elif spec['family'] == 'fixed_effects_regression':
            from .fixed_effects import render_review_section
        else:
            from .research_vintages import render_review_section
        technical_title={'en':'Technical receipt','ko':'기술 영수증','zh-Hans':'技术凭据','ja':'技術的な実行記録'}[locale]
        content=(feature_heading+feature_id+'<h3>'+source+'</h3>'
            +block({'source':spec['source'],'rights':spec['rights'],'study_sha256':receipt['review']['binding']['study_sha256']})
            +'<h3>'+result+'</h3><p>'+escape(receipt['interpretation'])+'</p>'
            +render_review_section(spec['request']['parameters'],receipt['result'],locale)+chart_svg(receipt['chart'])
            +'<details><summary>'+model+'</summary>'+block({'question':spec['question'],'request':spec['request']})+'</details>'
            +'<details class="technical-receipt"><summary>'+escape(technical_title)+'</summary>'+block(receipt)+'</details>')
    if spec['family'] in ('input_output','survey_replicate'):
        numeric = receipt['result']
        words = {
            'en': ('Sector / quantity', 'Direct', 'Indirect', 'Gross output', 'Exact value', 'Unit', 'Mean', 'Variance', 'Numerical verification', 'Technical receipt'),
            'ko': ('산업 / 통계량', '직접', '간접', '총생산 변화', '정확한 값', '단위', '평균', '분산', '수치 검산', '기술 영수증'),
            'zh-Hans': ('行业 / 统计量', '直接', '间接', '总产出变化', '精确值', '单位', '均值', '方差', '数值验证', '技术凭据'),
            'ja': ('産業 / 統計量', '直接', '間接', '総生産変化', '正確な値', '単位', '平均', '分散', '数値検証', '技術記録'),
        }[locale]
        if spec['family'] == 'input_output':
            headers = words[:4] + (words[5],)
            rows = zip(numeric['sector_names'], numeric['final_demand_direct_exact'],
                       numeric['indirect_output_change_exact'], numeric['gross_output_change_exact'],
                       [numeric['unit']]*len(numeric['sector_names']))
            limits = numeric['economic_claim']+' · '+numeric['induced_effect']
        else:
            headers = (words[0], words[4], words[5])
            rows = [(words[6], numeric['point_exact'], numeric['unit']),
                    (words[7], numeric['variance_exact'], numeric['unit']+' squared')]
            limits = numeric['status']+' · '+numeric['confidence_intervals']
        table = ('<table class="exact-study-values"><caption>'+escape(feature['name'])+'</caption><thead><tr>'
                 +''.join('<th scope="col">'+escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'
                 +''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)
                 +'</tbody></table>')
        content = (feature_heading+feature_id+'<h3>'+source+'</h3>'
                   +block({'source':spec['source'],'rights':spec['rights'],
                           'source_declared_sha256':spec['request']['parameters']['source_declared_sha256'],
                           'release_vintage':'NOT_ESTABLISHED'})
                   +'<h3>'+result+'</h3><p>'+escape(limits)+'</p>'+table+chart_svg(receipt['chart'])
                   +'<h3>'+escape(words[8])+'</h3><p>'+escape(numeric['independent_verification'])+'</p>'
                   +'<p>CAUSAL_AND_EXTERNAL_VALIDITY_NOT_ESTABLISHED</p>'
                   +'<details class="technical-receipt"><summary>'+escape(words[9])+'</summary>'+block(receipt)+'</details>')
    overview = render_research_workflow_preview(spec, receipt, locale)
    if as_section:
        return ('<section class="engine-study-details"><h2>'+title+'</h2><p>'+privacy+'</p>'
                +status+overview+content+'</section>')
    return ('<!doctype html><html lang="'+locale+'"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="Content-Security-Policy" content="default-src &#39;none&#39;; '
            'style-src &#39;sha256-'+_hash(STYLE)+'&#39;; base-uri &#39;none&#39;; form-action &#39;none&#39;">'
            '<title>'+title+'</title><style>'+STYLE+'</style></head><body><header><h1>'+title+'</h1>'
            '<p>'+privacy+'</p>'+status+'</header><main id="result">'+overview+content+'</main></body></html>')


ENGINE_STUDY_TEXT = {
    'en': 'Independent economic study. Recomputed when this page was generated; not a result of the selected scalar study or intervention. No input is attached to a host request. Keep this page private.',
    'zh-Hans': '独立经济研究。页面生成时已重新计算；不是所选标量研究或干预的结果。输入不会附加到主机请求。请妥善保密此页面。',
    'ko': '독립 경제 연구입니다. 페이지 생성 시 재계산했으며 선택한 스칼라 연구나 개입의 결과가 아닙니다. 입력은 호스트 요청에 첨부되지 않습니다. 페이지를 비공개로 보관하세요.',
    'ja': '独立した経済研究です。ページ生成時に再計算済みであり、選択したスカラー研究や介入の結果ではありません。入力はホスト要求に添付されません。このページは非公開で保管してください。',
}


def _engine_study_section(bundle, locale, sources, runtime=None):
    from . import engine_study
    try:
        verified = engine_study.verify(bundle, sources=sources, runtime=runtime)
        # Render only the bytes just verified, including after mutable-file
        # changes between reexecution and reading. No unverified result appears.
        blobs = {name: engine_study._read(Path(bundle)/name, engine_study.ARTIFACT_LIMIT)
                 for name in ('study.json', 'receipt.json')}
        if any(engine_study._sha(raw) != verified['artifacts'][name] for name, raw in blobs.items()):
            raise ValueError('ENGINE_STUDY_CHANGED_AFTER_VERIFY')
        spec, receipt = (engine_study._parse(blobs[name]) for name in ('study.json', 'receipt.json'))
        return ('<details class="independent-engine-study"><summary>'
                +escape(describe_feature(spec['family'], locale)['name']+' — '+spec['question'])+'</summary>'
                '<p>'+escape(ENGINE_STUDY_TEXT[locale])+'</p>'
                +render_engine_study_review(spec, receipt, locale, as_section=True)+'</details>')
    except (ValueError, OSError, RuntimeError, RecursionError, UnicodeError) as error:
        raise ScopeError('REVIEW_ENTRY_ENGINE_STUDY_INVALID') from error


EXPORT_TEXT = {
    'en': ('Download local request JSON', 'Prepared request only. Host consumption: NOT_ESTABLISHED. Approval and provider transfer: NOT_GRANTED. No selected study, model, run or comparison is attached. Keep this packet private.'),
    'zh-Hans': ('下载本地请求JSON', '仅准备请求。主机消费：NOT_ESTABLISHED。批准和提供方传输：NOT_GRANTED。不附加所选研究、模型、运行或比较。请妥善保密。'),
    'ko': ('로컬 요청 JSON 다운로드', '요청 준비만 완료했습니다. 호스트 소비: NOT_ESTABLISHED. 승인 및 제공자 전송: NOT_GRANTED. 선택한 연구, 모형, 실행 또는 비교는 첨부하지 않습니다. 이 묶음을 비공개로 보관하세요.'),
    'ja': ('ローカル要求JSONをダウンロード', '要求の準備のみです。ホスト消費：NOT_ESTABLISHED。承認とプロバイダー転送：NOT_GRANTED。選択した研究、モデル、実行、比較は添付しません。非公開で保管してください。'),
}
TEXT = {
    'en': ('Local research review', 'Study snapshot', 'Preserved model (optional)',
           'No model selected', 'Review selected inputs',
           'SNAPSHOT: reviews were prepared when this page was generated. Rerun the CLI to refresh changed inputs. '
           'All included reviews are stored in this page; keep it private. Selecting inputs grants no approval or execution.',
           'Select inputs and activate Review with the keyboard or pointer.',
           'Selected review snapshot displayed. Execution: BLOCKED. Approval: NOT_GRANTED.',
           'The initial review below is readable without JavaScript. Changing selections requires JavaScript; use conversation-preview to review other inputs without scripts.'),
    'zh-Hans': ('本地研究审查', '研究快照', '保存的模型（可选）', '不选择模型', '审查所选输入',
                'SNAPSHOT：审查在生成此页面时完成。输入更改后请重新运行CLI。页面包含所有审查内容，请妥善保密。选择输入不授予批准或执行权限。',
                '选择输入后，使用键盘或指针启动审查。',
                '已显示所选审查快照。执行：BLOCKED。批准：NOT_GRANTED。',
                '下方初始审查无需JavaScript即可阅读。更改选择需要JavaScript；无脚本审查其他输入请使用conversation-preview。'),
    'ko': ('로컬 연구 검토', '연구 스냅샷', '보존된 모형 (선택 사항)', '모형 선택 안 함', '선택한 입력 검토',
           'SNAPSHOT: 페이지 생성 시 검토한 결과입니다. 입력 변경 후 CLI를 다시 실행하세요. 모든 검토 내용이 이 페이지에 포함되므로 비공개로 보관하세요. 입력 선택은 승인 또는 실행 권한을 부여하지 않습니다.',
           '입력을 선택한 뒤 키보드 또는 포인터로 검토를 실행하세요.',
           '선택한 검토 스냅샷을 표시했습니다. 실행: BLOCKED. 승인: NOT_GRANTED.',
           '아래 초기 검토는 JavaScript 없이 읽을 수 있습니다. 선택 변경에는 JavaScript가 필요합니다. 다른 입력을 스크립트 없이 검토하려면 conversation-preview를 사용하세요.'),
    'ja': ('ローカル研究レビュー', '研究スナップショット', '保存されたモデル（任意）', 'モデルを選択しない', '選択した入力を確認',
           'SNAPSHOT：ページ生成時のレビューです。入力変更後はCLIを再実行してください。すべてのレビューを含むため非公開で保管してください。入力の選択は承認や実行権限を付与しません。',
           '入力を選び、キーボードまたはポインターで確認してください。',
           '選択したレビュースナップショットを表示しました。実行：BLOCKED。承認：NOT_GRANTED。',
           '下の初期レビューはJavaScriptなしで読めます。選択の変更にはJavaScriptが必要です。他の入力をスクリプトなしで確認するにはconversation-previewを使用してください。'),
}

INITIAL_TEXT = {
    'en': 'Initial snapshot: first listed study only, with any supplied intervention. No optional model, run or comparison selected. Execution: BLOCKED. Approval: NOT_GRANTED. To update this display, select inputs and activate Review.',
    'zh-Hans': '初始快照：仅显示列表中第一项研究及所提供的干预。不选择可选模型、运行或比较。执行：BLOCKED。批准：NOT_GRANTED。要更新显示，请选择输入并启动审查。',
    'ko': '초기 스냅샷: 목록의 첫 연구와 제공된 개입만 표시합니다. 선택 사항인 모형, 실행 또는 비교는 선택하지 않았습니다. 실행: BLOCKED. 승인: NOT_GRANTED. 표시를 바꾸려면 입력을 선택하고 검토를 실행하세요.',
    'ja': '初期スナップショット：一覧の最初の研究と指定された介入のみを表示します。任意のモデル、実行、比較は未選択です。実行：BLOCKED。承認：NOT_GRANTED。表示を更新するには入力を選択してレビューを実行してください。',
}

PENDING_TEXT = {
    'en': 'Selection changed. Previous results are hidden. Activate Review to display the selected snapshots. Approval: NOT_GRANTED.',
    'zh-Hans': '选择已更改。先前结果已隐藏。请启动审查以显示所选快照。批准：NOT_GRANTED。',
    'ko': '선택이 변경되었습니다. 이전 결과는 숨겨져 있습니다. 선택한 스냅샷을 보려면 검토를 실행하세요. 승인: NOT_GRANTED.',
    'ja': '選択が変更されました。前の結果は非表示です。選択したスナップショットを表示するにはレビューを実行してください。承認：NOT_GRANTED。',
}


RUN_SELECTION_TEXT = {
    'en': ('Preserved run (optional)', 'No run selected', 'Run verification unavailable',
           'Selected runs are checked against each study when this page is generated, including deterministic recomputation. Browser selection only displays snapshots. No new run is created; recorded parameter overrides are shown separately from the study review.'),
    'zh-Hans': ('保存的运行（可选）', '未选择运行', '运行验证不可用',
                '生成本页面时，会根据每项研究检查所选运行，包括确定性重新计算。浏览器选择仅显示快照。不会创建新运行；记录的参数覆盖值与研究审查分开显示。'),
    'ko': ('보존된 실행 (선택 사항)', '실행 선택 안 함', '실행 검증 불가',
           '페이지 생성 시 선택한 실행을 각 연구와 대조하여 검증하며 결정론적 재계산을 포함합니다. 브라우저에서 선택하면 스냅샷만 표시합니다. 새 실행은 생성하지 않으며 기록된 파라미터 재정의는 연구 검토와 별도로 표시합니다.'),
    'ja': ('保存済み実行（任意）', '実行未選択', '実行を検証できません',
           'ページ生成時に、決定論的な再計算を含め、選択した実行を各研究と照合します。ブラウザーでの選択はスナップショットを表示するだけです。新たな実行は作成せず、記録されたパラメータの上書きは研究レビューと別に表示します。'),
}

COMPARISON_SELECTION_TEXT = {
    'en': ('Portable comparison (optional)', 'No comparison selected', 'Comparison inspection unavailable',
           'Comparison bundles are checked for artifact consistency against each study when this page is generated, without engine recomputation. Comparison selection is independent of the optional run selection. Consistency does not establish authenticity, independent replication or empirical validity.'),
    'zh-Hans': ('便携比较（可选）', '不选择比较', '比较检查不可用',
                '生成页面时，根据每项研究检查比较包的文件一致性，不重新运行引擎。比较选择独立于可选运行选择。一致性不证明真实性、独立复现或实证有效性。'),
    'ko': ('이동식 비교 (선택 사항)', '비교 선택 안 함', '비교 검사 불가',
           '페이지 생성 시 각 연구와 비교 묶음의 산출물 일관성을 검사하며 엔진을 재계산하지 않습니다. 비교 선택은 선택적 실행 기록 선택과 독립적입니다. 일관성은 진위, 독립 재현 또는 실증적 타당성을 확립하지 않습니다.'),
    'ja': ('ポータブル比較（任意）', '比較を選択しない', '比較を検査できません',
           'ページ生成時に各研究と比較バンドルの成果物の整合性を確認し、エンジンの再計算は行いません。比較の選択は任意の実行記録の選択とは独立しています。整合性は真正性、独立した再現、実証的妥当性を確立しません。'),
}

STATUS_TEXT = {
    'en': ('New execution', 'Approval', 'LLM', 'Run verification failed',
           'Comparison inspection failed', 'Results', 'Engine recomputation'),
    'zh-Hans': ('新执行', '批准', '语言模型', '运行验证失败',
                '比较检查失败', '结果', '引擎重新计算'),
    'ko': ('새 실행', '승인', '언어 모형', '실행 검증 실패',
           '비교 검사 실패', '결과', '엔진 재계산'),
    'ja': ('新規実行', '承認', '言語モデル', '実行の検証に失敗しました',
           '比較の検査に失敗しました', '結果', 'エンジンの再計算'),
}

INTERVENTION_TEXT = {
    'en': 'Intended intervention: the supplied parameter overrides are reviewed against each study separately. Invalid overrides block that study review. Preserved runs and comparisons retain their recorded parameters; they are not results of this intervention.',
    'zh-Hans': '拟议干预：提供的参数覆盖值按每项研究分别审查。无效覆盖值会阻止该研究审查。保存的运行和比较保留其记录的参数，不是本次干预的结果。',
    'ko': '의도한 개입: 제공된 파라미터 재정의는 각 연구별로 검토합니다. 잘못된 값은 해당 연구 검토를 차단합니다. 보존된 실행과 비교는 기록된 파라미터를 유지하며 이 개입의 결과가 아닙니다.',
    'ja': '意図した介入：指定されたパラメータの上書きを研究ごとに確認します。無効な値はその研究のレビューをブロックします。保存済み実行と比較は記録されたパラメータを保持し、この介入の結果ではありません。',
}


MODEL_SOURCE_TEXT = {
    'en': 'Private page: full preserved model source is embedded, including in unselected snapshots. Keep this HTML private. Display only; execution, approval and provider transfer: NOT_GRANTED.',
    'zh-Hans': '私有页面：嵌入完整的已保存模型源文本，包括未选择的快照。请勿分享此HTML。仅供显示；执行、批准和向提供商传输：NOT_GRANTED。',
    'ko': '비공개 페이지: 선택하지 않은 스냅샷에도 보존된 모형 원문 전체가 포함됩니다. 이 HTML을 비공개로 보관하세요. 표시 전용이며 실행, 승인, 제공자 전송: NOT_GRANTED.',
    'ja': '非公開ページ：未選択のスナップショットにも保存済みモデルの全文が含まれます。このHTMLを非公開で保管してください。表示専用です。実行、承認、プロバイダー転送：NOT_GRANTED。',
}

INVALID_SELECTION_TEXT = {
    'en': 'INVALID_SELECTION: The selected snapshot is unavailable. Choose inputs and activate Review again. Results: NOT_SHOWN. Approval: NOT_GRANTED.',
    'zh-Hans': 'INVALID_SELECTION：所选快照不可用。请选择输入并重新启动审查。结果：NOT_SHOWN。批准：NOT_GRANTED。',
    'ko': 'INVALID_SELECTION: 선택한 스냅샷을 사용할 수 없습니다. 입력을 선택하고 검토를 다시 실행하세요. 결과: NOT_SHOWN. 승인: NOT_GRANTED.',
    'ja': 'INVALID_SELECTION：選択したスナップショットは利用できません。入力を選び、再度レビューを実行してください。結果：NOT_SHOWN。承認：NOT_GRANTED。',
}

NAVIGATION_TEXT = {
    'en': ('Inputs for displayed review', 'Change inputs'),
    'zh-Hans': ('当前审查的输入', '更改输入'),
    'ko': ('표시된 검토의 입력', '입력 변경'),
    'ja': ('表示中のレビューの入力', '入力を変更'),
}

NO_LLM_TEXT = {
    'en': 'Deterministic tools only. No LLM response was supplied; no LLM reasoning was performed. Preparing a host request does not run a model.',
    'zh-Hans': '仅使用确定性工具。未提供语言模型响应，未执行语言模型推理。准备主机请求不会运行模型。',
    'ko': '결정론적 도구만 사용합니다. LLM 응답이 제공되지 않았으며 LLM 추론을 수행하지 않았습니다. 호스트 요청 준비는 모형 실행이 아닙니다.',
    'ja': '決定論的ツールのみ。LLMの応答は提供されておらず、LLM推論は実行されていません。ホスト要求の準備はモデルを実行しません。',
}

# This fixed program only clones already escaped review elements. It has no
# evaluation, network or data parser and never interprets researcher content.
SCRIPT = """'use strict';
const form = document.getElementById('selection');
const result = document.getElementById('result');
const status = document.getElementById('status');
const reviewed = document.getElementById('reviewed-selection');
const edit = document.getElementById('change-inputs');
const selectionIds = ['study', 'model', 'run', 'comparison'];
let displayedSelection = ['s0', 'm0', 'r0', 'c0'];
function changeInputs() {
  form.hidden = false;
  reviewed.hidden = true;
  result.hidden = true;
  status.textContent = status.dataset.pending;
}
edit.hidden = false;
edit.addEventListener('click', function() {
  changeInputs();
  document.getElementById('study').focus();
});
form.addEventListener('change', changeInputs);
form.addEventListener('input', changeInputs);
form.addEventListener('reset', changeInputs);
window.addEventListener('pageshow', function() {
  if (selectionIds.some((id, index) =>
      document.getElementById(id).value !== displayedSelection[index])) changeInputs();
});
form.addEventListener('submit', function(event) {
  event.preventDefault();
  const study = document.getElementById('study').value;
  const model = document.getElementById('model').value;
  const selected = document.querySelector('template[data-review="' + study + '-' + model + '"]');
  const run = document.getElementById('run').value;
  const runReview = run === 'r0' ? null : document.querySelector('template[data-run="' + study + '-' + run + '"]');
  const comparison = document.getElementById('comparison').value;
  const comparisonReview = comparison === 'c0' ? null : document.querySelector('template[data-comparison="' + study + '-' + comparison + '"]');
  if (!selected || (run !== 'r0' && !runReview) || (comparison !== 'c0' && !comparisonReview)) {
    changeInputs();
    result.replaceChildren();
    status.textContent = status.dataset.invalid;
    document.getElementById('study').focus();
    return;
  }
  result.replaceChildren(selected.content.cloneNode(true));
  if (runReview) result.appendChild(runReview.content.cloneNode(true));
  if (comparisonReview) result.appendChild(comparisonReview.content.cloneNode(true));
  for (const id of selectionIds) {
    document.getElementById('reviewed-' + id).textContent = document.getElementById(id).selectedOptions[0].textContent;
  }
  displayedSelection = selectionIds.map(id => document.getElementById(id).value);
  form.hidden = true;
  reviewed.hidden = false;
  result.hidden = false;
  status.textContent = status.dataset.complete;
  result.focus();
});"""

STYLE = """[hidden]{display:none!important}
body{margin:auto;padding:1rem;max-width:70rem;font-family:system-ui,sans-serif;
line-height:1.5;background:#f6f5f1;color:#202521}header,form,#result,#reviewed-selection{background:white;
padding:1rem;margin-bottom:1rem;border:1px solid #bbc5bf;border-radius:.5rem}
form{display:flex;gap:1rem;flex-wrap:wrap;align-items:end}label{display:block;font-weight:600}
select,button{font:inherit;padding:.6rem;max-width:100%}form>div{min-width:0;max-width:100%}
table{display:block;max-width:100%;overflow-x:auto}th,td{padding:.3rem;text-align:start}
pre{white-space:pre-wrap;overflow-wrap:anywhere}dd{margin-inline-start:1rem}
summary{cursor:pointer;overflow-wrap:anywhere}
:focus-visible{outline:3px solid #27614b;outline-offset:3px}h1,h2,h3,dd,p{overflow-wrap:anywhere}
button{cursor:pointer;background:#234d3c;color:white;border:0;border-radius:.3rem}
.research-workflow-summary{display:grid;grid-template-columns:minmax(0,2fr) minmax(14rem,1fr);gap:1rem;max-width:100%;margin-bottom:1.5rem}
.research-workflow-summary article,.research-workflow-summary aside{background:#fff;border:1px solid #bbc5bf;border-radius:.45rem;padding:.8rem;margin-bottom:.6rem;min-width:0}
.research-workflow-summary aside{background:#f7f9f7}.research-workflow-summary ol{padding-inline-start:1.2rem;overflow-wrap:anywhere}
@media(max-width:60rem){.research-workflow-summary{grid-template-columns:minmax(0,1fr)}}
@media(max-width:40rem){body{padding:.5rem}form>div,select,button{width:100%}}"""


def _hash(text):
    return base64.b64encode(hashlib.sha256(text.encode('utf-8')).digest()).decode('ascii')


MODEL_DIFF_TEXT = {
    'en': 'Private page: model differences are embedded even in unselected snapshots. Keep this HTML private. Left/right selection establishes no revision chronology or semantic equivalence. Host transfer: NOT_GRANTED.',
    'zh-Hans': '\u79c1\u5bc6\u9875\u9762\uff1a\u672a\u9009\u62e9\u7684\u5feb\u7167\u4e5f\u5305\u542b\u6a21\u578b\u5dee\u5f02\u3002\u8bf7\u4fdd\u5bc6\u6b64 HTML\u3002\u5de6\u53f3\u9009\u62e9\u4e0d\u8bc1\u660e\u4fee\u8ba2\u65f6\u5e8f\u6216\u8bed\u4e49\u7b49\u4ef7\u3002\u4e3b\u673a\u4f20\u8f93\uff1aNOT_GRANTED\u3002',
    'ko': '\ube44\uacf5\uac1c \ud398\uc774\uc9c0: \uc120\ud0dd\ud558\uc9c0 \uc54a\uc740 \uc2a4\ub0c5\uc0f7\uc5d0\ub3c4 \ubaa8\ud615 \ucc28\uc774\uac00 \ud3ec\ud568\ub429\ub2c8\ub2e4. HTML\uc744 \ube44\uacf5\uac1c\ub85c \ubcf4\uad00\ud558\uc138\uc694. \uc88c\uc6b0 \uc120\ud0dd\uc740 \uac1c\uc815 \uc21c\uc11c\ub098 \uc758\ubbf8 \ub3d9\ub4f1\uc131\uc744 \uc785\uc99d\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4. \ud638\uc2a4\ud2b8 \uc804\uc1a1: NOT_GRANTED.',
    'ja': '\u975e\u516c\u958b\u30da\u30fc\u30b8: \u672a\u9078\u629e\u306e\u30b9\u30ca\u30c3\u30d7\u30b7\u30e7\u30c3\u30c8\u306b\u3082\u30e2\u30c7\u30eb\u306e\u5dee\u5206\u304c\u542b\u307e\u308c\u307e\u3059\u3002HTML\u3092\u975e\u516c\u958b\u3067\u4fdd\u7ba1\u3057\u3066\u304f\u3060\u3055\u3044\u3002\u5de6\u53f3\u306e\u9078\u629e\u306f\u6539\u8a02\u9806\u5e8f\u3084\u610f\u5473\u306e\u540c\u7b49\u6027\u3092\u8a3c\u660e\u3057\u307e\u305b\u3093\u3002\u30db\u30b9\u30c8\u8ee2\u9001: NOT_GRANTED\u3002',
}


def research_review_entry(workspace, study_files, model_artifact_ids=(), locale='en', run_ids=(), comparison_bundles=(), *, comparison_receipts=None, overrides=None, researcher_request=None, response_bytes=None, export_host_request=False, request_bytes=None, export_comparison_csv=False, export_comparison_html=False, export_comparison_pdf=False, model_show_source=False, model_compare_artifact_id=None, model_expected_sha256=None, imported_host_record=False, engine_study_bundles=(), engine_sources=None, engine_runtime=None, inverse_screen_specs=()):
    """Prepare bounded study/model reviews and study-bound run snapshots.

    All supplied inputs must pass the existing boundary before returning HTML.
    The study is the research contract; an optional .mod is inspected separately
    and is never represented as the study's executable implementation.
    Browser selections only reveal these snapshots and do not re-read disk.
    Submission replaces the selector with a summary of the displayed inputs;
    Change inputs restores the retained choices and hides the previous cards.
    Overrides apply only to intended study reviews, never preserved results.
    An optional local request/response pair is unauthenticated inspection data;
    study/model/run content is never appended to its prospective host request.
    Explicit export embeds only that prepared request as an inert local download;
    it does not consume the packet or authorize provider transfer.
    A saved request is an alternative to request text, requires a response, and
    must match the current host contract and selected locale before any review.
    comparison_receipts maps one-based comparison choices to explicit local
    unsigned receipt paths. Inspection never recomputes or authenticates them.
    export_comparison_csv embeds an optional private download inside each valid
    study/comparison snapshot, after study and receipt binding checks succeed.
    export_comparison_html adds the English evidence-first report from that same
    snapshot and inspected receipt; export_comparison_pdf adds its limited ASCII
    PDF equivalent. These downloads are never attached to the host.
    model_show_source opts into escaped full source in every model snapshot;
    unselected templates also contain that private text. No source enters host packets.
    model_compare_artifact_id instead selects an inert right-side revision diff
    against exactly one left artifact, shared across study choices. This private
    text is embedded in unselected templates too and establishes no chronology.
    model_expected_sha256 asserts the single selected (left) model's bytes;
    it does not authenticate origin or grant permission.
    imported_host_record marks caller-rendered, untrusted host text. It changes
    only the disclosure; it neither consumes a host packet nor grants authority.
    Explicit inverse_screen_specs opt into deterministic local regression and
    freshly recomputed evidence sections. They do not execute study models or
    untrusted code, grant study approval, or attach data to host requests.
    """
    if type(imported_host_record) is not bool:
        raise ScopeError('REVIEW_ENTRY_IMPORTED_HOST_MARKER_INVALID')
    if model_expected_sha256 is not None:
        if workspace is None or len(model_artifact_ids) != 1:
            raise ScopeError('REVIEW_ENTRY_MODEL_DIGEST_SELECTION_INVALID')
        if not isinstance(model_expected_sha256, str) or not SHA.fullmatch(model_expected_sha256):
            raise ScopeError('MOD_EXPECTED_SHA256_INVALID')
    if model_compare_artifact_id is not None and (
            workspace is None or len(model_artifact_ids) != 1 or model_show_source):
        raise ScopeError('REVIEW_ENTRY_MODEL_DIFF_SELECTION_INVALID')
    if type(model_show_source) is not bool or (model_show_source and not model_artifact_ids):
        raise ScopeError('REVIEW_ENTRY_MODEL_SOURCE_SELECTION_INVALID')
    if type(export_comparison_csv) is not bool or (export_comparison_csv and not comparison_bundles):
        raise ScopeError('REVIEW_ENTRY_COMPARISON_EXPORT_REQUIRES_SELECTION')
    if type(export_comparison_html) is not bool or (export_comparison_html and not comparison_bundles):
        raise ScopeError('REVIEW_ENTRY_COMPARISON_EXPORT_REQUIRES_SELECTION')
    if type(export_comparison_pdf) is not bool or (export_comparison_pdf and not comparison_bundles):
        raise ScopeError('REVIEW_ENTRY_COMPARISON_EXPORT_REQUIRES_SELECTION')
    if locale not in TEXT:
        raise ScopeError('REVIEW_ENTRY_LOCALE_UNSUPPORTED')
    if comparison_receipts is None:
        comparison_receipts = {}
    if (not isinstance(comparison_receipts, dict)
        or any(type(index) is not int or not 1 <= index <= len(comparison_bundles)
               or not isinstance(path, (str, Path)) or not str(path)
               for index, path in comparison_receipts.items())):
        raise ScopeError('REVIEW_ENTRY_COMPARISON_RECEIPTS_INVALID')
    if request_bytes is not None and (researcher_request is not None or export_host_request):
        raise ScopeError('REVIEW_ENTRY_REQUEST_INPUT_CONFLICT')
    has_request = researcher_request is not None or request_bytes is not None
    if not export_host_request and has_request != (response_bytes is not None):
        raise ScopeError('REVIEW_ENTRY_REQUEST_RESPONSE_PAIR_REQUIRED')
    conversation = ''
    if request_bytes is not None:
        request = parse_host_request(request_bytes)
        if request['locale'] != locale:
            raise ScopeError('REVIEW_ENTRY_REQUEST_LOCALE_MISMATCH')
        conversation = ('<p>request_file_sha256: <code>'
                        + hashlib.sha256(request_bytes).hexdigest() + '</code></p>')
    elif export_host_request or response_bytes is not None:
        request = prepare_host_request(researcher_request, locale)
    if export_host_request:
        label, disclosure = EXPORT_TEXT[locale]
        encoded = base64.b64encode(canonical(request)).decode('ascii')
        conversation = ('<section class="local-request-export"><p>' + escape(disclosure)
                        + '</p><a download="societybox-host-request.json" href="data:application/json;base64,'
                        + encoded + '">' + escape(label) + '</a></section>')
    if response_bytes is not None:
        turn = receive_host_turn(request, lambda _: response_bytes)
        conversation += ('<article class="local-host-review"><p>'
                        + escape(PREVIEW_TEXT[locale]['import']) + '</p>'
                        + render_host_turn_section(turn) + '</article>')
    if not study_files:
        raise ScopeError('REVIEW_ENTRY_STUDY_SELECTION_REQUIRED')
    if overrides is not None and workspace is None:
        raise ScopeError('STUDY_PREVIEW_OVERRIDES_REQUIRE_STUDY_AND_WORKSPACE')
    if engine_sources is not None and not engine_study_bundles:
        raise ScopeError('REVIEW_ENTRY_ENGINE_SOURCES_UNPAIRED')
    if engine_runtime is not None and not engine_study_bundles:
        raise ScopeError('REVIEW_ENTRY_ENGINE_RUNTIME_UNPAIRED')
    if not isinstance(inverse_screen_specs, (list, tuple)):
        raise ScopeError('REVIEW_ENTRY_INVERSE_SELECTION_INVALID')
    if any(len(choices) > MAX_CHOICES for choices in
           (study_files, model_artifact_ids, run_ids, comparison_bundles, engine_study_bundles, inverse_screen_specs)):
        raise ScopeError('REVIEW_ENTRY_SELECTION_LIMIT')
    if any(not isinstance(key, str) or not SHA.fullmatch(key) for key in run_ids):
        raise ScopeError('STUDY_RUN_ID_INVALID')
    labels = TEXT[locale]
    run_labels = RUN_SELECTION_TEXT[locale]
    comparison_labels = COMPARISON_SELECTION_TEXT[locale]
    status_labels = STATUS_TEXT[locale]
    navigation_labels = NAVIGATION_TEXT[locale]
    studies = [(Path(p).name, load_json(Path(p))) for p in study_files]
    if any(spec is None for _, spec in studies):
        raise ScopeError('STUDY_SCHEMA_UNSUPPORTED')
    # Request-local SHA256 identities of complete canonical proposals (including
    # rights). Named copies remain separate UI
    # choices, but must see the same reviewed bytes. Never retain these snapshots
    # across calls: a later request must recheck mutable sources and receipts.
    # Keep fixed-size keys rather than retaining extra copies of large inputs.
    study_keys = [hashlib.sha256(canonical(spec)).digest() for _, spec in studies]
    study_sections, run_sections, comparison_sections = {}, {}, {}
    models = [None, *model_artifact_ids]
    # One verified model snapshot per selection, shared across study choices.
    model_sections = {key: compared_model_section(
        workspace, key, model_compare_artifact_id, model_expected_sha256 or key, locale)
        if model_compare_artifact_id is not None else selected_model_section(
        None, None, model_expected_sha256, locale, workspace=workspace, artifact_id=key,
        show_source=model_show_source) for key in model_artifact_ids}
    templates = []
    initial_review = ''
    total_bytes = len(conversation.encode('utf-8'))
    engine_sections = []
    for bundle in engine_study_bundles:
        body = _engine_study_section(bundle, locale, engine_sources, engine_runtime)
        total_bytes += len(body.encode('utf-8'))
        if total_bytes > MAX_DOCUMENT_BYTES - 32768:
            raise ScopeError('REVIEW_ENTRY_DOCUMENT_TOO_LARGE')
        engine_sections.append(body)
    if inverse_screen_specs:
        from .mechanism_cli import load_input
        from .inverse_screen import screen_inverse_hypotheses
        from .inverse_investigation import render_inverse_screen
        for path in inverse_screen_specs:
            request = load_input(path)
            result = screen_inverse_hypotheses(request)
            body = render_inverse_screen(request, result, locale, as_section=True)
            total_bytes += len(body.encode('utf-8'))
            if total_bytes > MAX_DOCUMENT_BYTES - 32768:
                raise ScopeError('REVIEW_ENTRY_DOCUMENT_TOO_LARGE')
            engine_sections.append(body)
    for i, (_, spec) in enumerate(studies):
        # One review per distinct proposal, reused across displayed model choices.
        # Repeated reviews perform redundant data validation and can observe
        # different mutable input snapshots inside the same offline page.
        study_key = study_keys[i]
        if study_key not in study_sections:
            study_sections[study_key] = _study_section(
                spec, workspace=workspace, overrides=overrides, locale=locale)
        reviewed_study = study_sections[study_key]
        for j, model_id in enumerate(models):
            # Reuse the actual study/model review sections, without the preview's
            # synthetic host conversation or unrelated fixture tool proposal.
            body = reviewed_study
            if overrides is not None:
                body = ('<section class="intended-intervention"><p>'
                        + escape(INTERVENTION_TEXT[locale]) + '</p>' + body + '</section>')
            if model_id is not None:
                body += model_sections[model_id]
            if i == 0 and j == 0:
                # Reuse the escaped snapshot before any script runs; optional
                # artifacts remain unselected until explicit submission.
                initial_review = body
                total_bytes += len(initial_review.encode('utf-8'))
            template = f'<template data-review="s{i}-m{j}">{body}</template>'
            total_bytes += len(template.encode('utf-8'))
            if total_bytes > MAX_DOCUMENT_BYTES - 32768:
                raise ScopeError('REVIEW_ENTRY_DOCUMENT_TOO_LARGE')
            templates.append(template)
    # Comparison consistency is independent of optional run recomputation.
    # Reuse the portable boundary without executing or consulting local runs.
    # Prevalidated comparisons/receipts are shared across study choices.
    comparison_snapshots, comparison_records = {}, {}
    for i, (_, spec) in enumerate(studies):
        for k, folder in enumerate(comparison_bundles, 1):
            section_key = (study_keys[i], k)
            try:
                export_options = {'export_csv': True} if export_comparison_csv else {}
                if export_comparison_pdf:
                    export_options['export_pdf'] = True
                if export_comparison_html:
                    export_options['export_html'] = True
                if section_key in comparison_sections:
                    body = comparison_sections[section_key]
                elif len(studies) > 1:
                    # A single prevalidated byte snapshot per comparison ensures
                    # selection alternatives cannot observe differing file states.
                    if k not in comparison_snapshots:
                        try:
                            comparison_snapshots[k] = _comparison_preview._verified_comparison_snapshot(folder)
                        except ScopeError as failure:
                            comparison_snapshots[k] = failure
                    snapshot = comparison_snapshots[k]
                    if isinstance(snapshot, ScopeError):
                        raise snapshot
                    local_record = None
                    if (k in comparison_receipts and
                        canonical(spec) == canonical(snapshot[2]['baseline']['spec'])):
                        if k not in comparison_records:
                            try:
                                comparison_records[k] = _comparison_preview._inspect_receipt(
                                    snapshot, comparison_receipts[k])
                            except ScopeError as failure:
                                comparison_records[k] = failure
                        local_record = comparison_records[k]
                        if isinstance(local_record, ScopeError):
                            raise local_record
                    body = _comparison_preview._comparison_section_from_verified(
                        snapshot, spec, locale, local_record=local_record, **export_options)
                elif k in comparison_receipts:
                    body = comparison_section(folder, spec, locale,
                                              recomputation_receipt=comparison_receipts[k], **export_options)
                else:
                    body = comparison_section(folder, spec, locale, **export_options)
            except ScopeError as error:
                body = ('<section class="comparison-inspection-failure"><h2>'
                        + escape(comparison_labels[2]) + '</h2><p>'
                        + escape(status_labels[4]) + ' (COMPARISON_INSPECTION_FAILED): <code>'
                        + escape(str(error)) + '</code></p>'
                        + '<p>' + escape(status_labels[5]) + ': NOT_SHOWN. '
                        + escape(status_labels[6]) + ': NOT_RUN. '
                        + escape(status_labels[1]) + ': NOT_GRANTED.</p></section>')
            comparison_sections[section_key] = body
            template = f'<template data-comparison="s{i}-c{k}">{body}</template>'
            total_bytes += len(template.encode('utf-8'))
            if total_bytes > MAX_DOCUMENT_BYTES - 32768:
                raise ScopeError('REVIEW_ENTRY_DOCUMENT_TOO_LARGE')
            templates.append(template)
    # Verify each distinct proposal/run pair once, independent of named copies
    # and optional model choices. Cache failures too, never their raw receipts.
    # A rejected pair never includes results or unverified receipt content.
    for i, (_, spec) in enumerate(studies):
        for k, run_id in enumerate(run_ids, 1):
            section_key = (study_keys[i], run_id)
            try:
                if section_key not in run_sections:
                    run_sections[section_key] = selected_run_section(workspace, run_id, spec, None, locale)
                body = run_sections[section_key]
            except ScopeError as error:
                body = ('<section class="run-verification-failure"><h2>'
                        + escape(run_labels[2]) + '</h2><p><code>' + escape(run_id)
                        + '</code></p><p>' + escape(status_labels[3])
                        + ' (VERIFICATION_FAILED): <code>' + escape(str(error))
                        + '</code></p><p>' + escape(status_labels[5]) + ': NOT_SHOWN. '
                        + escape(status_labels[1]) + ': NOT_GRANTED.</p></section>')
            run_sections[section_key] = body
            template = f'<template data-run="s{i}-r{k}">{body}</template>'
            total_bytes += len(template.encode('utf-8'))
            if total_bytes > MAX_DOCUMENT_BYTES - 32768:
                raise ScopeError('REVIEW_ENTRY_DOCUMENT_TOO_LARGE')
            templates.append(template)
    csp = ("default-src 'none'; base-uri 'none'; form-action 'none'; "
           f"script-src 'sha256-{_hash(SCRIPT)}'; style-src 'sha256-{_hash(STYLE)}'")
    parts = ['<!doctype html><html lang="' + locale + '"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             '<meta name="referrer" content="no-referrer">',
             '<meta http-equiv="Content-Security-Policy" content="' + escape(csp, quote=True) + '">',
             '<title>' + escape(labels[0]) + '</title><style>' + STYLE + '</style></head><body>',
             '<header><h1>' + escape(labels[0]) + '</h1><p>' + escape(labels[5]) + '</p>',
             '<p>' + escape(run_labels[3]) + '</p>',
             '<p>' + escape(comparison_labels[3]) + '</p>',
             '<p>' + escape(status_labels[0]) + ': BLOCKED / NOT_RUN. '
             + escape(status_labels[1]) + ': NOT_GRANTED. '
             + escape(status_labels[2]) + ': '
             + ('NO_LLM. LLM_REASONING: NOT_PERFORMED.</p><p>'
                + escape(NO_LLM_TEXT[locale]) if response_bytes is None and not imported_host_record else
                'IMPORTED_RESPONSE_UNAUTHENTICATED. LIVE_HOST_CONSUMPTION: NOT_ESTABLISHED.')
             + '</p></header>',
             '<main>' + conversation + ('<p class="private-model-source">' + escape(MODEL_SOURCE_TEXT[locale]) + '</p>' if model_show_source else '')
             + ('<p class="private-model-diff">' + escape(MODEL_DIFF_TEXT[locale]) + '</p>' if model_compare_artifact_id is not None else '')
             + '<form id="selection"><div><label for="study">' + escape(labels[1]) + '</label>',
             '<select id="study" name="study">']
    for i, (name, _) in enumerate(studies):
        parts.append(f'<option value="s{i}">{i + 1}: {escape(name)}</option>')
    parts.extend(['</select></div><div><label for="model">' + escape(labels[2]) + '</label>',
                  '<select id="model" name="model"><option value="m0">' + escape(labels[3]) + '</option>'])
    for j, key in enumerate(model_artifact_ids, 1):
        parts.append(f'<option value="m{j}">{escape(key)}</option>')
    parts.extend(['</select></div><div><label for="run">' + escape(run_labels[0]) + '</label>',
                  '<select id="run" name="run"><option value="r0">' + escape(run_labels[1]) + '</option>'])
    for k, key in enumerate(run_ids, 1):
        parts.append(f'<option value="r{k}">{escape(key)}</option>')
    parts.extend(['</select></div><div><label for="comparison">' + escape(comparison_labels[0]) + '</label>',
                  '<select id="comparison" name="comparison"><option value="c0">' + escape(comparison_labels[1]) + '</option>'])
    for k, folder in enumerate(comparison_bundles, 1):
        parts.append(f'<option value="c{k}">{k}: {escape(Path(folder).name)}</option>')
    parts.extend(['</select></div><button type="submit">' + escape(labels[4]) + '</button></form>',
                  '<p id="status" role="status" aria-live="polite" data-complete="' + escape(labels[7], quote=True) + '" data-pending="' + escape(PENDING_TEXT[locale], quote=True) + '" data-invalid="' + escape(INVALID_SELECTION_TEXT[locale], quote=True) + '">',
                  escape(INITIAL_TEXT[locale]) + '</p>',
                  '<section id="reviewed-selection"><h2>' + escape(navigation_labels[0]) + '</h2><dl>'])
    for key, label, value in (
            ('study', labels[1], '1: ' + studies[0][0]),
            ('model', labels[2], labels[3]),
            ('run', run_labels[0], run_labels[1]),
            ('comparison', comparison_labels[0], comparison_labels[1])):
        parts.append('<dt>' + escape(label) + '</dt><dd id="reviewed-' + key + '">'
                     + escape(value) + '</dd>')
    parts.extend(['</dl><button id="change-inputs" type="button" hidden>'
                  + escape(navigation_labels[1]) + '</button></section>',
                  '<div id="result" tabindex="-1" aria-describedby="reviewed-selection">'
                  + initial_review + '</div>',
                  '<noscript><p>' + escape(labels[8]) + '</p></noscript>',
                  *templates, *engine_sections, '</main><script>' + SCRIPT + '</script></body></html>'])
    document = ''.join(parts)
    if len(document.encode('utf-8')) > MAX_DOCUMENT_BYTES:
        raise ScopeError('REVIEW_ENTRY_DOCUMENT_TOO_LARGE')
    return document
