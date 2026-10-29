"""Evidence projection and verified researcher view over the existing estimator.

No new solver/approval system. Local replay is explicitly not independent
replication. Assertions about source truth, causal effects, or full upstream
coverage stay open/blocked even when the exact calculation succeeds.
"""
from html import escape
import json
from pathlib import Path

from .feature_catalog import describe_feature
from .research_evidence import EvidenceLedger, UnknownRegistry, make_evidence, make_obligation, evaluate_coverage
from .workspace import ScopeError, canonical, digest

_CODE_FILES = ("inverse_screen.py", "mechanism_dynamic.py", "inverse_investigation.py",
               "inverse_numeric_verifier.py", "research_evidence.py", "workspace.py", "feature_catalog.py")
_LABELS = {
    "en": ("Research evidence", "Calculation", "Evidence and lineage", "Remaining questions",
           "Verification obligations", "Next investigation", "Exact holdout error / baseline",
           "Local recomputation is not independent replication. Data and release dates are declarations; causal identification is not established."),
    "ko": ("연구 증거", "계산 결과", "증거와 계보", "남은 미지수", "검증 의무", "다음 조사",
           "정확한 보류구간 오차 / 기준모형",
           "로컬 재계산은 독립 재현과 구별됩니다. 자료와 발표일은 선언값이며 인과 식별은 입증되지 않았습니다."),
    "zh-Hans": ("研究证据", "计算结果", "证据与谱系", "未解问题", "验证义务", "下一步调查",
                "精确保留样本误差 / 基准模型",
                "本地重新计算与独立复现有别。数据和发布日期为声明值，因果识别尚未确立。"),
    "ja": ("研究証拠", "計算結果", "証拠と系譜", "未解決の問い", "検証義務", "次の調査",
           "正確なホールドアウト誤差 / 基準モデル",
           "ローカル再計算と独立再現は区別されます。データと公開日は申告値であり、因果識別は確立されていません。"),
}
_LOCALES = ("en", "ko", "zh-Hans", "ja")
_CLAIM_NAMES = {
    "declared_calculation_executed": ("Declared calculation executed", "선언된 계산 실행", "声明的计算已执行", "申告された計算の実行"),
    "fitted_explanatory_model": ("Exploratory model fit", "탐색적 모형 적합", "探索模型拟合", "探索的モデルの適合"),
    "independent_numeric_parity": ("Independent arithmetic check", "독립 수치 검산", "独立算术核验", "独立した数値検算"),
    "source_and_rights_authentication": ("Source and rights authentication", "자료·사용권 인증", "来源与使用权认证", "資料・利用権の認証"),
    "latent_structure_identification": ("Latent structure identification", "잠재구조 식별", "潜在结构识别", "潜在構造の識別"),
    "causal_effect": ("Causal effect", "인과 효과", "因果效应", "因果効果"),
    "historical_as_of": ("Historical information set", "당시 실제 정보집합", "历史时点信息集", "当時の実際の情報集合"),
    "forecast_generalization": ("Forecast generalization", "예측의 외부 타당성", "预测泛化", "予測の外的妥当性"),
    "external_replication": ("External independent replication", "외부 독립 재현", "外部独立复现", "外部での独立再現"),
    "all_reference_public_api_integration": ("All-reference public API integration", "전체 레퍼런스 공개 API 통합", "全部参考项目公开API集成", "全参照プロジェクトの公開API統合"),
}
_QUESTION_NAMES = {
    "SOURCE_VINTAGE_RIGHTS": ("Are data, rights and release/access times authenticated?", "자료·사용권·발표시점이 인증되었는가?", "数据、使用权和发布时点是否已认证？", "資料・利用権・公開時点は認証済みか？"),
    "IDENTIFICATION": ("Which competing structures can independent observations distinguish?", "독립 관측으로 어떤 경쟁 구조를 구별할 수 있는가?", "独立观测可以区分哪些竞争结构？", "独立した観測でどの競合構造を区別できるか？"),
    "FORECAST_VALIDATION": ("Does the forecast generalize outside this exploratory sample?", "탐색 표본 밖에서도 예측이 유효한가?", "预测在探索样本之外是否有效？", "探索標本の外でも予測は有効か？"),
    "INDEPENDENT_REPLAY": ("Can an independently installed environment reproduce the study?", "독립 설치 환경에서 연구를 재현할 수 있는가?", "独立安装环境能否复现研究？", "独立したインストール環境で研究を再現できるか？"),
}
_NEXT_ACTIONS = {
    "declared_calculation_executed": ("Inspect exact coefficients, rank and information limits.", "정확한 계수·계수행렬의 rank·정보 한계를 검토하세요.", "检查精确系数、矩阵秩及信息限制。", "正確な係数・行列のランク・情報制約を確認してください。"),
    "fitted_explanatory_model": ("Review the fitted candidates and assumptions.", "적합된 후보와 가정을 검토하세요.", "审查拟合候选及假设。", "適合した候補と仮定を確認してください。"),
    "independent_numeric_parity": ("Inspect the Cramer oracle; seek external replication separately.", "Cramer 검산을 살펴보고 외부 독립 재현을 별도로 확인하세요.", "检查Cramer算术核验，另行开展外部独立复现。", "Cramer法の検算を確認し、外部独立再現を別途確認してください。"),
    "source_and_rights_authentication": ("Obtain reviewed source, rights and release/access records.", "검토된 원자료·사용권·발표 및 접근 기록을 확보하세요.", "获取经审查的原始资料、使用权及发布和访问记录。", "審査済みの原資料・利用権・公開とアクセス記録を取得してください。"),
    "latent_structure_identification": ("Challenge equivalent models, scale/rotation and negative controls.", "동치 모형·척도·회전 제약과 음성 대조군을 검증하세요.", "检验等价模型、尺度和旋转约束及负对照。", "同値モデル・尺度と回転の制約・負の対照を検証してください。"),
    "causal_effect": ("Specify and challenge an identification strategy.", "인과 식별 전략을 명시하고 반증하세요.", "明确因果识别策略并进行反证检验。", "因果識別戦略を明示し反証を試みてください。"),
    "historical_as_of": ("Authenticate actual release/access information at each origin.", "각 예측 원점의 실제 발표·접근 정보를 인증하세요.", "认证每个预测原点的实际发布和访问信息。", "各予測起点の実際の公開・アクセス情報を認証してください。"),
    "forecast_generalization": ("Run nested origin-wise validation and calibration.", "원점별 중첩 검증과 불확실성 보정을 수행하세요.", "执行按预测原点的嵌套验证和校准。", "起点ごとの入れ子検証と較正を実施してください。"),
    "external_replication": ("Obtain an independently installed and checked reproduction.", "독립 설치 및 검산을 거친 재현 증거를 확보하세요.", "取得独立安装并经核验的复现证据。", "独立したインストールと検算による再現証拠を取得してください。"),
    "all_reference_public_api_integration": ("Continue every eligible API with execution, rights, oracle and workflow evidence.", "수용 가능한 모든 API의 실행·권리·검산·연구 흐름 증거를 계속 확보하세요.", "持续取得所有合规API的执行、权利、核验和研究流程证据。", "利用可能な全APIの実行・権利・検算・研究手順の証拠を継続して取得してください。"),
}


def _code_receipt():
    hashes = {}
    for name in _CODE_FILES:
        path = Path(__file__).with_name(name)
        if path.is_symlink() or not path.is_file():
            raise ScopeError("INVESTIGATION_CODE_UNAVAILABLE")
        hashes[name] = digest(path.read_bytes())
    return hashes


def build_inverse_investigation(request, result, code_hashes):
    """Called only after a successful validation/calculation by inverse_screen."""
    subject = result["input_sha256"]
    study = request["study"]
    scope = dict(input_sha256=subject, data_kind=study["data_kind"],
                 source_declared_sha256=study["source_declared_sha256"],
                 series_units=study["series_units"], target=study["target"],
                 training_end_day=study["training_end_day"], search=request["search"],
                 time_policy="DECLARED_RELEASES_RETROSPECTIVE_NOT_AUTHENTICATED_AS_OF")
    producer = dict(tool_id="inverse.screen", version="1", code_sha256=digest(canonical(code_hashes)))
    declared = make_evidence(subject_sha256=subject, producer=producer,
        operation="RESEARCHER_INPUT_DECLARATION", authority="DECLARED_ONLY", epistemic_state="OBSERVED",
        proof_kind="INPUT_DECLARATION", scope=scope,
        coverage=dict(status="PARTIAL", unknown_dimensions=["source_authentication", "release_authentication"]),
        result=dict(request_sha256=subject, source_declared_sha256=study["source_declared_sha256"],
                    processing_rights=request["rights"], observation_count=len(study["observations"])),
        parent_evidence_ids=[], limitations=["Observed declaration, not authenticated economic observations or rights"])
    calculated = make_evidence(subject_sha256=subject, producer=producer,
        operation="EXACT_LAGGED_SCREEN", authority="LOCAL_NUMERIC_EXECUTION", epistemic_state="DERIVED",
        proof_kind="EXPLORATORY_REGRESSION", scope=scope,
        coverage=dict(status="PARTIAL", unknown_dimensions=["independent_numeric_oracle", "causal_identification",
                       "latent_identity", "unsearched_hypotheses", "external_forecast_validity"]),
        result=result, parent_evidence_ids=[declared["evidence_id"]],
        limitations=["Train-only exploratory specification search; multiple comparisons uncorrected",
                     "No original upstream statistical API execution; same-process arithmetic oracle is not external replication",
                     "Holdout uses declared release dates; no authenticated historical as-of inference"])
    evidence = EvidenceLedger([declared, calculated])
    unknowns = UnknownRegistry(evidence)

    def unknown(question, authority, proof, probe, blocked=False):
        entry = unknowns.register(question, scope, authority, proof)
        return unknowns.revise(entry["unknown_id"], entry["revision_digest"],
                               status="BLOCKED" if blocked else "OPEN", recommended_probes=[probe])

    source = unknown("Are source data, local rights and release/access vintages authenticated?",
        "AUTHENTICATED_SOURCE_REVIEW", "SOURCE_VINTAGE_RIGHTS", "Obtain reviewed source, rights and release/access records", True)
    latent = unknown("Which competing structures or latent labels are distinguished by independent observations?",
        "EXTERNAL_IDENTIFICATION", "IDENTIFICATION", "Test equivalent models, scale/rotation restrictions and negative controls")
    forecast = unknown("Does a prespecified origin-wise backtest generalize beyond this exploratory sample?",
        "INDEPENDENT_FORECAST_VALIDATION", "FORECAST_VALIDATION", "Run nested origin-wise selection with authenticated vintages and calibration")
    replication = unknown("Can a separately installed environment reproduce the study and its numerical oracle?",
        "EXTERNAL_REPLICATION", "INDEPENDENT_REPLAY", "Obtain clean installed and independently checked reproduction")

    obligations = []
    def add(claim, authority, proof, verifier, *, questions=(), blocks=(), deps=(), next_probe):
        ob = make_obligation(claim_id=claim, subject_sha256=subject, scope=scope,
            required_authority=authority, required_proof_kind=proof, verifier_id=verifier,
            dependencies=list(deps), unknown_ids=[u["unknown_id"] for u in questions],
            blocking_reasons=list(blocks), next_probe=next_probe)
        obligations.append(ob)
        return ob

    performed = add("declared_calculation_executed", "LOCAL_NUMERIC_EXECUTION", "CALCULATION_COMPLETED",
                    "inverse.screen", next_probe="Inspect exact coefficients, rank failures and information-set limitations")
    # This proof asserts only that the reviewed deterministic function returned.
    # It does not certify the broader partial scientific coverage above.
    execution = make_evidence(subject_sha256=subject, producer=producer,
        operation="CALCULATION_RETURNED", authority="LOCAL_NUMERIC_EXECUTION", epistemic_state="DERIVED",
        proof_kind="CALCULATION_COMPLETED", scope=scope,
        coverage=dict(status="COMPLETE", unknown_dimensions=[]),
        result=dict(obligation_id=performed["obligation_id"], revision=1,
                    previous_verification_id=None, outcome="PASS"),
        parent_evidence_ids=[calculated["evidence_id"]],
        limitations=["Function completion only; rank deficiency is a valid inconclusive calculation result"])
    evidence.merge([execution])
    fit_ob = add("fitted_explanatory_model", "LOCAL_NUMERIC_EXECUTION", "MODEL_FIT_RETURNED", "inverse.screen",
        deps=[performed["obligation_id"]], next_probe="Review rank and estimated candidate coefficients")
    if result["selected_candidate_id"] is not None:
        evidence.merge([make_evidence(subject_sha256=subject, producer=producer,
            operation="MODEL_FIT_RETURNED", authority="LOCAL_NUMERIC_EXECUTION", epistemic_state="DERIVED",
            proof_kind="MODEL_FIT_RETURNED", scope=scope, coverage=dict(status="COMPLETE", unknown_dimensions=[]),
            result=dict(obligation_id=fit_ob["obligation_id"], revision=1, previous_verification_id=None, outcome="PASS"),
            parent_evidence_ids=[calculated["evidence_id"]],
            limitations=["A fitted exploratory model is not an identified cause or externally valid prediction"])])
    numeric_ob = add("independent_numeric_parity", "INDEPENDENT_NUMERIC_TEST", "NUMERIC_PARITY", "independent.numeric",
        next_probe="Compare exact estimator and boundary cases against a separate numerical method")
    if result["numeric_verification"]["status"] != "PASS":
        raise ScopeError("INVESTIGATION_NUMERIC_VERIFICATION_REQUIRED")
    evidence.merge([make_evidence(subject_sha256=subject,
        producer=dict(producer, tool_id="independent.numeric"), operation="CRAMER_NUMERIC_VERIFICATION",
        authority="INDEPENDENT_NUMERIC_TEST", epistemic_state="DERIVED", proof_kind="NUMERIC_PARITY",
        scope=scope, coverage=dict(status="COMPLETE", unknown_dimensions=[]),
        result=dict(obligation_id=numeric_ob["obligation_id"], revision=1,
                    previous_verification_id=None, outcome="PASS"),
        parent_evidence_ids=[calculated["evidence_id"]],
        limitations=["Separate Cramer/determinant arithmetic implementation in the same process; not external replication",
                     "Parity covers this declared bounded screen, not source truth, causal effects or all upstream APIs"])])
    add("source_and_rights_authentication", "AUTHENTICATED_SOURCE_REVIEW", "SOURCE_VINTAGE_RIGHTS", "source.review",
        questions=[source], blocks=["Source/rights/release declarations are not authenticated"],
        next_probe=source["recommended_probes"][0])
    add("latent_structure_identification", "EXTERNAL_IDENTIFICATION", "IDENTIFICATION", "identification.review",
        questions=[latent], next_probe=latent["recommended_probes"][0])
    add("causal_effect", "EXTERNAL_IDENTIFICATION", "CAUSAL_DESIGN", "causal.review",
        questions=[latent], blocks=["No causal design, intervention or identification proof supplied"],
        next_probe="Specify and challenge the causal identification strategy before claiming a policy effect")
    add("historical_as_of", "AUTHENTICATED_SOURCE_REVIEW", "HISTORICAL_INFORMATION_SET", "vintage.review",
        questions=[source], blocks=["Actual release/access information set is unverified"],
        next_probe=source["recommended_probes"][0])
    add("forecast_generalization", "INDEPENDENT_FORECAST_VALIDATION", "FORECAST_VALIDATION", "forecast.review",
        questions=[forecast], next_probe=forecast["recommended_probes"][0])
    add("external_replication", "EXTERNAL_REPLICATION", "INDEPENDENT_REPLAY", "replication.review",
        questions=[replication], next_probe=replication["recommended_probes"][0])
    add("all_reference_public_api_integration", "SEMANTIC_API_ACCEPTANCE", "ALL_PUBLIC_APIS", "coverage.review",
        blocks=["Full semantic public API denominator and per-API runtime/rights/UI/clean-install acceptance not established"],
        next_probe="Continue all eligible reference capabilities with API-specific execution, oracle and workflow receipts")
    bundle = dict(schema="societybox-inverse-investigation-v1", status="SCIENTIFIC_CLAIMS_INCOMPLETE",
        product_feature=describe_feature("research_evidence"), input_sha256=subject,
        code_sha256=producer["code_sha256"], code_files=code_hashes,
        full_reference_integration="NOT_ESTABLISHED", evidence=evidence.records(),
        unknowns=unknowns.records(), unknown_histories={u["unknown_id"]: unknowns.history(u["unknown_id"]) for u in unknowns.records()},
        obligations=obligations, coverage=evaluate_coverage(obligations, evidence, unknowns),
        assurance="CONTENT_BINDING_AND_LOCAL_EXECUTION_NOT_SOURCE_OR_VERIFIER_IDENTITY_AUTHENTICATION")
    bundle["investigation_sha256"] = digest(canonical(bundle))
    return bundle


def verify_inverse_screen(request, result):
    """Recompute all numbers AND the complete evidence projection, without writes."""
    from .inverse_screen import screen_inverse_hypotheses
    actual = screen_inverse_hypotheses(request)
    try:
        same = canonical(actual) == canonical(result)
    except (ValueError, TypeError, RecursionError, UnicodeError):
        same = False
    if not same:
        raise ScopeError("INVESTIGATION_RECOMPUTATION_MISMATCH")
    return dict(schema="societybox-inverse-recomputation-v1", status="RECOMPUTED_LOCALLY",
                input_sha256=actual["input_sha256"], result_sha256=digest(canonical(actual)),
                investigation_sha256=actual["investigation"]["investigation_sha256"],
                independent_replication="NOT_ESTABLISHED", source_authentication="NOT_ESTABLISHED")


def render_inverse_screen(request, result, locale="en", *, as_section=False):
    """Render only a freshly recomputed result; no scripts or external resources."""
    if locale not in _LABELS:
        raise ScopeError("INVESTIGATION_LOCALE_INVALID")
    verify_inverse_screen(request, result)
    labels = _LABELS[locale]
    title, calculation, evidence_title, unknown_title, obligations_title, next_title, loss_title, disclaimer = labels
    inv = result["investigation"]
    language = _LOCALES.index(locale)
    def text(value):
        return escape(str(value), quote=True)
    def formatted(value):
        return "<pre>" + text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)) + "</pre>"
    parts = ['<!doctype html><html lang="' + locale + '"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">',
        '<title>' + text(title) + '</title><style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#172b35;background:#f4f7f8}section,details{background:white;border:1px solid #cdd9de;border-radius:8px;padding:1rem;margin:1rem 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}code{overflow-wrap:anywhere}a{color:#075b82}h1,h2{line-height:1.3}</style></head><body>',
        '<h1>' + text(title) + '</h1><p>' + text(disclaimer) + '</p>',
        '<p>' + text(describe_feature("dynamic_inverse", locale)["name"]) + '</p>',
        '<p><code>' + text(inv["investigation_sha256"]) + '</code></p>',
        '<section><h2>' + text(calculation) + '</h2>' + formatted({
            "status": result["status"], "selected_candidate_id": result["selected_candidate_id"],
            "selected_features": result["selected_features"], "coefficients_exact": result["selected_coefficients_exact"],
            "candidates": result["candidates"], "residual_patterns": result["residual_pattern_candidates"]})]
    if result["holdout"] is not None:
        parts.append('<h3>' + text(loss_title) + '</h3>' + formatted(result["holdout"]))
    parts.append('</section><section><h2>' + text(evidence_title) + '</h2>')
    for rec in inv["evidence"]:
        parts.append('<details id="' + rec["evidence_id"] + '"><summary><code>' + rec["evidence_id"] + '</code> ' + text(rec["operation"]) + '</summary>')
        for parent in rec["parent_evidence_ids"]:
            parts.append('<p><a href="#' + parent + '">' + parent + '</a></p>')
        parts.append(formatted(rec) + '</details>')
    parts.append('</section><section><h2>' + text(unknown_title) + '</h2>')
    for unk in inv["unknowns"]:
        question = _QUESTION_NAMES[unk["required_proof_kind"]][language]
        parts.append('<details id="' + unk["unknown_id"] + '"><summary>' + text(unk["status"] + ': ' + question) + '</summary>' + formatted(inv["unknown_histories"][unk["unknown_id"]]) + '</details>')
    parts.append('</section><section><h2>' + text(obligations_title) + '</h2>')
    for row in inv["coverage"]["obligations"]:
        parts.append('<details><summary>' + text(row["state"] + ': ' + _CLAIM_NAMES[row["claim_id"]][language]) + '</summary>')
        if row["verification_evidence_id"]:
            key = row["verification_evidence_id"]
            parts.append('<a href="#' + key + '">' + key + '</a>')
        parts.append(formatted(row) + '<p>' + text(next_title) + ': ' + text(_NEXT_ACTIONS[row["claim_id"]][language]) + '</p></details>')
    parts.append('</section><p>SCIENTIFIC_CLAIMS_INCOMPLETE; full_reference_integration: NOT_ESTABLISHED</p></body></html>')
    document = "".join(parts)
    if as_section:
        # The frame is fixed renderer-owned markup; all record text was escaped.
        body = document.split('<body>', 1)[1].rsplit('</body>', 1)[0]
        return '<section class="inverse-evidence-review" lang="' + locale + '">' + body + '</section>'
    return document
