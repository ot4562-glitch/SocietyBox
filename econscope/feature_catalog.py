"""Society Box functional names; upstream trademarks remain factual provenance.

These names identify *our research workflows*, not a fork, endorsement,
redistribution license, or full compatibility with any upstream library.
Internal study schema/engine identifiers are stable for existing users.
"""
from types import MappingProxyType

_NAMES = {
    "fixed_effects_regression": ("weighted-fixed-effects", ("Weighted fixed-effects regression", "가중 고정효과 회귀", "加权固定效应回归", "加重固定効果回帰")),
    "vintage_comparison": ("revision-availability-comparison", ("Revision and availability comparison", "자료 개정·이용시점 비교", "数据修订与可用时点比较", "データ改定・利用時点比較")),
    "input_output": ("static-input-output", ("Static input-output scenario", "정적 산업연관 시나리오", "静态投入产出情景", "静学産業連関シナリオ")),
    "survey_replicate": ("declared-replicate-variance", ("Declared replicate-weight variance", "선언된 반복가중치 분산", "声明的重复权重方差", "申告反復ウェイトの分散")),
    "consumption": ("intertemporal-budget", ("Intertemporal budget analysis", "기간 간 소비·예산 분석", "跨期消费与预算分析", "異時点間の消費・予算分析")),
    "allocation": ("resource-allocation", ("Resource allocation analysis", "자원 배분·제약 분석", "资源配置与约束分析", "資源配分・制約分析")),
    "service_queue": ("queue-dynamics", ("Queue and waiting-time analysis", "대기행렬·대기시간 분석", "排队与等待时间分析", "待ち行列・待ち時間分析")),
    "service_staffing": ("staffing-comparison", ("Staffing and service-cost comparison", "인력배치·서비스 비용 비교", "人员配置与服务成本比较", "人員配置・サービス費用比較")),
    "supply_network": ("supply-connectivity", ("Supply-route and disruption analysis", "공급망 경로·중단 분석", "供应路径与中断分析", "供給経路・遮断分析")),
    "supply_capacity": ("supply-throughput", ("Supply throughput and bottleneck analysis", "공급망 처리용량·병목 분석", "供应吞吐量与瓶颈分析", "供給処理能力・ボトルネック分析")),
    "supply_cost": ("supply-cost-allocation", ("Minimum-cost supply allocation", "최소비용 공급·수요 배분", "最低成本供需分配", "最小費用の需給配分")),
    "mathematical_program": ("constraint-optimization", ("Constraint optimization", "제약조건 최적화", "约束优化", "制約付き最適化")),
    "integrality_relaxation": ("integer-relaxation-bound", ("Integer-relaxation bound comparison", "정수·연속완화 상하한 비교", "整数与连续松弛界限比较", "整数・連続緩和の上下界比較")),
    "ols_reconstruction": ("regression-reconstruction", ("Regression coefficient reconstruction", "회귀계수 역추정·재구성", "回归系数反推与重建", "回帰係数の逆推定・再構成")),
    "dynamic_inverse": ("lagged-hypothesis-comparison", ("Lagged inverse hypothesis comparison", "시차 역추정·가설 비교", "滞后逆向推断与假设比较", "ラグ逆推定・仮説比較")),
    "daily_latent": ("latent-pattern-screening", ("Latent-pattern candidate screening", "잠재패턴 후보 탐색", "潜在模式候选筛查", "潜在パターン候補探索")),
    "research_evidence": ("research-evidence-review", ("Research evidence and verification review", "연구 증거·검증 의무 검토", "研究证据与验证审查", "研究証拠・検証義務の確認")),
    "coefficient_identification": ("coefficient-equivalence-ranges", ("Prediction-equivalent coefficient ranges", "동일 예측 계수 범위·식별 검토", "相同预测的系数范围与识别审查", "同一予測の係数範囲・識別検討")),
    "forecast_validation": ("originwise-forecast-evaluation", ("Origin-wise forecast validation", "원점별 재추정·예측 검증", "逐预测原点重估与验证", "予測起点ごとの再推定・検証")),
    "regression_sensitivity": ("weighted-regression-covariance", ("Weighted regression and covariance sensitivity", "가중회귀·공분산 민감도", "加权回归与协方差敏感性", "加重回帰・共分散の感度")),
}
_LOCALES = ("en", "ko", "zh-Hans", "ja")
_FEATURES = MappingProxyType(_NAMES)


def describe_feature(family, locale="en"):
    """Display-only metadata, never input validation, execution or source clearance."""
    if locale not in _LOCALES:
        raise ValueError("FEATURE_LOCALE_INVALID")
    if family not in _FEATURES:
        raise ValueError("FEATURE_UNKNOWN")
    feature_id, labels = _FEATURES[family]
    return {"feature_id": feature_id, "name": labels[_LOCALES.index(locale)],
            "naming": "SOCIETY_BOX_FUNCTION_NOT_UPSTREAM_BRAND",
            "upstream_attribution": "SEE_TECHNICAL_RECEIPTS_AND_LICENSES"}


def feature_catalog(locale="en"):
    return {family: describe_feature(family, locale) for family in _FEATURES}
