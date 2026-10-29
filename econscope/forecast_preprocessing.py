"""Exact predictor winsorization fitted only on an origin's training sample.

Quantiles use linear interpolation at (n-1)*p. Independent order-statistic
counts check their endpoints without reusing the sorted quantile selector.
This is independently authored arithmetic, not a native statistical API or
evidence of robust inference, causal validity or external forecast skill.
"""
from collections import Counter
from fractions import Fraction as F
from html import escape
import re

from .execution_control import checkpoint
from .mechanism_vintage import _date
from .workspace import canonical,digest

SCOPE='ORIGIN_AVAILABLE_COMMON_TRAINING_ROWS_ONLY'


def validate(plan):
    if (type(plan) is not dict or set(plan) != {'method','lower_probability','upper_probability','fit_scope'}
            or plan['method'] != 'WINSORIZE_TRAIN_QUANTILES' or plan['fit_scope'] != SCOPE):
        raise ValueError('FORECAST_PREPROCESSING_PLAN_INVALID')
    probabilities=[]
    for key in ('lower_probability','upper_probability'):
        value=plan[key]
        if (type(value) is not str or not 1<=len(value)<=64 or
                re.fullmatch(r'[0-9]+(?:/[0-9]+|\.[0-9]+)?',value,re.ASCII) is None):
            raise ValueError('FORECAST_PREPROCESSING_PROBABILITY_INVALID')
        try: probabilities.append(F(value))
        except (ValueError,ZeroDivisionError):
            raise ValueError('FORECAST_PREPROCESSING_PROBABILITY_INVALID') from None
    if not 0<=probabilities[0]<probabilities[1]<=1:
        raise ValueError('FORECAST_PREPROCESSING_PROBABILITY_ORDER_INVALID')
    return probabilities


def _quantile(values,p):
    ordered=sorted(values); position=(len(values)-1)*p
    lower=position.numerator//position.denominator; upper=min(lower+1,len(values)-1)
    return ordered[lower]+(position-lower)*(ordered[upper]-ordered[lower])


def _count_oracle(values,p):
    # Independent frequency-count oracle, not the raw-observation sort used
    # by _quantile. The old all-pairs <= scan became quadratic for each origin
    # and each predictor. Counts make it O(n log n) while preserving exact F.
    position=(len(values)-1)*p; rank=position.numerator//position.denominator
    frequencies=sorted(Counter(values).items())
    def order(index):
        cumulative=0
        for value,count in frequencies:
            cumulative+=count
            if cumulative>index:
                return value
        raise ValueError('FORECAST_PREPROCESSING_ORACLE_RANK_INVALID')
    left=order(rank); right=order(min(rank+1,len(values)-1))
    return (1-(position-rank))*left+(position-rank)*right


def transform(value,limits):
    if type(value) is not F:
        raise ValueError('FORECAST_PREPROCESSING_OBSERVED_RATIONAL_REQUIRED')
    return min(F(limits['upper_exact']),max(F(limits['lower_exact']),value))


def fit_predictors(columns,training_days,plan):
    lower,upper=validate(plan)
    if (type(columns) is not dict or not 1<=len(columns)<=24 or type(training_days) is not list
            or not 1<=len(training_days)<=128 or any(type(day) is not str for day in training_days)
            or sorted(set(training_days))!=training_days):
        raise ValueError('FORECAST_PREPROCESSING_TRAINING_SAMPLE_INVALID')
    for day in training_days: _date(day)
    features={}; processed={}
    for label,values in columns.items():
        checkpoint()
        if (type(label) is not str or not 1<=len(label)<=128 or type(values) is not list
                or len(values)!=len(training_days) or any(type(value) is not F for value in values)):
            raise ValueError('FORECAST_PREPROCESSING_TRAINING_COLUMN_INVALID')
        lo,hi=_quantile(values,lower),_quantile(values,upper)
        if lo!=_count_oracle(values,lower) or hi!=_count_oracle(values,upper):
            raise ValueError('FORECAST_PREPROCESSING_QUANTILE_ORACLE_MISMATCH')
        limits=dict(lower_exact=str(lo),upper_exact=str(hi),
                    clipped_training_values=sum(value<lo or value>hi for value in values))
        features[label]=limits
        processed[label]=[str(transform(value,limits)) for value in values]
    return dict(status='FITTED',method=plan['method'],fit_scope=SCOPE,
        lower_probability=str(lower),upper_probability=str(upper),quantile_definition='LINEAR_INTERPOLATION_AT_(N_MINUS_1)_P',
        quantile_oracle='INDEPENDENT_ORDER_STATISTIC_COUNTS',
        training_observations=len(training_days),training_days_sha256=digest(canonical(training_days)),
        raw_training_predictors_sha256=digest(canonical({label:list(map(str,values)) for label,values in columns.items()})),
        transformed_training_predictors_sha256=digest(canonical(processed)),features=features,
        target='UNTRANSFORMED',baseline='UNTRANSFORMED',
        coefficient_interpretation='DECLARED_WINSORIZED_PREDICTORS_NOT_RAW_TAIL_SLOPES_OR_CAUSAL_EFFECTS')


def render_review_section(result,locale):
    words={
        'en':('Training-only predictor preprocessing','Origin / horizon','Predictor','Lower bound','Upper bound','Training rows','Clipped training values',
              'Quantile limits are fitted on each origin’s common available training rows only. Targets and the baseline remain untransformed. Winsorization can change model selection and predictions; no robustness or causal guarantee is implied.'),
        'ko':('학습 표본 전용 설명변수 전처리','원점 / 예측기간','설명변수','하위 경계','상위 경계','학습 행 수','잘린 학습 값 수',
              '각 원점의 공통 이용 가능 학습 행으로만 분위수 경계를 추정합니다. 목표 값과 기준모형 값은 변환하지 않습니다. 윈저화는 후보 선택과 예측을 바꿀 수 있으며 강건성·인과 타당성을 보증하지 않습니다.'),
        'zh-Hans':('仅训练样本的预测变量预处理','原点 / 预测期','预测变量','下界','上界','训练行数','截尾训练值数',
              '分位数边界仅使用各原点共同可用的训练行估计。目标值与基准不变换。缩尾处理可能改变模型选择和预测，不保证稳健性或因果有效性。'),
        'ja':('訓練標本のみの説明変数前処理','起点 / 予測期間','説明変数','下限','上限','訓練行数','制限された訓練値数',
              '分位点の境界は各起点で共通して利用可能な訓練行のみから推定します。目的変数と基準値は変換しません。ウィンザー化は候補選択と予測を変え得ますが、頑健性や因果妥当性を保証しません。'),
    }[locale]
    rows=[]
    for cell in result['cells']:
        fitted=cell.get('preprocessing')
        if not fitted: continue
        for label,limits in fitted['features'].items():
            values=(cell['origin_day']+' / '+str(cell['horizon']),label,limits['lower_exact'],limits['upper_exact'],
                    fitted['training_observations'],limits['clipped_training_values'])
            rows.append('<tr>'+''.join('<td>'+escape(str(value))+'</td>' for value in values)+'</tr>')
    return ('<section class="forecast-preprocessing"><h3>'+escape(words[0])+'</h3><p>'+escape(words[7])+'</p>'
            +'<div class="table-scroll"><table><thead><tr>'+''.join('<th scope="col">'+escape(word)+'</th>' for word in words[1:7])
            +'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div></section>')
