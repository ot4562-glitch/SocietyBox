"""Authored display text for inert host-turn reviews; never model instructions.

Machine status values and untrusted payloads are deliberately not translated.
"""

_KEYS = (
    'title', 'heading', 'provider', 'binding', 'request', 'assistant', 'proposals',
    'empty', 'tool', 'arguments', 'limits', 'limits_text', 'result_sha256',
    'request_sha256', 'raw_response_sha256', 'consistency', 'assurance',
    'model_consumption', 'execution', 'authorization', 'report_evidence',
    'trust', 'claim_verification', 'id', 'status',
)

_TEXT = {
    'en': (
        'Untrusted host turn review', 'Offline review preview',
        'Provider: NONE; host_injection: NOT_IMPLEMENTED. '
        'This local record preview does not establish a connected model or model consumption. '
        'Offline fixtures can produce the same records.',
        'Review status and binding', 'Researcher request', 'Untrusted assistant text',
        'Blocked tool proposals', 'No tool proposals.', 'Proposed tool',
        'Untrusted arguments (JSON text)', 'Review limits',
        'Consistency checks do not verify scientific claims or model origin. '
        'Model-authored approval and tool arguments grant no authority. '
        'Study execution requires separate product review and explicit operator approval; '
        'reports require checked computation and source or assumption lineage.',
        'Result SHA256', 'Request SHA256', 'Raw response SHA256', 'Consistency',
        'Assurance', 'Model consumption', 'Execution', 'Authorization',
        'Report evidence eligibility', 'Trust', 'Claim verification', 'Proposal ID', 'Status',
    ),
    'zh-Hans': (
        '不可信主机轮次审查', '离线审查预览',
        '提供方：NONE；host_injection: NOT_IMPLEMENTED。此本地记录预览不能证明模型已连接或已接收输入。'
        '离线测试样例也能生成相同记录。',
        '审查状态与绑定', '研究者请求', '不可信助手文本', '已阻止的工具提议',
        '无工具提议。', '提议的工具', '不可信参数（JSON 文本）', '审查局限',
        '一致性检查不能验证科学主张或模型来源。模型生成的批准和工具参数不授予任何权限。'
        '研究执行需要单独的产品审查及操作员明确批准；报告需要经过核查的计算及来源或假设溯源。',
        '结果 SHA256', '请求 SHA256', '原始响应 SHA256', '一致性', '保证程度',
        '模型输入接收', '执行', '授权', '报告证据资格', '可信度', '主张验证', '提议标识', '状态',
    ),
    'ko': (
        '신뢰할 수 없는 호스트 대화 검토', '오프라인 검토 미리보기',
        '제공자: NONE; host_injection: NOT_IMPLEMENTED. 이 로컬 기록 미리보기는 모델 연결이나 '
        '모델의 입력 수신을 입증하지 않습니다. 오프라인 테스트 예제로도 같은 기록을 만들 수 있습니다.',
        '검토 상태 및 바인딩', '연구자 요청', '신뢰할 수 없는 어시스턴트 텍스트',
        '차단된 도구 제안', '도구 제안 없음.', '제안된 도구', '신뢰할 수 없는 인수(JSON 텍스트)',
        '검토 한계',
        '일관성 검사는 과학적 주장이나 모델 출처를 검증하지 않습니다. 모델이 작성한 승인과 도구 인수는 '
        '권한을 부여하지 않습니다. 연구 실행에는 별도 제품 검토와 운영자의 명시적 승인이 필요하며, '
        '보고서에는 검증된 계산 및 출처 또는 가정의 계보가 필요합니다.',
        '결과 SHA256', '요청 SHA256', '원본 응답 SHA256', '일관성', '보증 수준',
        '모델 입력 수신', '실행', '권한', '보고서 증거 적격성', '신뢰도', '주장 검증', '제안 ID', '상태',
    ),
    'ja': (
        '信頼できないホストターンのレビュー', 'オフラインレビューのプレビュー',
        'プロバイダー: NONE; host_injection: NOT_IMPLEMENTED。このローカル記録のプレビューは、'
        'モデルの接続や入力の受信を証明しません。オフラインのテスト例でも同じ記録を生成できます。',
        'レビュー状態とバインディング', '研究者のリクエスト', '信頼できないアシスタントのテキスト',
        'ブロックされたツール提案', 'ツール提案なし。', '提案されたツール',
        '信頼できない引数（JSON テキスト）', 'レビューの限界',
        '整合性チェックは科学的主張やモデルの出所を検証しません。モデルが作成した承認やツール引数は'
        '権限を付与しません。研究の実行には別途の製品レビューと操作者の明示的な承認が必要です。'
        '報告書には検証済みの計算と、出典または仮定の来歴が必要です。',
        '結果 SHA256', 'リクエスト SHA256', '生の応答 SHA256', '整合性', '保証水準',
        'モデルの入力受信', '実行', '権限', '報告書の証拠適格性', '信頼性', '主張の検証', '提案 ID', '状態',
    ),
}

REVIEW_LABELS = {locale: dict(zip(_KEYS, values, strict=True))
                 for locale, values in _TEXT.items()}

_BINDING_TEXT = {
    'en': ('Context check scope', 'Selected research inputs', 'Response/request binding',
           'Current context means the product host contract only. Selected studies, workspace data, '
           'models, runs and comparisons are not bound to this turn; their changes are not detected '
           'by this check. Pairing local request and response bytes does not prove the response '
           'was generated for that request.'),
    'zh-Hans': ('上下文检查范围', '所选研究输入', '响应与请求的绑定',
                '当前上下文仅指产品主机契约。所选研究、工作区数据、模型、运行和比较未绑定到此轮对话；此检查不检测其变化。'
                '配对本地请求与响应字节不能证明响应是针对该请求生成的。'),
    'ko': ('문맥 검사 범위', '선택한 연구 입력', '응답과 요청의 바인딩',
           '현재 문맥은 제품 호스트 계약만 뜻합니다. 선택한 연구, 작업공간 자료, 모형, 실행 및 비교는 이 대화에 바인딩되지 않으며 '
           '이 검사는 해당 변경을 감지하지 않습니다. 로컬 요청과 응답 바이트를 짝지어도 해당 요청에 대해 생성된 응답임을 입증하지 않습니다.'),
    'ja': ('文脈チェックの範囲', '選択した研究入力', '応答と要求の結び付け',
           '現在の文脈は製品ホスト契約のみを意味します。選択した研究、ワークスペースのデータ、モデル、実行、比較はこのターンに'
           '結び付けられておらず、このチェックはその変更を検出しません。ローカルの要求と応答のバイト列を組み合わせても、'
           'その要求に対して生成された応答であることは証明できません。'),
}

for _locale, (_scope, _inputs, _pairing, _limits) in _BINDING_TEXT.items():
    REVIEW_LABELS[_locale].update(context_scope=_scope, selected_research_inputs=_inputs,
                                 response_request_binding=_pairing)
    REVIEW_LABELS[_locale]['limits_text'] += ' ' + _limits
