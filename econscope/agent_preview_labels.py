"""Authored conversation examples and disclosures, never imported model text."""

PREVIEW_TEXT = {
    'en': {
        'heading': 'Research conversation preview',
        'request': 'What evidence, model assumptions and execution prerequisites must I review before a study run?',
        'assistant': 'OFFLINE FIXTURE: this text was authored for interface inspection; no LLM answered. '
        'Inspect sources, assumptions and the model with study-review. '
        'Study execution requires its separate supported workflow and explicit approval. '
        'The example proposal below is blocked and grants no permission.',
        'fixture': 'Fixed example conversation; no researcher study content enters this conversation. '
        'No live chat, LLM inference, study execution or approval occurs. '
        'Use study-review separately to inspect your own study. '
        'Do not use this fixture as report evidence.',
        'import': 'Locally supplied request and response bytes, paired for inspection only. '
        'Origin: UNAUTHENTICATED_LOCAL_RECORD. No proof that a model generated '
        'this response for this request. Exact response bytes are bound by SHA256; '
        'hash consistency does not authenticate origin or claims. '
        'No live chat, LLM inference, study execution or approval occurs. '
        'Study content is reviewed separately and is not added to the host request. '
        'Do not use this import as report evidence.',
    },
    'zh-Hans': {
        'heading': '研究对话预览',
        'request': '运行研究前，我必须审查哪些证据、模型假设和执行前提条件？',
        'assistant': 'OFFLINE FIXTURE: 此文本为界面检查而编写；没有 LLM 作答。'
        '请使用 study-review 检查来源、假设和模型。研究执行需要单独的受支持流程和明确批准。'
        '下方示例提议已被阻止，不授予任何权限。',
        'fixture': '固定示例对话；研究者的研究内容不会进入此对话。'
        '不会进行实时聊天、LLM 推理、研究执行或批准。'
        '请单独使用 study-review 检查自己的研究。不得将此样例用作报告证据。',
        'import': '本地提供的请求与响应字节仅配对用于检查。'
        '来源：UNAUTHENTICATED_LOCAL_RECORD。没有证据证明模型针对此请求生成了此响应。'
        '原始响应字节通过 SHA256 绑定；哈希一致性不能认证来源或主张。'
        '不会进行实时聊天、LLM 推理、研究执行或批准。'
        '研究内容单独审查，不会添加到主机请求中。不得将此导入记录用作报告证据。',
    },
    'ko': {
        'heading': '연구 대화 미리보기',
        'request': '연구 실행 전에 어떤 증거, 모델 가정 및 실행 전제 조건을 검토해야 합니까?',
        'assistant': 'OFFLINE FIXTURE: 이 텍스트는 인터페이스 점검용으로 작성되었으며 LLM이 응답하지 않았습니다. '
        'study-review로 출처, 가정 및 모델을 검토하세요. 연구 실행에는 별도의 지원 워크플로와 명시적 승인이 필요합니다. '
        '아래 예시 제안은 차단되었으며 어떠한 권한도 부여하지 않습니다.',
        'fixture': '고정된 예시 대화이며 연구자의 연구 내용은 이 대화에 포함되지 않습니다. '
        '실시간 채팅, LLM 추론, 연구 실행 또는 승인은 이루어지지 않습니다. '
        '자신의 연구는 study-review로 별도로 검토하세요. 이 예시를 보고서 증거로 사용하지 마세요.',
        'import': '로컬에서 제공된 요청과 응답 바이트는 점검용으로만 연결됩니다. '
        '출처: UNAUTHENTICATED_LOCAL_RECORD. 모델이 이 요청에 대해 이 응답을 생성했다는 증거는 없습니다. '
        '원본 응답 바이트는 SHA256으로 바인딩되며 해시 일치가 출처나 주장을 인증하지는 않습니다. '
        '실시간 채팅, LLM 추론, 연구 실행 또는 승인은 이루어지지 않습니다. '
        '연구 내용은 별도로 검토되며 호스트 요청에 추가되지 않습니다. 이 가져오기 기록을 보고서 증거로 사용하지 마세요.',
    },
    'ja': {
        'heading': '研究会話のプレビュー',
        'request': '研究の実行前に、どの証拠、モデルの仮定、実行の前提条件を確認する必要がありますか？',
        'assistant': 'OFFLINE FIXTURE: このテキストは画面の点検用に作成されたもので、LLM は応答していません。'
        'study-review で出典、仮定、モデルを確認してください。研究の実行には別途の対応ワークフローと明示的な承認が必要です。'
        '以下の例の提案はブロックされており、権限を付与しません。',
        'fixture': '固定された会話例です。研究者の研究内容はこの会話に含まれません。'
        'ライブチャット、LLM 推論、研究の実行や承認は行われません。'
        'ご自身の研究は study-review で別途確認してください。この例を報告書の証拠に使用しないでください。',
        'import': 'ローカルで提供されたリクエストと応答のバイト列を、点検用にのみ組み合わせています。'
        '出所: UNAUTHENTICATED_LOCAL_RECORD。このリクエストに対してモデルがこの応答を生成した証拠はありません。'
        '元の応答バイト列は SHA256 で結び付けられますが、ハッシュの一致は出所や主張を認証しません。'
        'ライブチャット、LLM 推論、研究の実行や承認は行われません。'
        '研究内容は別途レビューされ、ホストへのリクエストには追加されません。'
        'このインポート記録を報告書の証拠に使用しないでください。',
    },
}
