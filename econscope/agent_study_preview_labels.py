"""Display text for separate read-only study inspection; JSON stays verbatim."""

STUDY_PREVIEW_TEXT = {
    'en': {
        'heading': 'Selected study: read-only review',
        'limits': 'This study is separate from the conversation. Its content is '
        'not sent to a model. Structural validation does not establish scientific '
        'validity, verified sources, data rights or execution readiness.',
        'workspace': 'Workspace data checks',
        'numeric': 'Numeric CSV readiness',
        'execution': 'Execution',
        'authorization': 'Execution authorization',
        'approval_limits': 'Workspace binding does not grant permission; inspect the separate '
        'study-review workflow before explicit execution approval.',
        'sha256': 'Specification SHA256',
        'evidence': 'Source declarations and explicit assumptions',
        'model': 'Researcher-defined model and limitations',
        'validation': 'Validation, referenced-data requirements and missing prerequisites',
        'approval': 'Separate execution approval required',
        'bound': 'Workspace-bound review: local consistency only',
        'bound_limits': 'Stored bytes and declared publication dates were checked. '
        'Parameter overrides in this bound context replace only declared parameter values; '
        'the model card above retains the original specification and compares declared '
        'and effective values with units and declared evidence IDs. Effective values '
        'are bound proposals, not executed results or verified evidence. Changed overrides '
        'require a new review digest and separate execution approval. '
        'Numeric CSV inspection is separately labeled above and checks initializers only; '
        'transitions, source authenticity and rights remain unverified. '
        'This local review digest is not authenticated institutional approval.',
    },
    'zh-Hans': {
        'heading': '所选研究：只读审查',
        'limits': '此研究与对话分开。其内容不会发送给模型。结构验证不能确立科学有效性、'
        '来源已验证、数据权利或执行就绪。',
        'workspace': '工作区数据检查',
        'numeric': '数值 CSV 就绪状态',
        'execution': '执行',
        'authorization': '执行授权',
        'approval_limits': '工作区绑定不授予权限；明确批准执行前，请检查单独的 study-review 流程。',
        'sha256': '研究规范 SHA256',
        'evidence': '来源声明与明确假设',
        'model': '研究者定义的模型与局限',
        'validation': '验证、引用数据要求与缺失的前提条件',
        'approval': '需要单独的执行批准',
        'bound': '工作区绑定审查：仅检查本地一致性',
        'bound_limits': '已检查存储的字节与声明的发布日期。此绑定上下文中的参数覆盖仅替换已声明的参数值；'
        '上方模型卡保留原始规范，并比较声明值与有效值及其单位和声明的证据 ID。'
        '有效值是绑定的提议，并非执行结果或已验证证据。更改覆盖值需要新的审查摘要和单独的执行批准。'
        '数值 CSV 检查在上方单独标注，仅检查初始值；状态转移、来源真实性和权利仍未经验证。'
        '此本地审查摘要并非经认证的机构批准。',
    },
    'ko': {
        'heading': '선택한 연구: 읽기 전용 검토',
        'limits': '이 연구는 대화와 분리되어 있으며 내용은 모델에 전송되지 않습니다. '
        '구조 검증은 과학적 타당성, 출처 검증, 데이터 권리 또는 실행 준비 상태를 입증하지 않습니다.',
        'workspace': '작업 공간 데이터 검사',
        'numeric': '수치 CSV 준비 상태',
        'execution': '실행',
        'authorization': '실행 권한',
        'approval_limits': '작업 공간 바인딩은 권한을 부여하지 않습니다. 명시적 실행 승인 전에 '
        '별도의 study-review 워크플로를 검토하세요.',
        'sha256': '연구 명세 SHA256',
        'evidence': '출처 선언 및 명시적 가정',
        'model': '연구자 정의 모델 및 한계',
        'validation': '검증, 참조 데이터 요구 사항 및 누락된 전제 조건',
        'approval': '별도의 실행 승인 필요',
        'bound': '작업 공간에 바인딩된 검토: 로컬 일관성만 확인',
        'bound_limits': '저장된 바이트와 선언된 공표 날짜를 확인했습니다. 이 바인딩된 맥락에서 '
        '매개변수 재정의는 선언된 매개변수 값만 대체합니다. 위 모델 카드는 원본 명세를 유지하고 '
        '선언된 값과 유효 값을 단위 및 선언된 증거 ID와 함께 비교합니다. 유효 값은 바인딩된 '
        '제안이며 실행 결과나 검증된 증거가 아닙니다. 재정의 변경에는 새 검토 다이제스트와 별도의 '
        '실행 승인이 필요합니다. 수치 CSV 검사는 위에 별도로 표시되며 초기값만 확인합니다. '
        '상태 전이, 출처의 진위 및 권리는 여전히 검증되지 않았습니다. '
        '이 로컬 검토 다이제스트는 인증된 기관 승인이 아닙니다.',
    },
    'ja': {
        'heading': '選択した研究：読み取り専用レビュー',
        'limits': 'この研究は会話と分離され、内容はモデルに送信されません。構造検証は、科学的妥当性、'
        '出典の検証、データの権利、実行準備の完了を立証しません。',
        'workspace': 'ワークスペースのデータ検査',
        'numeric': '数値 CSV の準備状況',
        'execution': '実行',
        'authorization': '実行権限',
        'approval_limits': 'ワークスペースとの結び付けは権限を付与しません。明示的な実行承認の前に、'
        '別途 study-review のワークフローを確認してください。',
        'sha256': '研究仕様 SHA256',
        'evidence': '出典の申告と明示的な仮定',
        'model': '研究者定義のモデルと限界',
        'validation': '検証、参照データの要件と不足する前提条件',
        'approval': '別途の実行承認が必要',
        'bound': 'ワークスペースに結び付けたレビュー：ローカルの整合性のみ',
        'bound_limits': '保存されたバイト列と申告された公表日を確認しました。この結び付けられたコンテキストでの'
        'パラメータ上書きは、申告済みのパラメータ値のみを置き換えます。上のモデルカードは元の仕様を保持し、'
        '申告値と有効値を単位および申告された証拠 ID とともに比較します。有効値は結び付けられた提案であり、'
        '実行結果や検証済みの証拠ではありません。上書きの変更には新しいレビューのダイジェストと別途の実行承認が必要です。'
        '数値 CSV の検査は上に別途表示され、初期値のみを確認します。状態遷移、出典の真正性、権利は未検証です。'
        'このローカルのレビューダイジェストは、認証された機関の承認ではありません。',
    },
}
