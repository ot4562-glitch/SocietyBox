"""Dataset inspection from one existing study review; no reads or execution."""
from html import escape
import json


TEXT = {
    'en': ('Dataset evidence inspector', 'Dataset ID / content SHA256', 'Evidence references',
           'Preserved name', 'Bytes', 'Workspace checks', 'Numeric initializer',
           'Missing condition / outcome', 'Column', 'Aggregation',
           'Snapshot only. Source authenticity, scientific suitability and institutional rights: NOT_ESTABLISHED. Approval: NOT_GRANTED. No dataset cells are displayed.',
           'Review this study with a valid workspace to bind and check the declared dataset bytes.',
           'Workspace review failed for the study; no per-dataset byte verification is available. Resolve this condition and review again:',
           'Correct the declared column, input format or initializer bounds indicated by this condition and review again.',
           'Evidence-only reference; numeric initialization: NOT_APPLICABLE.'),
    'zh-Hans': ('数据集证据检查', '数据集ID / 内容SHA256', '证据引用', '保存的名称', '字节', '工作区检查',
                '数值初始值', '缺失条件 / 结果', '列', '聚合',
                '仅快照。来源真实性、科学适用性和机构权利：NOT_ESTABLISHED。批准：NOT_GRANTED。不显示数据单元格。',
                '请在有效工作区中审查此研究，以绑定并检查声明的数据集字节。',
                '研究的工作区审查失败；无法提供逐数据集字节验证。解决此条件后重新审查：',
                '请根据此条件修正声明的列、输入格式或初始值边界，然后重新审查。',
                '仅证据引用；数值初始化：NOT_APPLICABLE。'),
    'ko': ('데이터셋 근거 검사', '데이터셋 ID / 내용 SHA256', '근거 참조', '보존된 이름', '바이트', '작업공간 검사',
           '수치 초기값', '누락 조건 / 결과', '열', '집계',
           '스냅샷 전용입니다. 출처 진위, 과학적 적합성 및 기관 권리: NOT_ESTABLISHED. 승인: NOT_GRANTED. 데이터 셀은 표시하지 않습니다.',
           '유효한 작업공간에서 연구를 검토하여 선언된 데이터셋 바이트를 결합하고 검사하세요.',
           '연구의 작업공간 검토가 실패했습니다. 데이터셋별 바이트 검증은 제공되지 않습니다. 다음 조건을 해결하고 다시 검토하세요:',
           '이 조건에 따라 선언된 열, 입력 형식 또는 초기값 경계를 수정하고 다시 검토하세요.',
           '근거 전용 참조입니다. 수치 초기화: NOT_APPLICABLE.'),
    'ja': ('データセット根拠の検査', 'データセットID / 内容SHA256', '根拠参照', '保存済みの名前', 'バイト', 'ワークスペース検査',
           '数値初期値', '不足条件 / 結果', '列', '集計',
           'スナップショットのみ。出所の真正性、科学的適合性と機関の権利：NOT_ESTABLISHED。承認：NOT_GRANTED。データセルは表示しません。',
           '有効なワークスペースで研究をレビューし、宣言されたデータセットのバイトを結合して検査してください。',
           '研究のワークスペースレビューが失敗しました。データセット別のバイト検証は利用できません。この条件を解決して再確認してください：',
           'この条件に従って宣言された列、入力形式または初期値の境界を修正し、再確認してください。',
           '根拠のみの参照。数値初期化：NOT_APPLICABLE。'),
}

DECLARATION_TEXT = {
    'en': ('Researcher source declarations', 'Description', 'Locator (declared text)',
           'Processing rights', 'Source export conditions', 'Study export conditions',
           'Bound inventory declarations',
           'Declarations only; missing rights are NOT_DECLARED, not permission. Export conditions are not evaluated here. Institutional authorization: NOT_ESTABLISHED. Source verification: NOT_PERFORMED.'),
    'zh-Hans': ('研究者来源声明', '描述', '定位信息（声明文本）', '处理权利', '来源导出条件', '研究导出条件',
                '绑定清单声明', '仅为声明；缺失权利为 NOT_DECLARED，不代表许可。此处不评估导出条件。机构授权：NOT_ESTABLISHED。来源验证：NOT_PERFORMED。'),
    'ko': ('연구자 출처 선언', '설명', '위치 정보 (선언된 텍스트)', '처리 권리', '출처 내보내기 조건', '연구 내보내기 조건',
           '결합된 목록 선언', '선언 전용입니다. 누락된 권리는 NOT_DECLARED이며 허가가 아닙니다. 여기서는 내보내기 조건을 평가하지 않습니다. 기관 승인: NOT_ESTABLISHED. 출처 검증: NOT_PERFORMED.'),
    'ja': ('研究者の出所宣言', '説明', '所在情報（宣言テキスト）', '処理権利', '出所のエクスポート条件', '研究のエクスポート条件',
           '結合済み一覧の宣言', '宣言のみ。欠落した権利は NOT_DECLARED であり、許可ではありません。ここではエクスポート条件を評価しません。機関の承認：NOT_ESTABLISHED。出所検証：NOT_PERFORMED。'),
}


def dataset_section(review, locale):
    """Render declarations and checked identities without inventing partial checks.

    A failed workspace binding has no trustworthy per-object snapshot. Its error
    is study-wide, never attributed to each declared dataset as a local failure.
    Numeric outcomes join by state only after a successful binding.
    """
    labels = TEXT[locale]
    validation = next(c for c in review['cards'] if c['kind'] == 'validation')
    datasets = validation.get('data_requirements', {}).get('datasets', [])
    if not datasets:
        return ''
    bound = review.get('execution_review')
    declarations = DECLARATION_TEXT[locale]
    evidence = next(c['items'] for c in review['cards'] if c['kind'] == 'evidence')
    model = next(c for c in review['cards'] if c['kind'] == 'model')
    checks = {c['state']: c for c in validation['numeric_csv_readiness']['initializers']}

    def field(label, value):
        return '<dt>' + escape(label) + '</dt><dd>' + escape(str(value)) + '</dd>'

    def declaration(label, value):
        return field(label, json.dumps(value, ensure_ascii=False, sort_keys=True,
                                      allow_nan=False) if value is not None else 'NOT_DECLARED')

    parts = ['<section class="dataset-inspector"><h3>' + escape(labels[0])
             + '</h3><p>' + escape(labels[10]) + '</p>']
    for dataset in datasets:
        key = dataset['data_id']
        parts.append('<details class="dataset-evidence"><summary>' + escape(labels[1])
                     + ': <code>' + escape(key) + '</code></summary><dl>')
        parts.append(field(labels[2], ', '.join(dataset['evidence_refs'])))
        parts.append('</dl><section class="dataset-declarations"><h4>'
                     + escape(declarations[0]) + '</h4><p>' + escape(declarations[7]) + '</p>')
        for ref in dataset['evidence_refs']:
            source = evidence[ref]
            parts.extend(['<h5>' + escape(ref) + ': ' + escape(source['type']) + '</h5><dl>',
                          field(declarations[1], source['description']),
                          field(declarations[2], source.get('locator', 'NOT_DECLARED')),
                          declaration(declarations[3], source.get('processing_rights')),
                          declaration(declarations[4], source.get('export_rights')), '</dl>'])
        parts.extend(['<dl>', declaration(declarations[5], model['export_rights']), '</dl></section><dl>'])
        if bound is not None:
            item = bound['context']['inventory']['objects'][key]
            parts.extend([field(labels[3], item['name']), field(labels[4], item['size']),
                          field(labels[5], bound['data_checks'])])
            parts.append(declaration(declarations[6], {key: item[key] for key in (
                'publication_date_claim', 'publication_verified', 'license_label',
                'execution_permission', 'llm_upload_permission') if key in item}))
        else:
            parts.append(field(labels[5], 'NOT_VERIFIED'))
            condition = validation.get('workspace_failure')
            parts.append(field(labels[7], labels[12] + ' ' + condition if condition else labels[11]))
        parts.append('</dl>')
        for initializer in dataset['numeric_initializers']:
            check = checks.get(initializer['state'], {'status': 'NOT_RUN'})
            parts.extend(['<section><h4>' + escape(labels[6]) + ': '
                          + escape(initializer['state']) + '</h4><dl>',
                          field(labels[8], initializer['column']),
                          field(labels[9], initializer['aggregate']),
                          field(labels[7], check.get('reason', check['status'])), '</dl>'])
            if check['status'] == 'FAILED':
                parts.append('<p>' + escape(labels[13]) + '</p>')
            parts.append('</section>')
        if not dataset['numeric_initializers']:
            parts.append('<p>' + escape(labels[14]) + '</p>')
        parts.append('</details>')
    parts.append('</section>')
    return ''.join(parts)
