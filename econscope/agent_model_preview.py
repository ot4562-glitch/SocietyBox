"""Escaped read-only .mod review and opt-in preserved source; no host transfer."""
from html import escape
import difflib
import json

from .model_intake import review_mod, review_stored_mod, inspect_stored_mod_source
from .workspace import ScopeError


_SOURCE_LABELS = {
    'en': ('Preserved source text (untrusted)',
           'Full UTF-8 text, at most 1 MiB of original bytes. A leading UTF-8 BOM is omitted from display; SHA256 covers the original bytes. Display only; directives are not interpreted.'),
    'zh-Hans': ('已保存的源文本（不可信）',
                '完整UTF-8文本，原始字节最多1 MiB。显示时省略开头的UTF-8 BOM；SHA256涵盖原始字节。仅供显示；不解释指令。'),
    'ko': ('보존된 원문 (신뢰할 수 없음)',
           '원본 바이트가 최대 1 MiB인 전체 UTF-8 텍스트입니다. 표시할 때 선행 UTF-8 BOM을 생략하며 SHA256은 원본 바이트를 포함합니다. 표시 전용이며 지시문을 해석하지 않습니다.'),
    'ja': ('保存されたソース本文（信頼できない入力）',
           '元のバイトが最大1 MiBのUTF-8全文です。表示時は先頭のUTF-8 BOMを省略します。SHA256は元のバイト全体を対象とします。表示専用であり、指示文は解釈しません。'),
}


_KEYS = ('relative_path', 'bytes', 'sha256', 'encoding', 'schema', 'status',
         'syntax', 'includes_and_macros', 'source_rights', 'approval',
         'native_adapter', 'execution', 'native_reference_parity', 'os_isolation',
         'execution_status', 'blockers')
_STORED_LABELS = {
    'en': ('Artifact ID', 'Storage', 'Stored execution permission', 'Stored LLM upload permission'),
    'zh-Hans': ('工件ID', '存储', '已存执行权限', '已存LLM上传权限'),
    'ko': ('아티팩트 ID', '저장소', '저장된 실행 권한', '저장된 LLM 업로드 권한'),
    'ja': ('アーティファクトID', '保存先', '保存された実行権限', '保存されたLLMアップロード権限'),
}
_LABELS = {
    'en': ('Relative path', 'Byte count', 'SHA256', 'Encoding', 'Schema', 'Byte review',
           'Syntax', 'Includes and macros', 'Source rights', 'Approval',
           'Native adapter', 'Execution', 'Native reference parity', 'OS isolation',
           'Execution readiness', 'Execution blockers'),
    'zh-Hans': ('相对路径', '字节数', 'SHA256', '编码', '模式版本', '字节审查',
                '语法', '包含文件与宏', '来源权利', '批准', '原生适配器', '执行',
                '原生参考一致性', '操作系统隔离', '执行就绪状态', '执行阻碍'),
    'ko': ('상대 경로', '바이트 수', 'SHA256', '인코딩', '스키마', '바이트 검토',
           '문법', '포함 파일 및 매크로', '출처 권리', '승인', '네이티브 어댑터',
           '실행', '네이티브 참조 동등성', '운영체제 격리', '실행 준비 상태', '실행 차단 사유'),
    'ja': ('相対パス', 'バイト数', 'SHA256', '文字コード', 'スキーマ', 'バイト確認',
           '構文', 'インクルードとマクロ', '出典の権利', '承認', 'ネイティブアダプター',
           '実行', 'ネイティブ参照との一致', 'OS隔離', '実行準備状態', '実行の阻害要因'),
}
_TEXT = {
    'en': ('Selected researcher model: byte review only',
           'Unsigned snapshot of selected bytes; no syntax, safety or scientific validity is established. '
           'Source text, includes and macros are not executed or resolved. '
           'No model content is added to the host request. No execution, approval, LLM tool or transfer authority is granted. '
           'Filesystem checks are not OS isolation against hostile concurrent changes. '
           'Later execution must independently bind these bytes and dependencies in proven OS isolation.'),
    'zh-Hans': ('所选研究者模型：仅审查字节',
                '所选字节的未签名快照；不证明语法、安全性或科学有效性。源文本、包含文件和宏均未执行或解析。'
                '模型内容不会加入主机请求。不授予执行、批准、LLM工具或传输权限。'
                '文件系统检查不等于抵御恶意并发修改的操作系统隔离。后续执行必须在经证明的操作系统隔离中独立绑定这些字节及依赖。'),
    'ko': ('선택한 연구자 모델: 바이트 검토만',
           '선택한 바이트의 서명 없는 스냅샷이며 문법, 안전성 또는 과학적 타당성을 입증하지 않습니다. '
           '원문, 포함 파일 및 매크로를 실행하거나 해석하지 않습니다. 모델 내용은 호스트 요청에 추가되지 않습니다. '
           '실행, 승인, LLM 도구 또는 전송 권한을 부여하지 않습니다. 파일시스템 검사는 악의적 동시 변경에 대한 운영체제 격리가 아닙니다. '
           '향후 실행은 입증된 운영체제 격리에서 해당 바이트와 종속성을 독립적으로 바인딩해야 합니다.'),
    'ja': ('選択した研究者モデル：バイト確認のみ',
           '選択したバイトの署名のないスナップショットであり、構文、安全性、科学的妥当性を証明しません。'
           'ソース本文、インクルード、マクロは実行も解決もしていません。モデル内容はホスト要求に追加されません。'
           '実行、承認、LLMツール、転送の権限は付与されません。ファイルシステムの確認は悪意ある同時変更に対するOS隔離ではありません。'
           '後の実行では、実証されたOS隔離の下でこれらのバイトと依存関係を独立して結び付ける必要があります。'),
}


def selected_model_section(directory, relative_path, expected_sha256, locale, *,
                           workspace=None, artifact_id=None, show_source=False):
    """Build the card from a fresh intake, never a caller-supplied receipt."""
    if type(show_source) is not bool or (show_source and (artifact_id is None or workspace is None)):
        raise ScopeError('CONVERSATION_PREVIEW_MODEL_SOURCE_SELECTION_INVALID')
    source_text = None
    if show_source:
        receipt, source_text = inspect_stored_mod_source(workspace, artifact_id, expected_sha256)
    else:
        receipt = (review_stored_mod(workspace, artifact_id, expected_sha256)
                   if artifact_id is not None else
                   review_mod(directory, relative_path, expected_sha256))
    return _receipt_section(receipt, locale, source_text)


def _receipt_section(receipt, locale, source_text=None):
    labels = dict(zip(_KEYS, _LABELS[locale]))
    if 'artifact_id' in receipt['artifact']:
        labels['relative_path'] = {
            'en': 'Stored filename (unverified metadata)',
            'zh-Hans': '已存文件名（未验证的元数据）',
            'ko': '저장된 파일명 (미검증 메타데이터)',
            'ja': '保存されたファイル名（未検証のメタデータ）',
        }[locale]
    heading, limits = _TEXT[locale]
    values = {**receipt['artifact'], **{key: value for key, value in receipt.items()
                                     if key not in ('artifact', 'limitations')}}
    parts = ['<section><h2>', escape(heading), '</h2><p>', escape(limits), '</p><dl>']
    for key in _KEYS:
        value = values[key]
        if isinstance(value, list):
            value = ', '.join(value)
        parts.extend(['<dt>', escape(labels[key]), '</dt><dd>',
                      escape(str(value), quote=True), '</dd>'])
    for key, label in zip(('artifact_id', 'storage', 'execution_permission', 'llm_upload_permission'),
                          _STORED_LABELS[locale]):
        if key in receipt['artifact']:
            parts.extend(['<dt>', escape(label), '</dt><dd>',
                          escape(str(receipt['artifact'][key]), quote=True), '</dd>'])
    parts.append('</dl>')
    if source_text is not None:
        title, disclosure = _SOURCE_LABELS[locale]
        # Character references preserve original CR/LF through Windows stdout
        # newline translation and HTML input newline normalization.
        escaped_source = escape(source_text, quote=True).replace('\r', '&#13;').replace('\n', '&#10;')
        parts.extend(['<h3>', escape(title), '</h3><p>', escape(disclosure),
                      '</p><pre><code>', escaped_source, '</code></pre>'])
    parts.append('</section>')
    return ''.join(parts)


_DIFF_TEXT = {
    'en': ('Selected preserved model revision comparison',
           'Unsigned byte snapshots, left to right. At most 64 KiB and 2000 lines per input. '
           'Each diff line is a JSON string of one original line, including CR/LF and missing final newline. '
           'Leading UTF-8 BOM is omitted from text only. No semantic equivalence or safety is established.'),
    'zh-Hans': ('所选已保存模型版本比较',
                '从左到右的未签名字节快照。每个输入最多64 KiB和2000行。每个差异行是原始行的JSON字符串，'
                '包括CR/LF及末尾缺失的换行符。仅在文本中省略开头的UTF-8 BOM。不证明语义等价性或安全性。'),
    'ko': ('선택한 보존 모형 버전 비교',
           '왼쪽에서 오른쪽으로 비교하는 서명 없는 바이트 스냅샷입니다. 입력당 최대 64 KiB 및 2000줄입니다. '
           '차이의 각 줄은 CR/LF와 마지막 줄바꿈 유무를 포함한 원본 한 줄의 JSON 문자열입니다. '
           '텍스트에서만 선행 UTF-8 BOM을 생략합니다. 의미적 동등성이나 안전성을 입증하지 않습니다.'),
    'ja': ('選択した保存済みモデルの版比較',
           '左から右へ比較する署名のないバイトスナップショットです。各入力は最大64 KiB、2000行です。'
           '差分の各行は元の1行のJSON文字列で、CR/LFと末尾改行の有無を含みます。'
           'テキストのみ先頭のUTF-8 BOMを省略します。意味的な同等性や安全性は証明しません。'),
}


def compared_model_section(workspace, left_id, right_id, expected_sha256, locale):
    """Compare two inert verified snapshots with bounded diff work and output.

    IDs select exact bytes, not a historical ordering or authenticated authorship.
    No source is reread for the receipts. The pair is not an atomic filesystem
    snapshot, and does not authorize execution of either revision.
    """
    snapshots = [inspect_stored_mod_source(workspace, left_id, expected_sha256),
                 inspect_stored_mod_source(workspace, right_id, right_id)]
    records = []
    for receipt, text in snapshots:
        lines = text.splitlines(keepends=True)
        if receipt['artifact']['bytes'] > 64 * 1024 or len(lines) > 2000:
            raise ScopeError('MODEL_DIFF_TOO_LARGE')
        records.append([json.dumps(line, ensure_ascii=False) for line in lines])
    if left_id == right_id:
        status = 'IDENTICAL_BYTES'
    elif records[0] == records[1]:
        status = 'DIFFERENT_BYTES_SAME_DISPLAY_TEXT'
    else:
        status = 'DIFFERENT_BYTES'
    diff = '\n'.join(difflib.unified_diff(*records, fromfile=left_id,
                                        tofile=right_id, lineterm=''))
    if len(diff.encode('utf-8')) > 512 * 1024:
        raise ScopeError('MODEL_DIFF_TOO_LARGE')
    heading, limits = _DIFF_TEXT[locale]
    return ('<section><h2>' + escape(heading) + '</h2><p>' + escape(limits) +
            '</p><p>' + status + '</p>' +
            ''.join(_receipt_section(receipt, locale) for receipt, _ in snapshots) +
            '<pre><code>' + escape(diff, quote=True) + '</code></pre></section>')
