"""Explicit loopback study conversation; study files never enter host messages."""
from pathlib import Path

from .agent_bootstrap import LOCALES
from .agent_host import prepare_host_request
from .llm_host_transport import call_loopback_host
from .study_review import review
from .workspace import ScopeError, load_json


TEXT = {
    'en': {
        'NO_LLM': 'No LLM connected. Local study review remains available; no AI reasoning or tool execution occurred.',
        'LLM': 'Local host response is untrusted. Model instruction consumption is unverified; all tools remain blocked.',
        'BLOCKED': 'Local host unavailable or response rejected. NO_LLM local study review remains available.',
    },
    'ko': {
        'NO_LLM': 'LLM 연결 없음. 로컬 연구 검토는 가능하며 AI 추론이나 도구 실행은 수행되지 않았습니다.',
        'LLM': '로컬 호스트 응답은 신뢰되지 않습니다. 모델의 지침 소비는 미검증이며 모든 도구는 차단됩니다.',
        'BLOCKED': '로컬 호스트를 사용할 수 없거나 응답이 거부되었습니다. NO_LLM 로컬 연구 검토를 사용할 수 있습니다.',
    },
    'ja': {
        'NO_LLM': 'LLM 未接続。ローカル研究レビューは利用可能です。AI 推論やツール実行は行われていません。',
        'LLM': 'ローカルホストの応答は未信頼です。モデルの指示受信は未検証で、全ツールは遮断されています。',
        'BLOCKED': 'ローカルホストが利用不可、または応答が拒否されました。NO_LLM の研究レビューは利用可能です。',
    },
    'zh-Hans': {
        'NO_LLM': '未连接 LLM。仍可进行本地研究审查；未进行 AI 推理或工具执行。',
        'LLM': '本地主机响应不可信。模型是否接收指令尚未验证，所有工具仍被阻止。',
        'BLOCKED': '本地主机不可用或响应被拒绝。仍可使用 NO_LLM 本地研究审查。',
    },
}


def study_host(*, provider='none', model=None, base_url=None, request_text=None,
               allow_local_host=False, locale='en', file=None, workspace=None, overrides=None):
    """Review local files separately; only explicitly typed text may be sent.

    This client's literal-loopback restriction cannot attest that the selected
    provider is offline or prevent that provider from relaying content onward.
    Users must select an independently controlled local model for sensitive use.
    """
    if locale not in LOCALES:
        raise ScopeError('AGENT_LOCALE_UNSUPPORTED')
    if provider == 'none' and any((model is not None, base_url is not None,
                                 request_text is not None, allow_local_host)):
        raise ScopeError('STUDY_HOST_NO_LLM_OPTIONS_INVALID')
    if provider != 'none' and allow_local_host is not True:
        raise ScopeError('LLM_LOCAL_HOST_EXPLICIT_APPROVAL_REQUIRED')
    if workspace is not None and file is None:
        raise ScopeError('STUDY_HOST_WORKSPACE_REQUIRES_STUDY')
    if overrides is not None and (file is None or workspace is None):
        raise ScopeError('STUDY_HOST_OVERRIDES_REQUIRE_STUDY_AND_WORKSPACE')
    # Validate the local selection before contacting any provider.
    local_review = review(load_json(Path(file)), workspace, overrides) if file else None
    result = {
        'schema': 'societybox-study-host-v1', 'locale': locale,
        'status': 'READY', 'mode': 'NO_LLM', 'provider': provider,
        'provider_status': 'NOT_SELECTED', 'remote_transmission': 'DENY',
        'study_transfer': 'DENY', 'tool_authorization': 'NOT_GRANTED',
        'execution': 'NOT_RUN', 'claim_verification': 'NOT_RUN',
        'model_consumption': 'NOT_ESTABLISHED',
        'provider_onward_network_enforcement': 'NOT_ESTABLISHED',
        'study_review': local_review, 'host_result': None,
        'message': TEXT[locale]['NO_LLM'],
    }
    if provider == 'none':
        return result
    packet = prepare_host_request(request_text, locale)
    try:
        host = call_loopback_host(packet, provider=provider, model=model,
            base_url=base_url, allow_local_host=allow_local_host)
    except ScopeError as exc:
        # Policy/input errors remain explicit CLI refusals, never silent success.
        if exc.code not in (
                'LLM_LOCAL_PROVIDER_UNAVAILABLE', 'LLM_LOCAL_HTTP_REJECTED',
                'LLM_LOCAL_USAGE_LIMIT',
                'LLM_LOCAL_RESPONSE_TOO_LARGE', 'LLM_LOCAL_RESPONSE_INVALID',
                'LLM_LOCAL_MODEL_RESPONSE_NOT_VALIDATED'):
            raise
        result.update(status='BLOCKED', provider_status='UNKNOWN', reason=exc.code,
                      message=TEXT[locale]['BLOCKED'])
        return result
    result.update(mode='LLM', provider_status='RESPONSE_SCHEMA_VALIDATED',
                  model_consumption=host['model_instruction_consumption'],
                  host_result=host, message=TEXT[locale]['LLM'])
    return result
