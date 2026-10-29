"""Read-only offline HTML preview; no scripts, assets, tool actions or I/O."""
from html import escape
import json

from .agent_review import review_host_turn
from .agent_review_labels import REVIEW_LABELS


def _text(value):
    return escape(str(value), quote=True)


def _fields(values, labels):
    return "<dl>" + "".join(
        f"<dt>{_text(labels[key])}</dt><dd>{_text(value)}</dd>"
        for key, value in values.items()) + "</dl>"


def render_host_turn_html(result):
    """Wrap the same validated, inert section in a standalone document."""
    section = render_host_turn_section(result)
    locale = json.loads(result.request_json)['locale']
    return (
        '<!doctype html><html lang="' + _text(locale) + '"><head><meta charset="utf-8">'
        '<meta http-equiv="Content-Security-Policy" '
        'content="default-src &#x27;none&#x27;; base-uri &#x27;none&#x27;; '
        'form-action &#x27;none&#x27;">'
        '<meta name="referrer" content="no-referrer">'
        '<title>' + _text(REVIEW_LABELS[locale]['title']) + '</title></head><body><main>'
        + section + '</main></body></html>')


def render_host_turn_section(result):
    """Reinspect a bound turn and return inert HTML, never accepting display cards.

    All model content is escaped plain text, including JSON argument keys. This
    preview neither connects a provider nor certifies origin, evidence or approval.
    The returned document contains potentially sensitive full response text;
    storage/display/transfer policy remains the caller's responsibility.
    """
    review = review_host_turn(result)
    request = json.loads(result.request_json)
    locale = request['locale']
    labels = REVIEW_LABELS[locale]
    assistant, proposals, _ = review["cards"]
    parts = [
        '<h1>' + _text(labels['heading']) + '</h1>',
        '<p>' + _text(labels['provider']) + '</p>',
        '<section><h2>' + _text(labels['binding']) + '</h2>',
        _fields(review["binding"], labels),
        _fields({key: review[key] for key in ("execution", "authorization", "report_evidence")}, labels),
        '</section><section><h2>' + _text(labels['request']) + '</h2><pre><code>',
        _text(request["messages"][2]["content"]),
        '</code></pre></section><section><h2>' + _text(labels['assistant']) + '</h2>',
        _fields({key: assistant[key] for key in ("trust", "claim_verification")}, labels),
        '<pre><code>' + _text(assistant["text"]) + '</code></pre></section>',
        '<section><h2>' + _text(labels['proposals']) + '</h2>',
    ]
    if not proposals["items"]:
        parts.append('<p>' + _text(labels['empty']) + '</p>')
    for item in proposals["items"]:
        parts.extend([
            '<section><h3>' + _text(labels['tool']) + ': ' + _text(item["name"]) + '</h3>',
            _fields({key: item[key] for key in ("id", "trust", "authorization", "status")}, labels),
            '<p>' + _text(labels['arguments']) + '</p><pre><code>',
            _text(json.dumps(item["arguments"], ensure_ascii=False, allow_nan=False, indent=2)),
            '</code></pre></section>',
        ])
    parts.extend(['</section><section><h2>' + _text(labels['limits']) + '</h2><p>',
                  _text(labels['limits_text']), '</p></section>'])
    return "".join(parts)
