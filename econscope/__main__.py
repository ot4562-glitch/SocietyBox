"""No shell execution, no remote model calls; conservative offline CLI."""
import argparse
import json
import sys
from pathlib import Path

from .workspace import (
    ScopeError, get_root, ingest, ingest_directory, initialize, list_inventory, update_scope, load_json
)
from .runs import make_run, verify_run
from .ai import engine_catalog, make_ai_request, execute_plan
from .study_runs import diagnose as study_diagnose, execute as study_execute, verify as study_verify, report as study_report, compare as study_compare, replication_bundle as study_bundle
from .study_core import StudyCalculationError
from .study_review import review as study_review
from .study_capabilities import negotiate_capabilities
from .agent_bootstrap import agent_context, reference_capabilities, knowledge
from .agent_preview import conversation_preview
from .agent_host import prepare_host_request
from .model_intake import review_mod
from .study_runs import comparison_bundle, verify_comparison_bundle
from .study_charts import comparison_chart
from .study_html import comparison_report
from .study_pdf import comparison_pdf
from .study_csv import comparison_trajectory_csv
from .study_recompute import recompute_comparison_bundle, comparison_source_plan


class _ConsoleArgumentParser(argparse.ArgumentParser):
    def _print_message(self, message, file=None):
        target = file or sys.stderr
        encoding = getattr(target, 'encoding', None)
        if message and encoding:
            message = message.encode(encoding, 'backslashreplace').decode(encoding)
        super()._print_message(message, file)


def build_parser(prog="econscope"):
    parser = _ConsoleArgumentParser(prog=prog, description="Private offline economics and sociology research sandbox")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser('research-chat', help='Local chat with saved study versions, explicit approval and verified exact calculations; NO_LLM')
    a.add_argument('--workspace', required=True, help='New empty directory or an existing research-chat workspace')
    a.add_argument('--port', type=int, default=0)
    a.add_argument('--ttl-seconds', type=int, default=3600)
    a.add_argument('--runtime', help='Explicit reviewed local HiGHS runtime directory')
    a.add_argument('--sources', help='Explicit pinned source archive directory')
    a = sub.add_parser('wdi-replicate', help='Offline WDI descriptive arithmetic and CSV/SVG/ASCII PDF bundle')
    a.add_argument('--directory', required=True)
    a.add_argument('--cutoff', required=True)
    a.add_argument('--output', required=True, help='New output directory under an existing parent')
    a = sub.add_parser('wdi-snapshot-review', help='Offline pinned public GDP retrieval review; no historical as-of assurance')
    a.add_argument('--directory', required=True)
    a.add_argument('--cutoff', required=True, help='Offset timestamp at or after retrieval, never in the future')
    a = sub.add_parser('study-host', help='Local study review with optional explicit loopback LLM; no study transfer or tool execution')
    a.add_argument('--file', help='Optional local study JSON; never sent to host')
    a.add_argument('--workspace', help='Optional local read-only study binding; requires --file')
    a.add_argument('--overrides-file', help='Local JSON parameter overrides; requires study and workspace; never sent to host')
    a.add_argument('--provider', choices=('none', 'ollama', 'openai-compatible-local'), default='none')
    a.add_argument('--model', help='Explicit model ID at an independently controlled local host')
    a.add_argument('--base-url', help='Literal loopback HTTP IP and port only; remote destinations denied')
    a.add_argument('--request-text', help='Only this explicitly entered text is sent; do not include secrets or research data without independently verified provider isolation')
    a.add_argument('--allow-local-host', action='store_true', help='Opt in to this local text transfer; provider onward networking is not enforced by Society Box')
    a.add_argument('--locale', choices=('en', 'zh-Hans', 'ko', 'ja'), default='en')
    a = sub.add_parser("model-intake", help="Read-only bounded .mod byte review; no parsing, macros, solver execution or approval")
    a.add_argument("--directory", required=True, help="Existing local intake directory; no files are written")
    a.add_argument("--file", required=True, help="Relative .mod path using / separators, at most 1 MiB of UTF-8")
    a.add_argument("--expected-sha256", help="Optional exact lowercase SHA256; mismatch rejects review")
    a = sub.add_parser("inverse-screen", help="Offline automatic lagged-regression hypothesis screening and retrospective residual patterns; no causal identification or external transfer")
    a.add_argument("--spec", required=True, help="Bounded local JSON with declared series, release dates, training cutoff and explicitly selected search lags")
    a.add_argument("--format", choices=("json", "html"), default="json", help="Inline evidence JSON or recomputed offline researcher HTML")
    a.add_argument("--locale", choices=("en", "ko", "zh-Hans", "ja"), default="en")
    a.add_argument("--verify-result", help="Saved raw inverse-screen result JSON to reexecute and compare; excludes --format html")
    a = sub.add_parser("ols-reconstruct", help="Offline bounded one-regressor OLS inverse coefficients and forward replay; no causal, historical or original-spec claim")
    a.add_argument("--spec", required=True, help="Researcher-declared bounded OLS JSON with explicit classical/HC0-HC3 SE choice and optional sample_selection; returns residual CSV in JSON, no LLM/remote calls")
    a = sub.add_parser("synthetic-gaussian-states", help="Read-only 1–3 step synthetic fixed-parameter Gaussian math; no estimation, empirical forecast or execution authorization")
    a.add_argument("--spec", required=True, help="Local UTF-8 synthetic Gaussian JSON, at most 2 MiB; decimal strings, no code")
    a = sub.add_parser("mechanism-study-review", help="Read-only synthetic proposal contract; no estimation, forecast or execution")
    a.add_argument("--study", required=True, help="Local UTF-8 mechanism proposal JSON, at most 2 MiB; contains cutoff and input hashes")
    a.add_argument("--ledger", required=True, help="Local UTF-8 synthetic ledger JSON, at most 2 MiB")
    a.add_argument("--policy", required=True, help="Local UTF-8 synthetic policy JSON, at most 2 MiB")
    for command in ("ledger-review", "ledger-snapshot"):
        a = sub.add_parser(command, help="Read-only synthetic vintage ledger; no estimation, forecast or historical-access proof")
        a.add_argument("--ledger", required=True, help="Local UTF-8 ledger JSON, at most 2 MiB")
        a.add_argument("--policy", required=True, help="Local UTF-8 synthetic policy JSON, at most 2 MiB")
        a.add_argument("--as-of", required=True, help="Inclusive release cutoff with explicit UTC offset and seconds")
        if command == "ledger-snapshot":
            a.add_argument("--vintage-target", default="AS_OF", help="Only AS_OF is implemented; other targets block")
    a=sub.add_parser("research-review-entry", help="Print offline interactive HTML selecting study/model/run/comparison reviews; runs recomputed, comparisons checked for consistency only, no new run or approval")
    a.add_argument("--request-file", help="Explicit saved host-request-v1 JSON, at most 1 MiB; requires --response-file and matching --locale; excludes --request-text/--export-host-request; no provider calls")
    a.add_argument("--request-text", help="Bounded local researcher request; requires --response-file or --export-host-request; no inference or authority")
    a.add_argument("--export-host-request", action="store_true", help="Embed a local request JSON download; requires --request-text; no research attachments, provider transfer or host consumption")
    a.add_argument("--response-file", help="Optional untrusted host-response-v1 JSON, at most 64000 bytes; requires --request-text or --request-file; no provider calls")
    a.add_argument("--workspace", required=True)
    a.add_argument("--study-file", action="append", required=True, help="Local study JSON snapshot; repeat up to four times")
    a.add_argument("--engine-study-bundle", action="append", default=[], help="Independent economic study bundle; repeat up to four; explicitly recomputed before display, never attached to host requests")
    a.add_argument("--engine-sources", help="Reviewed pinned source archives for SimPy, NetworkX or Pyomo bundle reexecution; requires --engine-study-bundle")
    a.add_argument("--engine-runtime", help="Reviewed pinned local HiGHS runtime for explicit native bundle reexecution; requires --engine-study-bundle")
    a.add_argument("--inverse-screen-spec", action="append", default=[], help="Explicitly calculate a local declared inverse-screen spec and show recomputed evidence; repeat up to four per view; never sent to host")
    a.add_argument("--overrides-file", help="JSON map of intended parameter changes reviewed against each study; preserved runs/comparisons keep recorded parameters")
    a.add_argument("--model-artifact-id", action="append", default=[], help="Preserved .mod ID; repeat up to four times; byte review only")
    a.add_argument("--model-expected-sha256", help="Assert exact bytes of one selected (left) model; requires exactly one --model-artifact-id; no origin authentication")
    a.add_argument("--model-show-source", action="store_true", help="Embed escaped preserved source in private HTML, including unselected snapshots; requires --model-artifact-id; no execution or host inclusion")
    a.add_argument("--model-compare-artifact-id", help="Embed private escaped revision diff; requires exactly one left --model-artifact-id; exclusive with --model-show-source; 64 KiB/2000 lines per input; no execution or host inclusion")
    a.add_argument("--run-id", action="append", default=[], help="Preserved run ID; repeat up to four times; each study/run pair verified by deterministic recomputation during page generation")
    a.add_argument("--comparison-bundle", action="append", default=[], help="Portable comparison directory; repeat up to four times; study-bound artifact consistency only, no engine recomputation")
    a.add_argument("--export-comparison-csv", action="store_true", help="Opt-in local exact trajectory CSV download for selected comparisons; declared export rights and study/receipt binding required; no host transfer")
    a.add_argument("--export-comparison-html", action="store_true", help="Opt-in local evidence-first English HTML report download from the selected verified comparison snapshot; no host transfer")
    a.add_argument("--export-comparison-pdf", action="store_true", help="Opt-in private English ASCII PDF download; no multilingual fonts or host transfer")
    a.add_argument("--comparison-receipt", action="append", default=[], metavar="INDEX=PATH", help="Optional unsigned local receipt for the one-based comparison choice; repeat for distinct choices; binding inspection only, no execution or host transfer")
    a.add_argument("--locale", choices=("en", "zh-Hans", "ko", "ja"), default="en")
    a=sub.add_parser("conversation-preview", help="Print local review HTML; optional run verification recomputes supported deterministic model, no LLM or new approval")
    a.add_argument("--locale", choices=("en", "zh-Hans", "ko", "ja"), default="en")
    a.add_argument("--request-text", help="Local researcher request; requires --response-file, no inference")
    a.add_argument("--response-file", help="Local untrusted host-response-v1 UTF-8 JSON, at most 64000 bytes; requires --request-text")
    a.add_argument("--study-file", help="Optional local study JSON for read-only review; never sent to a model")
    a.add_argument("--run-id", help="Optional selected run; requires study and workspace; verifies by deterministic recomputation")
    a.add_argument("--comparison-bundle", help="Optional portable comparison directory; read-only artifact consistency only, no engine execution or host transfer")
    a.add_argument("--recomputation-receipt", help="Optional unsigned local record; requires --comparison-bundle; checks bindings without execution")
    a.add_argument("--workspace", help="Read-only workspace binding; requires --study-file or --model-artifact-id")
    a.add_argument("--model-artifact-id", help="Preserved .mod SHA256 ID; requires --workspace; exclusive with external model selection; no execution or upload")
    a.add_argument("--model-show-source", action="store_true", help="Opt-in escaped full source display, at most 1 MiB UTF-8; requires --model-artifact-id; no host inclusion or execution")
    a.add_argument("--model-compare-artifact-id", help="Opt-in read-only diff against a second preserved .mod; requires --model-artifact-id and --workspace; exclusive with --model-show-source; 64 KiB/2000 lines per input")
    a.add_argument("--overrides-file", help="Optional JSON map of declared parameter changes; requires --study-file and --workspace")
    a.add_argument("--model-directory", help="Local inert .mod intake directory; requires --model-file; never sent to host")
    a.add_argument("--model-file", help="Relative .mod path for byte-only review; requires --model-directory; no execution or approval")
    a.add_argument("--model-expected-sha256", help="Optional exact lowercase SHA256 for the selected .mod")
    a=sub.add_parser("study-capabilities", help="Inspect no-LLM study tools; no execution, provider connection or authorization")
    a.add_argument("--require", action="append", default=[], help="Required capability ID; repeat to negotiate a feature set")
    a=sub.add_parser("host-request", help="Export prospective offline host-request JSON to stdout; no model consumption, transfer authorization or execution")
    a.add_argument("--request-text", required=True, help="Researcher text, at most 64000 UTF-8 bytes; included verbatim in output")
    a.add_argument("--locale", choices=("en", "zh-Hans", "ko", "ja"), default="ko")
    a=sub.add_parser("agent-context", help="Read the LLM first-instruction asset and current offline capability/knowledge indexes; no connection")
    a.add_argument("--locale", choices=("en", "zh-Hans", "ko", "ja"), default="ko")
    a=sub.add_parser("reference-capabilities", help="Inventory every documented upstream feature family; no native execution")
    a.add_argument("--source", default=None, help="Optional source ID such as U-DYN or U-MESA")
    a=sub.add_parser("study-knowledge", help="Read original offline research method primers, not empirical data")
    a.add_argument("--topic", default=None, help="Optional topic ID from the index")
    a.add_argument("--locale", choices=("en", "zh-Hans", "ko", "ja"), default="ko")
    a=sub.add_parser("study-review", help="Inspect evidence and model card data without executing or approving it")
    a.add_argument("--file", required=True)
    a.add_argument("--workspace", default=None, help="Bind review to this workspace and referenced data")
    a.add_argument("--overrides-file", default=None)
    a=sub.add_parser("init", help="Create isolated workspace")
    a.add_argument("--workspace", required=True)
    a=sub.add_parser("ingest", help="Copy a researcher file as immutable, unexecuted bytes")
    a.add_argument("--workspace", required=True)
    a.add_argument("--file", required=True)
    a.add_argument("--published", default=None, help="User CLAIMED publication YYYY-MM-DD; never verified automatically")
    a.add_argument("--license", dest="license_label", default="UNKNOWN")
    a=sub.add_parser("ingest-dir", help="Copy a folder of researcher documents, skipping hidden/secret-like files")
    a.add_argument("--workspace", required=True)
    a.add_argument("--directory", required=True)
    a.add_argument("--dry-run", action="store_true")
    a=sub.add_parser("scope", help="Define region, topic and historical information cutoff")
    a.add_argument("--workspace", required=True)
    a.add_argument("--as-of", default=None)
    a.add_argument("--domain", choices=["economics", "sociology", "cross-disciplinary"], default="cross-disciplinary")
    a.add_argument("--region", default="unspecified")
    a.add_argument("--question", default=None)
    a.add_argument("--allow-undated", action="store_true", help="Explicitly allow unknown published date with warnings")
    a=sub.add_parser("inventory")
    a.add_argument("--workspace", required=True)
    a=sub.add_parser("engines")
    a.add_argument("--quantecon-reference", default=None, help="Read five pinned candidate files as inert bytes; no solver execution")
    a=sub.add_parser("plan")
    a.add_argument("--workspace", required=True)
    a.add_argument("--question", required=True)
    a.add_argument("--provider", choices=["deterministic","codex-export","chatgpt-export","local-llm-export"],default="deterministic")
    a.add_argument("--data-id", default=None)
    a=sub.add_parser("run")
    a.add_argument("--workspace", required=True)
    a.add_argument("--model", choices=["tax_transfer","opinion_diffusion"], required=True)
    a.add_argument("--data-id", default=None)
    a.add_argument("--tax-rate", default="0.15")
    a.add_argument("--steps", type=int, default=5)
    a.add_argument("--seed-share", default="0.2")
    a=sub.add_parser("execute-plan", help="Strictly validate one data-only plan exported by an assistant")
    a.add_argument("--workspace", required=True)
    a.add_argument("--file", required=True)
    a=sub.add_parser("study-check", help="Assess a researcher-authored arbitrary-variable declarative simulation model")
    a.add_argument("--file", required=True)
    a=sub.add_parser("study-run", help="Execute a reviewed, bounded model (never executes Python from input)")
    a.add_argument("--workspace", required=True)
    a.add_argument("--file", required=True)
    a.add_argument("--overrides-file", default=None, help="Optional JSON map of named parameter changes")
    a.add_argument("--approve-research-spec", action="store_true", help="Explicit CLI operator review acknowledgement")
    a.add_argument("--review-sha256", default=None, help="Exact execution review digest from study-review --workspace")
    a=sub.add_parser("study-verify", help="Reexecute a saved run and compare receipt hashes")
    a.add_argument("--workspace", required=True)
    a.add_argument("--run-id", required=True)
    a=sub.add_parser("study-compare", help="Contrast parameters under the same reviewed model and scope")
    a.add_argument("--workspace", required=True)
    a.add_argument("--baseline-id", required=True)
    a.add_argument("--alternative-id", required=True)
    a=sub.add_parser("study-compare-export", help="Export a rights-gated paired comparison with graphs and exact delta data")
    a.add_argument("--workspace", required=True)
    a.add_argument("--baseline-id", required=True)
    a.add_argument("--alternative-id", required=True)
    a=sub.add_parser("study-compare-source-plan", help="Inspect required source hashes and declared rights; print an empty local data-map template without reading data")
    a.add_argument("--bundle", required=True)
    a=sub.add_parser("study-compare-recompute", help="Recompute portable comparisons locally with the matching declarative engine")
    a.add_argument("--bundle", required=True)
    a.add_argument("--data-map", help="Local JSON mapping source SHA256 to path and optional published date; paths relative to current directory")
    a.add_argument("--data-map-relative-to", choices=("cwd", "map"), default="cwd",
                   help="Resolve relative source paths from the current directory (default) or data-map file's directory; no source search")
    a.add_argument("--receipt-output", help="Write a separate unsigned local recomputation receipt outside the bundle; conflicting existing files are preserved. Check the response receipt_output status for this attempt.")
    a=sub.add_parser("study-compare-verify", help="Check portable paired-bundle consistency offline; does not reexecute the engine")
    a.add_argument("--bundle", required=True)
    a=sub.add_parser("study-compare-chart", help="Write a local SVG for one variable from a verified paired bundle")
    a.add_argument("--bundle", required=True)
    a.add_argument("--variable", required=True)
    a.add_argument("--output", required=True)
    a=sub.add_parser("study-compare-report", help="Write a self-contained evidence-first HTML comparison; artifact consistency only")
    a.add_argument("--bundle", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--recomputation-receipt", help="Inspect an optional unsigned local recomputation record; no model execution")
    a=sub.add_parser("study-compare-trajectory-csv", help="Export exact paired trajectory cells and lineage; artifact consistency only")
    a.add_argument("--bundle", required=True)
    a.add_argument("--output", required=True)
    a=sub.add_parser("study-compare-pdf", help="Write an offline ASCII text PDF comparison; no rendered QA or engine execution")
    a.add_argument("--bundle", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--recomputation-receipt", help="Inspect an optional unsigned local record; no model execution")
    a=sub.add_parser("study-export", help="Create a replication folder only with explicit public-content export rights; raw bytes are excluded")
    a.add_argument("--workspace", required=True)
    a.add_argument("--run-id", required=True)
    a=sub.add_parser("study-report", help="Write an evidence-first Markdown report using verified run facts")
    a.add_argument("--workspace", required=True)
    a.add_argument("--run-id", required=True)
    a=sub.add_parser("verify")
    a.add_argument("--workspace", required=True)
    a.add_argument("--run-id", required=True)
    return parser


def run_cli(ns):
    cmd=ns.command
    if cmd == 'wdi-replicate':
        from .wdi_replication import export_wdi_replication
        return export_wdi_replication(ns.directory, ns.cutoff, ns.output)
    if cmd == 'wdi-snapshot-review':
        from .wdi_snapshot import review_snapshot
        return review_snapshot(ns.directory, ns.cutoff)
    if cmd == "inverse-screen":
        from .inverse_screen import screen_inverse_hypotheses
        from .mechanism_cli import load_input
        request = load_input(ns.spec)
        if ns.verify_result:
            if ns.format != "json":
                raise ScopeError("INVESTIGATION_VERIFY_FORMAT_INVALID")
            from .inverse_investigation import verify_inverse_screen
            return verify_inverse_screen(request, load_input(ns.verify_result))
        result = screen_inverse_hypotheses(request)
        if ns.format == "html":
            from .inverse_investigation import render_inverse_screen
            return render_inverse_screen(request, result, ns.locale)
        return result
    if cmd == "ols-reconstruct":
        from .mechanism_cli import run_ols_reconstruction_cli
        return run_ols_reconstruction_cli(ns)
    if cmd == "synthetic-gaussian-states":
        from .mechanism_cli import run_synthetic_gaussian_cli
        return run_synthetic_gaussian_cli(ns)
    if cmd == "mechanism-study-review":
        from .mechanism_cli import run_mechanism_study_cli
        return run_mechanism_study_cli(ns)
    if cmd in ("ledger-review", "ledger-snapshot"):
        from .mechanism_cli import run_ledger_cli
        return run_ledger_cli(ns)
    if cmd=="research-review-entry":
        from .research_review_entry import research_review_entry
        comparison_receipts = {}
        for value in ns.comparison_receipt:
            index, separator, path = value.partition('=')
            if (not separator or index not in ('1', '2', '3', '4') or not path
                or int(index) > len(ns.comparison_bundle) or int(index) in comparison_receipts):
                raise ScopeError("REVIEW_ENTRY_COMPARISON_RECEIPTS_INVALID")
            comparison_receipts[int(index)] = path
        if ns.request_file is not None and (ns.request_text is not None or ns.export_host_request):
            raise ScopeError("REVIEW_ENTRY_REQUEST_INPUT_CONFLICT")
        if ns.export_host_request and ns.request_text is None:
            raise ScopeError("AGENT_HOST_RESEARCHER_REQUEST_REQUIRED")
        has_request = ns.request_text is not None or ns.request_file is not None
        if not ns.export_host_request and has_request != (ns.response_file is not None):
            raise ScopeError("REVIEW_ENTRY_REQUEST_RESPONSE_PAIR_REQUIRED")
        request_bytes = None
        if ns.request_file is not None:
            from .agent_host import MAX_REQUEST_PACKET_BYTES
            with Path(ns.request_file).open('rb') as request_file:
                request_bytes = request_file.read(MAX_REQUEST_PACKET_BYTES + 1)
            if len(request_bytes) > MAX_REQUEST_PACKET_BYTES:
                raise ScopeError("AGENT_HOST_REQUEST_TOO_LARGE")
        response_bytes = None
        if ns.response_file is not None:
            from .agent_response import MAX_RESPONSE_BYTES
            with Path(ns.response_file).open('rb') as response_file:
                response_bytes = response_file.read(MAX_RESPONSE_BYTES + 1)
            if len(response_bytes) > MAX_RESPONSE_BYTES:
                raise ScopeError("AGENT_HOST_RESPONSE_TOO_LARGE")
        overrides = load_json(Path(ns.overrides_file)) if ns.overrides_file else None
        if ns.overrides_file and overrides is None:
            raise ScopeError("STUDY_OVERRIDE_UNDECLARED")
        return research_review_entry(ns.workspace, ns.study_file, ns.model_artifact_id, ns.locale, ns.run_id,
                                     comparison_bundles=ns.comparison_bundle, overrides=overrides,
                                     comparison_receipts=comparison_receipts, export_comparison_csv=ns.export_comparison_csv,
                                     export_comparison_html=ns.export_comparison_html, export_comparison_pdf=ns.export_comparison_pdf,
                                     researcher_request=ns.request_text, response_bytes=response_bytes,
                                     export_host_request=ns.export_host_request, request_bytes=request_bytes,
                                     model_show_source=ns.model_show_source,
                                     model_compare_artifact_id=ns.model_compare_artifact_id,
                                     model_expected_sha256=ns.model_expected_sha256,
                                     engine_study_bundles=ns.engine_study_bundle,
                                     engine_sources=ns.engine_sources, engine_runtime=ns.engine_runtime,
                                     inverse_screen_specs=ns.inverse_screen_spec)
    if cmd=="conversation-preview":
        if (ns.request_text is None) != (ns.response_file is None):
            raise ScopeError("CONVERSATION_PREVIEW_REQUEST_RESPONSE_PAIR_REQUIRED")
        response_bytes = None
        if ns.response_file is not None:
            from .agent_response import MAX_RESPONSE_BYTES
            with Path(ns.response_file).open('rb') as response_file:
                response_bytes = response_file.read(MAX_RESPONSE_BYTES + 1)
            if len(response_bytes) > MAX_RESPONSE_BYTES:
                raise ScopeError("AGENT_HOST_RESPONSE_TOO_LARGE")
        if ns.overrides_file and (not ns.study_file or not ns.workspace):
            raise ScopeError("STUDY_PREVIEW_OVERRIDES_REQUIRE_STUDY_AND_WORKSPACE")
        spec = load_json(Path(ns.study_file)) if ns.study_file else None
        if ns.study_file and spec is None:
            raise ScopeError("STUDY_SCHEMA_UNSUPPORTED")
        overrides = load_json(Path(ns.overrides_file)) if ns.overrides_file else None
        if ns.overrides_file and overrides is None:
            raise ScopeError("STUDY_OVERRIDE_UNDECLARED")
        return conversation_preview(ns.locale, study_spec=spec, workspace=ns.workspace,
                                    overrides=overrides, researcher_request=ns.request_text,
                                    response_bytes=response_bytes, run_id=ns.run_id,
                                    comparison_bundle=ns.comparison_bundle,
                                    recomputation_receipt=ns.recomputation_receipt,
                                    model_directory=ns.model_directory, model_file=ns.model_file,
                                    model_expected_sha256=ns.model_expected_sha256,
                                    model_artifact_id=ns.model_artifact_id,
                                    model_show_source=ns.model_show_source,
                                    model_compare_artifact_id=ns.model_compare_artifact_id)
    if cmd=="model-intake": return review_mod(ns.directory, ns.file, ns.expected_sha256)
    if cmd=="study-capabilities": return negotiate_capabilities(ns.require)
    if cmd=="host-request": return prepare_host_request(ns.request_text, ns.locale)
    if cmd == 'study-host':
        from .study_host import study_host
        overrides = load_json(__import__('pathlib').Path(ns.overrides_file)) if ns.overrides_file else None
        if ns.overrides_file and overrides is None:
            raise ScopeError('STUDY_OVERRIDE_UNDECLARED')
        return study_host(provider=ns.provider, model=ns.model, base_url=ns.base_url,
            request_text=ns.request_text, allow_local_host=ns.allow_local_host,
            locale=ns.locale, file=ns.file, workspace=ns.workspace, overrides=overrides)
    if cmd=="agent-context": return agent_context(ns.locale)
    if cmd=="reference-capabilities": return reference_capabilities(ns.source)
    if cmd=="study-knowledge": return knowledge(ns.topic, ns.locale)
    if cmd=="study-review":
        overrides=load_json(__import__("pathlib").Path(ns.overrides_file)) if ns.overrides_file else None
        if ns.overrides_file and overrides is None:
            raise ScopeError("STUDY_OVERRIDE_UNDECLARED")
        if overrides is not None and ns.workspace is None:
            raise ScopeError("STUDY_REVIEW_WORKSPACE_REQUIRED_FOR_OVERRIDES")
        return study_review(load_json(__import__("pathlib").Path(ns.file)), ns.workspace, overrides)
    if cmd=="init": return initialize(ns.workspace)
    if cmd=="ingest":
        return ingest(ns.workspace,ns.file,published=ns.published,license_label=ns.license_label)
    if cmd=="ingest-dir":
        return ingest_directory(ns.workspace,ns.directory,dry_run=ns.dry_run)
    if cmd=="scope":
        return update_scope(ns.workspace,as_of=ns.as_of,domain=ns.domain,
                            region=ns.region,question=ns.question,strict=not ns.allow_undated)
    if cmd=="inventory": return list_inventory(ns.workspace)
    if cmd=="engines": return engine_catalog(ns.quantecon_reference)
    if cmd=="plan":
        return make_ai_request(ns.workspace,ns.question,provider=ns.provider,data_id=ns.data_id)
    if cmd=="run":
        params=({"tax_rate":ns.tax_rate} if ns.model=="tax_transfer"
                else {"steps":ns.steps,"seed_share":ns.seed_share})
        return make_run(ns.workspace,model=ns.model,params=params,data_id=ns.data_id)
    if cmd=="execute-plan":
        return execute_plan(ns.workspace,ns.file)
    if cmd=="study-check": return study_diagnose(load_json(__import__("pathlib").Path(ns.file)))
    if cmd=="study-run":
        overrides=load_json(__import__("pathlib").Path(ns.overrides_file)) if ns.overrides_file else None
        if ns.overrides_file and overrides is None:
            raise ScopeError("STUDY_OVERRIDE_UNDECLARED")
        return study_execute(ns.workspace, load_json(__import__("pathlib").Path(ns.file)),
                             overrides=overrides, approved=ns.approve_research_spec,
                             review_sha256=ns.review_sha256)
    if cmd=="study-verify": return study_verify(ns.workspace,ns.run_id)
    if cmd=="study-report": return study_report(ns.workspace,ns.run_id)
    if cmd=="study-compare": return study_compare(ns.workspace,ns.baseline_id,ns.alternative_id)
    if cmd=="study-compare-export": return comparison_bundle(ns.workspace,ns.baseline_id,ns.alternative_id)
    if cmd=="study-compare-verify": return verify_comparison_bundle(ns.bundle)
    if cmd=="study-compare-source-plan": return comparison_source_plan(ns.bundle)
    if cmd=="study-compare-recompute":
        if ns.data_map_relative_to == "map" and not ns.data_map:
            raise ScopeError("STUDY_PORTABLE_SOURCE_MAP_REQUIRED")
        sources = load_json(Path(ns.data_map)) if ns.data_map else None
        if ns.data_map_relative_to == "map" and isinstance(sources, dict):
            base = Path(ns.data_map).resolve().parent
            # Only interpret path strings; the recomputation boundary still
            # validates the complete map and checks rights, bytes and cutoff.
            for entry in sources.values():
                if isinstance(entry, dict) and isinstance(entry.get("path"), str) and entry["path"]:
                    path = Path(entry["path"]).expanduser()
                    if not path.is_absolute():
                        entry["path"] = str(base / path)
        return recompute_comparison_bundle(ns.bundle,
            source_data=sources,
            receipt_output=ns.receipt_output)
    if cmd=="study-compare-chart": return comparison_chart(ns.bundle,ns.variable,ns.output)
    if cmd=="study-compare-report":
        return comparison_report(ns.bundle,ns.output, recomputation_receipt=ns.recomputation_receipt)
    if cmd=="study-compare-pdf":
        return comparison_pdf(ns.bundle,ns.output, recomputation_receipt=ns.recomputation_receipt)
    if cmd=="study-compare-trajectory-csv":
        return comparison_trajectory_csv(ns.bundle,ns.output)
    if cmd=="study-export": return study_bundle(ns.workspace,ns.run_id)
    if cmd=="verify": return verify_run(ns.workspace,ns.run_id)
    raise ScopeError("COMMAND_UNKNOWN")


def _recomputation_receipt_output(ns, result=None):
    """Bind CLI output to this attempt without exposing the local receipt path.

    An error can precede writing or follow publication (for example cleanup
    failure), so it cannot promise either deletion or preservation of a receipt.
    """
    if ns.command != "study-compare-recompute" or not ns.receipt_output:
        return {}
    if result is not None:
        return {"receipt_output": {"status": "RECORDED_FOR_THIS_ATTEMPT",
                                   "recorded_outcome": result["status"]}}
    return {"receipt_output": {
        "status": "NOT_CONFIRMED_FOR_THIS_ATTEMPT",
        "guidance": "An existing receipt may describe an earlier attempt. "
                    "Use this command response for the current outcome; retry with a fresh receipt path."}}


def main(prog="econscope"):
    ns=build_parser(prog=prog).parse_args()
    if ns.command == 'research-chat':
        from .research_chat_service import main as chat_main
        arguments=['--workspace', ns.workspace, '--port', str(ns.port), '--ttl-seconds', str(ns.ttl_seconds)]
        if ns.runtime is not None: arguments += ['--runtime', ns.runtime]
        if ns.sources is not None: arguments += ['--sources', ns.sources]
        return chat_main(arguments)
    try:
        result=run_cli(ns)
        if ns.command == 'study-host' and result['status'] == 'BLOCKED':
            print(json.dumps({'status': 'BLOCKED', 'data': result}, ensure_ascii=True,
                             sort_keys=True, allow_nan=False))
            return 2
        if ns.command in ("ledger-review", "ledger-snapshot", "mechanism-study-review", "synthetic-gaussian-states"):
            from .mechanism_cli import encode_result
            encoded, exit_code = encode_result(result)
            print(encoded)
            return exit_code
        if ns.command in ("conversation-preview", "research-review-entry") or (ns.command == "inverse-screen" and ns.format == "html"):
            # HTML character references preserve study Unicode across Windows pipes.
            print(result.encode("ascii", "xmlcharrefreplace").decode("ascii"))
            return 0
        # ASCII JSON escapes preserve Unicode exactly across console/pipe encodings.
        # Do not change the host stream settings or require a UTF-8 environment.
        if result.get("status") == "RECOMPUTATION_NOT_RUN":
            print(json.dumps({"status":"NOT_RUN","data":result,
                              **_recomputation_receipt_output(ns, result)},ensure_ascii=True,sort_keys=True,allow_nan=False))
            return 2
        print(json.dumps({"status":"OK","data":result,
                          **_recomputation_receipt_output(ns, result)},ensure_ascii=True,sort_keys=True,allow_nan=False))
        return 0
    except ScopeError as e:
        error = {"status":"INVALID", "reason":e.code, **_recomputation_receipt_output(ns)}
        if e.code == "STUDY_RUN_RECEIPT_UNAVAILABLE":
            error["recovery"] = {
                "receipt_verification": "NOT_COMPLETED",
                "instruction": "Select the workspace containing the saved run receipt and use its exact run ID. If no receipt exists, review the study and separately acknowledge execution before requesting reports or comparisons.",
            }
        if (e.code in ("DATA_FILE_MISSING_OR_UNSAFE", "DATA_HASH_MISMATCH")
            and ns.command in ("study-verify", "study-report", "study-export",
                               "study-compare", "study-compare-export")):
            error["recovery"] = {
                "receipt_verification": "NOT_COMPLETED",
                "source_verification": "NOT_COMPLETED",
                "instruction": "The referenced workspace data object is unavailable, unsafe or changed. Moving the original CSV alone does not affect its ingested copy. Select a trusted intact workspace backup, or, only if the object is absent, use ingest --workspace WORKSPACE --file SOURCE with legally available byte-identical data. Existing unsafe or mismatched objects require operator inspection; do not overwrite them or edit saved receipts, hashes or manifests. Then run study-verify for each selected run before retrying the report or export. Saved hashes alone do not verify results.",
            }
        if ns.command == "study-run" and isinstance(e, StudyCalculationError):
            error["calculation_failure"] = {**e.location, "execution":"FAILED_NO_RUN_SAVED"}
        print(json.dumps(error,sort_keys=True))
        return 2
    except (OSError, UnicodeError, OverflowError, ValueError, KeyError, TypeError) as e:
        # Fail closed without dumping private filesystem paths or dataset contents.
        print(json.dumps({"status":"ERROR","reason":"IO_OR_INPUT_FAILURE","kind":type(e).__name__,
                          **_recomputation_receipt_output(ns)},sort_keys=True))
        return 2

if __name__=="__main__":
    sys.exit(main())
