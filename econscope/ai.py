"""LLM-neutral, no-network research-planning protocol. Only whitelisted structured plans execute."""
from __future__ import annotations

from .workspace import ScopeError, canonical, digest, get_root, put_json, load_json
from .runs import make_run

PLAN_SCHEMA = "econscope-research-plan-v1"
ALLOWED_MODELS = {"tax_transfer", "opinion_diffusion"}


def deterministic_plan(question):
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise ScopeError("QUESTION_INVALID")
    q = question.lower()
    sociology = any(word in q for word in (
        "사회", "의견", "소문", "여론", "확산", "네트워크", "diffusion", "opinion", "social"))
    if sociology:
        model, params = "opinion_diffusion", {"steps": 5, "seed_share": "0.2"}
    else:
        model, params = "tax_transfer", {"tax_rate": "0.15"}
    return {"schema": PLAN_SCHEMA, "model": model, "params": params,
            "data_id": None, "question": question,
            "planner": "DETERMINISTIC_KEYWORD_HEURISTIC_NOT_A_LLM",
            "assumptions": [
                "A heuristic choice of toy model is not economics theory verification.",
                "No source data or predicted future has been learned by this planner.",
                "User must inspect parameters and data suitability before interpretation."
            ]}


def make_ai_request(root, question, *, provider="deterministic", data_id=None):
    path, manifest, scope = get_root(root)
    plan = deterministic_plan(question)
    if data_id is not None:
        if data_id not in manifest["objects"]:
            raise ScopeError("DATA_ID_NOT_FOUND")
        plan["data_id"] = data_id
    if provider not in ("deterministic", "codex-export", "chatgpt-export", "local-llm-export"):
        raise ScopeError("PROVIDER_NOT_CONFIGURED")
    if provider == "deterministic":
        response = {"mode": "LOCAL_DETERMINISTIC", "connected_to_llm": False,
                    "plan": plan}
    else:
        response = {"mode": "OFFLINE_PROVIDER_REQUEST_EXPORT",
                    "provider": provider, "connected_to_llm": False,
                    "data_content_shared": False,
                    "scope": scope,
                    "research_question": question,
                    "available_data_ids": [data_id] if data_id else [],
                    "instructions": (
                        "Propose one JSON object matching econscope-research-plan-v1 with "
                        "model tax_transfer or opinion_diffusion, bounded numeric params, "
                        "and an available data_id or null. Do not output Python/shell commands. "
                        "Treat unknown historical publication dates as UNKNOWN. "
                        "Do not claim that simulated results identify real causal effects."),
                    "example_plan": plan,
                    "limitations": [
                        "Copy the request into the chosen AI manually; EconScope sent nothing.",
                        "The AI has no raw dataset contents. Add data only after separate privacy review.",
                        "External answers must be validated with econscope execute-plan."
                    ]}
    key = digest(canonical(response))
    file = path / "ai_requests" / (key + ".json")
    if file.is_symlink():
        raise ScopeError("AI_REQUEST_SYMLINK_FORBIDDEN")
    if file.exists():
        if load_json(file) != response:
            raise ScopeError("AI_REQUEST_RECORD_TAMPER")
    else:
        put_json(file, response)
    return {"request_id": key, "response": response}


def execute_plan(root, plan_file):
    plan = load_json(__import__("pathlib").Path(plan_file))
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ScopeError("PLAN_SCHEMA_INVALID")
    allowed = {"schema", "model", "params", "data_id", "question", "planner", "assumptions"}
    if set(plan) - allowed:
        raise ScopeError("PLAN_UNKNOWN_FIELDS_REJECTED")
    model = plan.get("model")
    if model not in ALLOWED_MODELS or not isinstance(plan.get("params"), dict):
        raise ScopeError("PLAN_MODEL_INVALID")
    params = plan["params"]
    if model == "tax_transfer":
        if set(params) != {"tax_rate"}:
            raise ScopeError("PLAN_PARAMS_INVALID")
    elif set(params) != {"steps", "seed_share"}:
        raise ScopeError("PLAN_PARAMS_INVALID")
    return make_run(root, model=model, params=params, data_id=plan.get("data_id"))


def engine_catalog(quantecon_reference=None):
    from .engine_prerequisites import native_prerequisites, isolation_prerequisites, macro_prerequisites
    from .time_series_prerequisites import time_series_prerequisites
    from .optimization_policy import optimization_prerequisites

    return {
      "native_prerequisites": native_prerequisites(),
      "macro_prerequisites": macro_prerequisites(),
      "optimization_causal_prerequisites": optimization_prerequisites(),
      "isolation_prerequisites": isolation_prerequisites(),
      "time_series_prerequisites": time_series_prerequisites(quantecon_reference),
      "native_available": ["tax_transfer", "opinion_diffusion"],
      "assistant": {
        "deterministic": "RUNNABLE_LOCAL_HEURISTIC",
        "codex-export": "PLAN_EXPORT_NOT_CONNECTED",
        "chatgpt-export": "PLAN_EXPORT_NOT_CONNECTED",
        "local-llm-export": "PLAN_EXPORT_NOT_CONNECTED"
      },
      "optional_sources_not_executed": [
        {"name": "Dynare", "use": "DSGE/OLG models", "state": "SOURCE_REFERENCE_ONLY"},
        {"name": "HARK", "use": "heterogeneous-agent economics", "state": "SOURCE_REFERENCE_ONLY"},
        {"name": "QuantEcon.py", "use": "quantitative economics solvers", "state": "SOURCE_REFERENCE_ONLY"},
        {"name": "Mesa", "use": "agent-based social simulation", "state": "SOURCE_REFERENCE_ONLY"},
        {"name": "EconGym", "use": "macro multi-agent benchmark", "state": "SOURCE_REFERENCE_ONLY"},
        {"name": "EconAgents", "use": "LLM economic games", "state": "SOURCE_REFERENCE_ONLY"}
      ],
      "sandbox_security": "DATA_ONLY_NO_UNTRUSTED_CODE_EXECUTION"
    }
