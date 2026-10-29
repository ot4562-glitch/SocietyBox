"""Deterministic computational lineage, never an evidence truth certificate.

Callers must verify the original receipt before constructing or checking a graph.
Edges point from a computed observation to its prerequisites. Dependencies are
syntactic; they do not establish causal identification or scientific validity.
"""
import ast
import copy

from .workspace import ScopeError, canonical, digest


def build_claim_graph(record, run_id):
    if digest(canonical(record)) != run_id:
        raise ScopeError("STUDY_RUN_RECEIPT_TAMPERED")
    return _project_claim_graph(record, run_id)


def _project_claim_graph(record, run_id):
    """Internal projection only; does not authenticate an exported receipt."""
    spec, results = record["spec"], record["results"]
    model = spec["model"]
    nodes, edges = {}, set()

    def node(key, kind, **fields):
        nodes[key] = {"kind": kind, **copy.deepcopy(fields)}
        return key

    def link(key, dependencies):
        edges.update((key, dep) for dep in dependencies)

    def basis(key, definition):
        link(key, ["evidence/" + ref for ref in definition["evidence"]])

    def inputs(expr, step):
        names = {n.id for n in ast.walk(ast.parse(expr, mode="eval"))
                 if isinstance(n, ast.Name)}
        return ["parameter/" + name if name in model["parameters"]
                else f"cell/{step}/{name}" for name in sorted(names)
                if name in model["parameters"] or name in model["states"]]

    for name, entry in sorted(spec["evidence"].items()):
        node("evidence/" + name, "source_or_assumption", declaration=entry,
             verification="NOT_INDEPENDENTLY_VERIFIED")
    for name, entry in sorted(model["parameters"].items()):
        key = node("parameter/" + name, "effective_parameter", definition=entry,
                   value=results["parameters"][name],
                   overridden=name in record["overrides"])
        basis(key, entry)
    for name, entry in sorted(model["states"].items()):
        key = node("initial/" + name, "initial_definition", definition=entry,
                   source_lineage=results["source_lineage"][name])
        basis(key, entry)
    for name, rule in sorted(model["rules"].items()):
        key = node("rule/" + name, "transition_definition", definition=rule,
                   timing="SYNCHRONOUS_PREVIOUS_STEP_STATE")
        basis(key, rule)
    for event in model["events"]:
        key = node("detector/" + event["id"], "detector_definition", definition=event,
                   timing="POST_TRANSITION_STATE")
        basis(key, event)
    for index, row in enumerate(results["trajectory"]):
        for name, value in sorted(row["state"].items()):
            key = node(f"cell/{row['step']}/{name}", "state_cell", value=value,
                       unit=model["states"][name]["unit"], step=row["step"],
                       receipt_pointer=f"/results/trajectory/{index}/state/{name}",
                       report_anchor="computed-state-table")
            if row["step"] == 0:
                link(key, ["initial/" + name])
            else:
                rule = model["rules"][name]
                link(key, ["rule/" + name] + inputs(rule["expr"], row["step"] - 1))
    for index, event in enumerate(results["events"]):
        key = node(f"event/{index}", "computed_event", observation=event,
                   receipt_pointer=f"/results/events/{index}",
                   report_anchor=f"claim-event-{index}")
        link(key, ["detector/" + event["event_id"]] + inputs(event["detector"], event["step"]))
    return {"schema": "societybox-claim-graph-v1", "run_id": run_id,
            "spec_sha256": record["spec_sha256"],
            "engine_code_sha256": record["engine_code_sha256"],
            "scope": "COMPUTATIONAL_LINEAGE_NOT_EMPIRICAL_OR_CAUSAL_VALIDATION",
            "time_unit": model["time_unit"],
            "nodes": nodes,
            "edges": [{"from": a, "to": b} for a, b in sorted(edges)]}


def validate_claim_graph(graph, record, run_id):
    """Reject altered/missing/extra nodes, values, edges, or report references."""
    if canonical(graph) != canonical(build_claim_graph(record, run_id)):
        raise ScopeError("STUDY_CLAIM_GRAPH_MISMATCH")
    return {"valid": True, "scope": "EXACT_RECEIPT_PROJECTION_NOT_SOURCE_VERIFICATION"}
