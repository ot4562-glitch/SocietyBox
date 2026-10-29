"""Conservative declarative research calculations.

A model is written by a researcher/LLM, never selected from an economic template.
This limited expression interpreter does NOT execute generated Python, and is not
a container or a hardened sandbox for other files or libraries.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from decimal import (Context, Decimal, DecimalException, InvalidOperation,
                     DivisionByZero, Overflow, Underflow, ROUND_HALF_EVEN, localcontext)
import re

from .workspace import ScopeError, SHA, canonical, digest, retrieve, admission
from .study_units import audit_expression
from .study_csv_input import rows as csv_rows

SCHEMA = "societybox-study-v1"
ENGINE = "societybox-scalar-v0.1"
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,39}\Z")
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
LIMIT = Decimal("1000000000000")
EVIDENCE_TYPES = {"USER_ASSUMPTION", "USER_DATA_UNVERIFIED", "LITERATURE_UNVERIFIED"}
MODES = {"THOUGHT_EXPERIMENT", "HISTORICAL_RECONSTRUCTION", "COUNTERFACTUAL"}


class StudyCalculationError(ScopeError):
    """A failed model operation, with identifiers but no input values or paths."""

    def __init__(self, code, phase, step, target):
        super().__init__(code)
        self.location = {"phase": phase, "step": step, "target": target}


@contextmanager
def calculation_stage(phase, step, target):
    try:
        yield
    except (ScopeError, DecimalException) as exc:
        code = exc.code if isinstance(exc, ScopeError) else "STUDY_ARITHMETIC_ERROR"
        raise StudyCalculationError(code, phase, step, target) from exc


def fail(reason):
    raise ScopeError(reason)


def exact(item, required, optional, reason):
    if not isinstance(item, dict) or not required <= item.keys() or not item.keys() <= (required | optional):
        fail(reason)


def bounded_str(s, maximum=500, minimum=1):
    if not isinstance(s, str) or not minimum <= len(s.strip()) <= maximum:
        fail("STUDY_TEXT_INVALID")
    return s


def name(s):
    if not isinstance(s, str) or not NAME.fullmatch(s) or s in {"min", "max", "abs"}:
        fail("STUDY_VARIABLE_NAME_INVALID")


def number(value):
    if type(value) not in (str, int, Decimal):
        fail("STUDY_NUMBER_MUST_BE_DECIMAL_STRING")
    try:
        val = Decimal(str(value))
    except (InvalidOperation, ValueError):
        fail("STUDY_NUMBER_INVALID")
    if not val.is_finite() or val.copy_abs() > LIMIT:
        fail("STUDY_NUMBER_NONFINITE_OR_TOO_LARGE")
    return val


def finite(val):
    if not isinstance(val, Decimal) or not val.is_finite() or val.copy_abs() > LIMIT:
        fail("STUDY_RESULT_OUT_OF_BOUNDS")
    return val


def refs(ls, evidence):
    if not isinstance(ls, list) or not 1 <= len(ls) <= 12:
        fail("STUDY_EVIDENCE_REF_REQUIRED")
    if len(ls) != len(set(str(v) for v in ls)):
        fail("STUDY_EVIDENCE_REF_DUPLICATE")
    if any(type(x) is not str or x not in evidence for x in ls):
        fail("STUDY_EVIDENCE_REF_UNKNOWN")


def expression(text, allowed, event=False):
    if not isinstance(text, str) or not 1 <= len(text) <= 180:
        fail("STUDY_EXPRESSION_LENGTH_INVALID")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        fail("STUDY_EXPRESSION_SYNTAX")
    nodes = list(ast.walk(tree))
    if len(nodes) > 60:
        fail("STUDY_EXPRESSION_TOO_COMPLEX")
    ok = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
        ast.Div, ast.UAdd, ast.USub, ast.Constant, ast.Name, ast.Load,
        ast.Call, ast.Compare, ast.Gt, ast.GtE, ast.Lt, ast.LtE,
        ast.Eq, ast.NotEq, ast.BoolOp, ast.And, ast.Or
    )
    for n in nodes:
        if not isinstance(n, ok):
            fail("STUDY_EXPRESSION_SYNTAX_FORBIDDEN")
        if isinstance(n, ast.Name) and n.id not in allowed and n.id not in {"min", "max", "abs"}:
            fail("STUDY_EXPRESSION_NAME_UNKNOWN")
        if isinstance(n, ast.Constant):
            if type(n.value) is not int:
                # Noninteger constants must be declared as sourced parameters.
                fail("STUDY_DECIMAL_LITERAL_NEEDS_SOURCED_PARAMETER")
            number(n.value)
        if isinstance(n, ast.Call):
            if (not isinstance(n.func, ast.Name) or n.func.id not in {"min","max","abs"}
                or n.keywords or not 1 <= len(n.args) <= 6):
                fail("STUDY_FUNCTION_CALL_NOT_ALLOWED")
            if n.func.id == "abs" and len(n.args) != 1:
                fail("STUDY_ABS_ARITY_INVALID")
        if isinstance(n, ast.Compare) and len(n.ops) != 1:
            fail("STUDY_CHAIN_COMPARISON_UNSUPPORTED")
    if event and not any(isinstance(n, ast.Compare) for n in nodes):
        fail("STUDY_EVENT_MUST_COMPARE")
    return tree


def evaluate(node, env):
    if isinstance(node, ast.Expression):
        return evaluate(node.body, env)
    if isinstance(node, ast.Constant):
        return number(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            fail("STUDY_EXPRESSION_NAME_UNKNOWN")
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        v = evaluate(node.operand, env)
        if type(v) is bool:
            fail("STUDY_NUMERIC_TYPE_REQUIRED")
        return finite(+v if isinstance(node.op, ast.UAdd) else -v)
    if isinstance(node, ast.BinOp):
        a, b = evaluate(node.left, env), evaluate(node.right, env)
        if type(a) is bool or type(b) is bool:
            fail("STUDY_NUMERIC_TYPE_REQUIRED")
        try:
            if isinstance(node.op, ast.Add):
                return finite(a+b)
            if isinstance(node.op, ast.Sub):
                return finite(a-b)
            if isinstance(node.op, ast.Mult):
                return finite(a*b)
            if isinstance(node.op, ast.Div):
                if not b:
                    fail("STUDY_DIVISION_BY_ZERO")
                return finite(a/b)
        except (InvalidOperation, DivisionByZero, Overflow):
            fail("STUDY_ARITHMETIC_ERROR")
        fail("STUDY_OPERATOR_NOT_ALLOWED")
    if isinstance(node, ast.Call):
        args = [evaluate(x, env) for x in node.args]
        if any(type(x) is bool for x in args):
            fail("STUDY_NUMERIC_TYPE_REQUIRED")
        if node.func.id == "min": return finite(min(args))
        if node.func.id == "max": return finite(max(args))
        if node.func.id == "abs": return finite(abs(args[0]))
        fail("STUDY_FUNCTION_NOT_ALLOWED")
    if isinstance(node, ast.Compare):
        a, b = evaluate(node.left, env), evaluate(node.comparators[0], env)
        if type(a) is bool or type(b) is bool:
            fail("STUDY_NUMERIC_TYPE_REQUIRED")
        op = node.ops[0]
        if isinstance(op, ast.Gt): return a > b
        if isinstance(op, ast.GtE): return a >= b
        if isinstance(op, ast.Lt): return a < b
        if isinstance(op, ast.LtE): return a <= b
        if isinstance(op, ast.Eq): return a == b
        if isinstance(op, ast.NotEq): return a != b
        fail("STUDY_OPERATOR_NOT_ALLOWED")
    if isinstance(node, ast.BoolOp):
        args = [evaluate(x, env) for x in node.values]
        if any(type(x) is not bool for x in args):
            fail("STUDY_BOOLEAN_TYPE_REQUIRED")
        return all(args) if isinstance(node.op, ast.And) else any(args)
    fail("STUDY_EXPRESSION_SYNTAX_FORBIDDEN")


def validate(spec):
    exact(spec,
          {"schema", "question", "title", "inference", "evidence", "model", "limitations"},
          {"export_rights"}, "STUDY_SCHEMA_INVALID")
    if spec["schema"] != SCHEMA:
        fail("STUDY_SCHEMA_VERSION_INVALID")
    bounded_str(spec["question"], 1500, 10)
    bounded_str(spec["title"], 160, 5)
    if not isinstance(spec["inference"], str) or spec["inference"] not in MODES:
        fail("STUDY_INFERENCE_INVALID")
    if not isinstance(spec["limitations"], list) or not 1 <= len(spec["limitations"]) <= 20:
        fail("STUDY_LIMITATIONS_REQUIRED")
    for v in spec["limitations"]:
        bounded_str(v, 500, 8)
    evidence = spec["evidence"]
    if not isinstance(evidence, dict) or not 1 <= len(evidence) <= 120:
        fail("STUDY_EVIDENCE_REQUIRED")
    for k, v in evidence.items():
        if not isinstance(k, str) or not ID.fullmatch(k):
            fail("STUDY_EVIDENCE_ID_INVALID")
        exact(v, {"type","description"}, {"locator","data_id","export_rights","processing_rights"}, "STUDY_EVIDENCE_SCHEMA_INVALID")
        if not isinstance(v["type"], str) or v["type"] not in EVIDENCE_TYPES:
            fail("STUDY_EVIDENCE_TRUST_ESCALATION")
        bounded_str(v["description"], 800, 8)
        if "locator" in v: bounded_str(v["locator"], 400)
        if "data_id" in v:
            if v["type"] != "USER_DATA_UNVERIFIED" or not isinstance(v["data_id"], str) or not SHA.fullmatch(v["data_id"]):
                fail("STUDY_DATA_EVIDENCE_INVALID")
        if "processing_rights" in v:
            rights = v["processing_rights"]
            exact(rights, {"local_processing", "basis"}, set(), "STUDY_DATA_PROCESSING_RIGHTS_INVALID")
            if ("data_id" not in v or
                    not isinstance(rights["local_processing"], str) or
                    rights["local_processing"] not in {"PERMITTED", "DENIED", "UNKNOWN"} or
                    not isinstance(rights["basis"], str) or
                    not 1 <= len(rights["basis"]) <= 800 or not rights["basis"].strip()):
                fail("STUDY_DATA_PROCESSING_RIGHTS_INVALID")
    model = spec["model"]
    exact(model, {"states","parameters","rules","events","steps","time_unit"},
          set(), "STUDY_MODEL_SCHEMA_INVALID")
    if type(model["steps"]) is not int or not 1 <= model["steps"] <= 200:
        fail("STUDY_STEP_BUDGET_INVALID")
    if not isinstance(model["time_unit"], str) or model["time_unit"] not in {"step", "day", "month", "quarter", "year"}:
        fail("STUDY_TIME_UNIT_INVALID")
    states, params, rules, events = (model[k] for k in ("states","parameters","rules","events"))
    if not isinstance(states, dict) or not 1 <= len(states) <= 24:
        fail("STUDY_STATES_REQUIRED")
    if not isinstance(params, dict) or len(params) > 36:
        fail("STUDY_PARAMS_INVALID")
    for k in set(states) | set(params):
        name(k)
    if set(states) & set(params):
        fail("STUDY_VARIABLE_COLLISION")
    if not isinstance(rules, dict) or set(rules) != set(states):
        fail("STUDY_RULE_MUST_COVER_ALL_STATES")
    allowed = set(states) | set(params)
    for k, v in states.items():
        exact(v, {"initial","unit","evidence"}, {"min","max"}, "STUDY_STATE_SCHEMA_INVALID")
        refs(v["evidence"], evidence)
        bounded_str(v["unit"], 80)
        if type(v["initial"]) is dict:
            exact(v["initial"], {"data_id","column","aggregate"}, set(), "STUDY_DATA_INITIAL_SCHEMA_INVALID")
            inp=v["initial"]
            if not isinstance(inp["data_id"], str) or not SHA.fullmatch(inp["data_id"]):
                fail("STUDY_DATA_INITIAL_ID_INVALID")
            bounded_str(inp["column"], 100)
            if not isinstance(inp["aggregate"], str) or inp["aggregate"] not in {"mean", "sum", "min", "max", "count"}:
                fail("STUDY_AGGREGATION_INVALID")
            if not any(evidence[ref].get("data_id")==inp["data_id"] for ref in v["evidence"]):
                fail("STUDY_INITIAL_DATA_UNCITED")
        else:
            number(v["initial"])
        for z in ("min", "max"):
            if z in v: number(v[z])
        if "min" in v and "max" in v and number(v["min"]) > number(v["max"]):
            fail("STUDY_BOUNDS_INVALID")
    for k,v in params.items():
        exact(v, {"value","unit","evidence"}, set(), "STUDY_PARAMETER_SCHEMA_INVALID")
        number(v["value"])
        bounded_str(v["unit"], 80)
        refs(v["evidence"], evidence)
    labels={**{k:v["unit"] for k,v in states.items()}, **{k:v["unit"] for k,v in params.items()}}
    for k, v in rules.items():
        exact(v, {"expr","evidence"}, set(), "STUDY_RULE_SCHEMA_INVALID")
        refs(v["evidence"], evidence)
        parsed=expression(v["expr"], allowed)
        audit_expression(parsed, labels, expected_unit=states[k]["unit"])
    if not isinstance(events, list) or len(events)>24:
        fail("STUDY_EVENT_LIST_INVALID")
    used=set()
    for e in events:
        exact(e, {"id","label","when","evidence"}, set(), "STUDY_EVENT_SCHEMA_INVALID")
        if not isinstance(e["id"],str) or not ID.fullmatch(e["id"]) or e["id"] in used:
            fail("STUDY_EVENT_ID_INVALID")
        used.add(e["id"])
        bounded_str(e["label"], 160)
        refs(e["evidence"], evidence)
        audit_expression(expression(e["when"], allowed, event=True), labels, event=True)
    return {
        "valid_for_bounded_computation": True,
        "model_hash": digest(canonical(spec)),
        "engine": ENGINE,
        "research_status": "CONDITIONAL_SIMULATION_NOT_EMPIRICALLY_VALIDATED",
        "limitations": [
            "Theoretical mechanisms and source claims were not independently validated.",
            "Only symbolic units are checked; physical unit meaning, conversions and simulation time scaling are not verified.",
            "This engine supports scalar discrete-time expressions only; no generated Python is executed.",
            "Computational repeatability does not establish causal identification or external validity."
        ]
    }


def value_in_bounds(val, state):
    finite(val)
    if "min" in state and val < number(state["min"]):
        fail("STUDY_STATE_BELOW_BOUND")
    if "max" in state and val > number(state["max"]):
        fail("STUDY_STATE_ABOVE_BOUND")


def dataset_availability(item, scope):
    """Check declared publication timing, without authenticating its truth.

    This is a dataset-level claim only: observation dates, revisions, vintages
    and row-level information sets are not supported temporal validations.
    """
    availability=admission(item, scope)
    if scope.get("as_of") and scope.get("strict_historical") and availability in (
            "UNKNOWN_PUBLICATION_DATE","CLAIMED_RELEASE_AFTER_CUTOFF"):
        fail("STUDY_DATA_NOT_AVAILABLE_AT_CUTOFF")
    return availability


def dataset_references(spec):
    """All dataset IDs, including evidence not used in numeric aggregation."""
    data_ids = {entry["data_id"] for entry in spec["evidence"].values() if "data_id" in entry}
    data_ids.update(state["initial"]["data_id"] for state in spec["model"]["states"].values()
                    if isinstance(state["initial"], dict))
    return sorted(data_ids)


def temporal_assurance(scope=None, data_ids=None):
    """Describe the narrow publication check, never historical truth assurance.

    Supply data_ids only after workspace checking, or for a computed receipt.
    None means checks were not run; an empty list means no data were referenced.
    """
    status = "NOT_RUN"
    if scope is not None and data_ids is not None:
        if not data_ids:
            status = "NOT_APPLICABLE_NO_DATASETS"
        elif not scope.get("as_of"):
            status = "NOT_APPLICABLE_NO_CUTOFF"
        elif not scope.get("strict_historical"):
            status = "NOT_ENFORCED_NON_STRICT_SCOPE"
        else:
            status = "PASS_DECLARED_DATES_ONLY"
    return {"publication_cutoff_check": status,
            "as_of": scope.get("as_of") if scope is not None else None,
            "publication_authentication": "NOT_PERFORMED",
            "row_information_sets": "UNSUPPORTED",
            "revision_vintages": "UNSUPPORTED",
            "observation_dates": "UNSUPPORTED",
            "leakage_free_prediction": "NOT_ESTABLISHED"}


def check_processing_rights(spec):
    """Honor every source restriction, without authenticating legal rights.

    Every referenced dataset needs an explicit permitted processing declaration.
    Check all aliases before reading any data, including evidence-only inputs.
    Call after structural validation.
    """
    for entry in spec["evidence"].values():
        rights = entry.get("processing_rights")
        if "data_id" in entry:
            if rights is None or rights['local_processing'] != 'PERMITTED':
                fail("STUDY_DATA_PROCESSING_RIGHTS_DENIED_OR_UNKNOWN")


def review_datasets(root, spec, scope):
    """Check every declared data reference, including evidence-only inputs."""
    check_processing_rights(spec)
    data_ids = dataset_references(spec)
    for key in sorted(data_ids):
        _, item, _ = retrieve(root, key)
        dataset_availability(item, scope)
    return sorted(data_ids)


def aggregate_from_dataset(root, cfg, scope):
    # Keep direct callers from silently treating unsupported operations as max.
    op = cfg.get("aggregate")
    if not isinstance(op, str) or op not in {"mean", "sum", "min", "max", "count"}:
        fail("STUDY_AGGREGATION_INVALID")
    raw, item, _ = retrieve(root, cfg["data_id"])
    availability = dataset_availability(item, scope)
    if len(raw) > 4*1024*1024:
        fail("STUDY_CSV_SIZE_UNSUPPORTED")
    try: s=raw.decode("utf-8-sig")
    except UnicodeError: fail("STUDY_CSV_UTF8_REQUIRED")
    if "\x00" in s: fail("STUDY_CSV_BINARY_FORBIDDEN")
    reader=csv_rows(s)
    headers=next(reader, None)
    if not headers or len(headers)!=len(set(headers)) or cfg["column"] not in headers:
        fail("STUDY_DATA_COLUMN_MISSING_OR_DUPLICATE")
    column=headers.index(cfg["column"])
    vals=[]
    for row in reader:
        if not row:  # Standard DictReader ignores empty physical records.
            continue
        if len(vals)>=10000 or len(row)!=len(headers):
            fail("STUDY_CSV_INVALID_ROW_OR_LIMIT")
        vals.append(number(row[column]) if op!="count" else Decimal(1))
    if not vals: fail("STUDY_DATA_NO_ROWS")
    if op in ("sum","count"): v=sum(vals,Decimal(0))
    elif op=="mean": v=sum(vals,Decimal(0))/Decimal(len(vals))
    elif op=="min": v=min(vals)
    else: v=max(vals)
    return finite(v), {"data_id":cfg["data_id"],"column":cfg["column"],
       "aggregate":op,"rows":len(vals),"availability":availability,
       "publisher_verified":False,"sha256":digest(raw)}


def numerical_context():
    """A fresh execution context, also the source of policy disclosures."""
    return Context(prec=28, rounding=ROUND_HALF_EVEN, Emin=-999999,
                   Emax=999999, capitals=1, clamp=0, flags=[],
                   traps=[InvalidOperation, DivisionByZero, Overflow, Underflow])


def numerical_assurance():
    """Fixed arithmetic policy, never a per-run accuracy certificate."""
    context = numerical_context()
    return {
        "policy": {"arithmetic": "DECIMAL", "precision": context.prec,
                   "rounding": context.rounding, "Emin": context.Emin,
                   "Emax": context.Emax, "clamp": context.clamp,
                   "traps": sorted(signal.__name__ for signal, enabled in context.traps.items() if enabled),
                   "absolute_value_limit": str(LIMIT),
                   "ordinary_inexact_arithmetic": "ROUNDED_NOT_REJECTED",
                   "csv_sum_reduction": "LEFT_TO_RIGHT_FROM_ZERO_IN_FILE_ORDER",
                   "csv_mean_reduction": "CSV_SUM_THEN_DIVIDE_BY_ROW_COUNT",
                   "csv_order_invariance": "NOT_GUARANTEED",
                   "exact_subnormals": "PERMITTED",
                   "declared_state_bounds": "ENFORCED"},
        "scope": "FIXED_EXECUTION_POLICY_NOT_PER_RUN_ERROR_MEASUREMENT",
        "convergence": "UNSUPPORTED",
        "stability": "UNSUPPORTED",
        "error_bounds": "UNSUPPORTED",
        "numerical_accuracy": "NOT_ESTABLISHED",
    }


def calculate(spec, root, overrides=None):
    # Own the entire numeric context, including CSV initialization, rather than
    # inheriting a caller's rounding, exponent bounds or traps. Inexact ordinary
    # division is permitted at 28 digits; tiny inexact underflow must not silently
    # become a zero-valued state or event input.
    with localcontext(numerical_context()) as context:
        try:
            result = _calculate(spec, root, overrides)
            result['numerical_diagnostics'] = {
                'raised_flags': sorted(signal.__name__ for signal, raised in context.flags.items() if raised),
                'scope': 'WHOLE_CALCULATION_STICKY_FLAGS_NOT_ERROR_BOUNDS',
                'numerical_accuracy': 'NOT_ESTABLISHED',
            }
            return result
        except DecimalException:
            fail("STUDY_ARITHMETIC_ERROR")


def _calculate(spec, root, overrides=None):
    validate(spec)
    from .workspace import get_root
    _,_,scope=get_root(root)
    review_datasets(root, spec, scope)
    m=spec["model"]
    overrides={} if overrides is None else overrides
    if not isinstance(overrides,dict) or set(overrides)-set(m["parameters"]):
        fail("STUDY_OVERRIDE_UNDECLARED")
    params={k:number(v["value"]) for k,v in m["parameters"].items()}
    for k,v in overrides.items(): params[k]=number(v)
    states={}
    source_lineage={}
    for key,item in m["states"].items():
        # Report only the validated state identifier, never source cells,
        # column names or paths, just as for later model operations.
        with calculation_stage("INITIALIZATION", 0, key):
            inp=item["initial"]
            if isinstance(inp,dict):
                states[key],source_lineage[key]=aggregate_from_dataset(root,inp,scope)
            else:
                states[key]=number(inp)
                source_lineage[key]={"kind":"USER_ASSUMPTION","refs":item["evidence"]}
            value_in_bounds(states[key],item)
    rule_trees={k:expression(v["expr"],set(states)|set(params)) for k,v in m["rules"].items()}
    event_trees={e["id"]:expression(e["when"],set(states)|set(params),event=True) for e in m["events"]}
    history=[{"step":0,"state":{k:str(v) for k,v in sorted(states.items())}}]
    events=[]
    for step in range(1,m["steps"]+1):
        prev={**states,**params}
        next_state={}
        for key,tree in rule_trees.items():
            with calculation_stage("TRANSITION", step, key):
                val=evaluate(tree,prev)
                if type(val) is bool: fail("STUDY_RULE_MUST_RETURN_NUMBER")
                value_in_bounds(val,m["states"][key])
            next_state[key]=val
        states=next_state
        history.append({"step":step,"state":{k:str(v) for k,v in sorted(states.items())}})
        env={**states,**params}
        for ev in m["events"]:
            with calculation_stage("DETECTOR", step, ev["id"]):
                val=evaluate(event_trees[ev["id"]],env)
                if type(val) is not bool: fail("STUDY_EVENT_MUST_BE_BOOLEAN")
            if val:
                vars_used=sorted(set(states)&{z.id for z in ast.walk(event_trees[ev["id"]]) if isinstance(z,ast.Name)})
                events.append({"step":step,"event_id":ev["id"],"label":ev["label"],
                               "detector":ev["when"],"refs":ev["evidence"],
                               "trigger_state":{k:str(states[k]) for k in vars_used},
                               "event_scope":"INSIDE_DECLARED_MODEL_ONLY"})
                if len(events)>600: fail("STUDY_EVENT_OUTPUT_LIMIT")
    return {"trajectory":history,"events":events,"source_lineage":source_lineage,
            "parameters":{k:str(v) for k,v in sorted(params.items())},
            "time_unit":m["time_unit"],"steps":m["steps"],
            "scientific_scope":"MODEL_CONDITIONAL_NOT_REAL_WORLD_FORECAST"}
