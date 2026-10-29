"""Exact, independently authored static input-output impact experiment.

Prototype mathematical core for a common public economic method. This is NOT
IMPLAN compatibility, its proprietary data, observed employment/GDP effects,
a price model, or an authenticated real-world regional multiplier.

No proprietary imports, network, external data, LLM, solver, licensing bypass
or disk writes. Only its own trusted code bytes are read for provenance.
Per-invocation finite dimensions do not bound the product's eventual coverage.
"""
from fractions import Fraction as F
from hashlib import sha256
from itertools import permutations
from pathlib import Path
import re

from .research_evidence import make_evidence
from .workspace import ScopeError, canonical, digest

_SHA = re.compile(r"[a-f0-9]{64}\Z")
_NUM = re.compile(r"-?(?:0|[1-9][0-9]{0,9})(?:/[1-9][0-9]{0,9})?\Z")
_COLUMNS = ("schema", "sectors", "coefficient_matrix", "final_demand_change",
            "unit", "source_declared_sha256", "rights")
SCHEMA = "societybox-input-output-conditional-v1"


def _fail(code):
    raise ScopeError("INPUT_OUTPUT_" + code)


def _number(raw):
    if type(raw) is not str or not _NUM.fullmatch(raw):
        _fail("NUMBER_INVALID")
    return F(raw)


def _validate(q):
    if type(q) is not dict or set(q) != set(_COLUMNS) or q["schema"] != SCHEMA:
        _fail("SCHEMA_INVALID")
    names = q["sectors"]
    # This exact-math exploratory invocation is bounded, not the number of
    # sectors, geographies or economics APIs the eventual app may support.
    if (type(names) is not list or not 1 <= len(names) <= 8
            or any(type(name) is not str or not 1 <= len(name) <= 80
                   or any(ord(c) < 32 for c in name) for name in names)
            or len(set(names)) != len(names)):
        _fail("SECTORS_INVALID")
    if (type(q["unit"]) is not str or not 1 <= len(q["unit"]) <= 100
            or any(ord(c) < 32 for c in q["unit"])):
        _fail("UNIT_INVALID")
    if type(q["source_declared_sha256"]) is not str or not _SHA.fullmatch(q["source_declared_sha256"]):
        _fail("SOURCE_INVALID")
    rights = q["rights"]
    if (type(rights) is not dict or set(rights) != {"local_processing", "report_export"}
            or rights["local_processing"] != "PERMITTED"
            or rights["report_export"] != "PERMITTED"):
        _fail("RIGHTS_REQUIRED")
    n = len(names)
    rows = q["coefficient_matrix"]
    shock = q["final_demand_change"]
    if (type(rows) is not list or len(rows) != n
            or any(type(row) is not list or len(row) != n for row in rows)
            or type(shock) is not list or len(shock) != n):
        _fail("DIMENSIONS_INVALID")
    a = [[_number(x) for x in row] for row in rows]
    if any(x < 0 for row in a for x in row):
        _fail("INPUT_COEFFICIENT_NEGATIVE")
    d = [_number(x) for x in shock]
    # A[i,j] = input of sector i per unit of sector j's gross output.
    return names, a, d


def _inverse(matrix):
    n = len(matrix)
    aug = [[*row, *(F(int(i == j)) for j in range(n))]
           for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = next((row for row in range(column, n) if aug[row][column]), None)
        if pivot is None:
            return None
        aug[column], aug[pivot] = aug[pivot], aug[column]
        scale = aug[column][column]
        aug[column] = [value / scale for value in aug[column]]
        for i in range(n):
            if i != column:
                factor = aug[i][column]
                aug[i] = [v - factor * w for v, w in zip(aug[i], aug[column])]
    return [row[n:] for row in aug]


def _dot(left, right):
    return sum((x * y for x, y in zip(left, right)), F(0))


def _det_leibniz(matrix):
    """Independent determinant oracle for small matrices, not the inverse path."""
    n = len(matrix)
    total = F(0)
    for p in permutations(range(n)):
        inversions = sum(p[i] > p[j] for i in range(n) for j in range(i+1, n))
        product = F(-1 if inversions % 2 else 1)
        for i in range(n):
            product *= matrix[i][p[i]]
        total += product
    return total


def _oracle(b, inverse, demand, output):
    n = len(b)
    # Always check two-sided exact identities and original economic balance.
    if any(_dot(b[i], [inverse[j][k] for j in range(n)]) != F(int(i == k))
           for i in range(n) for k in range(n)):
        _fail("INVERSE_IDENTITY")
    if any(_dot([inverse[i][j] for j in range(n)], [b[j][k] for j in range(n)])
           != F(int(i == k)) for i in range(n) for k in range(n)):
        _fail("INVERSE_IDENTITY")
    if any(_dot(b[i], output) != demand[i] for i in range(n)):
        _fail("EXACT_BALANCE")
    if n > 5:
        return {"status": "EXACT_IDENTITIES_ONLY",
                "independent_determinant": "NOT_RUN_PER_INVOCATION_BUDGET"}
    determinant = _det_leibniz(b)
    if determinant == 0:
        _fail("NONPRODUCTIVE_SYSTEM")
    for j in range(n):
        replaced = [list(row) for row in b]
        for i in range(n):
            replaced[i][j] = demand[i]
        if _det_leibniz(replaced) != determinant * output[j]:
            _fail("CRAMER_PARITY")
    return {"status": "INDEPENDENT_CRAMER_AND_EXACT_IDENTITIES",
            "determinant_exact": str(determinant)}


def validate(q):
    """Validate declared inputs without calculating or granting execution."""
    _validate(q)


def calculate(q):
    """Compute scenario accounting, never a verified realized economic effect."""
    names, a, demand = _validate(q)
    # Evidence must refer to the executable's actual source, never a fixed
    # version-looking token that survives source changes unnoticed.
    source_code_sha256 = sha256(Path(__file__).read_bytes()).hexdigest()
    n = len(names)
    b = [[F(int(i == j)) - a[i][j] for j in range(n)] for i in range(n)]
    inverse = _inverse(b)
    # For nonnegative A, a nonnegative (I-A)^-1 implies a productive
    # nonsingular M-matrix (rho(A)<1), avoiding unstable multiplier stories.
    if inverse is None or any(v < 0 for row in inverse for v in row):
        _fail("NONPRODUCTIVE_SYSTEM")
    output = [_dot(row, demand) for row in inverse]
    indirect = [x - d for x, d in zip(output, demand)]
    oracle = _oracle(b, inverse, demand, output)
    if sha256(Path(__file__).read_bytes()).hexdigest() != source_code_sha256:
        _fail("CODE_CHANGED_DURING_RUN")
    subject = digest(canonical(q))
    evidence = make_evidence(
        subject_sha256=subject,
        producer={"tool_id": "societybox.static_io", "version": "prototype-1",
                  "code_sha256": source_code_sha256},
        operation="EXACT_STATIC_INPUT_OUTPUT_SCENARIO",
        authority="LOCAL_EXACT_ARITHMETIC", epistemic_state="DERIVED",
        proof_kind="FINITE_MATRIX_IDENTITY", scope={"sectors": list(names), "unit": q["unit"]},
        coverage={"status": "PARTIAL",
                  "unknown_dimensions": ["source_authenticity", "production_stability",
                                         "regional_data_rights", "induced_effect",
                                         "behavioral_or_causal_validity"]},
        result={"output_change_exact": list(map(str, output)),
                "oracle_status": oracle["status"]},
        parent_evidence_ids=[],
        limitations=["Input coefficients and final demand are researcher-declared",
                     "Static fixed technology; no prices, substitution, capacity or household closure",
                     "Direct and indirect are conditional accounting, not validated observed impact"])
    return {"schema": "societybox-input-output-result-v1", "status": "CONDITIONAL_SCENARIO",
            "source_authentication": "NOT_ESTABLISHED",
            "native_execution": "NOT_RUN",
            "economic_claim": "STATIC_FIXED_COEFFICIENT_ACCOUNTING_NOT_CAUSAL",
            "sector_names": list(names), "unit": q["unit"], "input_sha256": subject,
            "leontief_inverse_exact": [[str(v) for v in row] for row in inverse],
            "final_demand_direct_exact": list(map(str, demand)),
            "gross_output_change_exact": list(map(str, output)),
            "indirect_output_change_exact": list(map(str, indirect)),
            "total_direct_exact": str(sum(demand)),
            "total_indirect_exact": str(sum(indirect)),
            "total_change_exact": str(sum(output)),
            "induced_effect": "NOT_ESTIMATED_HOUSEHOLD_CLOSURE_UNDECLARED",
            "oracle": oracle, "evidence": evidence,
            "statistical_and_causal_validity": "NOT_ESTABLISHED"}


def verify(q, result):
    """Same-source deterministic recomputation, not external independence."""
    if type(result) is not dict or canonical(calculate(q)) != canonical(result):
        _fail("RECOMPUTATION_MISMATCH")
    return {"status": "RECOMPUTED", "scope": "SAME_CODE_AND_DECLARED_INPUT_ONLY"}

