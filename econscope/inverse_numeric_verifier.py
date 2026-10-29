"""Independent arithmetic oracle for the existing bounded inverse screen.

Uses Leibniz determinants/Cramer's rule, never the estimator's elimination,
design builder or residual code. It checks every generated single/joint model,
rank, score, selection, coefficients, residuals and one-step holdout cells.
This is same-process verification of validated declarations, not external
replication, source authentication, statistical inference or upstream parity.
"""
from fractions import Fraction as F
from itertools import permutations
import sys

from .workspace import ScopeError, canonical, digest


def _require(condition):
    if not condition:
        raise ScopeError("INVERSE_NUMERIC_ORACLE_MISMATCH")


def _determinant(matrix):
    size = len(matrix)
    total = F(0)
    for order in permutations(range(size)):
        inversions = sum(order[i] > order[j] for i in range(size) for j in range(i+1, size))
        term = F(-1 if inversions % 2 else 1)
        for i, j in enumerate(order):
            term *= matrix[i][j]
        total += term
    return total


def verify_computed_screen(request, result):
    """Internal check after the existing input/rights/release validation.

    External saved results must go through verify_inverse_screen, which reruns
    the original validator and all calculations. This function grants no rights.
    """
    try:
        return _verify(request, result)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError) as exc:
        if isinstance(exc, ScopeError):
            raise
        raise ScopeError("INVERSE_NUMERIC_ORACLE_MISMATCH") from None


def _verify(request, result):
    study, search = request["study"], request["search"]
    rows = study["observations"]
    _require(24 <= len(rows) <= 128 and 1 <= len(search["lags"]) <= 3)
    start = max(search["lags"])
    end = next(i for i, row in enumerate(rows) if row["day"] == study["training_end_day"])
    _require(start <= 7 and end - start + 1 >= 12 and len(rows) - end - 1 >= 8)
    target = study["target"]
    names = [name for name in sorted(study["series_units"])
             if name != target or search["include_target_history"]]
    descriptors = [("screen_%02d" % (index+1), [{"series": name, "lag": lag}])
                   for index, (name, lag) in enumerate((name, lag) for name in names for lag in search["lags"])]
    _require(1 <= len(descriptors) <= 24)
    targets = [F(row["values"][target]) for row in rows]
    fits = {}
    expected = []

    def compute(candidate_id, features):
        # At most 3 parameters for this screen's singles + top-two joint;
        # this is an algorithm-specific resource contract, not an API cap.
        width = len(features) + 1
        _require(2 <= width <= 3)
        x = [[F(1)] + [F(rows[t-feature["lag"]]["values"][feature["series"]]) for feature in features]
             for t in range(start, len(rows))]
        training = x[:end-start+1]
        y = targets[start:end+1]
        gram = [[sum(row[i]*row[j] for row in training) for j in range(width)] for i in range(width)]
        rhs = [sum(row[j]*value for row, value in zip(training, y)) for j in range(width)]
        denominator = _determinant(gram)
        rec = dict(candidate_id=candidate_id, features=features, status="RANK_DEFICIENT")
        if denominator:
            beta = [_determinant([[rhs[r] if c == j else gram[r][c] for c in range(width)]
                                  for r in range(width)]) / denominator for j in range(width)]
            residuals = [value - sum(v*b for v, b in zip(row, beta)) for row, value in zip(training, y)]
            sse = sum(e*e for e in residuals)
            df = len(training) - width
            score = sse/df
            cells = []
            for t, row in zip(range(end+1, len(rows)), x[len(training):]):
                pred = sum(v*b for v, b in zip(row, beta))
                cells.append(dict(origin_day=rows[t-1]["day"], target_day=rows[t]["day"],
                                  prediction_exact=str(pred), observed_exact=str(targets[t]),
                                  residual_exact=str(targets[t]-pred)))
            mse = sum(F(cell["residual_exact"])**2 for cell in cells) / len(cells)
            rec.update(status="FIT", training_score_exact=str(score), training_sse_exact=str(sse),
                       training_degrees_of_freedom=df)
            fits[candidate_id] = dict(beta=beta, residuals=residuals, score=score,
                                      cells=cells, mse=mse, features=features)
        expected.append(rec)

    for candidate_id, features in descriptors:
        compute(candidate_id, features)
    singles = sorted((fit["score"], len(fit["features"]), key) for key, fit in fits.items())
    if len(singles) >= 2:
        joint = fits[singles[0][2]]["features"] + fits[singles[1][2]]["features"]
        compute("joint_top_two", joint)
    ranking = sorted((fit["score"], len(fit["features"]), key) for key, fit in fits.items())
    winner = ranking[0][2] if ranking else None
    _require(canonical(result["candidates"]) == canonical(expected))
    _require(result["candidate_count"] == len(expected) and result["common_max_lag"] == start)
    _require(result["selected_candidate_id"] == winner)
    _require(result["status"] == ("READY" if winner else "INDETERMINATE"))
    if winner is None:
        _require(result["selected_features"] == [] and result["selected_coefficients_exact"] is None
                 and result["selected_training_residuals_exact"] is None and result["holdout"] is None)
    else:
        fit = fits[winner]
        _require(result["selected_features"] == fit["features"])
        _require(result["selected_coefficients_exact"] == list(map(str, fit["beta"])))
        _require(result["selected_training_residuals_exact"] == list(map(str, fit["residuals"])))
        baseline = sum((targets[t]-targets[t-1])**2 for t in range(end+1, len(rows))) / (len(rows)-end-1)
        expected_holdout = dict(evaluation="RETROSPECTIVE_ROLLING_ONE_STEP_DECLARED_RELEASES_ONLY",
            model_mse_exact=str(fit["mse"]), naive_previous_target_mse_exact=str(baseline),
            rows=fit["cells"], was_used_for_selection=False)
        _require(canonical(result["holdout"]) == canonical(expected_holdout))
    receipt = dict(schema="societybox-inverse-numeric-verification-v1", status="PASS",
        input_sha256=digest(canonical(request)), method="EXACT_CRAMERS_RULE_LEIBNIZ_DETERMINANTS",
        candidates_checked=len(expected), fitted_candidates=len(fits),
        selected_candidate_id=winner, tolerance="EXACT_RATIONAL_EQUALITY",
        independence="SEPARATE_ARITHMETIC_IMPLEMENTATION_SAME_PROCESS_NOT_EXTERNAL_REPLICATION",
        runtime=dict(implementation=sys.implementation.name, python=sys.version.split()[0], platform=sys.platform),
        upstream_statistical_api="NOT_EXECUTED", source_authentication="NOT_ESTABLISHED",
        causal_identification="NOT_ESTABLISHED")
    receipt["verification_sha256"] = digest(canonical(receipt))
    return receipt
