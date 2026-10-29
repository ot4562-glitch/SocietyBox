"""Small transparent toy models, intended for reproducible computation not empirical inference."""
import csv
import io
import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN

from .workspace import ScopeError

MAX_ROWS = 10000
Q = Decimal("0.000001")


def value(text, lo, hi, code):
    try:
        d = Decimal(str(text))
    except (InvalidOperation, ValueError, TypeError):
        raise ScopeError(code) from None
    if not d.is_finite() or d < lo or d > hi:
        raise ScopeError(code)
    return d


def number(v):
    return str(v.quantize(Q, rounding=ROUND_HALF_EVEN))


def parse_csv(data, model):
    if len(data) > 32 * 1024 * 1024:
        raise ScopeError("CSV_TOO_LARGE")
    try:
        txt = data.decode("utf-8-sig")
    except UnicodeError:
        raise ScopeError("CSV_UTF8_REQUIRED") from None
    if "\x00" in txt:
        raise ScopeError("CSV_BINARY_REJECTED")
    if model == "tax_transfer":
        required = ("income", "propensity")
    elif model == "opinion_diffusion":
        required = ("opinion", "threshold")
    else:
        raise ScopeError("MODEL_UNKNOWN")
    stream = io.StringIO(txt, newline="")
    reader = csv.DictReader(stream)
    if reader.fieldnames != list(required):
        raise ScopeError("CSV_COLUMNS_MISMATCH")
    rows = []
    for row in reader:
        if len(rows) >= MAX_ROWS or None in row or any(v is None for v in row.values()):
            raise ScopeError("CSV_ROWS_INVALID")
        if model == "tax_transfer":
            row = {"income": value(row["income"], 0, 1_000_000_000, "INCOME_INVALID"),
                   "propensity": value(row["propensity"], 0, 1, "PROPENSITY_INVALID")}
        else:
            opinion = value(row["opinion"], 0, 1, "OPINION_INVALID")
            if opinion not in (0, 1):
                raise ScopeError("OPINION_NOT_BINARY")
            row = {"opinion": int(opinion),
                   "threshold": value(row["threshold"], 0, 1, "THRESHOLD_INVALID")}
        rows.append(row)
    if not rows:
        raise ScopeError("CSV_EMPTY")
    return rows


def synthetic_rows(model):
    if model == "tax_transfer":
        return [{"income": Decimal(x), "propensity": Decimal(p)}
                for x, p in [(100, ".7"), (150, ".8"), (250, ".9"), (350, ".7"), (1000, ".6")]]
    if model == "opinion_diffusion":
        return [{"opinion": x, "threshold": Decimal(".5")}
                for x in (0, 0, 1, 0, 0, 0, 1, 0, 0, 0)]
    raise ScopeError("MODEL_UNKNOWN")


def tax_transfer(rows, tax_rate):
    rate = value(tax_rate, 0, 1, "TAX_RATE_OUT_OF_RANGE")
    n = Decimal(len(rows))
    income = sum((r["income"] for r in rows), Decimal(0))
    total_tax = income * rate
    equal_transfer = total_tax / n
    after = [r["income"] * (1-rate) + equal_transfer for r in rows]
    consumption = sum((r["propensity"] * y for r, y in zip(rows, after)), Decimal(0))
    income_after_total = sum(after, Decimal(0))
    if abs(income_after_total-income) > Decimal("0.00001"):
        raise ScopeError("BUDGET_NOT_BALANCED")
    # Distribution after a balanced proportional tax and equal lump-sum transfer.
    ordered = sorted(after)
    # Ranked Gini computation is O(n log n), not an O(n^2) pairwise loop.
    weighted = sum(((Decimal(2*i-len(ordered)-1) * y)
                    for i, y in enumerate(ordered, start=1)), Decimal(0))
    gini = weighted / (n * income) if income else Decimal(0)
    return {"model": "tax_transfer", "people": len(rows),
            "tax_rate": number(rate), "total_income": number(income),
            "tax_revenue": number(total_tax), "equal_transfer": number(equal_transfer),
            "total_modeled_consumption": number(consumption),
            "after_tax_transfer_gini": number(gini),
            "budget_identity": "PASS",
            "limitations": [
                "Toy static equal-transfer model: income and spending propensities are fixed.",
                "No behavioral, labor-supply, price, equilibrium or causal-policy estimation."
            ]}


def opinion_diffusion(rows, *, steps, seed_share):
    if type(steps) is not int or not 0 <= steps <= 500:
        raise ScopeError("STEPS_INVALID")
    share = value(seed_share, 0, 1, "SEED_SHARE_INVALID")
    opinions = [r["opinion"] for r in rows]
    n = len(rows)
    # In this toy model the 'intervention' seeds the first floor(n * share) agents.
    injected = int((Decimal(n) * share).to_integral_value(rounding="ROUND_FLOOR"))
    for i in range(injected):
        opinions[i] = 1
    history = [sum(opinions)]
    for _ in range(steps):
        new = list(opinions)
        for i, row in enumerate(rows):
            if n < 3:
                continue
            neighbors = (opinions[(i-1)%n], opinions[(i+1)%n])
            opposite_fraction = Decimal(2-sum(neighbors))/2 if opinions[i] == 1 else Decimal(sum(neighbors))/2
            if opposite_fraction > 0 and opposite_fraction >= row["threshold"]:
                new[i] = 1-opinions[i]
        opinions = new
        history.append(sum(opinions))
    return {"model": "opinion_diffusion", "people": n, "steps": steps,
            "seed_share": number(share), "initial_adopters_after_intervention": history[0],
            "final_adopters": history[-1],
            "adopters_per_step": history,
            "limitations": [
                "Synthetic ring-neighbor binary-opinion toy model, not a calibrated social network.",
                "This does not estimate real opinions, persuasion, policy impact, or individual behavior."
            ]}


def compute(model, raw_data, params):
    rows = parse_csv(raw_data, model) if raw_data is not None else synthetic_rows(model)
    if model == "tax_transfer":
        if set(params) != {"tax_rate"}:
            raise ScopeError("MODEL_PARAMS_INVALID")
        return tax_transfer(rows, params["tax_rate"])
    if model == "opinion_diffusion":
        if set(params) != {"steps", "seed_share"}:
            raise ScopeError("MODEL_PARAMS_INVALID")
        return opinion_diffusion(rows, steps=params["steps"], seed_share=params["seed_share"])
    raise ScopeError("MODEL_UNKNOWN")
