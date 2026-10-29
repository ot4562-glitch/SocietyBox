"""Independent public mathematics, not proprietary input-output product parity."""
import copy
from hashlib import sha256
from pathlib import Path
import unittest
from unittest.mock import patch

from econscope import input_output_impact as io
from econscope.workspace import ScopeError


def case():
    return {"schema": io.SCHEMA,
            "sectors": ["steel", "machinery"],
            # Column j purchases A[i][j] units from supplier sector i.
            "coefficient_matrix": [["0", "1/2"], ["1/4", "0"]],
            "final_demand_change": ["100", "0"],
            "unit": "declared million currency units (fixed-price scenario)",
            "source_declared_sha256": "a" * 64,
            "rights": {"local_processing": "PERMITTED", "report_export": "PERMITTED"}}


class InputOutputImpactTests(unittest.TestCase):
    def test_exact_two_sector_leontief_decomposition_and_inverse(self):
        result = io.calculate(case())
        self.assertEqual(result["native_execution"], "NOT_RUN")
        self.assertEqual(result["source_authentication"], "NOT_ESTABLISHED")
        self.assertEqual(result["oracle"]["status"], "INDEPENDENT_CRAMER_AND_EXACT_IDENTITIES")
        self.assertEqual(result["leontief_inverse_exact"],
                         [["8/7", "4/7"], ["2/7", "8/7"]])
        self.assertEqual(result["gross_output_change_exact"], ["800/7", "200/7"])
        self.assertEqual(result["final_demand_direct_exact"], ["100", "0"])
        self.assertEqual(result["indirect_output_change_exact"], ["100/7", "200/7"])
        self.assertEqual(result["total_direct_exact"], "100")
        self.assertEqual(result["total_indirect_exact"], "300/7")
        self.assertEqual(result["total_change_exact"], "1000/7")
        self.assertEqual(result["induced_effect"],
                         "NOT_ESTIMATED_HOUSEHOLD_CLOSURE_UNDECLARED")
        self.assertIn("behavioral_or_causal_validity",
                      result["evidence"]["coverage"]["unknown_dimensions"])
        self.assertEqual(io.verify(case(), result)["status"], "RECOMPUTED")

    def test_permuting_sector_order_permutates_effects_not_totals(self):
        spec = case(); original = io.calculate(spec)
        spec["sectors"] = list(reversed(spec["sectors"]))
        spec["coefficient_matrix"] = [list(reversed(row)) for row in reversed(spec["coefficient_matrix"])]
        spec["final_demand_change"] = list(reversed(spec["final_demand_change"]))
        changed = io.calculate(spec)
        self.assertEqual(changed["gross_output_change_exact"],
                         list(reversed(original["gross_output_change_exact"])))
        self.assertEqual(changed["total_change_exact"], original["total_change_exact"])

    def test_zero_and_negative_final_demand_changes_are_accounting_not_realized(self):
        spec = case(); spec["final_demand_change"] = ["0", "0"]
        self.assertEqual(io.calculate(spec)["total_change_exact"], "0")
        spec["final_demand_change"] = ["-100", "0"]
        result = io.calculate(spec)
        self.assertEqual(result["total_change_exact"], "-1000/7")
        self.assertEqual(result["statistical_and_causal_validity"], "NOT_ESTABLISHED")

    def test_nonproductive_singular_negative_inputs_rights_and_shapes_block(self):
        variants = []
        for rows in ([[["1", "0"], ["0", "0"]],
                      [["2", "0"], ["0", "0"]],
                      [["-1/4", "0"], ["0", "0"]]]):
            changed = case(); changed["coefficient_matrix"] = rows; variants.append(changed)
        changed = case(); changed["final_demand_change"] = ["10"]; variants.append(changed)
        changed = case(); changed["coefficient_matrix"][0][0] = "1/0"; variants.append(changed)
        changed = case(); changed["sectors"] = ["steel", "steel"]; variants.append(changed)
        changed = case(); changed["rights"]["report_export"] = "DENIED"; variants.append(changed)
        changed = case(); changed["sectors"][0] = "\nEvil"; variants.append(changed)
        changed = case(); changed["final_demand_change"][0] = 100; variants.append(changed)
        for spec in variants:
            with self.subTest(spec=str(spec)[:100]), self.assertRaises(ScopeError):
                io.calculate(spec)

    def test_large_exploratory_invocation_does_not_pretend_independent_oracle(self):
        n = 7; spec = case()
        spec["sectors"] = ["s" + str(i) for i in range(n)]
        spec["coefficient_matrix"] = [
            ["1/2" if i == j else "0" for j in range(n)] for i in range(n)]
        spec["final_demand_change"] = ["1"] + ["0"] * (n - 1)
        result = io.calculate(spec)
        self.assertEqual(result["gross_output_change_exact"], ["2"] + ["0"] * (n - 1))
        self.assertEqual(result["oracle"]["status"], "EXACT_IDENTITIES_ONLY")
        self.assertEqual(result["oracle"]["independent_determinant"],
                         "NOT_RUN_PER_INVOCATION_BUDGET")

    def test_evidence_code_sha_matches_real_source_and_fails_if_changed_mid_run(self):
        result = io.calculate(case())
        self.assertEqual(result["evidence"]["producer"]["code_sha256"],
                         sha256(Path(io.__file__).read_bytes()).hexdigest())
        # No worktree mutation. Simulate a source replacing itself during the
        # calculation and demand an explicit blocker rather than a false receipt.
        with patch.object(Path, "read_bytes", side_effect=[b"before", b"after"]):
            with self.assertRaisesRegex(ScopeError, "CODE_CHANGED_DURING_RUN"):
                io.calculate(case())

    def test_recomputation_detects_output_evidence_or_status_mutation(self):
        spec = case(); original = io.calculate(spec)
        for path, value in (
                (("total_change_exact",), "1"),
                (("native_execution",), "EXECUTED"),
                (("evidence", "authority"), "REAL_WORLD_AUTHENTICATED"),
                (("induced_effect",), "100")):
            changed = copy.deepcopy(original); target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaisesRegex(
                    ScopeError, "RECOMPUTATION_MISMATCH"):
                io.verify(spec, changed)


if __name__ == "__main__":
    unittest.main()
