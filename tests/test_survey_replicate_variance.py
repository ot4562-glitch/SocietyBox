"""Declared survey replicate weights are not regression analytic weights."""
import copy
from hashlib import sha256
from pathlib import Path
import unittest

from econscope import survey_replicate_variance as survey
from econscope.workspace import ScopeError


def case(method="BRR_DECLARED"):
    return {"schema": survey.SCHEMA, "values": ["0", "10"],
            "base_weights": ["1", "1"],
            "replicate_weights": [["2", "0"], ["0", "2"]],
            "method": method, "unit": "declared households, index units",
            "source_declared_sha256": "c" * 64,
            "rights": {"local_processing": "PERMITTED", "report_export": "PERMITTED"}}


class SurveyReplicateVarianceTests(unittest.TestCase):
    def test_brr_declared_exact_mean_variance_not_validated_design(self):
        spec = case(); result = survey.calculate(spec)
        self.assertEqual(result["point_exact"], "5")
        self.assertEqual(result["replicate_points_exact"], ["0", "10"])
        self.assertEqual(result["variance_exact"], "25")
        self.assertEqual(result["native_execution"], "NOT_RUN")
        self.assertEqual(result["design_authentication"], "NOT_ESTABLISHED")
        self.assertEqual(result["confidence_intervals"], "NOT_COMPUTED_DESIGN_NOT_VERIFIED")
        self.assertEqual(survey.verify(spec, result)["status"], "RECOMPUTED")

    def test_jackknife_center_differs_from_brr_point_center(self):
        spec = case(); spec["replicate_weights"] = [["1", "0"], ["1", "9"]]
        brr = survey.calculate(spec)
        spec["method"] = "DELETE_ONE_JACKKNIFE_DECLARED"
        jk = survey.calculate(spec)
        self.assertEqual(jk["replicate_points_exact"], ["0", "9"])
        self.assertEqual(brr["variance_exact"], "41/2")
        self.assertEqual(jk["variance_exact"], "81/4")

    def test_translation_changes_point_only(self):
        spec = case(); old = survey.calculate(spec)
        spec["values"] = ["7", "17"]
        new = survey.calculate(spec)
        self.assertEqual(new["point_exact"], "12")
        self.assertEqual(new["variance_exact"], old["variance_exact"])
        self.assertEqual(new["replicate_points_exact"], ["7", "17"])

    def test_invalid_rights_weights_methods_and_dimensions_rejected(self):
        variants = []
        def altered(path, value):
            q = case(); target = q
            for key in path[:-1]: target = target[key]
            target[path[-1]] = value
            variants.append(q)
        altered(("rights", "report_export"), "DENIED")
        altered(("base_weights", 0), "0")
        altered(("replicate_weights", 0, 0), "-1")
        altered(("replicate_weights", 0), ["0", "0"])
        altered(("values", 0), "nan")
        altered(("values", 1), 10)
        altered(("method",), "BRR_VERIFIED")
        altered(("method",), ["BRR_DECLARED"])
        altered(("unit",), "age\nunit")
        altered(("replicate_weights",), [["2", "0"]])
        altered(("source_declared_sha256",), "bad")
        altered(("values",), ["0"])
        for q in variants:
            with self.subTest(case=str(q)[:150]), self.assertRaises(ScopeError):
                survey.calculate(q)

    def test_tamper_detection_and_source_code_receipt(self):
        spec = case(); result = survey.calculate(spec)
        self.assertEqual(result["evidence"]["producer"]["code_sha256"],
                         sha256(Path(survey.__file__).read_bytes()).hexdigest())
        for key, changed_value in (
                ("variance_exact", "0"), ("design_authentication", "VERIFIED"),
                ("confidence_intervals", "[0,10]")):
            mutated = copy.deepcopy(result); mutated[key] = changed_value
            with self.subTest(key=key), self.assertRaisesRegex(
                    ScopeError, "RECOMPUTATION_MISMATCH"):
                survey.verify(spec, mutated)


if __name__ == "__main__":
    unittest.main()
