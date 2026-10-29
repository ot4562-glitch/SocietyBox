"""Evidence integrity is separate from scientific authority and source truth."""
import copy
import unittest

from econscope import research_evidence as rea
from econscope.workspace import ScopeError, canonical, digest


def evidence(**changes):
    fields = dict(subject_sha256="a" * 64,
                  producer={"tool_id": "verifier.exact", "version": "1", "code_sha256": "b" * 64},
                  operation="CHECK", authority="LOCAL_NUMERIC_EXECUTION",
                  epistemic_state="DERIVED", proof_kind="NUMERIC_REPLAY",
                  scope={"sample": "train", "vintage": "DECLARED_ONLY"},
                  coverage={"status": "COMPLETE", "unknown_dimensions": []},
                  result={"value": "3/2"}, parent_evidence_ids=[], limitations=["No source authentication"])
    fields.update(changes)
    return rea.make_evidence(**fields)


def obligation(**changes):
    fields = dict(claim_id="replay", subject_sha256="a" * 64,
                  scope={"sample": "train", "vintage": "DECLARED_ONLY"},
                  required_authority="LOCAL_NUMERIC_EXECUTION", required_proof_kind="NUMERIC_REPLAY",
                  verifier_id="verifier.exact", dependencies=[], unknown_ids=[], blocking_reasons=[],
                  next_probe="Rerun the same input independently")
    fields.update(changes)
    return rea.make_obligation(**fields)


def verification(ob, revision=1, previous=None, outcome="PASS", **changes):
    return evidence(result={"obligation_id": ob["obligation_id"], "revision": revision,
                            "previous_verification_id": previous, "outcome": outcome}, **changes)


class EvidenceTests(unittest.TestCase):
    def test_content_addressed_copied_and_idempotent(self):
        rec = evidence()
        ledger = rea.EvidenceLedger([rec])
        ledger.merge([rec])
        rec["result"]["value"] = "99"
        saved = ledger.records()
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["result"]["value"], "3/2")
        self.assertEqual(saved[0]["evidence_id"], "ev_" + digest(canonical(
            {k: v for k, v in saved[0].items() if k != "evidence_id"})))
        saved[0]["result"]["value"] = "other"
        self.assertEqual(ledger.records()[0]["result"]["value"], "3/2")

    def test_merge_validates_everything_before_mutating(self):
        ledger = rea.EvidenceLedger([evidence()])
        before = ledger.records()
        good = evidence(operation="NEW")
        bad = evidence(operation="TAMPER")
        bad["result"]["value"] = "changed"
        with self.assertRaisesRegex(ScopeError, "EVIDENCE_DIGEST"):
            ledger.merge([good, bad])
        self.assertEqual(before, ledger.records())

    def test_forward_parent_allowed_but_missing_parent_rejected(self):
        parent = evidence()
        child = evidence(operation="CHILD", parent_evidence_ids=[parent["evidence_id"]])
        ledger = rea.EvidenceLedger([child, parent])
        self.assertEqual(len(ledger.records()), 2)
        missing = evidence(parent_evidence_ids=["ev_" + "c" * 64])
        with self.assertRaisesRegex(ScopeError, "PARENT"):
            ledger.merge([missing])
        self.assertEqual(len(ledger.records()), 2)

    def test_nonfinite_non_json_extra_field_and_bad_types_rejected(self):
        for changes in ({"result": {"x": float("nan")}}, {"result": {1: "bad"}},
                        {"result": {"x": (1, 2)}}, {"scope": []},
                        {"epistemic_state": "TRUE"}, {"parent_evidence_ids": "ev_bad"}):
            with self.subTest(changes=repr(changes)), self.assertRaises(ScopeError):
                evidence(**changes)
        rec = evidence()
        rec["extra"] = "field"
        with self.assertRaises(ScopeError):
            rea.EvidenceLedger([rec])

    def test_scope_and_provenance_changes_fork_id(self):
        first = evidence()
        for change in ({"scope": {"sample": "test"}}, {"subject_sha256": "d" * 64},
                       {"authority": "DECLARED_ONLY"}):
            self.assertNotEqual(first["evidence_id"], evidence(**change)["evidence_id"])


class UnknownTests(unittest.TestCase):
    def test_stable_id_revision_chain_and_stale_write_refusal(self):
        ledger = rea.EvidenceLedger()
        registry = rea.UnknownRegistry(ledger)
        first = registry.register("Which mechanism?", {"sample": "train"},
                                  "EXTERNAL_REPLICATION", "IDENTIFICATION")
        self.assertEqual(first, registry.register("Which mechanism?", {"sample": "train"},
                                                  "EXTERNAL_REPLICATION", "IDENTIFICATION"))
        next_revision = registry.revise(first["unknown_id"], first["revision_digest"],
                                        status="INVESTIGATING", recommended_probes=["negative control"])
        self.assertEqual(next_revision["revision"], 2)
        self.assertEqual(next_revision["previous_revision_digest"], first["revision_digest"])
        self.assertEqual(next_revision["unknown_id"], first["unknown_id"])
        with self.assertRaisesRegex(ScopeError, "REVISION_CONFLICT"):
            registry.revise(first["unknown_id"], first["revision_digest"], status="OPEN")

    def test_resolution_requires_matching_noninferred_complete_evidence(self):
        ledger = rea.EvidenceLedger()
        registry = rea.UnknownRegistry(ledger)
        unk = registry.register("Numerical value?", {"sample": "train"},
                                "LOCAL_NUMERIC_EXECUTION", "NUMERIC_REPLAY")
        for changes in ({"epistemic_state": "INFERRED"}, {"authority": "DECLARED_ONLY"},
                        {"coverage": {"status": "PARTIAL", "unknown_dimensions": ["sample"]}},
                        {"scope": {"sample": "test"}}):
            rec = evidence(scope={"sample": "train"}) if not changes else evidence(
                **dict({"scope": {"sample": "train"}}, **changes))
            ledger.merge([rec])
            with self.assertRaisesRegex(ScopeError, "RESOLUTION"):
                registry.revise(unk["unknown_id"], unk["revision_digest"], status="RESOLVED",
                                resolution_evidence_ids=[rec["evidence_id"]])
        valid = evidence(scope={"sample": "train"}, result={"unknown_id": unk["unknown_id"],
            "revision": 1, "previous_verification_id": None, "outcome": "PASS"})
        ledger.merge([valid])
        resolved = registry.revise(unk["unknown_id"], unk["revision_digest"], status="RESOLVED",
                                  resolution_evidence_ids=[valid["evidence_id"]])
        self.assertEqual(resolved["status"], "RESOLVED")
        self.assertEqual(len(registry.history(unk["unknown_id"])), 2)

    def test_unresolved_counterevidence_cannot_be_hidden_by_resolution(self):
        rec = evidence(scope={"sample": "train"})
        ledger = rea.EvidenceLedger([rec])
        registry = rea.UnknownRegistry(ledger)
        unk = registry.register("Value?", {"sample": "train"}, "LOCAL_NUMERIC_EXECUTION", "NUMERIC_REPLAY")
        opposed = registry.revise(unk["unknown_id"], unk["revision_digest"], status="CONTRADICTED",
                                  contradicting_evidence_ids=[rec["evidence_id"]])
        with self.assertRaisesRegex(ScopeError, "RESOLUTION"):
            registry.revise(unk["unknown_id"], opposed["revision_digest"], status="RESOLVED",
                            resolution_evidence_ids=[rec["evidence_id"]])

    def test_a_later_failed_verification_invalidates_old_resolution(self):
        ledger = rea.EvidenceLedger()
        registry = rea.UnknownRegistry(ledger)
        unk = registry.register("Value?", {"sample": "train"}, "LOCAL_NUMERIC_EXECUTION", "NUMERIC_REPLAY")
        good = evidence(scope={"sample": "train"}, result={"unknown_id": unk["unknown_id"],
            "revision": 1, "previous_verification_id": None, "outcome": "PASS"})
        ledger.merge([good])
        resolved = registry.revise(unk["unknown_id"], unk["revision_digest"], status="RESOLVED",
                                  resolution_evidence_ids=[good["evidence_id"]])
        self.assertTrue(registry.is_resolved(unk["unknown_id"]))
        bad = evidence(scope={"sample": "train"}, result={"unknown_id": unk["unknown_id"],
            "revision": 2, "previous_verification_id": good["evidence_id"], "outcome": "FAIL"})
        ledger.merge([bad])
        self.assertFalse(registry.is_resolved(unk["unknown_id"]))
        with self.assertRaisesRegex(ScopeError, "RESOLUTION"):
            registry.revise(unk["unknown_id"], resolved["revision_digest"], status="RESOLVED")

    def test_scope_equality_does_not_conflate_boolean_and_integer(self):
        ob = obligation(scope={"selection": True})
        rec = verification(ob, scope={"selection": 1})
        result = rea.evaluate_coverage([ob], rea.EvidenceLedger([rec]))
        self.assertEqual(result["obligations"][0]["state"], "UNKNOWN")


class CoverageTests(unittest.TestCase):
    def test_many_matching_and_unrelated_evidence_keep_baseline_digest(self):
        # Historical all-pairs coverage oracle baseline, before lookup indexing.
        # This guards the exact obligation order, evidence IDs, verdicts and hash.
        obs = [obligation(claim_id="bench_" + str(i)) for i in range(220)]
        ledger = rea.EvidenceLedger([verification(ob) for ob in obs])
        result = rea.evaluate_coverage(obs, ledger)
        self.assertEqual(result["evaluation_sha256"],
                         "4377b08d0586e1c44ef53bbb04dfb8184c6056b593deee12abcc03f608f5b014")
        self.assertTrue(all(row["state"] == "PASS" for row in result["obligations"]))

    def test_deep_dependency_chain_preserves_baseline_ordered_hash(self):
        obs = []
        for i in range(120):
            obs.append(obligation(
                claim_id="chain_" + str(i),
                dependencies=[obs[-1]["obligation_id"]] if obs else []))
        result = rea.evaluate_coverage(list(reversed(obs)),
                                       rea.EvidenceLedger([verification(ob) for ob in obs]))
        self.assertEqual(result["evaluation_sha256"],
                         "eb1692b84bbf375fcd75ad976adcbeb7413f28a069d42bb37d8c27ff3065b13f")
        self.assertTrue(all(row["state"] == "PASS" for row in result["obligations"]))

    def test_empty_is_unknown_not_pass(self):
        ob = obligation()
        result = rea.evaluate_coverage([ob], rea.EvidenceLedger())
        self.assertEqual(result["obligations"][0]["state"], "UNKNOWN")
        self.assertEqual(result["status"], "INCOMPLETE")

    def test_latest_failure_wins_regardless_of_insertion_order(self):
        ob = obligation()
        old = verification(ob)
        new = verification(ob, 2, old["evidence_id"], "FAIL")
        result = rea.evaluate_coverage([ob], rea.EvidenceLedger([new, old]))
        self.assertEqual(result["obligations"][0]["state"], "FAIL")
        self.assertEqual(result["obligations"][0]["verification_evidence_id"], new["evidence_id"])

    def test_exact_authority_scope_and_proof_required(self):
        ob = obligation()
        for changes in ({"authority": "EXTERNAL_REPLICATION"}, {"epistemic_state": "INFERRED"},
                        {"scope": {"sample": "test"}}, {"proof_kind": "CAUSAL_EFFECT"},
                        {"subject_sha256": "d" * 64},
                        {"coverage": {"status": "UNKNOWN", "unknown_dimensions": []}}):
            with self.subTest(changes=changes):
                result = rea.evaluate_coverage([ob], rea.EvidenceLedger([verification(ob, **changes)]))
                self.assertEqual(result["obligations"][0]["state"], "UNKNOWN")

    def test_stale_success_cannot_mask_latest_lower_authority(self):
        ob = obligation()
        old = verification(ob)
        new = verification(ob, 2, old["evidence_id"], authority="DECLARED_ONLY")
        result = rea.evaluate_coverage([ob], rea.EvidenceLedger([old, new]))
        self.assertEqual(result["obligations"][0]["state"], "UNKNOWN")

    def test_broken_or_conflicting_verifier_revision_rejected(self):
        ob = obligation()
        old = verification(ob)
        for new in (verification(ob, 2, "ev_" + "e" * 64), verification(ob, outcome="FAIL")):
            with self.assertRaisesRegex(ScopeError, "VERIFICATION_CHAIN"):
                rea.evaluate_coverage([ob], rea.EvidenceLedger([old, new]))

    def test_dependencies_unknowns_and_blockers_do_not_disappear(self):
        first = obligation()
        second = obligation(claim_id="causal", dependencies=[first["obligation_id"]])
        ledger = rea.EvidenceLedger([verification(second)])
        result = rea.evaluate_coverage([second, first], ledger)
        self.assertEqual([r["state"] for r in result["obligations"]], ["UNKNOWN", "UNKNOWN"])
        blocked = obligation(blocking_reasons=["Official release metadata unavailable"])
        result = rea.evaluate_coverage([blocked], rea.EvidenceLedger([verification(blocked)]))
        self.assertEqual(result["obligations"][0]["state"], "BLOCKED")
        missing = obligation(dependencies=["obl_" + "f" * 64])
        with self.assertRaisesRegex(ScopeError, "DEPENDENCY"):
            rea.evaluate_coverage([missing], rea.EvidenceLedger())

    def test_complete_only_for_matching_local_claim_not_scientific_truth(self):
        ob = obligation()
        result = rea.evaluate_coverage([ob], rea.EvidenceLedger([verification(ob)]))
        self.assertEqual(result["status"], "COMPLETE_FOR_DECLARED_OBLIGATIONS")
        self.assertEqual(result["scope"], "RECORD_CONSISTENCY_NOT_VERIFIER_OR_SOURCE_AUTHENTICATION")


if __name__ == "__main__":
    unittest.main()
