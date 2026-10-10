"""Offline test suite for the CAD Companion pipeline.

Runs with NO API key and NO network: every test exercises either a pure-Python CAD
tool or graph logic with the LLM replaced by a stub. This is what proves a fresh
clone behaves before anyone spends quota.

    .\\.venv\\Scripts\\python.exe -m pip install pytest
    .\\.venv\\Scripts\\python.exe -m pytest -q
"""
import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pytest

import cad_companion as cc

PARTS = Path(__file__).parent / "parts"
STL = PARTS / "bracket.stl"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def sample_part():
    """Generate the demo bracket once per session if it is missing."""
    if not STL.exists():
        from shapely.geometry import Polygon

        import trimesh

        profile = Polygon([(0, 0), (120, 0), (120, 12), (2.5, 12), (2.5, 72), (0, 72)])
        STL.parent.mkdir(parents=True, exist_ok=True)
        trimesh.creation.extrude_polygon(profile, height=60.0).export(STL)
    return STL


@pytest.fixture
def props(sample_part):
    return cc.extract_geometry(str(sample_part), "mm", "aisi_4130")


@pytest.fixture
def fake_state(sample_part, props):
    """A minimal but complete AgentState, as the reporter would receive it."""
    state = cc.make_initial_state(str(sample_part), "mm", "aisi_4130", None, "bracket")
    state["extracted_properties"] = props
    state["compliance_status"] = [{
        "rule_ref": "R1", "rule_checked": "mass <= 0.5 kg", "status": "FAIL",
        "reasoning": "mass exceeds limit", "source": "rulebook p.1", "limit_text": "shall not exceed 0.50 kg",
        "property_key": "mass_kg", "measured_value": props["mass_kg"],
        "comparator": "<=", "limit_value": 0.5, "validation_notes": [],
    }]
    state["stress_hotspots"] = [
        {"id": "H1", "heuristic": "thin_wall", "estimated_severity": "HIGH", "severity_score": 0.9,
         "location": {"xyz_mm": [1, 2, 3]}, "geometric_reason": "thin", "reviewer_status": "pending"},
    ]
    state["fea_job"] = {"status": "NOT_CONFIGURED", "message": "stub"}
    state["human_review"] = {"approved": True, "reviewer": "t", "notes": "", "timestamp_utc": "now"}
    state["completed"] = list(cc.PIPELINE)
    return state


# ---------------------------------------------------------------------------
# 1. Geometry extraction (pure Python)
# ---------------------------------------------------------------------------
class TestGeometry:
    def test_volume_matches_hand_calculation(self, props):
        # L-profile 120x12 + 2.5x60 = 1590 mm^2, extruded 60 mm
        assert props["volume_mm3"] == pytest.approx(1590 * 60, rel=1e-6)

    def test_mass_from_material_density(self, props):
        assert props["mass_kg"] == pytest.approx(95400 * 1e-9 * 7850.0, rel=1e-9)
        assert props["material"] == "aisi_4130"

    def test_mass_is_null_without_material(self, sample_part):
        assert cc.extract_geometry(str(sample_part), "mm")["mass_kg"] is None

    def test_bbox_and_watertight(self, props):
        assert props["bounding_box_xyz_mm"] == pytest.approx([120, 72, 60])
        assert props["is_watertight"] is True
        assert props["volume_source"] == "mesh"

    def test_centre_of_gravity_is_inside_bbox(self, props):
        lo, hi = props["bbox_min_mm"], props["bbox_max_mm"]
        assert all(l <= c <= h for l, c, h in zip(lo, props["center_of_gravity_mm"], hi))

    def test_unit_scaling(self, sample_part):
        mm = cc.extract_geometry(str(sample_part), "mm")
        cm = cc.extract_geometry(str(sample_part), "cm")
        assert cm["volume_mm3"] == pytest.approx(mm["volume_mm3"] * 1000, rel=1e-6)

    def test_unknown_material_rejected(self, sample_part):
        with pytest.raises(cc.CADToolError, match="Unknown material"):
            cc.extract_geometry(str(sample_part), "mm", "unobtanium")

    def test_missing_file_rejected(self):
        with pytest.raises(cc.CADToolError, match="not found"):
            cc.extract_geometry("nope/missing.stl", "mm")

    def test_non_mesh_file_rejected(self, tmp_path):
        bad = tmp_path / "part.stl"
        bad.write_text("not an stl")
        with pytest.raises(cc.CADToolError):
            cc.extract_geometry(str(bad), "mm")

    def test_explicit_density_overrides_material(self, sample_part):
        p = cc.extract_geometry(str(sample_part), "mm", "aisi_4130", density_kg_m3=1000.0)
        assert p["density_kg_m3"] == 1000.0
        assert p["mass_kg"] == pytest.approx(95400 * 1e-9 * 1000.0)


# ---------------------------------------------------------------------------
# 2. Stress heuristics (pure Python)
# ---------------------------------------------------------------------------
class TestHeuristics:
    def test_sharp_corner_found_at_expected_location(self, sample_part):
        r = cc.check_sharp_internal_radii(str(sample_part), "mm")
        assert r["summary"]["sharp_concave_edges"] >= 1
        worst = r["hotspots"][0]
        assert worst["heuristic"] == "sharp_internal_radius"
        x, y, z = worst["location"]["xyz_mm"]
        assert (x, y) == pytest.approx((2.5, 12.0), abs=0.5)
        assert 0 < z < 60

    def test_section_change_found_on_both_axes(self, sample_part):
        r = cc.check_section_changes(str(sample_part), "mm")
        axes = {h["location"]["axis"] for h in r["hotspots"]}
        assert {"x", "y"} <= axes
        assert all(h["severity_score"] <= 1 for h in r["hotspots"])

    def test_thin_wall_matches_measured_thickness(self, sample_part):
        r = cc.check_thin_walls(str(sample_part), "mm")
        assert r["summary"]["median_wall_mm"] == pytest.approx(12.0, abs=0.5)
        assert r["hotspots"], "expected the 2.5 mm wall to be flagged"
        assert r["summary"]["thin_threshold_mm"] == pytest.approx(3.0, abs=0.1)

    def test_hotspot_schema_is_complete(self, sample_part):
        for tool in (cc.check_sharp_internal_radii, cc.check_section_changes, cc.check_thin_walls):
            for h in tool(str(sample_part), "mm")["hotspots"]:
                assert {"heuristic", "location", "estimated_severity", "severity_score",
                        "geometric_reason"} <= set(h)
                assert cc._severity_label(h["severity_score"]) == h["estimated_severity"]

    def test_max_hotspots_respected(self, sample_part):
        for tool in (cc.check_sharp_internal_radii, cc.check_section_changes, cc.check_thin_walls):
            assert len(tool(str(sample_part), "mm", max_hotspots=2)["hotspots"]) <= 2


# ---------------------------------------------------------------------------
# 3. Compliance validator (the anti-hallucination core)
# ---------------------------------------------------------------------------
class TestValidator:
    REF = {"R1": {"source": "rulebook p.1"}}
    FLAT = {"mass_kg": 0.7489, "bbox_x_mm": 120.0}

    def _finding(self, **kw):
        base = dict(rule_ref="R1", rule_checked="mass limit", status="PASS",
                    reasoning="ok", property_key="mass_kg", measured_value=0.7489,
                    comparator="<=", limit_value=2.0)
        base.update(kw)
        return cc.RuleFinding(**base)

    def test_correct_pass_survives(self):
        out = cc.validate_finding(self._finding(), self.FLAT, self.REF)
        assert out["status"] == "PASS"
        assert out["validation_notes"] == []

    def test_llm_wrong_verdict_is_overridden(self):
        # model claims PASS but mass 0.7489 > 0.5 limit
        out = cc.validate_finding(self._finding(status="PASS", limit_value=0.5), self.FLAT, self.REF)
        assert out["status"] == "FAIL"
        assert any("overridden" in n for n in out["validation_notes"])

    def test_llm_invented_measurement_is_replaced(self):
        out = cc.validate_finding(self._finding(measured_value=0.111), self.FLAT, self.REF)
        assert out["measured_value"] == pytest.approx(0.7489)
        assert any("replaced" in n for n in out["validation_notes"])

    def test_unknown_property_key_becomes_not_verifiable(self):
        out = cc.validate_finding(self._finding(property_key="magic_kg"), self.FLAT, self.REF)
        assert out["status"] == "NOT_VERIFIABLE"

    def test_unknown_rule_ref_becomes_not_verifiable(self):
        out = cc.validate_finding(self._finding(rule_ref="R999"), self.FLAT, self.REF)
        assert out["status"] == "NOT_VERIFIABLE"

    def test_pass_without_limit_becomes_not_verifiable(self):
        out = cc.validate_finding(self._finding(limit_value=None, comparator=None), self.FLAT, self.REF)
        assert out["status"] == "NOT_VERIFIABLE"

    @pytest.mark.parametrize("value,comp,limit,expected", [
        (0.5, "<=", 0.5, True), (0.6, "<=", 0.5, False), (0.5, ">=", 0.5, True),
        (0.49, "<", 0.5, True), (0.51, ">", 0.5, True), (0.5, "==", 0.5, True),
    ])
    def test_comparators(self, value, comp, limit, expected):
        assert cc._compare(value, comp, limit) is expected


# ---------------------------------------------------------------------------
# 4. Verdict + reporter guards
# ---------------------------------------------------------------------------
class TestVerdict:
    def _state(self, fake_state, **kw):
        s = dict(fake_state)
        s.update(kw)
        return s

    def test_fail_forces_non_compliant(self, fake_state):
        assert cc.compute_verdict(fake_state) == "NON_COMPLIANT"

    def test_clean_part_is_approved(self, fake_state):
        s = self._state(fake_state, compliance_status=[], stress_hotspots=[])
        assert cc.compute_verdict(s) == "APPROVED"

    def test_serious_hotspot_triggers_warning(self, fake_state):
        s = self._state(fake_state, compliance_status=[])
        assert cc.compute_verdict(s) == "APPROVED_WITH_WARNINGS"

    def test_dismissed_hotspot_clears_warning(self, fake_state):
        hs = [{**fake_state["stress_hotspots"][0], "reviewer_status": "dismissed"}]
        s = self._state(fake_state, compliance_status=[], stress_hotspots=hs)
        assert cc.compute_verdict(s) == "APPROVED"

    def test_not_verifiable_triggers_warning(self, fake_state):
        """One unverified rule among checked ones => warnings, not INCOMPLETE."""
        checked = {**fake_state["compliance_status"][0], "status": "PASS", "rule_ref": "R2"}
        unverified = {**fake_state["compliance_status"][0], "status": "NOT_VERIFIABLE", "rule_ref": "R3"}
        s = self._state(fake_state, compliance_status=[checked, unverified], stress_hotspots=[])
        assert cc.compute_verdict(s) == "APPROVED_WITH_WARNINGS"

    def test_nothing_evaluated_is_incomplete_not_approved(self, fake_state):
        """A run where the rulebook/model never worked must not read as a near-miss."""
        s = self._state(fake_state, compliance_status=[
            {**fake_state["compliance_status"][0], "status": "NOT_VERIFIABLE"}], stress_hotspots=[])
        assert cc.compute_verdict(s) == "INCOMPLETE"

    def test_fail_still_wins_over_incomplete(self, fake_state):
        s = self._state(fake_state, compliance_status=[
            {**fake_state["compliance_status"][0], "status": "NOT_VERIFIABLE"},
            {**fake_state["compliance_status"][0], "status": "FAIL"}])
        assert cc.compute_verdict(s) == "NON_COMPLIANT"

    def test_empty_findings_is_not_incomplete(self, fake_state):
        assert cc.compute_verdict(self._state(fake_state, compliance_status=[])) in (
            "APPROVED", "APPROVED_WITH_WARNINGS")


class TestReporterGuards:
    def test_grounded_number_passes(self):
        assert cc.ungrounded_numbers("mass is 0.7489 kg", {"mass": 0.7489}) == []

    def test_invented_number_is_flagged(self):
        assert cc.ungrounded_numbers("safety factor of 3.5", {"mass": 0.7489}) == ["3.5"]

    def test_small_integers_are_exempt(self):
        assert cc.ungrounded_numbers("see rule 3 and item 7", {"mass": 1}) == []

    def test_precision_must_match_context(self):
        # 0.75 matches 0.7489 rounded to 2 dp (by design: "to the stated precision")
        assert cc.ungrounded_numbers("0.75 kg", {"mass": 0.7489}) == []
        # ...but at 4 dp, 0.7488 does not equal 0.7489
        assert cc.ungrounded_numbers("0.7488 kg", {"mass": 0.7489}) == ["0.7488"]

    def test_fallback_narrative_never_invents(self, fake_state):
        n = cc._fallback_narrative(fake_state, "NON_COMPLIANT")
        assert "NON_COMPLIANT" in n.executive_summary
        assert n.recommendations


# ---------------------------------------------------------------------------
# 5. Graph routing
# ---------------------------------------------------------------------------
class TestRouting:
    def test_pipeline_order_is_fixed(self):
        assert cc.PIPELINE == ["geometry_extractor", "compliance_checker",
                               "stress_reviewer", "human_review", "reporter"]

    def test_routes_to_first_unfinished_spoke(self, fake_state):
        assert cc.route_from_orchestrator({"completed": [], "error": None, "human_review": {}}) == "geometry_extractor"
        assert cc.route_from_orchestrator({"completed": ["geometry_extractor"],
                                           "error": None, "human_review": {}}) == "compliance_checker"

    def test_error_routes_to_failure_handler(self, fake_state):
        assert cc.route_from_orchestrator({"completed": [], "error": "boom", "human_review": {}}) == "failure_handler"

    def test_rejection_routes_to_failure_handler(self, fake_state):
        s = {"completed": list(cc.PIPELINE), "error": None,
             "human_review": {"approved": False, "reviewer": "e", "notes": "no"}}
        assert cc.route_from_orchestrator(s) == "failure_handler"

    def test_complete_graph_routes_to_end(self, fake_state):
        assert cc.route_from_orchestrator({"completed": list(cc.PIPELINE),
                                           "error": None, "human_review": {"approved": True}}) == cc.END

    def test_graph_compiles_with_all_nodes(self):
        g = cc.build_graph()
        assert set(g.get_graph().nodes.keys()) >= set(cc.PIPELINE) | {"orchestrator", "failure_handler"}

    def test_failure_handler_reports_error(self, fake_state):
        out = cc.failure_handler_node({**fake_state, "error": "[x] boom", "human_review": {}})
        assert out["verdict"] == "FAILED"
        assert "boom" in out["final_report"]

    def test_failure_handler_reports_rejection(self, fake_state):
        out = cc.failure_handler_node({**fake_state, "error": None})
        assert out["verdict"] == "REJECTED"


# ---------------------------------------------------------------------------
# 6. End-to-end with a stubbed LLM (no network, no quota)
# ---------------------------------------------------------------------------
class TestEndToEndStubbed:
    def test_full_run_produces_report_without_any_llm(self, monkeypatch, sample_part, tmp_path):
        """The whole graph runs with the LLM stubbed out - proves the pipeline is
        tool-driven and that a rate-limited key still yields a usable review."""
        monkeypatch.setattr(cc.Settings, "from_env",
                            classmethod(lambda cls: cc.Settings(
                                llm_provider="openai", openai_model="x", anthropic_model="x",
                                gemini_model="x", embedding_provider="openai", embedding_model="x",
                                gemini_embedding_model="x", rulebook_db_dir=str(tmp_path / "db"),
                                rulebook_collection="t", reports_dir=str(tmp_path / "reports"))))

        class BoomLLM:
            def invoke(self, *a, **k):
                raise RuntimeError("simulated rate limit")

            def bind_tools(self, *a, **k):
                return self

            def with_structured_output(self, *a, **k):
                return self

        monkeypatch.setattr(cc, "get_llm", lambda: BoomLLM())
        monkeypatch.setattr(cc, "get_rulebook_retriever", lambda **k: (_ for _ in ()).throw(
            cc.CADToolError("index empty")))

        state = cc.make_initial_state(str(sample_part), "mm", "aisi_4130", None, "bracket")
        final = cc.run_review(state, reviewer_callback=lambda _p: {
            "approved": True, "reviewer": "ci", "notes": "stubbed", "dismiss_hotspots": []})

        # No LLM => no rule could be evaluated, so the verdict must say so plainly.
        assert final["verdict"] == "INCOMPLETE"
        assert all(f["status"] == "NOT_VERIFIABLE" for f in final["compliance_status"])
        assert "INCOMPLETE" in final["final_report"]
        assert final["extracted_properties"]["mass_kg"] == pytest.approx(0.74889, abs=1e-9)
        assert len(final["stress_hotspots"]) >= 3            # heuristics ran without an LLM
        assert Path(final["report_path"]).exists()
        assert "Executive summary" in final["final_report"]
        assert "heuristics" in final["final_report"].lower() or "screening" in final["final_report"].lower()

    def test_report_survives_llm_outage_at_final_step(self, monkeypatch, sample_part, tmp_path):
        """A rate limit at the reporter must still yield a report (coded narrative)."""
        monkeypatch.setattr(cc, "get_llm", lambda: type("L", (), {
            "invoke": lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("rl")),
            "bind_tools": lambda self, *a, **k: self,
            "with_structured_output": lambda self, *a, **k: self})())
        monkeypatch.setattr(cc, "get_rulebook_retriever", lambda **k: (_ for _ in ()).throw(
            cc.CADToolError("index empty")))

        state = cc.make_initial_state(str(sample_part), "mm", "aisi_4130")
        final = cc.run_review(state, reviewer_callback=lambda _p: {
            "approved": True, "reviewer": "ci", "notes": "", "dismiss_hotspots": []})
        # Report exists despite the outage, and the verdict does not pretend it was reviewed.
        assert final["verdict"] == "INCOMPLETE"
        assert final["report_path"] and Path(final["report_path"]).exists()
        assert "INCOMPLETE" in final["final_report"]

    def test_human_gate_rejection_stops_before_report(self, monkeypatch, sample_part, tmp_path):
        monkeypatch.setattr(cc, "get_llm", lambda: type("L", (), {
            "invoke": lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("rl")),
            "bind_tools": lambda self, *a, **k: self,
            "with_structured_output": lambda self, *a, **k: self})())
        monkeypatch.setattr(cc, "get_rulebook_retriever", lambda **k: (_ for _ in ()).throw(
            cc.CADToolError("index empty")))

        state = cc.make_initial_state(str(sample_part), "mm", "aisi_4130")
        final = cc.run_review(state, reviewer_callback=lambda _p: {
            "approved": False, "reviewer": "e", "notes": "needs work", "dismiss_hotspots": []})
        assert final["verdict"] == "REJECTED"
        assert final.get("report_path") is None


# ---------------------------------------------------------------------------
# 7. Quota / error message quality
# ---------------------------------------------------------------------------
class TestErrorMessages:
    def test_quota_error_is_human_readable(self):
        msg = cc.describe_llm_error(RuntimeError(
            "RESOURCE_EXHAUSTED: 429 {'quotaId': 'FreeTier', 'quotaValue': 20} "
            "'model': 'gemini-3.5-flash' 'retryDelay': '16430s'"))
        assert "rate limit" in msg.lower()
        assert "gemini-3.5-flash" in msg
        assert "{" not in msg                      # no raw JSON

    def test_no_credits_message(self):
        msg = cc.describe_llm_error(RuntimeError("429 credit_balance_exhausted"))
        assert "credits" in msg.lower()

    def test_bad_model_message(self):
        msg = cc.describe_llm_error(RuntimeError("404 NOT_FOUND models/gemini-9-turbo is no longer available"))
        assert "gemini-9-turbo" in msg
        assert "GEMINI_MODEL" in msg

    def test_invalid_key_message(self):
        msg = cc.describe_llm_error(RuntimeError("401 API_KEY_INVALID"))
        assert "API key" in msg
