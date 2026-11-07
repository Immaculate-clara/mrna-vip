import json

import pytest
from pydantic import ValidationError

from src import config
from src.report_generator import (OfflineGuidelineLLM, ReportGenerationError, extract_max_dose_mcg,
                                  generate_report, get_llm_client, guideline_sentences,
                                  retrieve_for_patient, save_report)
from src.schemas import ClinicalSafetyReport

GOOD = dict(Eligible_For_Vaccine=True, Recommended_Dose_mg=0.05, Reasoning_Steps=["ok"],
            FDA_Safety_Warnings_Cited=[], Disclaimer="x")


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), 0

    def generate(self, system, user, ctx):
        self.calls += 1
        return self.responses.pop(0)


@pytest.fixture
def chunks(index, patient):
    return retrieve_for_patient(index, patient)[0]


def test_schema_rejects_bad_input():  # FR-4.1
    assert ClinicalSafetyReport(**GOOD)
    for bad in [{k: v for k, v in GOOD.items() if k != "Reasoning_Steps"},
                {**GOOD, "Recommended_Dose_mg": "lots"}, {**GOOD, "Eligible_For_Vaccine": "yes"},
                {**GOOD, "Extra": 1}]:
        with pytest.raises(ValidationError):
            ClinicalSafetyReport(**bad)


def test_prompt_contains_only_supplied_facts_and_rules(index, patient):  # FR-4.2
    from src.report_generator import GenerationContext, SYSTEM_PROMPT, build_user_prompt
    chunks, _ = retrieve_for_patient(index, patient)
    prompt = build_user_prompt(GenerationContext(patient, "Low", 0.2, chunks))
    assert "Retrieved guideline excerpts" in prompt and chunks[0] in prompt
    assert "Never invent" in SYSTEM_PROMPT and "student engineering project" in SYSTEM_PROMPT


def test_exactly_one_retry_then_success(patient, chunks):  # FR-4.3
    llm = ScriptedLLM("not json at all", json.dumps(GOOD))
    res = generate_report(patient, "Low", 0.2, chunks, llm)
    assert llm.calls == 2 and res.attempts == 2


def test_exactly_one_retry_then_failure(patient, chunks):
    llm = ScriptedLLM("bad", "still bad", "never reached")
    with pytest.raises(ReportGenerationError):
        generate_report(patient, "Low", 0.2, chunks, llm)
    assert llm.calls == 2


def test_invented_citation_is_rejected(patient, chunks):
    fake = {**GOOD, "FDA_Safety_Warnings_Cited": ["Doses up to 900 micrograms are safe."]}
    llm = ScriptedLLM(json.dumps(fake), json.dumps(fake))
    with pytest.raises(ReportGenerationError, match="not found"):
        generate_report(patient, "Low", 0.2, chunks, llm)


def test_dose_over_guideline_bound_forces_ineligible(patient, chunks):  # FR-4.4
    assert extract_max_dose_mcg(chunks) == 100
    over = {**GOOD, "Recommended_Dose_mg": 0.12}  # 120 mcg
    res = generate_report({**patient, "Vaccine_Dose_mcg": 120}, "Low", 0.2, chunks,
                          ScriptedLLM(json.dumps(over)))
    assert res.report.Eligible_For_Vaccine is False and res.report.Recommended_Dose_mg == 0


def test_disclaimer_always_canonical(patient, chunks):  # FR-4.5
    for blank in ["", "something else"]:
        r = generate_report(patient, "Low", 0.2, chunks,
                            ScriptedLLM(json.dumps({**GOOD, "Disclaimer": blank or "a"}))).report
        assert r.Disclaimer == config.DISCLAIMER


def test_no_chunks_gives_conservative_default(patient):
    res = generate_report(patient, "Low", 0.1, [], ScriptedLLM())
    assert res.report.Eligible_For_Vaccine is False and res.attempts == 0


def test_offline_engine_end_to_end_and_files(tmp_path, index, patient):  # FR-4.6
    p = {**patient, "Vaccine_Dose_mcg": 130, "Pre_Existing_Conditions": 3, "Prior_Reaction_History": "Moderate"}
    chunks, _ = retrieve_for_patient(index, p)
    res = generate_report(p, "High", 0.9, chunks, OfflineGuidelineLLM())
    assert not res.report.Eligible_For_Vaccine and res.report.FDA_Safety_Warnings_Cited
    sents = guideline_sentences(chunks)
    assert all(c in sents for c in res.report.FDA_Safety_Warnings_Cited)
    j, m = save_report(res.report, p, "High", 0.9, tmp_path)
    assert j.stem == m.stem and j.exists() and "Disclaimer" in j.read_text()
    assert config.DISCLAIMER in m.read_text()


def test_low_risk_standard_patient_is_eligible(index, patient):
    chunks, _ = retrieve_for_patient(index, {**patient, "Pre_Existing_Conditions": 0})
    res = generate_report({**patient, "Pre_Existing_Conditions": 0}, "Low", 0.2, chunks,
                          OfflineGuidelineLLM())
    assert res.report.Eligible_For_Vaccine and res.report.Recommended_Dose_mg == pytest.approx(0.06)


def test_llm_provider_validation(monkeypatch):
    assert isinstance(get_llm_client("offline"), OfflineGuidelineLLM)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_llm_client("openai")
    with pytest.raises(ValueError):
        get_llm_client("nope")
