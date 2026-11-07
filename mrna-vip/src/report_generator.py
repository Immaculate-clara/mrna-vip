"""Module 4 - structured, guideline-grounded clinical safety report.

Flow: patient -> ML risk (Module 1) -> retrieval (Module 3) -> prompt -> LLM ->
schema validation + grounding check (one retry) -> deterministic guardrails -> JSON + Markdown.

Satisfies FR-4.1 .. FR-4.6. Educational demo only - never real clinical guidance.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import pandas as pd
from pydantic import ValidationError

from src import config
from src.data_pipeline import load_clean
from src.rag_index import _SENTENCE_RE, GuidelineIndex
from src.risk_model import load_model, predict_risk
from src.schemas import ClinicalSafetyReport

SYSTEM_PROMPT = """You are a clinical safety reporting assistant for a student engineering project.
Rules you must follow without exception:
1) Never recommend a dose that contradicts the retrieved guideline text.
2) Never invent a clinical metric, statistic, or guideline that is not present in the retrieved
   guideline excerpts. Use ONLY the risk score and guideline excerpts supplied below.
3) If the excerpts do not contain enough information to answer confidently, set
   Eligible_For_Vaccine to false (the conservative default), set Recommended_Dose_mg to 0 and
   explain why in Reasoning_Steps instead of guessing.
4) Always include the Disclaimer field stating this is a student engineering project and not
   real clinical guidance.
5) FDA_Safety_Warnings_Cited must contain verbatim sentences copied from the excerpts.
6) Recommended_Dose_mg is in MILLIGRAMS (1 mg = 1000 mcg).
Respond with a single JSON object and nothing else."""


class ReportGenerationError(RuntimeError):
    """Raised when the LLM output is still invalid after the single permitted retry (FR-4.3)."""


class ReportValidationError(ValueError):
    """Schema or grounding failure on one LLM response."""


@dataclass
class GenerationContext:
    patient: dict
    risk_label: str
    risk_probability: float
    chunks: list[str]


class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str, context: GenerationContext) -> str: ...


# --------------------------------------------------------------------------
# Guideline text helpers
# --------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def guideline_sentences(chunks: list[str]) -> list[str]:
    """Verbatim sentence/line pieces of the retrieved chunks (safe to cite)."""
    out: list[str] = []
    for chunk in chunks:
        for piece in re.split(r"\n|" + _SENTENCE_RE.pattern, chunk):
            piece = piece.strip()
            if piece and piece not in out:
                out.append(piece)
    return out


def extract_max_dose_mcg(chunks: list[str]) -> float | None:
    """Dose bound stated in the retrieved text, e.g. 'Doses above 100 micrograms ...'."""
    for chunk in chunks:
        m = re.search(r"doses?\s+(?:above|exceeding)\s+(\d+(?:\.\d+)?)\s*(?:micrograms?|mcg)",
                      chunk, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def _find(chunks: list[str], pattern: str) -> str | None:
    for sentence in guideline_sentences(chunks):
        if re.search(pattern, sentence, flags=re.IGNORECASE):
            return sentence
    return None


# --------------------------------------------------------------------------
# Retrieval for a patient profile
# --------------------------------------------------------------------------
def patient_summary(p: dict) -> str:
    return (f"Patient {p.get('Patient_ID', '?')}: age {p['Age']}, biomarker "
            f"{p['Biomarker_Level']:.2f} mg/L, {p['Pre_Existing_Conditions']} pre-existing "
            f"condition(s), candidate dose {p['Vaccine_Dose_mcg']:.1f} mcg, prior reaction "
            f"history: {p['Prior_Reaction_History']}")


def build_queries(p: dict) -> list[str]:
    return [
        f"dosing guidance and dose limits for a candidate dose of {p['Vaccine_Dose_mcg']:.0f} micrograms",
        f"age band reference dose for a {p['Age']} year old patient",
        f"precautions for {p['Pre_Existing_Conditions']} pre-existing conditions requiring secondary review",
        "dose adjustment for elevated inflammatory biomarker level",
        f"prior {p['Prior_Reaction_History']} reaction history contraindications and warnings",
    ]


def retrieve_for_patient(index: GuidelineIndex, p: dict, k: int = 2) -> tuple[list[str], float]:
    """Union of top-k chunks over the profile queries (de-duplicated). Returns total latency."""
    seen: dict[int, str] = {}
    total = 0.0
    for q in build_queries(p):
        results, latency = index.query(q, k=k)
        total += latency
        for r in results:
            seen.setdefault(r.chunk_id, r.text)
    return [seen[i] for i in sorted(seen)], total


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------
def build_user_prompt(ctx: GenerationContext, feedback: str | None = None) -> str:
    """Only Module-1 output and Module-3 chunks are supplied as factual grounding (FR-4.2)."""
    excerpts = "\n---\n".join(ctx.chunks)
    schema = json.dumps(ClinicalSafetyReport.model_json_schema(), indent=2)
    prompt = (
        f"Patient data: {patient_summary(ctx.patient)}\n"
        f"Predicted reaction-risk label (ML model): {ctx.risk_label} "
        f"(probability {ctx.risk_probability:.2f})\n\n"
        f"Retrieved guideline excerpts:\n{excerpts}\n\n"
        f"Return one JSON object matching this JSON schema:\n{schema}\n\n"
        "Generate the structured clinical safety report now."
    )
    if feedback:
        prompt += f"\n\nYour previous answer was rejected: {feedback}\nFix it and answer again."
    return prompt


# --------------------------------------------------------------------------
# LLM clients
# --------------------------------------------------------------------------
class LangChainLLM:
    """OpenAI or Anthropic chat model through LangChain (needs the optional extras + API key)."""

    def __init__(self, provider: str):
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            self.model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            self.model = ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
                                       temperature=0)
        else:
            raise ValueError(f"Unknown provider {provider!r}")

    def generate(self, system_prompt: str, user_prompt: str, context: GenerationContext) -> str:
        content = self.model.invoke([("system", system_prompt), ("user", user_prompt)]).content
        if isinstance(content, list):  # content-block style responses
            content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        return content


class OfflineGuidelineLLM:
    """Deterministic, rule-based stand-in for an LLM - no network, no API key.

    It follows the same contract as a real model: it reasons ONLY over the retrieved
    excerpts (a rule fires only if its guideline sentence was actually retrieved) and
    returns a JSON string that then goes through the same validation as a real LLM's.
    """

    def generate(self, system_prompt: str, user_prompt: str, context: GenerationContext) -> str:
        p, chunks = context.patient, context.chunks
        dose = float(p["Vaccine_Dose_mcg"])
        age, n_cond = int(p["Age"]), int(p["Pre_Existing_Conditions"])
        bio, prior = float(p["Biomarker_Level"]), p["Prior_Reaction_History"]
        steps = [f"ML risk classifier output: {context.risk_label} reaction risk "
                 f"(probability {context.risk_probability:.2f})."]
        cites: list[str] = []
        flags: list[str] = []

        def rule(triggered: bool, pattern: str, message: str) -> None:
            sentence = _find(chunks, pattern) if triggered else None
            if sentence:
                flags.append(message)
                steps.append(message)
                if sentence not in cites:
                    cites.append(sentence)

        bound = extract_max_dose_mcg(chunks)
        rule(bound is not None and dose > bound, r"doses? above \d+",
             f"Candidate dose {dose:.1f} mcg exceeds the {bound:g} mcg guideline bound; "
             "secondary clinician review is required.")
        rule(age >= 65 and n_cond >= 2, r"65 and older",
             f"Age {age} with {n_cond} pre-existing conditions: flagged for secondary clinician review.")
        rule(n_cond >= 3, r"three or more relevant pre-existing",
             f"{n_cond} pre-existing conditions (3 or more) requires secondary review.")
        rule(prior == "Moderate", r"prior history of a moderate reaction",
             "Prior moderate reaction to an mRNA product requires secondary review.")
        elevated = bio >= config.ELEVATED_BIOMARKER_MG_L
        rule(elevated, r"inflammatory biomarker level meaningfully above",
             f"Biomarker {bio:.2f} mg/L is above the project's working 'elevated' threshold "
             f"({config.ELEVATED_BIOMARKER_MG_L} mg/L - an assumption, the guideline gives no number); "
             "dosing should be deferred or use the lowest appropriate tier.")
        rule(bound is not None and dose > bound and elevated, r"exceeding \d+ micrograms combined",
             "High candidate dose combined with an elevated biomarker requires secondary review.")
        if context.risk_label == "High":
            flags.append("high ML risk")
            steps.append("The ML model predicts HIGH reaction risk; withholding the dose pending "
                         "clinician review is the conservative default.")

        monitor = _find(chunks, r"monitored for at least")
        if monitor:
            steps.append("Standard post-administration monitoring applies if a dose is given.")
            cites.append(monitor)
        steps.append("No allergy history is available in the data, so absolute contraindications "
                     "cannot be excluded; a clinician must confirm them.")

        eligible = not flags
        if eligible:
            steps.append("No guideline precaution was triggered and ML risk is low: eligible at the "
                         "candidate dose.")
        else:
            steps.append("Not eligible at this time: one or more precautions require clinician review.")
        return json.dumps({
            "Eligible_For_Vaccine": eligible,
            "Recommended_Dose_mg": round(dose / config.MCG_PER_MG, 4) if eligible else 0.0,
            "Reasoning_Steps": steps,
            "FDA_Safety_Warnings_Cited": cites,
            "Disclaimer": config.DISCLAIMER,
        })


def get_llm_client(provider: str | None = None) -> LLMClient:
    """Provider from argument or LLM_PROVIDER env var: offline (default) | openai | anthropic."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    provider = (provider or os.getenv("LLM_PROVIDER", "offline")).split("#")[0].strip().lower()
    if provider == "offline":
        return OfflineGuidelineLLM()
    key = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider)
    if key is None:
        raise ValueError(f"Unknown LLM provider {provider!r} (use offline, openai or anthropic)")
    if not os.getenv(key):
        raise RuntimeError(f"{key} is not set; put it in .env or use LLM_PROVIDER=offline")
    return LangChainLLM(provider)


# --------------------------------------------------------------------------
# Validation, guardrails, generation
# --------------------------------------------------------------------------
def parse_and_validate(raw: str, chunks: list[str]) -> ClinicalSafetyReport:
    """Schema validation (FR-4.3) plus grounding: every citation must be verbatim retrieved text."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ReportValidationError("response contains no JSON object")
    try:
        report = ClinicalSafetyReport.model_validate_json(text[start:end + 1])
    except ValidationError as exc:
        raise ReportValidationError(f"schema validation failed: {exc}") from exc
    corpus = _norm(" ".join(chunks))
    invented = [c for c in report.FDA_Safety_Warnings_Cited if _norm(c) not in corpus]
    if invented:
        raise ReportValidationError(f"citation(s) not found in retrieved text: {invented}")
    return report


def conservative_report(reason: str) -> ClinicalSafetyReport:
    return ClinicalSafetyReport(
        Eligible_For_Vaccine=False, Recommended_Dose_mg=0.0, Reasoning_Steps=[reason],
        FDA_Safety_Warnings_Cited=[], Disclaimer=config.DISCLAIMER)


def apply_guardrails(report: ClinicalSafetyReport, chunks: list[str]) -> ClinicalSafetyReport:
    """Deterministic post-checks that no LLM output can bypass (FR-4.4, FR-4.5)."""
    updates: dict = {"Disclaimer": config.DISCLAIMER}  # always the canonical notice
    bound = extract_max_dose_mcg(chunks)
    dose_mcg = report.Recommended_Dose_mg * config.MCG_PER_MG
    if bound is not None and dose_mcg > bound + 1e-9:
        updates.update(
            Eligible_For_Vaccine=False, Recommended_Dose_mg=0.0,
            Reasoning_Steps=report.Reasoning_Steps + [
                f"GUARDRAIL: recommended dose {dose_mcg:.1f} mcg exceeds the guideline bound of "
                f"{bound:g} mcg; recommendation rejected and patient marked not eligible."])
    return report.model_copy(update=updates)


@dataclass
class GenerationResult:
    report: ClinicalSafetyReport
    llm_latency_s: float
    attempts: int


def generate_report(patient: dict, risk_label: str, risk_probability: float, chunks: list[str],
                    llm: LLMClient) -> GenerationResult:
    """Generate and validate; retry exactly once on validation failure (FR-4.3)."""
    if not chunks:  # not enough grounding -> conservative default, no LLM call (rule 3)
        return GenerationResult(conservative_report(
            "No guideline text was retrieved, so a safe recommendation cannot be made."), 0.0, 0)
    ctx = GenerationContext(patient, risk_label, risk_probability, chunks)
    feedback, last_error, latency = None, None, 0.0
    for attempt in (1, 2):
        start = time.perf_counter()
        raw = llm.generate(SYSTEM_PROMPT, build_user_prompt(ctx, feedback), ctx)
        latency += time.perf_counter() - start
        try:
            report = parse_and_validate(raw, chunks)
            return GenerationResult(apply_guardrails(report, chunks), latency, attempt)
        except ReportValidationError as exc:
            feedback = last_error = str(exc)
    raise ReportGenerationError(f"LLM output invalid after 1 retry: {last_error}")


# --------------------------------------------------------------------------
# Persistence (FR-4.6)
# --------------------------------------------------------------------------
def render_markdown(report: ClinicalSafetyReport, patient: dict, risk_label: str,
                    risk_probability: float) -> str:
    verdict = "ELIGIBLE" if report.Eligible_For_Vaccine else "NOT ELIGIBLE"
    lines = [
        "# Clinical Safety Report", "", f"> **{report.Disclaimer}**", "",
        f"**Patient:** {patient_summary(patient)}  ",
        f"**ML risk:** {risk_label} (p = {risk_probability:.2f})  ",
        f"**Decision:** {verdict}  ",
        f"**Recommended dose:** {report.Recommended_Dose_mg:g} mg "
        f"({report.Recommended_Dose_mg * config.MCG_PER_MG:g} mcg)", "", "## Reasoning",
    ]
    lines += [f"{i}. {s}" for i, s in enumerate(report.Reasoning_Steps, 1)]
    lines += ["", "## Guideline excerpts cited"]
    lines += [f"> {c}" for c in report.FDA_Safety_Warnings_Cited] or ["_None_"]
    return "\n".join(lines) + "\n"


def save_report(report: ClinicalSafetyReport, patient: dict, risk_label: str,
                risk_probability: float, out_dir: Path = config.REPORT_DIR) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{patient.get('Patient_ID', 'patient')}_{datetime.now():%Y%m%dT%H%M%S%f}"
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report, patient, risk_label, risk_probability),
                       encoding="utf-8")
    return json_path, md_path


# --------------------------------------------------------------------------
# End-to-end for patients
# --------------------------------------------------------------------------
def pick_sample_patients(df: pd.DataFrame, test_ids: list[str], n: int = 5) -> list[dict]:
    """A deliberately diverse handful from the held-out set (high dose, moderate history, ...)."""
    pool = df[df[config.ID_COL].isin(test_ids)]
    masks = [pool["Vaccine_Dose_mcg"] > 100,
             pool[config.CATEGORICAL_COL] == "Moderate",
             (pool["Age"] >= 65) & (pool[config.COUNT_COL] >= 2),
             pool["Biomarker_Level"] >= config.ELEVATED_BIOMARKER_MG_L,
             (pool["Vaccine_Dose_mcg"] <= 100) & (pool[config.COUNT_COL] == 0)
             & (pool[config.CATEGORICAL_COL] == "None")]
    picked: list[str] = []
    for m in masks:
        for pid in pool.loc[m, config.ID_COL]:
            if pid not in picked:
                picked.append(pid)
                break
    for pid in pool[config.ID_COL]:
        if len(picked) >= n:
            break
        if pid not in picked:
            picked.append(pid)
    return pool[pool[config.ID_COL].isin(picked[:n])].to_dict("records")


def run_for_patient(patient: dict, model, index: GuidelineIndex, llm: LLMClient,
                    out_dir: Path = config.REPORT_DIR) -> dict:
    risk_label, prob = predict_risk(model, patient)
    chunks, rag_latency = retrieve_for_patient(index, patient)
    result = generate_report(patient, risk_label, prob, chunks, llm)
    json_path, md_path = save_report(result.report, patient, risk_label, prob, out_dir)
    return {"patient_id": patient.get("Patient_ID"), "risk_label": risk_label,
            "risk_probability": prob, "eligible": result.report.Eligible_For_Vaccine,
            "rag_latency_s": rag_latency, "llm_latency_s": result.llm_latency_s,
            "attempts": result.attempts, "json": str(json_path), "md": str(md_path)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--patient-id", action="append", help="Patient_ID from the clean CSV (repeatable)")
    ap.add_argument("--provider", default=None, help="offline | openai | anthropic")
    args = ap.parse_args()

    df = load_clean()
    model = load_model()
    index = GuidelineIndex.load() if (config.VECTOR_STORE_DIR / "manifest.json").exists() \
        else GuidelineIndex.build()
    llm = get_llm_client(args.provider)
    if args.patient_id:
        patients = df[df[config.ID_COL].isin(args.patient_id)].to_dict("records")
    else:
        for old in config.REPORT_DIR.glob("*.[jm][sd]*"):  # fresh sample set; no stale duplicates
            old.unlink()
        test_ids = pd.read_csv(config.TEST_PREDICTIONS_PATH)[config.ID_COL].tolist()  # IDs only
        patients = pick_sample_patients(df, test_ids)
    runs = [run_for_patient(p, model, index, llm) for p in patients]
    for r in runs:
        print(f"{r['patient_id']}: risk={r['risk_label']} ({r['risk_probability']:.2f}) "
              f"eligible={r['eligible']} rag={r['rag_latency_s']*1000:.1f}ms "
              f"llm={r['llm_latency_s']*1000:.1f}ms -> {Path(r['md']).name}")
    config.LATENCY_LOG_PATH.write_text(json.dumps({
        "llm_client": type(llm).__name__, "retrieval_backend": index.backend,
        "rag_retrieval_s": [r["rag_latency_s"] for r in runs],
        "llm_generation_s": [r["llm_latency_s"] for r in runs],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
