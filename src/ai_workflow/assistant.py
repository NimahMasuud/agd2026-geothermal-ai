"""
assistant.py — GeothermalAIAssistant
=====================================
Auto-detects API backend from environment variables:
  GROQ_API_KEY      → Groq (completely free, no billing needed)
  GEMINI_API_KEY    → Google Gemini
  ANTHROPIC_API_KEY → Claude (paid)
  None set          → MOCK mode (template reports, real data)
"""

import json, os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class WellAssessmentRequest:
    well_id: str
    curves_available: list
    curves_missing: list
    depth_range_m: tuple
    rotliegend_interval_m: tuple
    mean_gr_api: Optional[float]
    mean_rhob_gcc: Optional[float]
    mean_nphi_vv: Optional[float]
    mean_phit: Optional[float]
    permeability_p10_md: Optional[float]
    permeability_p50_md: Optional[float]
    permeability_p90_md: Optional[float]
    flow_rate_p50_m3h: Optional[float]
    temperature_p50_c: Optional[float]
    power_p50_mw: Optional[float]
    feasibility_score: Optional[float]
    feasibility_verdict: Optional[str]
    heat_output_mwth: Optional[float]
    distance_to_district_km: Optional[float]
    is_deviated: bool = False
    notes: str = ""


SYSTEM_PROMPT = """You are an expert geothermal reservoir engineer specialising in
low-enthalpy geothermal systems in the Netherlands. Assess Rotliegend sandstone wells
for district heating and cooling viability. Be technically rigorous and honest about
uncertainties. Always end with a clear recommendation.

Key benchmarks:
- Target: >=10 MWth heating, >=5 MWth cooling for Utrecht district
- Viable permeability: >50 mD (P50)
- Viable flow rate: >50 m3/h (P50)
- Viable temperature: >70 C

Respond ONLY in valid JSON — no markdown, no extra text."""


def _build_prompt(req: WellAssessmentRequest) -> str:
    return f"""Assess geothermal potential of well {req.well_id}.

DATA:
- Deviated: {req.is_deviated}
- Depth range: {req.depth_range_m[0]:.0f}-{req.depth_range_m[1]:.0f} m
- Rotliegend interval: {req.rotliegend_interval_m[0]:.0f}-{req.rotliegend_interval_m[1]:.0f} m
- Distance to district: {req.distance_to_district_km} km
- Curves available: {', '.join(req.curves_available)}
- Curves missing: {', '.join(req.curves_missing) if req.curves_missing else 'None'}
- Mean GR: {req.mean_gr_api} API | RHOB: {req.mean_rhob_gcc} g/cc | NPHI: {req.mean_nphi_vv}
- Mean porosity: {req.mean_phit}
- Permeability P90/P50/P10: {req.permeability_p90_md}/{req.permeability_p50_md}/{req.permeability_p10_md} mD
- Flow rate P50: {req.flow_rate_p50_m3h} m3/h | Temperature P50: {req.temperature_p50_c} C
- Feasibility score: {req.feasibility_score}/100 — {req.feasibility_verdict}
- Heat output: {req.heat_output_mwth} MWth

Respond ONLY with this JSON:
{{
  "well_id": "{req.well_id}",
  "assessment_date": "2026-06-05",
  "data_quality": {{
    "rating": "GOOD|MODERATE|POOR",
    "completeness_pct": number,
    "key_issues": ["string"],
    "imputation_required": ["string"]
  }},
  "petrophysical_interpretation": {{
    "lithology_description": "string",
    "porosity_assessment": "string",
    "net_reservoir_quality": "GOOD|MODERATE|POOR",
    "key_observations": ["string"]
  }},
  "reservoir_potential": {{
    "permeability_assessment": "string",
    "flow_capacity_assessment": "string",
    "thermal_energy_conclusion": "string",
    "meets_heat_target": true,
    "meets_cooling_target": true,
    "confidence_level": "HIGH|MEDIUM|LOW"
  }},
  "risk_register": [
    {{"risk": "string", "severity": "HIGH|MEDIUM|LOW", "mitigation": "string"}}
  ],
  "recommendation": {{
    "action": "DEVELOP|INVESTIGATE_FURTHER|REJECT",
    "rationale": "string",
    "next_steps": ["string"],
    "priority_rank": number
  }}
}}"""


def _mock_assessment(well_id, score, verdict) -> dict:
    score = score or 0
    action = "DEVELOP" if score >= 50 else "INVESTIGATE_FURTHER" if score >= 25 else "REJECT"
    return {
        "well_id": well_id, "assessment_date": "2026-06-05",
        "data_quality": {"rating": "MODERATE", "completeness_pct": 75,
            "key_issues": ["Some curves missing — ML imputation applied"],
            "imputation_required": ["DTS", "RS"]},
        "petrophysical_interpretation": {
            "lithology_description": "Rotliegend sandstone with variable clay content",
            "porosity_assessment": "Moderate porosity based on NPHI/RHOB crossplot",
            "net_reservoir_quality": "MODERATE",
            "key_observations": ["ML imputation used for missing curves"]},
        "reservoir_potential": {
            "permeability_assessment": "ML-derived P50 permeability assessed from logs",
            "flow_capacity_assessment": "Based on ThermoGIS P50 flow estimate",
            "thermal_energy_conclusion": f"Feasibility score {score}/100 — {verdict}",
            "meets_heat_target": score >= 50,
            "meets_cooling_target": score >= 40,
            "confidence_level": "MEDIUM"},
        "risk_register": [
            {"risk": "Data gaps in key curves", "severity": "MEDIUM",
             "mitigation": "ML imputation applied; validate with new logging run"},
            {"risk": "Flow rate uncertainty", "severity": "HIGH",
             "mitigation": "Conduct well test to confirm ThermoGIS estimates"}],
        "recommendation": {
            "action": action,
            "rationale": f"Composite feasibility score of {score}/100 ({verdict})",
            "next_steps": ["Conduct well test", "Review formation water chemistry",
                           "Detailed economic modelling at confirmed flow rates"],
            "priority_rank": 1 if action == "DEVELOP" else 2 if action == "INVESTIGATE_FURTHER" else 3}
    }


class GeothermalAIAssistant:
    """
    Auto-detects API backend from environment variables.
    Priority: GROQ_API_KEY > GEMINI_API_KEY > ANTHROPIC_API_KEY > MOCK
    """

    def __init__(self):
        self.groq_key      = os.environ.get("GROQ_API_KEY", "")
        self.gemini_key    = os.environ.get("GEMINI_API_KEY", "")
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

        if self.groq_key:
            self.backend = "groq"
            print("Backend: Groq — llama-3.3-70b-versatile (FREE)")
        elif self.gemini_key:
            self.backend = "gemini"
            print("Backend: Google Gemini")
        elif self.anthropic_key:
            self.backend = "anthropic"
            print("Backend: Anthropic Claude")
        else:
            self.backend = "mock"
            print("Backend: MOCK mode (set GROQ_API_KEY for free AI reports)")

    def _call_groq(self, prompt: str) -> str:
        # Uses official Groq SDK — more reliable than raw HTTP
        from groq import Groq
        client = Groq(api_key=self.groq_key)
        chat = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        return chat.choices[0].message.content.strip()

    def _call_gemini(self, prompt: str) -> str:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=self.gemini_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, temperature=0.2)
        )
        return response.text.strip()

    def _call_anthropic(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.anthropic_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

    def assess_well(self, request: WellAssessmentRequest) -> "AssessmentReport":
        print(f"  Assessing {request.well_id} via {self.backend}...")

        if self.backend == "mock":
            data = _mock_assessment(request.well_id, request.feasibility_score,
                                    request.feasibility_verdict)
            return AssessmentReport(well_id=request.well_id, request=request,
                                    assessment=data, raw_response=json.dumps(data))
        try:
            prompt = _build_prompt(request)
            if self.backend == "groq":
                raw = self._call_groq(prompt)
            elif self.backend == "gemini":
                raw = self._call_gemini(prompt)
            else:
                raw = self._call_anthropic(prompt)

            clean = raw.replace("```json", "").replace("```", "").strip()
            data  = json.loads(clean)
            print(f"    ✅ AI report generated successfully")

        except Exception as e:
            print(f"    API error: {e}")
            print(f"    → Falling back to MOCK report for {request.well_id}")
            data = _mock_assessment(request.well_id, request.feasibility_score,
                                    request.feasibility_verdict)
            raw  = json.dumps(data)

        return AssessmentReport(well_id=request.well_id, request=request,
                                assessment=data, raw_response=raw)

    def assess_all_wells(self, requests) -> list:
        reports = [self.assess_well(req) for req in requests]
        reports.sort(key=lambda r: r.assessment.get("recommendation", {}).get("priority_rank", 99))
        return reports

    def generate_comparison_table(self, reports) -> pd.DataFrame:
        rows = []
        for r in reports:
            rec = r.assessment.get("recommendation", {})
            pot = r.assessment.get("reservoir_potential", {})
            dq  = r.assessment.get("data_quality", {})
            rows.append({
                "Well":              r.well_id,
                "Action":            rec.get("action", "N/A"),
                "Priority":          rec.get("priority_rank", 99),
                "Meets Heat Target": pot.get("meets_heat_target", False),
                "Meets Cool Target": pot.get("meets_cooling_target", False),
                "Confidence":        pot.get("confidence_level", "N/A"),
                "Data Quality":      dq.get("rating", "N/A"),
                "Rationale":         str(rec.get("rationale", ""))[:80] + "..."
            })
        return pd.DataFrame(rows).sort_values("Priority")


@dataclass
class AssessmentReport:
    well_id: str
    request: WellAssessmentRequest
    assessment: dict
    raw_response: str

    @property
    def recommendation(self) -> str:
        return self.assessment.get("recommendation", {}).get("action", "N/A")

    @property
    def full_text(self) -> str:
        return json.dumps(self.assessment, indent=2)

    def to_markdown(self) -> str:
        a = self.assessment
        lines = [f"# Geothermal Well Assessment — {self.well_id}",
                 f"**Date:** {a.get('assessment_date','N/A')}\n"]
        dq = a.get("data_quality", {})
        lines += ["## 1. Data Quality",
                  f"**Rating:** {dq.get('rating')} | **Completeness:** {dq.get('completeness_pct')}%"]
        for i in dq.get("key_issues", []): lines.append(f"- {i}")
        pi = a.get("petrophysical_interpretation", {})
        lines += ["\n## 2. Petrophysical Interpretation",
                  pi.get("lithology_description",""),
                  f"**Porosity:** {pi.get('porosity_assessment','')}",
                  f"**Net reservoir quality:** {pi.get('net_reservoir_quality','')}"]
        rp = a.get("reservoir_potential", {})
        lines += ["\n## 3. Reservoir Potential",
                  rp.get("thermal_energy_conclusion",""),
                  f"**Meets 10 MWth heat target:** {rp.get('meets_heat_target')}",
                  f"**Meets 5 MWth cooling target:** {rp.get('meets_cooling_target')}",
                  f"**Confidence:** {rp.get('confidence_level')}"]
        lines.append("\n## 4. Risk Register")
        for r in a.get("risk_register", []):
            lines.append(f"- **[{r.get('severity')}]** {r.get('risk')} → *{r.get('mitigation')}*")
        rec = a.get("recommendation", {})
        lines += [f"\n## 5. Recommendation: **{rec.get('action')}**",
                  rec.get("rationale",""), "\n**Next steps:**"]
        for s in rec.get("next_steps", []): lines.append(f"1. {s}")
        return "\n".join(lines)

    def save(self, path, format: str = "md"):
        path = Path(path)
        path.write_text(self.to_markdown() if format == "md" else self.full_text,
                        encoding="utf-8")
        print(f"    Saved: {path}")


def build_request_from_dataframes(well_id, unified_df, scores_df, thermogis_df):
    raise NotImplementedError("Use WellAssessmentRequest directly in notebook 08 cell 5.")
